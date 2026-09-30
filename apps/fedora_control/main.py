#!/usr/bin/env python3
"""Tletl v5 — app de control de Fedora. CLIENTE DELGADO.

NO contiene lógica de clasificación, guard, critic ni temporal: todo eso vive en
tletl_core.pipeline. Esta app solo: abre cámara, realza baja luz, corre el
detector de manos (apps.common.hand_tracker, dual-hand), pide al pipeline el
gesto estable de cada mano, traduce el gesto de la mano dominante a llamadas de
FedoraActions, dibuja el panel y escribe el bus.

La mano dominante controla Fedora; la otra (mod) se escribe al bus para Blender.

mediapipe NO se importa aquí: lo hace apps/common/hand_tracker.py de forma lazy
al crear el backend. Así el módulo se importa (y smoke-testea) sin mediapipe.

Decisiones de v5.2 (todas con test en tests/test_fedora_app.py):
  - CLI > config > defaults: los argumentos tienen default None y caen a
    config/tletl.toml ([camera], [tracker], [fedora], [paths], [bus]).
  - Pausa de seguridad por FIST: exige FIST estable durante [fedora].safety_hold
    (o conf >= 0.90 durante la mitad). Antes bastaba UN frame y agarrar la taza
    pausaba el control (VALIDACION_FISICA_v5.md §3).
  - Modo CURSOR: tap vs drag con [fedora].tap_max / drag_min en una máquina de
    estados pura (PinchClickDrag). Con 0.24/0.32 s fijos y ~15 fps el filtro
    temporal hacía que un PINCH estable durara >= 0.3 s: nunca había click.
  - --headless: sin ventana (para alimentar el bus de Blender/CAD).
  - [camera].proc_width: el detector ve un frame reducido; los landmarks son
    normalizados así que nada más cambia.
"""

from __future__ import annotations

import argparse
import inspect
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tletl_core import __version__ as CORE_VERSION
from tletl_core.bus import TletlStateBus
from tletl_core.config import bank_path_from_config, bus_path_from_config, load_config
from tletl_core.features import extract_live_features
from tletl_core.geometry import palm_center
from tletl_core.intent import gesture_to_common_intent
from tletl_core.lowlight import LowLightEnhancer
from tletl_core.pipeline import HandResult, TletlPipeline
from tletl_core.state import TletlFrameState, TletlHandState
from tletl_core.temporal import CursorDelta, Hold, MotionTracker
from apps.common.gesture_hint import GestureHint, hint_for
from apps.common.hand_tracker import (
    HandDetection,
    HandTracker,
    describe_backend,
    draw_landmarks,
    format_backend_summary,
)
from apps.fedora_control.actions import FedoraActions

APP = "Tletl v5 Core Fedora"
MODES = ["NAVEGADOR", "CURSOR", "VENTANAS"]
CLICK_GAP = 0.28            # s mínimos entre dos clicks del modo CURSOR (anti doble click)
SAFETY_FAST_CONF = 0.90     # FIST con esta confianza pausa a la mitad de safety_hold
DEFAULT_FOURCC = "MJPG"     # formato de captura V4L2 (muchas webcams solo dan 30 fps a 720p en MJPG)
HEADLESS_STATUS_EVERY = 2.0  # s entre líneas de estado en terminal cuando no hay ventana


# ---------------------------------------------------------------------------
# Piezas puras (testeables sin cámara)
# ---------------------------------------------------------------------------

def draw_panel(frame: np.ndarray, lines: Sequence[str], color: Tuple[int, int, int], *,
               zones: bool = True) -> None:
    """Panel semitransparente con `lines`. `zones` dibuja las franjas del modo NAVEGADOR
    (scroll arriba/abajo al 39 %/61 % de la altura); solo tiene sentido en ese modo."""
    h, w = frame.shape[:2]
    panel_h = 22 + 25 * max(1, len(lines))
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, min(h - 10, 10 + panel_h)), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    y = 35
    for i, line in enumerate(lines):
        c = color if i == 0 else (255, 255, 255)
        cv2.putText(frame, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 2, cv2.LINE_AA)
        y += 25
    if zones:
        cv2.line(frame, (0, int(h * 0.39)), (w, int(h * 0.39)), (255, 255, 0), 1)
        cv2.line(frame, (0, int(h * 0.61)), (w, int(h * 0.61)), (255, 255, 0), 1)


def split_hands(detections: Sequence[HandDetection], dominant_label: str
                ) -> Tuple[Optional[HandDetection], Optional[HandDetection]]:
    """Separa las manos detectadas en (dom, mod).

    dom = la mano cuya lateralidad coincide con `dominant_label` (si hay varias,
    la de mayor score); si ninguna coincide, la primera. mod = la primera mano
    distinta de dom (nunca el mismo objeto). Una sola mano -> siempre dom.
    """
    dets = list(detections or [])
    if not dets:
        return None, None
    matches = [i for i, d in enumerate(dets) if d.handedness == dominant_label]
    dom_idx = max(matches, key=lambda i: dets[i].score) if matches else 0
    dom = dets[dom_idx]
    mod = next((d for i, d in enumerate(dets) if i != dom_idx), None)
    return dom, mod


def detector_size(width: int, height: int, proc_width: int) -> Tuple[int, int]:
    """Tamaño al que se reduce el frame SOLO para el detector (aspecto intacto).

    proc_width <= 0 o frame ya más estrecho -> sin cambio. Los landmarks salen
    normalizados (0..1), así que las features, el panel y el bus no se enteran.
    Ganancia: se ahorra la conversión BGR->RGB, el realce de baja luz y la copia
    a 1280x720 (~4x menos píxeles); el HandLandmarker ya redimensiona por dentro,
    así que su tiempo apenas cambia. Medido aquí: ~1-3 ms/frame a 720p.
    """
    if proc_width <= 0 or width <= proc_width:
        return int(width), int(height)
    scale = proc_width / float(width)
    return int(proc_width), max(1, int(round(height * scale)))


def safety_pause_due(progress: float, conf: float, *, fast_conf: float = SAFETY_FAST_CONF) -> bool:
    """Pausa de seguridad: FIST estable durante TODO safety_hold (progress >= 1), o
    FIST muy seguro (conf >= fast_conf) durante al menos la mitad. Un puño de un
    solo frame (agarrar la taza) ya no pausa."""
    return progress >= 1.0 or (conf >= fast_conf and progress >= 0.5)


class PinchClickDrag:
    """Máquina de estados tap-vs-drag del modo CURSOR (pura, reloj inyectable).

    Llama `update(pinching)` en CADA frame con el PINCH estable de la mano dominante:
      - PINCH que dura <= tap_max s          -> "click" al soltar
      - PINCH que alcanza drag_min s         -> "drag_on" (mouse_down) y "drag_off" al soltar
      - entre tap_max y drag_min             -> nada (zona muerta, evita clicks accidentales)
      - dos clicks a menos de click_gap s    -> el segundo se ignora
    El tiempo se cuenta desde el PRIMER frame en que el PINCH es estable (el
    llamador pasa el gesto estable del filtro temporal, no el crudo).
    `release()` fuerza el fin (mano perdida, control OFF, cambio de modo).
    """

    def __init__(self, tap_max: float = 0.35, drag_min: float = 0.40, click_gap: float = CLICK_GAP,
                 clock: Callable[[], float] = time.monotonic):
        self.tap_max = float(tap_max)
        self.drag_min = float(drag_min)
        self.click_gap = float(click_gap)
        self._clock = clock
        self.pinch_t0: Optional[float] = None
        self.dragging = False
        self.last_click = -1e9

    def update(self, pinching: bool) -> Optional[str]:
        now = self._clock()
        if pinching:
            if self.pinch_t0 is None:
                self.pinch_t0 = now
                return None
            if not self.dragging and now - self.pinch_t0 >= self.drag_min:
                self.dragging = True
                return "drag_on"
            return None
        if self.pinch_t0 is None:
            return None
        held = now - self.pinch_t0
        self.pinch_t0 = None
        if self.dragging:
            self.dragging = False
            return "drag_off"
        if held <= self.tap_max and now - self.last_click >= self.click_gap:
            self.last_click = now
            return "click"
        return None

    def release(self) -> Optional[str]:
        self.pinch_t0 = None
        if self.dragging:
            self.dragging = False
            return "drag_off"
        return None


class FpsMeter:
    """FPS suavizado (EMA) para el panel; el instantáneo salta demasiado."""

    def __init__(self, alpha: float = 0.15):
        self.alpha = float(alpha)
        self.prev: Optional[float] = None
        self.fps = 0.0

    def tick(self, now: float) -> float:
        if self.prev is not None:
            inst = 1.0 / max(now - self.prev, 1e-6)
            self.fps = inst if self.fps <= 0 else (1 - self.alpha) * self.fps + self.alpha * inst
        self.prev = now
        return self.fps


def make_process_features(pipeline: Any) -> Callable[[Dict[str, float], str, Optional[str]], HandResult]:
    """Devuelve process(features, hand, hint) que pasa `hint=` SOLO si el pipeline
    lo acepta (la firma nueva es process_features(features, hand, hint=None); la
    vieja no tiene hint). Se inspecciona una vez al arrancar."""
    accepts_hint = False
    try:
        params = inspect.signature(pipeline.process_features).parameters
        accepts_hint = "hint" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        accepts_hint = False

    def process(features: Dict[str, float], hand: str, hint: Optional[str] = None) -> HandResult:
        if hint is not None and accepts_hint:
            return pipeline.process_features(features, hand=hand, hint=hint)
        return pipeline.process_features(features, hand=hand)

    return process


def apply_cli_overrides(args: argparse.Namespace, cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """CLI > env/toml. Escribe los overrides no-None en cfg y devuelve cfg.

    Antes argparse tenía sus propios defaults (960x540, det_conf 0.72...) y
    [camera]/[tracker] del toml se ignoraban en silencio.
    """
    mapping = (
        ("camera", "camera", "index"), ("width", "camera", "width"), ("height", "camera", "height"),
        ("fps", "camera", "fps"), ("proc_width", "camera", "proc_width"), ("fourcc", "camera", "fourcc"),
        ("det_conf", "tracker", "det_conf"), ("track_conf", "tracker", "track_conf"),
        ("backend", "tracker", "backend"), ("action_conf", "fedora", "action_conf"),
    )
    for attr, section, key in mapping:
        value = getattr(args, attr, None)
        if value is not None:
            cfg.setdefault(section, {})[key] = value
    if getattr(args, "k", None):
        cfg["classifier"]["k"] = int(args.k)
    if getattr(args, "dry_run", False):
        cfg["fedora"]["dry_run"] = True
    if getattr(args, "gesture_hint", False):
        cfg["tracker"]["gesture_hint"] = True
    if getattr(args, "window", False):
        cfg["camera"]["headless"] = False
    elif getattr(args, "headless", False):
        cfg["camera"]["headless"] = True
    return cfg


def effective_bank_path(args: argparse.Namespace, cfg: Dict[str, Dict[str, Any]]) -> Path:
    """--bank (relativo al cwd) > [paths].bank / TLETL_GESTURE_BANK > datasets/ del repo."""
    if getattr(args, "bank", None):
        return Path(args.bank).expanduser().resolve()
    return bank_path_from_config(cfg)


def short_path(path: Path) -> str:
    """~ en vez del HOME para que quepa en el panel."""
    try:
        return "~/" + str(Path(path).relative_to(Path.home()))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Cámara
# ---------------------------------------------------------------------------

def open_camera(index: int, width: int, height: int, fps: int, fourcc: str = DEFAULT_FOURCC):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir cámara {index}. Prueba --camera 1 o --camera 2.")
    # El formato va ANTES del tamaño/fps: en V4L2 decide qué fps ofrece la cámara
    # (YUYV suele topar en ~10-15 fps a 720p; MJPG da 30). Si la cámara no lo
    # soporta, set() falla en silencio y se queda el formato por defecto.
    if fourcc and len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def camera_report(cap: Any) -> str:
    """'1280x720@30 MJPG' con lo que la cámara aceptó de verdad (no lo pedido)."""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    code = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ") or "?"
    return f"{w}x{h}@{fps:.0f} {fourcc}"


def _hand_state_from(pipeline: TletlPipeline, res: HandResult, side: str,
                     palm: Optional[Tuple[float, float]], feats: Dict[str, float]) -> TletlHandState:
    hs = pipeline.to_hand_state(res, side=side, features=feats)
    hs.palm = palm
    return hs


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    cfg = apply_cli_overrides(args, load_config(args.config))
    cam, trk, fed = cfg["camera"], cfg["tracker"], cfg["fedora"]
    dominant = fed["dominant_hand"]
    dry_run = bool(fed["dry_run"])
    headless = bool(cam.get("headless", False))
    proc_width = int(cam.get("proc_width", 0) or 0)
    action_conf = float(fed["action_conf"])
    safety_hold = float(fed["safety_hold"])
    bus_path = bus_path_from_config(cfg)
    bank = effective_bank_path(args, cfg)
    frame_period = 1.0 / max(float(cam["fps"]), 1.0)

    pipeline = TletlPipeline(bank, config=cfg)
    process = make_process_features(pipeline)
    lowlight = LowLightEnhancer(clip_limit=cfg["lowlight"]["clip_limit"],
                                gamma_dark=cfg["lowlight"]["gamma_dark"]) if cfg["lowlight"]["enabled"] else None

    tracker: Optional[HandTracker] = None
    hint: Optional[GestureHint] = None
    cap = None
    bus: Optional[TletlStateBus] = None
    actions = FedoraActions(dry_run=dry_run)
    pinch = PinchClickDrag(tap_max=float(fed["tap_max"]), drag_min=float(fed["drag_min"]))

    def release_pinch(reason: str) -> None:
        nonlocal last_action
        if pinch.release() == "drag_off":
            actions.mouse_up()
            last_action = f"Drag OFF ({reason})"

    motion = MotionTracker()
    cursor = CursorDelta()
    hold_toggle = Hold()
    hold_mode = Hold()
    hold_safety = Hold()
    fps_meter = FpsMeter()

    mode_i = 0
    control = False
    last_action = "-"
    last_click = last_scroll = last_tab = last_big = last_toggle = 0.0
    bus_errors = 0
    last_bus_ok = 0.0
    last_status = 0.0
    window = APP
    window_sized = False

    try:
        # Resumen puro ANTES de crear nada: si algo falla después, el usuario ya
        # sabe qué backend/modelo se intentó (y puede confirmar backend=tasks).
        print(f"[TLETL] {format_backend_summary(describe_backend(trk))}")
        tracker = HandTracker.create(trk, num_hands=int(cam["max_num_hands"]))
        hint = GestureHint.create(trk, num_hands=int(cam["max_num_hands"]))
        cap = open_camera(int(cam["index"]), int(cam["width"]), int(cam["height"]), int(cam["fps"]),
                          str(cam.get("fourcc", DEFAULT_FOURCC) or ""))
        bus = TletlStateBus(bus_path, include_features=bool(cfg["bus"]["include_features"]),
                            udp_target=cfg["bus"]["udp_target"] or None)

        print(f"[TLETL] core {CORE_VERSION} | backend:{tracker.backend_name} | hint:{'on' if hint else 'off'} "
              f"| cámara {cam['index']}: pedido {cam['width']}x{cam['height']}@{cam['fps']}, "
              f"obtenido {camera_report(cap)} | proc_width:{proc_width or 'off'}")
        print(f"[TLETL] bus -> {bus_path}" + (f" (+UDP {cfg['bus']['udp_target']})" if cfg["bus"]["udp_target"] else "")
              + f" | banco: {bank}")
        if headless:
            print("[TLETL] headless: sin ventana. Atajos de teclado (q/t/m) NO disponibles; Ctrl+C para salir.")
        else:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            loop_t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print("No pude leer frame.")
                break
            frame = cv2.flip(frame, 1)   # espejo: la lateralidad de MediaPipe asume selfie (ver hand_tracker)
            now = time.time()
            fps = fps_meter.tick(now)

            # Frame para el detector: reducido a proc_width + realce de baja luz.
            # La ventana muestra el frame completo sin realzar (el realce es para el detector).
            pw, ph = detector_size(frame.shape[1], frame.shape[0], proc_width)
            proc = frame if (pw, ph) == (frame.shape[1], frame.shape[0]) else \
                cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
            luma_mode = "off"
            if lowlight is not None:
                proc = lowlight.enhance(proc)
                luma_mode = lowlight.last_mode
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

            detections = tracker.process(rgb)
            hints = hint.process(rgb, tracker.last_timestamp_ms) if hint is not None else []
            dom_det, mod_det = split_hands(detections, dominant)

            dom_res = HandResult()
            mod_res = HandResult()
            dom_state = TletlHandState()
            mod_state = TletlHandState()
            cx = cy = 0.0
            swipe = None
            dom_hint = mod_hint = None
            hand_seen = dom_det is not None

            if dom_det is not None:
                lm = dom_det.landmarks
                feats = extract_live_features(lm)
                cx, cy = palm_center(lm)
                motion.update(cx, cy)
                swipe = motion.swipe()
                dom_hint = hint_for(hints, dom_det.handedness)
                dom_res = process(feats, "dom", dom_hint)
                dom_state = _hand_state_from(pipeline, dom_res, dom_det.handedness, (cx, cy), feats)
                if not headless:
                    draw_landmarks(frame, dom_det)
                    px, py = int(cx * frame.shape[1]), int(cy * frame.shape[0])
                    cv2.circle(frame, (px, py), 12, (0, 255, 255), 2)
                    cv2.putText(frame, f"{dom_res.stable_gesture} {dom_res.confidence:.2f}",
                                (px - 70, py - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                motion.reset(); cursor.reset(); hold_toggle.reset(); hold_mode.reset(); hold_safety.reset()
                pipeline.reset("dom")
                release_pinch("mano perdida")

            if mod_det is not None:
                lm2 = mod_det.landmarks
                feats2 = extract_live_features(lm2)
                mcx, mcy = palm_center(lm2)
                mod_hint = hint_for(hints, mod_det.handedness)
                mod_res = process(feats2, "mod", mod_hint)
                mod_state = _hand_state_from(pipeline, mod_res, mod_det.handedness, (mcx, mcy), feats2)
                if not headless:
                    draw_landmarks(frame, mod_det, color=(200, 120, 0))
            else:
                pipeline.reset("mod")

            g = dom_res.stable_gesture
            conf = dom_res.confidence

            # hold OPEN_PALM = toggle control
            toggle_progress = 0.0
            if hand_seen and g == "OPEN_PALM":
                required = 2.0 if control else 1.25
                toggle_progress = hold_toggle.progress("OPEN_PALM", required)
                if toggle_progress >= 1.0 and now - last_toggle > 1.0:
                    control = not control
                    last_toggle = now
                    hold_toggle.reset(); pipeline.reset("dom")
                    last_action = "Control ON" if control else "Control OFF"
                    if not control:
                        release_pinch("control OFF")
                    print(f"[TLETL] {last_action}")
            else:
                hold_toggle.reset()

            # Pausa de seguridad: FIST sostenido (safety_hold), no un puño de un frame.
            safety_progress = 0.0
            if control and g == "FIST" and conf > 0.55:
                safety_progress = hold_safety.progress("FIST", safety_hold)
                if safety_pause_due(safety_progress, conf):
                    control = False
                    last_action = "Pausa seguridad"
                    hold_safety.reset()
                    release_pinch("pausa seguridad")
                    print("[TLETL] Pausa de seguridad (FIST sostenido)")
            else:
                hold_safety.reset()

            # THREE = cambiar modo
            mode_progress = 0.0
            if control and g == "THREE":
                mode_progress = hold_mode.progress("THREE", 0.85)
                if mode_progress >= 1.0 and now - last_big > 1.0:
                    mode_i = (mode_i + 1) % len(MODES)
                    last_big = now
                    hold_mode.reset(); pipeline.reset("dom"); cursor.reset()
                    release_pinch("cambio de modo")
                    last_action = f"Modo {MODES[mode_i]}"
            else:
                hold_mode.reset()

            mode = MODES[mode_i]
            can_act = control and hand_seen and dom_res.ok and conf >= action_conf
            pinching = False

            if can_act:
                if mode == "NAVEGADOR":
                    if g == "POINT" and now - last_scroll > 0.11:
                        if cy < 0.22:
                            actions.page_up(); last_action = "PageUp"; last_scroll = now + 0.12
                        elif cy > 0.78:
                            actions.page_down(); last_action = "PageDown"; last_scroll = now + 0.12
                        elif cy < 0.39:
                            actions.scroll_up(); last_action = "Scroll arriba"; last_scroll = now
                        elif cy > 0.61:
                            actions.scroll_down(); last_action = "Scroll abajo"; last_scroll = now
                    elif g == "PINCH" and now - last_click > 0.34:
                        actions.click_left(); last_action = "Click"; last_click = now
                    elif g == "VICTORY" and swipe in {"LEFT", "RIGHT"} and now - last_tab > 0.65:
                        actions.next_tab() if swipe == "RIGHT" else actions.prev_tab()
                        last_action = "Tab siguiente" if swipe == "RIGHT" else "Tab anterior"
                        last_tab = now
                    elif g == "OPEN_PALM" and swipe in {"LEFT", "RIGHT"} and toggle_progress < 0.55 and now - last_tab > 0.75:
                        actions.back() if swipe == "LEFT" else actions.forward()
                        last_action = "Atrás" if swipe == "LEFT" else "Adelante"
                        last_tab = now

                elif mode == "CURSOR":
                    if g in {"POINT", "OPEN_PALM", "PINCH"}:
                        dx, dy = cursor.update(cx, cy)
                        actions.mousemove(dx, dy)
                    else:
                        cursor.reset()
                    pinching = g == "PINCH"

                elif mode == "VENTANAS":
                    if g == "OPEN_PALM" and swipe == "UP" and now - last_big > 0.9:
                        actions.overview(); last_big = now; last_action = "Overview"
                    elif g == "PINCH" and now - last_click > 0.38:
                        actions.enter(); last_click = now; last_action = "Enter"
            else:
                cursor.reset()
                # Un arrastre en curso sobrevive a frames inestables (ok=False) mientras
                # el gesto estable siga siendo PINCH; así no se suelta a mitad de camino.
                pinching = pinch.dragging and control and hand_seen and mode == "CURSOR" and g == "PINCH"

            event = pinch.update(pinching)
            if event == "click":
                actions.click_left(); last_action = "Click"
            elif event == "drag_on":
                actions.mouse_down(); last_action = "Drag ON"
            elif event == "drag_off":
                actions.mouse_up(); last_action = "Drag OFF"

            # construir state dom + mod y escribir bus
            state = TletlFrameState(
                version=5, app_version=f"{CORE_VERSION}-core-fedora", timestamp=now,
                frame_width=frame.shape[1], frame_height=frame.shape[0], fps=fps,
                dom=dom_state, mod=mod_state, mode=mode, action=last_action,
                selected=control, grabbed=(g == "PINCH" and dom_res.ok),
                extra={"swipe": swipe or "-", "control": control, "lowlight": luma_mode,
                       "backend": tracker.backend_name, "hint": dom_hint or "", "headless": headless},
            )
            state.intent = gesture_to_common_intent(state)
            try:
                bus.write(state)
                last_bus_ok = now
            except OSError as exc:
                bus_errors += 1
                if bus_errors == 1:
                    print(f"[TLETL] ERROR escribiendo el bus {bus_path}: {exc} (se cuenta en el panel)")
            bus_age_ms = int((now - last_bus_ok) * 1000) if last_bus_ok else -1

            if headless:
                if now - last_status >= HEADLESS_STATUS_EVERY:
                    last_status = now
                    print(f"[TLETL] {'ON ' if control else 'OFF'} {mode} fps:{fps:4.1f} "
                          f"dom:{g}:{conf:.2f}{'*' if dom_res.ok else ''} mod:{mod_res.stable_gesture} "
                          f"hint:{dom_hint or '-'} intent:{state.intent.name} bus_err:{bus_errors}")
                # cap del bucle a los fps de la cámara (si la cámara entrega más rápido)
                sleep_for = frame_period - (time.time() - loop_t0)
                if sleep_for > 0:
                    time.sleep(sleep_for)
                continue

            state_txt = "ON" if control else "OFF"
            color = (0, 255, 0) if control else (0, 0, 255)
            lines = [
                f"TLETL v5 {state_txt} | MODO:{mode} | FPS:{fps:.1f} | DRY:{dry_run} | LowLight:{luma_mode} | backend:{tracker.backend_name}",
                f"DOM[{dom_state.side}]:{g} raw:{dom_res.raw_gesture} conf:{conf:.2f} ok:{dom_res.ok} hint:{dom_hint or '-'}",
                (f"MOD[{mod_state.side}]:{mod_res.stable_gesture} hint:{mod_hint or '-'} (al bus para Blender)"
                 if mod_state.present else "MOD: (sin segunda mano)"),
                f"Guard:{dom_res.guard_reason} | Critic:{dom_res.critic_reason} | Intent:{state.intent.name} | Swipe:{swipe or '-'}",
                f"Action:{last_action} | Bus:{short_path(bus_path)} age:{bus_age_ms}ms err:{bus_errors}",
                "OPEN_PALM hold=ON/OFF | FIST hold=seguridad | THREE=modo | Q/ESC=salir | t=toggle | m=modo",
            ]
            if toggle_progress > 0:
                lines.append(f"Toggle: {int(toggle_progress * 100)}%")
            if mode_progress > 0:
                lines.append(f"Mode: {int(mode_progress * 100)}%")
            if safety_progress > 0:
                lines.append(f"Seguridad (FIST): {int(safety_progress * 100)}%")
            draw_panel(frame, lines, color, zones=(mode == "NAVEGADOR"))
            cv2.circle(frame, (frame.shape[1] - 34, 34), 17, color, -1)

            if not window_sized:
                cv2.resizeWindow(window, frame.shape[1], frame.shape[0])
                window_sized = True
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                break
            if key == ord("t"):
                control = not control
                last_action = "Control ON tecla" if control else "Control OFF tecla"
                hold_toggle.reset()
                if not control:
                    release_pinch("control OFF tecla")
            elif key == ord("m"):
                mode_i = (mode_i + 1) % len(MODES)
                cursor.reset()
                release_pinch("cambio de modo")
                last_action = f"Modo {MODES[mode_i]} tecla"
    finally:
        # Pase lo que pase (excepción incluida) el botón del ratón no se queda pulsado
        # y la cámara/ventana/modelos se liberan.
        if pinch.dragging:
            actions.mouse_up()
            pinch.release()
        if cap is not None:
            cap.release()
        if not headless:
            cv2.destroyAllWindows()
        if hint is not None:
            hint.close()
        if tracker is not None:
            tracker.close()
        if bus is not None:
            bus.close()
        if hasattr(pipeline, "close"):
            pipeline.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=APP,
                                 epilog="Todo default viene de config/tletl.toml (y env TLETL_*); la CLI lo sobreescribe.")
    ap.add_argument("--bank", default=None, help="banco JSONL (default: [paths].bank / TLETL_GESTURE_BANK / datasets/)")
    ap.add_argument("--config", default=None, help="ruta a tletl.toml (default: config/tletl.toml)")
    ap.add_argument("--camera", type=int, default=None, help="índice de cámara ([camera].index / TLETL_CAMERA)")
    ap.add_argument("--width", type=int, default=None, help="[camera].width")
    ap.add_argument("--height", type=int, default=None, help="[camera].height")
    ap.add_argument("--fps", type=int, default=None, help="[camera].fps")
    ap.add_argument("--proc-width", type=int, default=None, help="ancho del frame para el detector, 0 = completo ([camera].proc_width)")
    ap.add_argument("--fourcc", default=None, help='formato de captura V4L2, p.ej. MJPG; "" para no tocarlo ([camera].fourcc)')
    ap.add_argument("--backend", choices=["auto", "legacy", "tasks"], default=None,
                    help="detector: auto | legacy (mp.solutions) | tasks (HandLandmarker) ([tracker].backend / TLETL_TRACKER_BACKEND)")
    ap.add_argument("--det-conf", type=float, default=None, help="[tracker].det_conf")
    ap.add_argument("--track-conf", type=float, default=None, help="[tracker].track_conf")
    ap.add_argument("--action-conf", type=float, default=None, help="confianza mínima para actuar ([fedora].action_conf)")
    ap.add_argument("--k", type=int, default=None, help="override de k del KNN (default: [classifier].k)")
    ap.add_argument("--dry-run", action="store_true", help="imprime las acciones sin ejecutarlas (TLETL_DRY_RUN)")
    ap.add_argument("--headless", action="store_true",
                    help="sin ventana OpenCV: solo procesa y escribe el bus ([camera].headless / TLETL_HEADLESS)")
    ap.add_argument("--window", action="store_true", help="fuerza la ventana aunque config/launcher pidan headless")
    ap.add_argument("--gesture-hint", action="store_true",
                    help="segunda opinión del Gesture Recognizer ([tracker].gesture_hint / TLETL_GESTURE_HINT)")
    return ap


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except KeyboardInterrupt:
        print("\n[TLETL] Interrumpido.")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        print("Tips: prueba --camera 1, ./launchers/tletl-fetch-models.sh si falta el modelo, "
              "o TLETL_DRY_RUN=1 ./launchers/tletl-fedora-safe.sh")
        raise
