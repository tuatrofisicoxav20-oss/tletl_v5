#!/usr/bin/env python3
"""Tletl v5 — app de control de Fedora. CLIENTE DELGADO.

NO contiene lógica de clasificación, guard, critic ni temporal: todo eso vive en
tletl_core.pipeline. Esta app solo: abre cámara, realza baja luz, corre MediaPipe
(dual-hand), pide al pipeline el gesto estable de cada mano, traduce el gesto de la
mano dominante a llamadas de FedoraActions, dibuja el panel y escribe el bus.

La mano dominante controla Fedora; la otra (mod) se escribe al bus para Blender.

mediapipe se importa de forma lazy dentro de run(): así el módulo se puede importar
(y smoke-testear) en entornos sin mediapipe (p.ej. Python 3.14).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from tletl_core.bus import TletlStateBus
from tletl_core.config import load_config
from tletl_core.features import extract_live_features
from tletl_core.geometry import palm_center, points_from_mediapipe
from tletl_core.intent import gesture_to_common_intent
from tletl_core.lowlight import LowLightEnhancer
from tletl_core.pipeline import HandResult, TletlPipeline
from tletl_core.state import TletlFrameState, TletlHandState
from tletl_core.temporal import CursorDelta, Hold, MotionTracker
from apps.fedora_control.actions import FedoraActions

APP = "Tletl v5 Core Fedora"
MODES = ["NAVEGADOR", "CURSOR", "VENTANAS"]


def draw_panel(frame: np.ndarray, lines, color: Tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 232), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    y = 35
    for i, line in enumerate(lines):
        c = color if i == 0 else (255, 255, 255)
        cv2.putText(frame, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 2, cv2.LINE_AA)
        y += 25
    cv2.line(frame, (0, int(h * 0.39)), (w, int(h * 0.39)), (255, 255, 0), 1)
    cv2.line(frame, (0, int(h * 0.61)), (w, int(h * 0.61)), (255, 255, 0), 1)


def open_camera(index: int, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir cámara {index}. Prueba --camera 1 o --camera 2.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def split_hands(result, dominant_label: str):
    """Separa las manos detectadas en (dom_landmarks, mod_landmarks).

    Usa multi_handedness para asignar la mano dominante. Si solo hay una mano,
    va a dom. Devuelve landmarks crudos de MediaPipe (o None).
    """
    lms = result.multi_hand_landmarks or []
    if not lms:
        return None, None
    handed = result.multi_handedness or []

    labels = []
    for i in range(len(lms)):
        lbl = "Right"
        if i < len(handed):
            try:
                lbl = handed[i].classification[0].label
            except Exception:
                lbl = "Right"
        labels.append(lbl)

    # índice de la mano dominante; si ninguna coincide, la primera
    dom_idx = next((i for i, lbl in enumerate(labels) if lbl == dominant_label), 0)
    dom = lms[dom_idx]
    # mod = primera mano distinta de dom (garantiza que nunca sean el mismo landmark)
    mod = next((lms[i] for i in range(len(lms)) if i != dom_idx), None)
    return dom, mod


def _hand_state_from(pipeline: TletlPipeline, res: HandResult, side: str,
                     palm: Optional[Tuple[float, float]], feats: Dict[str, float]) -> TletlHandState:
    hs = pipeline.to_hand_state(res, side=side, features=feats)
    hs.palm = palm
    return hs


def run(args: argparse.Namespace) -> int:
    import mediapipe as mp  # lazy: solo se necesita en vivo

    cfg = load_config(args.config)
    # overrides de CLI sobre config
    if args.k:
        cfg["classifier"]["k"] = args.k
    if args.dry_run:
        cfg["fedora"]["dry_run"] = True
    dominant = cfg["fedora"]["dominant_hand"]
    dry_run = bool(cfg["fedora"]["dry_run"])
    bus_path = os.environ.get("TLETL_STATE_PATH", cfg["blender"]["bus_path"])

    bank = Path(args.bank).expanduser().resolve()
    pipeline = TletlPipeline(bank, config=cfg)
    lowlight = LowLightEnhancer(clip_limit=cfg["lowlight"]["clip_limit"],
                                gamma_dark=cfg["lowlight"]["gamma_dark"]) if cfg["lowlight"]["enabled"] else None
    cap = open_camera(args.camera, args.width, args.height, args.fps)
    actions = FedoraActions(dry_run=dry_run)
    bus = TletlStateBus(bus_path)

    motion = MotionTracker()
    cursor = CursorDelta()
    hold_toggle = Hold()
    hold_mode = Hold()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    mode_i = 0
    control = False
    last_action = "-"
    last_click = last_scroll = last_tab = last_big = last_toggle = 0.0
    drag = False
    pinch_t0: Optional[float] = None

    window = APP
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.width, args.height)

    prev_t = time.time()
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=cfg["camera"]["max_num_hands"],
        model_complexity=1,
        min_detection_confidence=args.det_conf,
        min_tracking_confidence=args.track_conf,
    ) as hands_model:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No pude leer frame.")
                break
            frame = cv2.flip(frame, 1)
            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            luma_mode = "off"
            if lowlight is not None:
                frame = lowlight.enhance(frame)
                luma_mode = lowlight.last_mode

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands_model.process(rgb)
            dom_lmk, mod_lmk = split_hands(result, dominant)

            dom_res = HandResult()
            mod_res = HandResult()
            dom_state = TletlHandState()
            mod_state = TletlHandState()
            cx = cy = 0.0
            swipe = None
            hand_seen = dom_lmk is not None

            if dom_lmk is not None:
                mp_draw.draw_landmarks(frame, dom_lmk, mp_hands.HAND_CONNECTIONS)
                lm = points_from_mediapipe(dom_lmk)
                feats = extract_live_features(lm)
                cx, cy = palm_center(lm)
                motion.update(cx, cy)
                swipe = motion.swipe()
                dom_res = pipeline.process_features(feats, hand="dom")
                dom_state = _hand_state_from(pipeline, dom_res, dominant, (cx, cy), feats)
                px, py = int(cx * frame.shape[1]), int(cy * frame.shape[0])
                cv2.circle(frame, (px, py), 12, (0, 255, 255), 2)
                cv2.putText(frame, f"{dom_res.stable_gesture} {dom_res.confidence:.2f}",
                            (px - 70, py - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                motion.reset(); cursor.reset(); hold_toggle.reset(); hold_mode.reset()
                pipeline.reset("dom")
                if drag:
                    actions.mouse_up(); drag = False
                pinch_t0 = None

            if mod_lmk is not None:
                mp_draw.draw_landmarks(frame, mod_lmk, mp_hands.HAND_CONNECTIONS)
                lm2 = points_from_mediapipe(mod_lmk)
                feats2 = extract_live_features(lm2)
                mcx, mcy = palm_center(lm2)
                mod_res = pipeline.process_features(feats2, hand="mod")
                other = "Left" if dominant == "Right" else "Right"
                mod_state = _hand_state_from(pipeline, mod_res, other, (mcx, mcy), feats2)
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
                    if not control and drag:
                        actions.mouse_up(); drag = False
                    print(f"[TLETL] {last_action}")
            else:
                hold_toggle.reset()

            # safety FIST
            if control and g == "FIST" and conf > 0.55:
                control = False
                last_action = "Pausa seguridad"
                if drag:
                    actions.mouse_up(); drag = False
                pinch_t0 = None

            # THREE = cambiar modo
            mode_progress = 0.0
            if control and g == "THREE":
                mode_progress = hold_mode.progress("THREE", 0.85)
                if mode_progress >= 1.0 and now - last_big > 1.0:
                    mode_i = (mode_i + 1) % len(MODES)
                    last_big = now
                    hold_mode.reset(); pipeline.reset("dom"); cursor.reset()
                    last_action = f"Modo {MODES[mode_i]}"
            else:
                hold_mode.reset()

            mode = MODES[mode_i]
            can_act = control and hand_seen and dom_res.ok and conf >= args.action_conf

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
                    if g == "PINCH":
                        if pinch_t0 is None:
                            pinch_t0 = now
                        if not drag and now - pinch_t0 > 0.32:
                            actions.mouse_down(); drag = True; last_action = "Drag ON"
                    else:
                        if pinch_t0 is not None:
                            held = now - pinch_t0
                            if drag:
                                actions.mouse_up(); drag = False; last_action = "Drag OFF"
                            elif held < 0.24 and now - last_click > 0.28:
                                actions.click_left(); last_click = now; last_action = "Click"
                        pinch_t0 = None

                elif mode == "VENTANAS":
                    if g == "OPEN_PALM" and swipe == "UP" and now - last_big > 0.9:
                        actions.overview(); last_big = now; last_action = "Overview"
                    elif g == "PINCH" and now - last_click > 0.38:
                        actions.enter(); last_click = now; last_action = "Enter"
            else:
                cursor.reset()
                if drag and g != "PINCH":
                    actions.mouse_up(); drag = False; pinch_t0 = None

            # construir state dom + mod y escribir bus
            state = TletlFrameState(
                version=5, app_version="5.0-core-fedora", timestamp=now,
                frame_width=frame.shape[1], frame_height=frame.shape[0], fps=fps,
                dom=dom_state, mod=mod_state, mode=mode, action=last_action,
                selected=control, grabbed=(g == "PINCH" and dom_res.ok),
                extra={"swipe": swipe or "-", "control": control, "lowlight": luma_mode},
            )
            state.intent = gesture_to_common_intent(state)
            bus.write(state)

            state_txt = "ON" if control else "OFF"
            color = (0, 255, 0) if control else (0, 0, 255)
            lines = [
                f"TLETL v5 {state_txt} | MODO:{mode} | FPS:{fps:.1f} | DRY:{dry_run} | LowLight:{luma_mode}",
                f"DOM[{dom_state.side}]:{dom_res.stable_gesture} raw:{dom_res.raw_gesture} conf:{dom_res.confidence:.2f} ok:{dom_res.ok}",
                f"MOD[{mod_state.side}]:{mod_res.stable_gesture} (al bus para Blender)" if mod_state.present else "MOD: (sin segunda mano)",
                f"Guard:{dom_res.guard_reason} | Critic:{dom_res.critic_reason} | Intent:{state.intent.name} | Swipe:{swipe or '-'}",
                f"Action:{last_action} | Bus:{bus_path}",
                "OPEN_PALM hold=ON/OFF | FIST=seguridad | THREE=modo | Q/ESC=salir | t=toggle | m=modo",
            ]
            if toggle_progress > 0:
                lines.append(f"Toggle: {int(toggle_progress * 100)}%")
            if mode_progress > 0:
                lines.append(f"Mode: {int(mode_progress * 100)}%")
            draw_panel(frame, lines, color)
            cv2.circle(frame, (frame.shape[1] - 34, 34), 17, color, -1)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                break
            if key == ord("t"):
                control = not control
                last_action = "Control ON tecla" if control else "Control OFF tecla"
            elif key == ord("m"):
                mode_i = (mode_i + 1) % len(MODES)
                last_action = f"Modo {MODES[mode_i]} tecla"

    if drag:
        actions.mouse_up()
    cap.release()
    cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=APP)
    ap.add_argument("--bank", default=os.environ.get("TLETL_GESTURE_BANK", "datasets/tletl_gesture_bank_v2_features.jsonl"))
    ap.add_argument("--config", default=None, help="ruta a tletl.toml (default: config/tletl.toml)")
    ap.add_argument("--camera", type=int, default=int(os.environ.get("TLETL_CAMERA", "0")))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--det-conf", type=float, default=0.72)
    ap.add_argument("--track-conf", type=float, default=0.72)
    ap.add_argument("--action-conf", type=float, default=0.52)
    ap.add_argument("--k", type=int, default=0, help="override de k (0 = usar config)")
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except KeyboardInterrupt:
        print("\n[TLETL] Interrumpido.")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        print("Tips: prueba --camera 1, o TLETL_DRY_RUN=1 ./launchers/tletl-fedora-safe.sh")
        raise
