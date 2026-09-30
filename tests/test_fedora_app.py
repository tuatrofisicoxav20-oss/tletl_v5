from __future__ import annotations

"""Tests de la app Fedora SIN mediapipe ni cámara.

Verifican que el módulo importa (mediapipe es lazy, dentro del tracker) y que
las piezas puras funcionan: parser, config CLI > toml, split_hands sobre
HandDetection, PinchClickDrag (tap/drag), pausa de seguridad por hold,
tamaño del frame del detector, wrapper de hint, FedoraActions.
"""

import importlib
import shutil
import subprocess
import sys
import unittest.mock as mock

import numpy as np
import pytest

from apps.common.hand_tracker import HandDetection
from tletl_core.geometry import Point


def _det(label="Right", score=0.9):
    return HandDetection([Point(0.4 + 0.01 * j, 0.5, 0.0) for j in range(21)], label, score)


# ── import / parser ──────────────────────────────────────────────────────────

def test_import_without_mediapipe():
    # El módulo debe importarse aunque mediapipe no esté instalado.
    for key in list(sys.modules):
        if key.startswith("apps.fedora_control") or key.startswith("apps.common"):
            del sys.modules[key]
    with mock.patch.dict(sys.modules, {"mediapipe": None}):
        main = importlib.import_module("apps.fedora_control.main")
        assert hasattr(main, "build_parser")
        assert hasattr(main, "run")
        assert main.MODES == ["NAVEGADOR", "CURSOR", "VENTANAS"]
    for key in list(sys.modules):
        if key.startswith("apps.fedora_control") or key.startswith("apps.common"):
            del sys.modules[key]


def test_parser_defaults_are_none_so_config_wins():
    from apps.fedora_control.main import build_parser
    ns = build_parser().parse_args([])
    assert ns.camera is None and ns.width is None and ns.height is None and ns.fps is None
    assert ns.det_conf is None and ns.track_conf is None and ns.action_conf is None
    assert ns.bank is None and ns.backend is None and ns.k is None
    assert ns.dry_run is False and ns.headless is False and ns.window is False
    ns2 = build_parser().parse_args(["--dry-run", "--k", "9", "--backend", "tasks", "--headless"])
    assert ns2.dry_run is True and ns2.k == 9 and ns2.backend == "tasks" and ns2.headless is True
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--backend", "yolo"])


def test_cli_overrides_config_but_config_is_default(tmp_path):
    from apps.fedora_control.main import apply_cli_overrides, build_parser, effective_bank_path
    from tletl_core.config import bank_path_from_config, load_config

    cfg = load_config(tmp_path / "nope.toml")
    cfg["camera"]["index"] = 3
    cfg["tracker"]["det_conf"] = 0.61
    ns = build_parser().parse_args([])
    out = apply_cli_overrides(ns, cfg)
    assert out is cfg
    assert cfg["camera"]["index"] == 3 and cfg["camera"]["width"] == 1280
    assert cfg["tracker"]["det_conf"] == 0.61 and cfg["fedora"]["action_conf"] == 0.52
    assert cfg["camera"]["headless"] is False and cfg["classifier"]["k"] == 13
    assert effective_bank_path(ns, cfg) == bank_path_from_config(cfg)

    ns = build_parser().parse_args(["--camera", "1", "--width", "640", "--height", "360", "--fps", "15",
                                    "--det-conf", "0.5", "--track-conf", "0.4", "--action-conf", "0.7",
                                    "--backend", "legacy", "--proc-width", "0", "--k", "7", "--dry-run",
                                    "--headless", "--gesture-hint", "--bank", "mi_banco.jsonl"])
    apply_cli_overrides(ns, cfg)
    assert (cfg["camera"]["index"], cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fps"]) == (1, 640, 360, 15)
    assert cfg["tracker"]["det_conf"] == 0.5 and cfg["tracker"]["track_conf"] == 0.4
    assert cfg["fedora"]["action_conf"] == 0.7 and cfg["fedora"]["dry_run"] is True
    assert cfg["tracker"]["backend"] == "legacy" and cfg["tracker"]["gesture_hint"] is True
    assert cfg["camera"]["proc_width"] == 0 and cfg["camera"]["headless"] is True
    assert cfg["classifier"]["k"] == 7
    assert effective_bank_path(ns, cfg).name == "mi_banco.jsonl" and effective_bank_path(ns, cfg).is_absolute()


def test_window_flag_beats_headless_config(tmp_path):
    from apps.fedora_control.main import apply_cli_overrides, build_parser
    from tletl_core.config import load_config

    cfg = load_config(tmp_path / "nope.toml")
    cfg["camera"]["headless"] = True
    apply_cli_overrides(build_parser().parse_args(["--headless", "--window"]), cfg)
    assert cfg["camera"]["headless"] is False
    cfg["camera"]["headless"] = True
    apply_cli_overrides(build_parser().parse_args([]), cfg)
    assert cfg["camera"]["headless"] is True   # sin flags manda la config


# ── FedoraActions ────────────────────────────────────────────────────────────

def test_actions_dry_run_does_not_call_system(capsys, monkeypatch):
    from apps.fedora_control.actions import FedoraActions
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no debe ejecutar")))
    act = FedoraActions(dry_run=True)
    act.click_left()
    act.scroll_up()
    out = capsys.readouterr().out
    assert "[DRY] click_left: ydotool click 0xC0" in out
    assert "scroll_up" in out


def test_actions_without_ydotool_warn_once_and_noop(capsys, monkeypatch):
    from apps.fedora_control.actions import FedoraActions
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no debe ejecutar")))
    act = FedoraActions(dry_run=False)
    assert act.available is False
    assert "ydotool" in capsys.readouterr().out
    for _ in range(5):
        act.mousemove(3, 4)
        act.click_left()
    assert capsys.readouterr().out == ""     # ni un mensaje por acción


def test_actions_run_ydotool_and_report_failure_once(capsys, monkeypatch):
    from apps.fedora_control.actions import FedoraActions, key_combo_sequence
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ydotool")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0 if cmd[1] != "click" else 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    act = FedoraActions(dry_run=False)
    assert act.available is True
    act.next_tab()
    act.prev_tab()
    act.back()
    assert calls[0] == ["ydotool", "key", "29:1", "15:1", "15:0", "29:0"]
    assert calls[1] == ["ydotool", "key", "29:1", "42:1", "15:1", "15:0", "42:0", "29:0"]
    assert calls[2] == ["ydotool", "key", "56:1", "105:1", "105:0", "56:0"]
    assert key_combo_sequence(29, 15) == "29:1 15:1 15:0 29:0"
    assert key_combo_sequence() == ""
    act.click_left()
    act.click_left()
    out = capsys.readouterr().out
    assert out.count("[ydotool error]") == 1 and act.failures == 2
    act.mousemove(0, 0)
    assert len(calls) == 5   # dx=dy=0 no ejecuta nada


# ── split_hands ──────────────────────────────────────────────────────────────

def test_split_hands_assigns_dominant():
    from apps.fedora_control.main import split_hands
    dets = [_det("Left"), _det("Right")]
    dom, mod = split_hands(dets, "Right")
    assert dom is dets[1] and mod is dets[0]


def test_split_hands_single_hand_goes_dom():
    from apps.fedora_control.main import split_hands
    dets = [_det("Left")]
    dom, mod = split_hands(dets, "Right")
    assert dom is dets[0] and mod is None


def test_split_hands_no_hands():
    from apps.fedora_control.main import split_hands
    assert split_hands([], "Right") == (None, None)
    assert split_hands(None, "Right") == (None, None)


def test_split_hands_two_non_dominant_are_distinct():
    """Regresión: si ninguna mano es la dominante, dom y mod NO deben ser el mismo objeto."""
    from apps.fedora_control.main import split_hands
    dets = [_det("Left"), _det("Left")]
    dom, mod = split_hands(dets, "Right")
    assert dom is dets[0] and mod is dets[1]
    assert dom is not mod


def test_split_hands_two_same_label_picks_higher_score():
    from apps.fedora_control.main import split_hands
    dets = [_det("Right", 0.6), _det("Right", 0.95)]
    dom, mod = split_hands(dets, "Right")
    assert dom is dets[1] and mod is dets[0]


# ── PinchClickDrag ───────────────────────────────────────────────────────────

class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _pcd():
    from apps.fedora_control.main import PinchClickDrag
    clock = _Clock()
    return PinchClickDrag(tap_max=0.35, drag_min=0.40, click_gap=0.28, clock=clock), clock


def test_short_pinch_is_click():
    p, clock = _pcd()
    assert p.update(True) is None          # primer frame estable: arranca el conteo
    clock.t = 0.15
    assert p.update(True) is None
    clock.t = 0.25
    assert p.update(False) == "click"
    assert not p.dragging and p.pinch_t0 is None


def test_long_pinch_is_drag_then_release():
    p, clock = _pcd()
    p.update(True)
    clock.t = 0.30
    assert p.update(True) is None          # aún por debajo de drag_min
    clock.t = 0.45
    assert p.update(True) == "drag_on"
    assert p.dragging
    clock.t = 0.90
    assert p.update(True) is None          # sigue arrastrando, sin repetir el evento
    clock.t = 1.00
    assert p.update(False) == "drag_off"
    assert not p.dragging
    clock.t = 1.05
    assert p.update(False) is None


def test_dead_zone_between_tap_and_drag_does_nothing():
    p, clock = _pcd()
    p.update(True)
    clock.t = 0.37
    assert p.update(False) is None


def test_no_double_click_within_gap():
    p, clock = _pcd()
    p.update(True); clock.t = 0.10; assert p.update(False) == "click"
    clock.t = 0.15; p.update(True); clock.t = 0.30
    assert p.update(False) is None          # 0.20 s desde el click anterior < 0.28
    clock.t = 0.50; p.update(True); clock.t = 0.60
    assert p.update(False) == "click"       # ya pasó el gap


def test_release_ends_drag_and_resets():
    p, clock = _pcd()
    assert p.release() is None
    p.update(True); clock.t = 0.5; assert p.update(True) == "drag_on"
    assert p.release() == "drag_off"
    assert p.release() is None and p.pinch_t0 is None
    # tap medido desde el PRIMER frame estable, aunque hubiera pasado tiempo ocioso antes
    clock.t = 5.0; p.update(True); clock.t = 5.2
    assert p.update(False) == "click"


def test_pinch_state_machine_at_30fps_tap_and_drag():
    """Con el filtro temporal (size 7, min 4) un PINCH estable dura ~lo mismo que el real,
    solo desplazado ~3-4 frames. A 30 fps: 5 frames estables (0.17 s) = UN click;
    30 frames estables (1 s) = drag_on + drag_off y ningún click."""
    p, clock = _pcd()
    step = 1.0 / 30.0
    events = []
    for _ in range(5):
        events.append(p.update(True)); clock.t += step
    events.append(p.update(False))
    assert events.count("click") == 1 and events[-1] == "click"
    assert "drag_on" not in events and "drag_off" not in events

    clock.t += 1.0
    events = []
    for _ in range(30):
        events.append(p.update(True)); clock.t += step
    events.append(p.update(False))
    assert events.count("drag_on") == 1 and events.count("drag_off") == 1 and "click" not in events
    assert events.index("drag_on") in (12, 13)       # 0.40 s a 30 fps (tolerancia float)
    assert events[-1] == "drag_off"
    # el frame estable "de más" tras soltar no genera un segundo evento
    clock.t += step
    assert p.update(False) is None


# ── seguridad, tamaño de proceso, fps, hint wrapper ──────────────────────────

def test_safety_pause_requires_hold_or_high_confidence():
    from apps.fedora_control.main import safety_pause_due
    assert safety_pause_due(1.0, 0.60) is True       # FIST sostenido todo safety_hold
    assert safety_pause_due(0.5, 0.95) is True       # muy seguro: basta la mitad
    assert safety_pause_due(0.5, 0.80) is False
    assert safety_pause_due(0.3, 0.99) is False      # un puño de un frame ya no pausa
    assert safety_pause_due(0.0, 1.00) is False


def test_detector_size_keeps_aspect():
    from apps.fedora_control.main import detector_size
    assert detector_size(1280, 720, 640) == (640, 360)
    assert detector_size(1920, 1080, 640) == (640, 360)
    assert detector_size(640, 480, 640) == (640, 480)
    assert detector_size(320, 240, 640) == (320, 240)
    assert detector_size(1280, 720, 0) == (1280, 720)


def test_fps_meter_smooths():
    from apps.fedora_control.main import FpsMeter
    m = FpsMeter(alpha=0.5)
    assert m.tick(0.0) == 0.0
    for i in range(1, 30):
        fps = m.tick(i / 30.0)
    assert abs(fps - 30.0) < 0.5


def test_process_features_wrapper_tolerates_old_and_new_signatures():
    from apps.fedora_control.main import make_process_features

    class OldPipeline:
        def __init__(self):
            self.calls = []

        def process_features(self, features, hand="dom"):
            self.calls.append((hand, "no-hint"))
            return "old"

    class NewPipeline:
        def __init__(self):
            self.calls = []

        def process_features(self, features, hand="dom", hint=None):
            self.calls.append((hand, hint))
            return "new"

    old = OldPipeline()
    process = make_process_features(old)
    assert process({}, "dom", "FIST") == "old" and process({}, "mod", None) == "old"
    assert old.calls == [("dom", "no-hint"), ("mod", "no-hint")]

    new = NewPipeline()
    process = make_process_features(new)
    assert process({}, "dom", "FIST") == "new" and process({}, "mod", None) == "new"
    assert new.calls == [("dom", "FIST"), ("mod", None)]


def test_short_path_uses_tilde(tmp_path, monkeypatch):
    from pathlib import Path
    from apps.fedora_control.main import short_path
    home = Path.home()
    assert short_path(home / ".tletl" / "tletl_state.json") == "~/.tletl/tletl_state.json"
    assert short_path(Path("/nope/x.json")) == "/nope/x.json"


# ── panel ────────────────────────────────────────────────────────────────────

def test_draw_panel_zones_only_when_asked():
    pytest.importorskip("cv2")
    from apps.fedora_control.main import draw_panel
    h, w = 200, 300
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    draw_panel(frame, ["hola"], (0, 255, 0), zones=False)
    assert not frame[int(h * 0.39)].any() and not frame[int(h * 0.61)].any()
    frame2 = np.zeros((h, w, 3), dtype=np.uint8)
    draw_panel(frame2, ["hola"], (0, 255, 0), zones=True)
    assert frame2[int(h * 0.39)].any() and frame2[int(h * 0.61)].any()
    frame3 = np.zeros((h, w, 3), dtype=np.uint8)
    draw_panel(frame3, ["a"] * 8, (0, 0, 255))   # firma vieja (3 posicionales) sigue valiendo
    assert frame3.any()


# ── smoke de run(): cámara falsa + tracker falso, sin mediapipe ──────────────

class _FakeCap:
    """Entrega N frames y luego (False, None) para que el bucle termine solo."""

    def __init__(self, n, w=1280, h=720):
        self.n, self.w, self.h = n, w, h
        self.released = False

    def read(self):
        if self.n <= 0:
            return False, None
        self.n -= 1
        return True, np.full((self.h, self.w, 3), 90, dtype=np.uint8)

    def get(self, prop):
        return 0.0

    def release(self):
        self.released = True


def _run_smoke(monkeypatch, tmp_path, extra_args, keys=()):
    from tests.conftest import make_landmarks
    from apps.common.hand_tracker import HandTracker
    from apps.fedora_control import main

    class Backend:
        name = "fake"

        def __init__(self):
            self.frames = []
            self.closed = False

        def detect(self, rgb, ts):
            self.frames.append(rgb.shape)
            return [HandDetection(make_landmarks("OPEN_PALM"), "Right", 0.9),
                    HandDetection(make_landmarks("FIST"), "Left", 0.9)]

        def close(self):
            self.closed = True

    backend = Backend()
    cap = _FakeCap(12)
    monkeypatch.setenv("TLETL_STATE_PATH", str(tmp_path / "bus.json"))
    monkeypatch.setattr(main.HandTracker, "create", classmethod(lambda cls, cfg, num_hands=2, **k: HandTracker(backend)))
    monkeypatch.setattr(main.GestureHint, "create", classmethod(lambda cls, cfg, num_hands=2, **k: None))
    monkeypatch.setattr(main, "describe_backend", lambda cfg: {"backend": "fake"})
    monkeypatch.setattr(main, "open_camera", lambda *a, **k: cap)
    pressed = list(keys)
    monkeypatch.setattr(main.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(main.cv2, "resizeWindow", lambda *a, **k: None)
    monkeypatch.setattr(main.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(main.cv2, "destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr(main.cv2, "waitKey", lambda *a, **k: pressed.pop(0) if pressed else 255)
    args = main.build_parser().parse_args(["--dry-run", "--fps", "500", "--config", str(tmp_path / "nope.toml"), *extra_args])
    rc = main.run(args)
    return rc, backend, cap, main


def test_run_headless_smoke_writes_bus(monkeypatch, tmp_path, capsys):
    import json
    import time as real_time
    t0 = real_time.time()
    rc, backend, cap, main = _run_smoke(monkeypatch, tmp_path, ["--headless", "--proc-width", "640"])
    assert rc == 0 and cap.released and backend.closed
    assert backend.frames and all(shape == (360, 640, 3) for shape in backend.frames)   # detector ve el frame reducido
    data = json.loads((tmp_path / "bus.json").read_text(encoding="utf-8"))
    assert data["extra"]["backend"] == "fake" and data["extra"]["headless"] is True
    assert t0 <= data["timestamp"] <= real_time.time()
    assert data["dom"]["present"] and data["dom"]["side"] == "Right"
    assert data["mod"]["present"] and data["mod"]["side"] == "Left"
    # pipeline REAL + banco real: la palma sintética sale como OPEN_PALM estable tras 12 frames
    assert data["dom"]["gesture"] == data["dom"]["stable_gesture"] == "OPEN_PALM" and data["dom"]["critic_ok"] is True
    assert data["intent"]["name"] == "RELEASE_OR_TOGGLE" and data["intent"]["active"] is True
    assert data["dom"]["features"] == {}                     # include_features=false por default
    assert data["frame_width"] == 1280 and data["mode"] == "NAVEGADOR"
    assert data["app_version"].endswith("-core-fedora") and data["extra"]["hint"] == ""
    out = capsys.readouterr().out
    assert "headless" in out and "backend" in out


def test_run_window_smoke_handles_keys(monkeypatch, tmp_path, capsys):
    rc, backend, cap, main = _run_smoke(monkeypatch, tmp_path, ["--proc-width", "0"],
                                        keys=[ord("t"), ord("m"), ord("t"), 255, ord("q")])
    assert rc == 0 and cap.released and backend.closed
    assert backend.frames and backend.frames[0] == (720, 1280, 3)   # proc_width 0 = frame completo
    assert cap.n == 12 - 5                                          # 'q' cortó el bucle en el frame 5


# ── run() en modo CURSOR con un guion de PINCH: tap = 1 click, 1 s = drag on/off ─────

class _StepClock:
    """Reloj determinista: avanza exactamente un frame (1/fps) por cada cap.read()."""

    def __init__(self, fps=30.0, t0=100.0):
        self.t = t0
        self.step = 1.0 / fps

    def __call__(self):
        return self.t

    def tick(self):
        self.t += self.step


class _ScriptedCap(_FakeCap):
    def __init__(self, n, clock):
        super().__init__(n)
        self.clock = clock

    def read(self):
        ok, frame = super().read()
        if ok:
            self.clock.tick()
        return ok, frame


def test_run_cursor_pinch_tap_and_drag_through_temporal_filter(monkeypatch, tmp_path):
    """Bucle real de run() (ventana simulada, teclas t+m = control ON + modo CURSOR) con:
      - tracker falso (una mano Right por frame),
      - pipeline falso cuyo KNN es un GUION de gestos crudos por frame pero cuyo filtro
        temporal es el REAL (size 7, min_count 4): el PINCH estable aparece 3 frames
        después del crudo y persiste 3 frames tras soltar,
      - reloj inyectado a 30 fps en time.time / PinchClickDrag / Hold / MotionTracker,
      - FedoraActions que solo registra etiquetas, bus que guarda cada frame.
    5 frames crudos de PINCH => exactamente UN click_left; 30 frames (1 s) => mouse_down +
    mouse_up y ningún click. El bus lleva timestamp, backend, gesto dom, grabbed e intent."""
    import functools
    import json
    from tests.conftest import make_landmarks
    from apps.common.hand_tracker import HandTracker
    from apps.fedora_control import main
    from apps.fedora_control.actions import FedoraActions
    from tletl_core.bus import TletlStateBus
    from tletl_core.classifier import Prediction
    from tletl_core.pipeline import HandResult, TletlPipeline
    from tletl_core.temporal import Hold, MotionTracker, TemporalFilter

    script = ["NEUTRAL"] * 10 + ["PINCH"] * 5 + ["NEUTRAL"] * 15 + ["PINCH"] * 30 + ["NEUTRAL"] * 12
    n_frames = len(script)

    class ScriptedPipeline:
        def __init__(self):
            self.i = 0
            self.filters = {}
            self.resets = []
            self.closed = False

        def process_features(self, features, hand="dom", hint=None):
            raw = "NEUTRAL"
            if hand == "dom":
                raw = script[min(self.i, n_frames - 1)]
                self.i += 1
            pred = Prediction(label=raw, raw_label=raw, confidence=0.9, margin=0.3,
                              orientation="UP", reason="", ok=True, votes={})
            tf = self.filters.setdefault(hand, TemporalFilter(size=7, min_count=4))
            stable = tf.update(pred)
            return HandResult(present=True, raw_gesture=raw, guarded_gesture=raw,
                              stable_gesture=stable.label, ok=bool(stable.ok), confidence=0.9,
                              margin=0.3, orientation="UP", reason=stable.reason, hint=hint)

        def to_hand_state(self, result, side="unknown", features=None):
            return TletlPipeline.to_hand_state(self, result, side, features)   # no usa self

        def reset(self, hand=None):
            self.resets.append(hand)
            for h, tf in self.filters.items():
                if hand is None or h == hand:
                    tf.reset()

        def close(self):
            self.closed = True

    class RecordingActions(FedoraActions):
        labels = []

        def run(self, cmd, label):
            type(self).labels.append(label)

    class RecordingBus(TletlStateBus):
        frames = []

        def write(self, state):
            type(self).frames.append(json.loads(self.serialize(state)))
            super().write(state)

    class Backend:
        name = "fake"

        def detect(self, rgb, ts):
            return [HandDetection(make_landmarks("PINCH"), "Right", 0.9)]

        def close(self):
            pass

    RecordingActions.labels = []
    RecordingBus.frames = []
    clock = _StepClock(fps=30.0)
    cap = _ScriptedCap(n_frames, clock)
    pipeline = ScriptedPipeline()

    monkeypatch.setenv("TLETL_STATE_PATH", str(tmp_path / "bus.json"))
    monkeypatch.setattr(main, "time", mock.Mock(time=clock, monotonic=clock, sleep=lambda s: None))
    monkeypatch.setattr(main, "PinchClickDrag", functools.partial(main.PinchClickDrag, clock=clock))
    monkeypatch.setattr(main, "Hold", functools.partial(Hold, clock=clock))
    monkeypatch.setattr(main, "MotionTracker", functools.partial(MotionTracker, clock=clock))
    monkeypatch.setattr(main, "TletlPipeline", lambda bank, config=None, **k: pipeline)
    monkeypatch.setattr(main, "FedoraActions", RecordingActions)
    monkeypatch.setattr(main, "TletlStateBus", RecordingBus)
    monkeypatch.setattr(main.HandTracker, "create", classmethod(lambda cls, cfg, num_hands=2, **k: HandTracker(Backend(), clock=clock)))
    monkeypatch.setattr(main.GestureHint, "create", classmethod(lambda cls, cfg, num_hands=2, **k: None))
    monkeypatch.setattr(main, "describe_backend", lambda cfg: {"backend": "fake"})
    monkeypatch.setattr(main, "open_camera", lambda *a, **k: cap)
    keys = [ord("t"), ord("m")]          # frame 0: control ON; frame 1: modo CURSOR
    for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
        monkeypatch.setattr(main.cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(main.cv2, "waitKey", lambda *a, **k: keys.pop(0) if keys else 255)

    args = main.build_parser().parse_args(["--dry-run", "--config", str(tmp_path / "nope.toml")])
    assert main.run(args) == 0
    assert pipeline.closed and cap.released

    # Acciones: UN click por el tap, UN mouse_down + UN mouse_up por el arrastre, nada más
    # (mousemove no aparece: la palma no se mueve => dx=dy=0 no ejecuta).
    assert RecordingActions.labels == ["click_left", "mouse_down", "mouse_up"]

    frames = RecordingBus.frames
    assert len(frames) == n_frames
    stable = [f["dom"]["gesture"] for f in frames]
    # filtro temporal real: crudo 10..14 -> estable 13..17 ; crudo 30..59 -> estable 33..62
    assert [i for i, g in enumerate(stable) if g == "PINCH"] == list(range(13, 18)) + list(range(33, 63))
    actions = [f["action"] for f in frames]
    assert actions[18] == "Click" and "Click" not in actions[:18]
    assert actions[63] == "Drag OFF" and actions[-1] == "Drag OFF"
    assert actions.index("Drag ON") in (45, 46) and actions[62] == "Drag ON"
    assert "Click" not in actions[actions.index("Drag ON"):]           # el click no reaparece tras el drag
    # las teclas se leen DESPUÉS de escribir el bus: el efecto se ve en el frame siguiente
    assert actions[0] == "-" and actions[1] == "Control ON tecla" and actions[2] == "Modo CURSOR tecla"

    assert frames[0]["mode"] == "NAVEGADOR" and all(f["mode"] == "CURSOR" for f in frames[2:])
    for i in (13, 40):
        assert frames[i]["grabbed"] is True and frames[i]["intent"]["name"] == "GRAB_OR_SELECT"
        assert frames[i]["selected"] is True and frames[i]["dom"]["critic_ok"] is True
    assert frames[20]["grabbed"] is False and frames[20]["intent"]["name"] == "IDLE"
    assert all(f["extra"]["backend"] == "fake" and f["extra"]["control"] is True for f in frames[1:])
    assert all(f["dom"]["side"] == "Right" and f["dom"]["present"] for f in frames)
    stamps = [f["timestamp"] for f in frames]
    assert stamps == sorted(stamps) and len(set(stamps)) == n_frames and stamps[0] > 100.0
    assert stamps[-1] - stamps[0] == pytest.approx((n_frames - 1) / 30.0)
    # nunca se perdió la mano ni cambió el control/modo por gesto (solo el reset de "mod"
    # de cada frame sin segunda mano)
    assert set(pipeline.resets) == {"mod"}
    final = json.loads((tmp_path / "bus.json").read_text(encoding="utf-8"))
    assert final == frames[-1]
