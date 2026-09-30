"""Tests de apps/common/hand_tracker.py y apps/common/gesture_hint.py.

Todo corre SIN modelo real ni mediapipe (resultados falsos, backends falsos,
descargador falso). Las dos pruebas de integración del final usan el modelo
.task real y se saltan limpiamente si falta mediapipe, el modelo o la
librería nativa.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from apps.common import gesture_hint as gh
from apps.common import hand_tracker as ht
from apps.common.hand_tracker import (
    HAND_CONNECTIONS,
    HandDetection,
    HandTracker,
    MonotonicTimestamps,
    TrackerModelMissing,
    TrackerUnavailable,
    describe_backend,
    detections_from_legacy,
    detections_from_tasks,
    ensure_model_file,
    format_backend_summary,
    normalize_handedness,
    resolve_model_path,
    select_backend,
)
from tletl_core.geometry import Point

ROOT = Path(__file__).resolve().parent.parent
VENV_MODELS = ROOT / ".venv" / "models"


# ── helpers: resultados falsos de ambos backends ─────────────────────────────

def _pt(x, y, z=0.0):
    return types.SimpleNamespace(x=x, y=y, z=z)


def _hand_points(offset: float, n: int = 21):
    return [_pt(offset + 0.01 * j, 0.5 - 0.005 * j, -0.001 * j) for j in range(n)]


def _legacy_result(labels, scores=None, n_points: int = 21):
    scores = scores or [0.9] * len(labels)
    hands = [types.SimpleNamespace(landmark=_hand_points(0.1 * i, n_points)) for i in range(len(labels))]
    handed = [types.SimpleNamespace(classification=[types.SimpleNamespace(label=lbl, score=scores[i], index=0)])
              for i, lbl in enumerate(labels)]
    return types.SimpleNamespace(multi_hand_landmarks=hands, multi_handedness=handed)


def _tasks_result(labels, scores=None, n_points: int = 21):
    scores = scores or [0.9] * len(labels)
    hands = [_hand_points(0.1 * i, n_points) for i in range(len(labels))]
    handed = [[types.SimpleNamespace(category_name=lbl, display_name=lbl, score=scores[i], index=0)]
              for i, lbl in enumerate(labels)]
    return types.SimpleNamespace(hand_landmarks=hands, handedness=handed, hand_world_landmarks=[])


def _detection(label="Right", score=0.9, n=21):
    return HandDetection([Point(0.3 + 0.01 * j, 0.4, 0.0) for j in range(n)], label, score)


class FakeBackend:
    name = "fake"

    def __init__(self, detections=None):
        self.calls = []
        self.closed = False
        self.detections = detections if detections is not None else []

    def detect(self, rgb, timestamp_ms):
        self.calls.append((rgb.shape, timestamp_ms))
        return list(self.detections)

    def close(self):
        self.closed = True


# ── conexiones y conversión ──────────────────────────────────────────────────

def test_hand_connections_match_mediapipe_topology():
    expected = {
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
    }
    assert set(HAND_CONNECTIONS) == expected
    assert len(HAND_CONNECTIONS) == 21
    assert {i for c in HAND_CONNECTIONS for i in c} == set(range(21))


def test_legacy_conversion_keeps_order_and_values():
    dets = detections_from_legacy(_legacy_result(["Left", "Right"], [0.8, 0.95]))
    assert [d.handedness for d in dets] == ["Left", "Right"]
    assert [d.score for d in dets] == [0.8, 0.95]
    assert all(len(d.landmarks) == 21 and d.is_complete for d in dets)
    assert isinstance(dets[0].landmarks[0], Point)
    assert dets[1].landmarks[3].x == pytest.approx(0.1 + 0.03)
    assert dets[1].landmarks[3].z == pytest.approx(-0.003)


def test_tasks_conversion_keeps_order_and_values():
    dets = detections_from_tasks(_tasks_result(["Right", "Left"], [0.7, 0.99]))
    assert [d.handedness for d in dets] == ["Right", "Left"]
    assert [d.score for d in dets] == [0.7, 0.99]
    assert all(len(d.landmarks) == 21 for d in dets)
    assert dets[0].landmarks[5].y == pytest.approx(0.5 - 0.025)


def test_conversion_empty_results():
    assert detections_from_legacy(types.SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)) == []
    assert detections_from_tasks(types.SimpleNamespace(hand_landmarks=[], handedness=[])) == []
    assert detections_from_tasks(object()) == []


def test_conversion_without_handedness_defaults_to_right():
    res = _tasks_result(["Left"])
    res.handedness = []
    dets = detections_from_tasks(res)
    assert dets[0].handedness == "Right" and dets[0].score == 0.0
    res2 = _legacy_result(["Left"])
    res2.multi_handedness = None
    assert detections_from_legacy(res2)[0].handedness == "Right"


def test_handedness_is_never_swapped():
    """El frame ya va espejado, así que la etiqueta de MediaPipe se respeta tal cual (ver docstring)."""
    assert detections_from_legacy(_legacy_result(["Left"]))[0].handedness == "Left"
    assert detections_from_tasks(_tasks_result(["Left"]))[0].handedness == "Left"
    assert detections_from_tasks(_tasks_result(["Right"]))[0].handedness == "Right"


def test_normalize_handedness():
    assert normalize_handedness("left") == "Left"
    assert normalize_handedness("LEFT") == "Left"
    assert normalize_handedness("Right") == "Right"
    assert normalize_handedness(None) == "Right"
    assert normalize_handedness("") == "Right"


# ── selección de backend ─────────────────────────────────────────────────────

def test_select_backend_auto_prefers_tasks(monkeypatch):
    monkeypatch.setattr(ht, "tasks_available", lambda: True)
    monkeypatch.setattr(ht, "legacy_available", lambda: True)
    assert select_backend("auto") == "tasks"
    assert select_backend(None) == "tasks"


def test_select_backend_auto_falls_back_to_legacy(monkeypatch):
    monkeypatch.setattr(ht, "tasks_available", lambda: False)
    monkeypatch.setattr(ht, "legacy_available", lambda: True)
    assert select_backend("auto") == "legacy"


def test_select_backend_auto_without_mediapipe_raises(monkeypatch):
    monkeypatch.setattr(ht, "tasks_available", lambda: False)
    monkeypatch.setattr(ht, "legacy_available", lambda: False)
    with pytest.raises(TrackerUnavailable):
        select_backend("auto")


def test_select_backend_explicit_and_invalid(monkeypatch):
    monkeypatch.setattr(ht, "tasks_available", lambda: (_ for _ in ()).throw(AssertionError("no debe sondear")))
    assert select_backend("tasks") == "tasks"
    assert select_backend(" Legacy ") == "legacy"
    with pytest.raises(ValueError):
        select_backend("yolo")


def test_describe_backend_is_pure_and_complete(monkeypatch, tmp_path):
    monkeypatch.setenv("TLETL_HOME", str(tmp_path))
    monkeypatch.delenv("TLETL_TRACKER_MODEL", raising=False)
    monkeypatch.delenv("TLETL_GESTURE_MODEL", raising=False)
    monkeypatch.setattr(ht, "tasks_available", lambda: True)
    monkeypatch.setattr(ht, "legacy_available", lambda: False)
    monkeypatch.setattr(ht, "mediapipe_version", lambda: "9.9.9")
    info = describe_backend({"backend": "auto", "auto_download": False, "gesture_hint": True})
    assert info["backend"] == "tasks" and info["tasks_available"] and not info["legacy_available"]
    assert info["model_path"] == str(tmp_path / "models" / "hand_landmarker.task")
    assert info["model_present"] is False
    assert info["gesture_model_path"] == str(tmp_path / "models" / "gesture_recognizer.task")
    assert info["auto_download"] is False and info["gesture_hint"] is True
    assert info["mediapipe_version"] == "9.9.9"
    summary = format_backend_summary(info)
    assert "backend=tasks" in summary and "FALTA" in summary and "9.9.9" in summary

    monkeypatch.setattr(ht, "tasks_available", lambda: False)
    monkeypatch.setattr(ht, "legacy_available", lambda: True)
    assert describe_backend({"backend": "auto"})["backend"] == "legacy"
    monkeypatch.setattr(ht, "legacy_available", lambda: False)
    assert describe_backend({"backend": "auto"})["backend"] == "none"
    assert describe_backend({"backend": "legacy"})["backend"] == "legacy"


# ── rutas de modelos y descarga ──────────────────────────────────────────────

def test_resolve_model_path_default_env_and_config(monkeypatch, tmp_path):
    monkeypatch.setenv("TLETL_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TLETL_TRACKER_MODEL", raising=False)
    assert ht.default_models_dir() == tmp_path / "home" / "models"
    assert resolve_model_path("") == tmp_path / "home" / "models" / "hand_landmarker.task"
    assert resolve_model_path(None, "gesture_recognizer.task") == tmp_path / "home" / "models" / "gesture_recognizer.task"
    # config apunta a un archivo
    assert resolve_model_path(str(tmp_path / "x.task")) == tmp_path / "x.task"
    # config apunta a un directorio existente -> se añade el nombre
    d = tmp_path / "dir"
    d.mkdir()
    assert resolve_model_path(str(d)) == d / "hand_landmarker.task"
    assert resolve_model_path(str(d) + "/") == d / "hand_landmarker.task"
    # env gana sobre config
    monkeypatch.setenv("TLETL_TRACKER_MODEL", str(tmp_path / "env.task"))
    assert resolve_model_path(str(tmp_path / "x.task")) == tmp_path / "env.task"


def test_ensure_model_file_downloads_with_fake_downloader(tmp_path):
    dest = tmp_path / "models" / "hand_landmarker.task"
    calls = []

    def fake_download(name, path):
        calls.append((name, Path(path)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"x" * 10)

    logs = []
    out = ensure_model_file(dest, "hand_landmarker.task", auto_download=True, downloader=fake_download, log=logs.append)
    assert out == dest and dest.is_file()
    assert calls == [("hand_landmarker.task", dest)]
    # segunda vez: ya existe, no se descarga
    ensure_model_file(dest, "hand_landmarker.task", auto_download=True, downloader=fake_download)
    assert len(calls) == 1


def test_ensure_model_file_without_auto_download_raises_with_hint(tmp_path):
    dest = tmp_path / "hand_landmarker.task"
    with pytest.raises(TrackerModelMissing) as exc:
        ensure_model_file(dest, "hand_landmarker.task", auto_download=False, downloader=lambda n, p: None)
    assert "tletl-fetch-models" in str(exc.value)
    assert not dest.exists()


def test_ensure_model_file_download_failure_raises_with_hint(tmp_path):
    dest = tmp_path / "hand_landmarker.task"

    def boom(name, path):
        raise ConnectionError("sin red")

    with pytest.raises(TrackerModelMissing) as exc:
        ensure_model_file(dest, "hand_landmarker.task", auto_download=True, downloader=boom)
    assert "sin red" in str(exc.value) and "tletl-fetch-models" in str(exc.value)

    def liar(name, path):  # "descarga" que no deja archivo
        return None

    with pytest.raises(TrackerModelMissing):
        ensure_model_file(dest, "hand_landmarker.task", auto_download=True, downloader=liar)


# ── HandTracker.create con backends falsos ───────────────────────────────────

class _FakeTasksBackend:
    name = "tasks"
    created = []

    def __init__(self, model_path, num_hands, det_conf, track_conf):
        type(self).created.append((Path(model_path), num_hands, det_conf, track_conf))

    def detect(self, rgb, ts):
        return []

    def close(self):
        pass


class _FakeLegacyBackend:
    name = "legacy"
    created = []

    def __init__(self, num_hands, det_conf, track_conf, model_complexity=1):
        type(self).created.append((num_hands, det_conf, track_conf, model_complexity))

    def detect(self, rgb, ts):
        return []

    def close(self):
        pass


@pytest.fixture
def fake_backends(monkeypatch, tmp_path):
    _FakeTasksBackend.created = []
    _FakeLegacyBackend.created = []
    monkeypatch.setattr(ht, "_TasksBackend", _FakeTasksBackend)
    monkeypatch.setattr(ht, "_LegacyBackend", _FakeLegacyBackend)
    monkeypatch.setenv("TLETL_HOME", str(tmp_path))
    monkeypatch.delenv("TLETL_TRACKER_MODEL", raising=False)
    return tmp_path


def _fake_downloader(name, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"model")


def test_create_auto_uses_tasks_and_auto_downloads(fake_backends, monkeypatch):
    monkeypatch.setattr(ht, "tasks_available", lambda: True)
    monkeypatch.setattr(ht, "legacy_available", lambda: True)
    logs = []
    t = HandTracker.create({"backend": "auto", "det_conf": 0.5, "track_conf": 0.6, "auto_download": True},
                           num_hands=2, downloader=_fake_downloader, log=logs.append)
    assert t.backend_name == "tasks"
    model, hands, det, track = _FakeTasksBackend.created[0]
    assert model == fake_backends / "models" / "hand_landmarker.task" and model.is_file()
    assert (hands, det, track) == (2, 0.5, 0.6)
    assert _FakeLegacyBackend.created == []
    assert any("backend tasks" in line for line in logs)


def test_create_auto_falls_back_to_legacy_when_tasks_creation_fails(fake_backends, monkeypatch):
    class Boom(_FakeTasksBackend):
        def __init__(self, *a, **k):
            raise TrackerUnavailable("libEGL ausente")

    monkeypatch.setattr(ht, "_TasksBackend", Boom)
    monkeypatch.setattr(ht, "tasks_available", lambda: True)
    monkeypatch.setattr(ht, "legacy_available", lambda: True)
    logs = []
    t = HandTracker.create({"backend": "auto", "model_complexity": 0}, num_hands=1,
                           downloader=_fake_downloader, log=logs.append)
    assert t.backend_name == "legacy"
    assert _FakeLegacyBackend.created == [(1, 0.72, 0.72, 0)]
    assert any("legacy" in line and "libEGL" in line for line in logs)


def test_create_auto_model_missing_without_download_falls_back_or_raises(fake_backends, monkeypatch):
    monkeypatch.setattr(ht, "tasks_available", lambda: True)
    monkeypatch.setattr(ht, "legacy_available", lambda: True)
    t = HandTracker.create({"backend": "auto", "auto_download": False}, downloader=_fake_downloader, log=lambda s: None)
    assert t.backend_name == "legacy"          # legacy existe -> respaldo
    monkeypatch.setattr(ht, "legacy_available", lambda: False)
    with pytest.raises(TrackerModelMissing) as exc:  # mediapipe 1.x sin modelo ni descarga -> error claro
        HandTracker.create({"backend": "auto", "auto_download": False}, downloader=_fake_downloader, log=lambda s: None)
    assert "tletl-fetch-models" in str(exc.value)


def test_create_forced_tasks_propagates_errors(fake_backends, monkeypatch):
    monkeypatch.setattr(ht, "legacy_available", lambda: True)
    with pytest.raises(TrackerModelMissing):
        HandTracker.create({"backend": "tasks", "auto_download": False}, downloader=_fake_downloader)
    t = HandTracker.create({"backend": "tasks"}, downloader=_fake_downloader, log=lambda s: None)
    assert t.backend_name == "tasks"


def test_create_forced_legacy_never_touches_models(fake_backends, monkeypatch):
    monkeypatch.setattr(ht, "tasks_available", lambda: True)

    def no_download(name, path):
        raise AssertionError("legacy no descarga modelos")

    t = HandTracker.create({"backend": "legacy", "det_conf": 0.4, "track_conf": 0.3, "model_complexity": 2},
                           num_hands=2, downloader=no_download)
    assert t.backend_name == "legacy"
    assert _FakeLegacyBackend.created == [(2, 0.4, 0.3, 2)]


def test_create_auto_without_tasks_uses_legacy(fake_backends, monkeypatch):
    monkeypatch.setattr(ht, "tasks_available", lambda: False)
    monkeypatch.setattr(ht, "legacy_available", lambda: True)
    logs = []
    assert HandTracker.create({"backend": "auto"}, log=logs.append).backend_name == "legacy"
    assert any("Tasks API" in line for line in logs)
    monkeypatch.setattr(ht, "legacy_available", lambda: False)
    with pytest.raises(TrackerUnavailable):
        HandTracker.create({"backend": "auto"}, log=logs.append)


# ── timestamps y process ─────────────────────────────────────────────────────

def test_timestamps_strictly_increasing_with_stalled_clock():
    stamps = MonotonicTimestamps(clock=lambda: 100.0)
    seen = [stamps.next() for _ in range(5)]
    assert seen == sorted(seen) and len(set(seen)) == 5
    assert seen[0] >= 1
    # timestamps externos repetidos o hacia atrás se corrigen
    assert stamps.next(3) > seen[-1]
    last = stamps.last
    assert stamps.next(last) == last + 1


def test_timestamps_follow_clock():
    t = {"now": 0.0}
    stamps = MonotonicTimestamps(clock=lambda: t["now"])
    assert stamps.next() == 1
    t["now"] = 0.5
    assert stamps.next() == 501
    t["now"] = 0.5004
    assert stamps.next() == 502   # mismo ms -> +1


def test_tracker_process_uses_backend_and_monotonic_timestamps():
    backend = FakeBackend([_detection("Left", 0.8)])
    tracker = HandTracker(backend, clock=lambda: 5.0)
    frame = np.zeros((36, 64, 3), dtype=np.uint8)
    d1 = tracker.process(frame)
    d2 = tracker.process(frame)
    d3 = tracker.process(frame, timestamp_ms=1)
    assert tracker.backend_name == "fake"
    assert len(d1) == 1 and d1[0].handedness == "Left"
    ts = [c[1] for c in backend.calls]
    assert ts == sorted(ts) and len(set(ts)) == 3
    assert tracker.frames == 3 and tracker.last_timestamp_ms == ts[-1]
    tracker.close()
    tracker.close()
    assert backend.closed and tracker.closed


def test_tracker_drops_incomplete_hands():
    backend = FakeBackend([_detection("Right", 0.9, n=21), _detection("Left", 0.9, n=5)])
    tracker = HandTracker(backend)
    dets = tracker.process(np.zeros((10, 10, 3), dtype=np.uint8))
    assert len(dets) == 1 and dets[0].handedness == "Right"


def test_tracker_rejects_bad_frames():
    tracker = HandTracker(FakeBackend())
    with pytest.raises(ValueError):
        tracker.process(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        tracker.process("no soy un frame")  # type: ignore[arg-type]


def test_tracker_context_manager_closes():
    backend = FakeBackend()
    with HandTracker(backend, name="ctx") as tracker:
        assert tracker.backend_name == "ctx"
    assert backend.closed


# ── dibujo ───────────────────────────────────────────────────────────────────

def test_draw_landmarks_draws_skeleton():
    pytest.importorskip("cv2")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    det = HandDetection([Point(0.2 + 0.03 * j, 0.3 + 0.02 * j, 0.0) for j in range(21)], "Right", 0.9)
    ht.draw_landmarks(frame, det)
    assert frame.any() and frame.shape == (120, 160, 3)
    # landmarks fuera del frame no revientan
    off = HandDetection([Point(-0.5 + 0.1 * j, 1.5, 0.0) for j in range(21)], "Left", 0.5)
    ht.draw_landmarks(frame, off, color=(255, 0, 0))
    assert ht.landmarks_to_pixels(det, 160, 120)[0] == (32, 36)


# ── gesture_hint ─────────────────────────────────────────────────────────────

def test_map_mp_gesture_table():
    assert gh.map_mp_gesture("Closed_Fist") == "FIST"
    assert gh.map_mp_gesture("Open_Palm") == "OPEN_PALM"
    assert gh.map_mp_gesture("Pointing_Up") == "POINT"
    assert gh.map_mp_gesture("Victory") == "VICTORY"
    for other in ("None", "Thumb_Up", "Thumb_Down", "ILoveYou", "", None, "bogus"):
        assert gh.map_mp_gesture(other) is None


def _gesture_result(pairs):
    """pairs: [(handedness, gesture_name, score), ...]"""
    gestures = [[types.SimpleNamespace(category_name=g, score=s, display_name=g, index=0)] if g is not None else []
                for _, g, s in pairs]
    handed = [[types.SimpleNamespace(category_name=h, score=0.9, display_name=h, index=0)] for h, _, _ in pairs]
    return types.SimpleNamespace(gestures=gestures, handedness=handed, hand_landmarks=[[] for _ in pairs])


def test_hints_from_result_maps_and_filters_by_score():
    res = _gesture_result([("Left", "Closed_Fist", 0.9), ("Right", "Thumb_Up", 0.9), ("Right", None, 0.0)])
    assert gh.hints_from_result(res) == [("Left", "FIST"), ("Right", None), ("Right", None)]
    low = _gesture_result([("Right", "Open_Palm", 0.3)])
    assert gh.hints_from_result(low, min_score=0.5) == [("Right", None)]
    assert gh.hints_from_result(low, min_score=0.0) == [("Right", "OPEN_PALM")]
    assert gh.hints_from_result(object()) == []


def test_hint_for_picks_matching_hand():
    hints = [("Left", "FIST"), ("Right", "POINT")]
    assert gh.hint_for(hints, "Right") == "POINT"
    assert gh.hint_for(hints, "Left") == "FIST"
    assert gh.hint_for([], "Right") is None


def test_hint_enabled_reads_only_merged_config(monkeypatch):
    """La env var se aplica en load_config (ENV_MAP); el helper no la relee, así
    `TLETL_GESTURE_HINT=0` no puede anular un `--gesture-hint` explícito."""
    monkeypatch.setenv("TLETL_GESTURE_HINT", "0")
    assert gh.hint_enabled({}) is False
    assert gh.hint_enabled({"gesture_hint": True}) is True      # CLI ya aplicado al dict
    assert gh.hint_enabled({"gesture_hint": "yes"}) is True
    assert gh.hint_enabled({"gesture_hint": "0"}) is False


def test_hint_precedence_cli_over_env_over_toml(monkeypatch, tmp_path):
    from tletl_core.config import load_config
    from apps.fedora_control.main import apply_cli_overrides, build_parser

    monkeypatch.delenv("TLETL_GESTURE_HINT", raising=False)
    cfg = load_config(tmp_path / "nope.toml")
    assert gh.hint_enabled(cfg["tracker"]) is False               # toml default: off
    monkeypatch.setenv("TLETL_GESTURE_HINT", "1")
    cfg = load_config(tmp_path / "nope.toml")
    assert gh.hint_enabled(cfg["tracker"]) is True                # env enciende
    monkeypatch.setenv("TLETL_GESTURE_HINT", "0")
    cfg = load_config(tmp_path / "nope.toml")
    apply_cli_overrides(build_parser().parse_args(["--gesture-hint"]), cfg)
    assert gh.hint_enabled(cfg["tracker"]) is True                # CLI gana a env=0


def test_gesture_model_path_follows_tracker_model(monkeypatch, tmp_path):
    monkeypatch.delenv("TLETL_GESTURE_MODEL", raising=False)
    monkeypatch.setenv("TLETL_TRACKER_MODEL", str(tmp_path / "m" / "hand_landmarker.task"))
    assert gh.resolve_gesture_model_path({}) == tmp_path / "m" / "gesture_recognizer.task"
    monkeypatch.setenv("TLETL_GESTURE_MODEL", str(tmp_path / "g.task"))
    assert gh.resolve_gesture_model_path({}) == tmp_path / "g.task"


def test_gesture_hint_create_disabled_or_missing_model_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("TLETL_HOME", str(tmp_path))
    monkeypatch.delenv("TLETL_GESTURE_HINT", raising=False)
    monkeypatch.delenv("TLETL_TRACKER_MODEL", raising=False)
    monkeypatch.delenv("TLETL_GESTURE_MODEL", raising=False)
    assert gh.GestureHint.create({"gesture_hint": False}) is None
    logs = []
    assert gh.GestureHint.create({"gesture_hint": True, "auto_download": False}, log=logs.append) is None
    assert any("tletl-fetch-models" in line for line in logs)


def test_gesture_hint_create_auto_downloads_then_builds(monkeypatch, tmp_path):
    """Con auto_download el modelo se descarga (descargador falso) y luego se intenta crear el
    recognizer; si mediapipe no lo puede crear (modelo falso) devuelve None sin lanzar."""
    monkeypatch.setenv("TLETL_HOME", str(tmp_path))
    monkeypatch.delenv("TLETL_GESTURE_HINT", raising=False)
    monkeypatch.delenv("TLETL_TRACKER_MODEL", raising=False)
    monkeypatch.delenv("TLETL_GESTURE_MODEL", raising=False)
    downloaded = []

    def fake(name, path):
        downloaded.append(name)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"not a real model")

    result = gh.GestureHint.create({"gesture_hint": True, "auto_download": True}, downloader=fake, log=lambda s: None)
    assert downloaded == ["gesture_recognizer.task"]
    assert (tmp_path / "models" / "gesture_recognizer.task").is_file()
    assert result is None or isinstance(result, gh.GestureHint)
    if result is not None:
        result.close()


def test_gesture_hint_process_with_fake_recognizer():
    class FakeRecognizer:
        def __init__(self):
            self.calls = []
            self.closed = False

        def recognize_for_video(self, image, ts):
            self.calls.append((image.shape, ts))
            return _gesture_result([("Right", "Victory", 0.8)])

        def close(self):
            self.closed = True

    rec = FakeRecognizer()
    hint = gh.GestureHint(rec, min_score=0.5, image_factory=lambda rgb: rgb, clock=lambda: 1.0)
    frame = np.zeros((36, 64, 3), dtype=np.uint8)
    assert hint.process(frame) == [("Right", "VICTORY")]
    assert hint.process(frame, timestamp_ms=1) == [("Right", "VICTORY")]
    ts = [c[1] for c in rec.calls]
    assert ts == sorted(ts) and len(set(ts)) == 2
    hint.close()
    assert rec.closed


# ── integración con el modelo real (se salta limpiamente si no se puede) ─────

def _real_model(name: str) -> Path:
    candidate = VENV_MODELS / name
    if candidate.is_file():
        return candidate
    fallback = ht.default_models_dir() / name
    if fallback.is_file():
        return fallback
    pytest.skip(f"modelo {name} no descargado (./launchers/tletl-fetch-models.sh)")


def test_tasks_backend_real_model_black_frame(monkeypatch):
    pytest.importorskip("mediapipe")
    model = _real_model("hand_landmarker.task")
    monkeypatch.setenv("TLETL_TRACKER_MODEL", str(model))
    try:
        tracker = HandTracker.create({"backend": "tasks", "det_conf": 0.5, "track_conf": 0.5, "auto_download": False},
                                     num_hands=2, log=lambda s: None)
    except (OSError, TrackerUnavailable) as exc:
        pytest.skip(f"MediaPipe Tasks no carga aquí: {exc}")
    try:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        assert tracker.process(frame) == []
        assert tracker.process(frame) == []
        assert tracker.backend_name == "tasks"
        assert tracker.frames == 2
    finally:
        tracker.close()


def test_gesture_hint_real_model_black_frame(monkeypatch):
    pytest.importorskip("mediapipe")
    model = _real_model("gesture_recognizer.task")
    monkeypatch.setenv("TLETL_GESTURE_MODEL", str(model))
    monkeypatch.setenv("TLETL_GESTURE_HINT", "1")
    hint = gh.GestureHint.create({"auto_download": False}, num_hands=2, log=lambda s: None)
    if hint is None:
        pytest.skip("GestureRecognizer no se pudo crear aquí")
    try:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        assert hint.process(frame) == []
        assert hint.process(frame) == []
    finally:
        hint.close()


def test_tasks_backend_real_model_noise_frames_and_timestamps(monkeypatch):
    """Frames de ruido (no negros) con timestamps externos repetidos/hacia atrás: no revienta,
    devuelve listas y MediaPipe recibe timestamps estrictamente crecientes."""
    pytest.importorskip("mediapipe")
    model = _real_model("hand_landmarker.task")
    monkeypatch.setenv("TLETL_TRACKER_MODEL", str(model))
    try:
        tracker = HandTracker.create({"backend": "tasks", "auto_download": False}, num_hands=2, log=lambda s: None)
    except (OSError, TrackerUnavailable) as exc:
        pytest.skip(f"MediaPipe Tasks no carga aquí: {exc}")
    rng = np.random.default_rng(0)
    used = []
    try:
        for ts in (5, 5, 3, None, 10):
            dets = tracker.process(rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8), timestamp_ms=ts)
            assert isinstance(dets, list)
            used.append(tracker.last_timestamp_ms)
    finally:
        tracker.close()
    assert used == sorted(used) and len(set(used)) == 5


def test_tasks_backend_corrupt_model_is_tracker_unavailable_with_hint(monkeypatch, tmp_path):
    """Regresión: un .task corrupto/truncado salía como RuntimeError crudo de MediaPipe
    ("Unable to open zip archive"), sin pista y sin pasar por el respaldo legacy de auto."""
    pytest.importorskip("mediapipe")
    if not ht.tasks_available():
        pytest.skip("Tasks API no disponible")
    bad = tmp_path / "models" / "hand_landmarker.task"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"garbage" * 100)
    monkeypatch.setenv("TLETL_HOME", str(tmp_path))
    monkeypatch.delenv("TLETL_TRACKER_MODEL", raising=False)

    with pytest.raises(TrackerUnavailable) as exc:
        HandTracker.create({"backend": "tasks", "auto_download": False}, log=lambda s: None)
    assert str(bad) in str(exc.value) and "--force" in str(exc.value)

    # auto: el fallo al crear el landmarker cae a legacy si existe (docstring del módulo)...
    _FakeLegacyBackend.created = []
    monkeypatch.setattr(ht, "_LegacyBackend", _FakeLegacyBackend)
    monkeypatch.setattr(ht, "legacy_available", lambda: True)
    logs = []
    t = HandTracker.create({"backend": "auto", "auto_download": False}, log=logs.append)
    assert t.backend_name == "legacy" and len(_FakeLegacyBackend.created) == 1
    assert any("legacy" in line and str(bad) in line for line in logs)
    # ...y si no existe (mediapipe 1.x) el error claro se propaga, no un RuntimeError anónimo.
    monkeypatch.setattr(ht, "legacy_available", lambda: False)
    with pytest.raises(TrackerUnavailable):
        HandTracker.create({"backend": "auto", "auto_download": False}, log=lambda s: None)
