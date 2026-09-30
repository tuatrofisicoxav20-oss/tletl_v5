"""tests/test_adaptive.py

Tests para tletl_core.adaptive.AdaptiveGestureMemory.
"""

import json


from tletl_core.adaptive import AdaptiveGestureMemory


SAMPLE_FEATURES = {
    "angle_0": 0.12,
    "angle_1": 0.34,
    "angle_2": 0.56,
    "angle_3": 0.78,
    "ratio_0": 1.23,
    # posición absoluta — debe ser excluida
    "palm_center_x": 320.0,
    "palm_center_y": 240.0,
    "screen_x": 0.5,
    "screen_y": 0.5,
    "cx": 160.0,
    "cy": 120.0,
}


# ---------------------------------------------------------------------------
# enabled=False → observe() devuelve False y NO crea archivo
# ---------------------------------------------------------------------------

def test_disabled_observe_returns_false(tmp_path):
    db = tmp_path / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=False)
    result = mem.observe("FIST", SAMPLE_FEATURES, confidence=0.9)
    assert result is False


def test_disabled_does_not_create_file(tmp_path):
    db = tmp_path / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=False)
    mem.observe("FIST", SAMPLE_FEATURES, confidence=0.9)
    assert not db.exists(), "enabled=False no debe crear ningún archivo"


# ---------------------------------------------------------------------------
# enabled=True, confidence alta → guarda; archivo existe; sin palm_center_x
# ---------------------------------------------------------------------------

def test_enabled_high_confidence_saves(tmp_path):
    db = tmp_path / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.78)
    result = mem.observe("FIST", SAMPLE_FEATURES, confidence=0.9)
    assert result is True


def test_enabled_file_exists_after_save(tmp_path):
    db = tmp_path / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.78)
    mem.observe("FIST", SAMPLE_FEATURES, confidence=0.9)
    assert db.exists(), "El archivo JSON debería existir tras guardar"


def test_saved_sample_excludes_palm_center_x(tmp_path):
    db = tmp_path / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.78)
    mem.observe("FIST", SAMPLE_FEATURES, confidence=0.9)

    data = json.loads(db.read_text(encoding="utf-8"))
    sample = data["gestures"]["FIST"][0]
    assert "palm_center_x" not in sample, (
        "La muestra no debe contener 'palm_center_x'"
    )


def test_saved_sample_excludes_all_position_keys(tmp_path):
    db = tmp_path / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.78)
    mem.observe("FIST", SAMPLE_FEATURES, confidence=0.9)

    data = json.loads(db.read_text(encoding="utf-8"))
    sample = data["gestures"]["FIST"][0]
    for banned_key in ("palm_center_x", "palm_center_y", "screen_x", "screen_y", "cx", "cy"):
        assert banned_key not in sample, f"La muestra no debe contener '{banned_key}'"


def test_saved_sample_contains_allowed_keys(tmp_path):
    db = tmp_path / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.78)
    mem.observe("FIST", SAMPLE_FEATURES, confidence=0.9)

    data = json.loads(db.read_text(encoding="utf-8"))
    sample = data["gestures"]["FIST"][0]
    assert "angle_0" in sample


# ---------------------------------------------------------------------------
# enabled=True, confidence baja → no guarda
# ---------------------------------------------------------------------------

def test_low_confidence_does_not_save(tmp_path):
    db = tmp_path / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.78)
    result = mem.observe("FIST", SAMPLE_FEATURES, confidence=0.5)
    assert result is False


def test_low_confidence_no_file(tmp_path):
    db = tmp_path / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.78)
    mem.observe("FIST", SAMPLE_FEATURES, confidence=0.5)
    assert not db.exists(), "Confianza baja no debe crear archivo"


# ---------------------------------------------------------------------------
# Cap de muestras por gesto
# ---------------------------------------------------------------------------

def test_cap_not_exceeded(tmp_path):
    db = tmp_path / "adaptive.json"
    cap = 10
    mem = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.5, max_samples_per_gesture=cap)

    for i in range(cap + 5):
        mem.observe("OPEN_PALM", {"feat_a": float(i), "feat_b": float(i) * 0.1}, confidence=0.9)

    counts = mem.counts()
    assert counts.get("OPEN_PALM", 0) <= cap, (
        f"Se excedió el cap: {counts['OPEN_PALM']} > {cap}"
    )


def test_cap_exactly_at_limit(tmp_path):
    db = tmp_path / "adaptive.json"
    cap = 5
    mem = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.5, max_samples_per_gesture=cap)

    for i in range(cap):
        mem.observe("VICTORY", {"feat_a": float(i)}, confidence=0.9)

    assert mem.counts().get("VICTORY", 0) == cap


# ---------------------------------------------------------------------------
# Persistencia: los datos se cargan al instanciar de nuevo
# ---------------------------------------------------------------------------

def test_persistence_reload(tmp_path):
    db = tmp_path / "adaptive.json"

    mem1 = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.5)
    mem1.observe("FIST", {"feat_a": 1.0, "feat_b": 2.0}, confidence=0.9)

    mem2 = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.5)
    assert mem2.counts().get("FIST", 0) >= 1, "Los datos deben persistir entre instancias"


# ---------------------------------------------------------------------------
# Escritura throttled (antes: un JSON completo a disco por cada observe/frame)
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _disk_count(db, gesture="FIST") -> int:
    data = json.loads(db.read_text(encoding="utf-8"))
    return len(data["gestures"].get(gesture, []))


def test_first_save_is_immediate_then_throttled(tmp_path):
    clock = FakeClock()
    db = tmp_path / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.5,
                                save_interval=2.0, clock=clock)

    assert mem.observe("FIST", {"a": 1.0}, confidence=0.9) is True
    assert db.exists() and _disk_count(db) == 1        # primera escritura: inmediata
    assert mem.dirty is False

    clock.advance(0.5)
    assert mem.observe("FIST", {"a": 2.0}, confidence=0.9) is True
    assert mem.counts()["FIST"] == 2                   # en memoria sí
    assert _disk_count(db) == 1                        # en disco todavía no
    assert mem.dirty is True

    clock.advance(1.0)                                 # 1.5 s < 2.0 s
    mem.observe("FIST", {"a": 3.0}, confidence=0.9)
    assert _disk_count(db) == 1 and mem.dirty

    clock.advance(0.6)                                 # 2.1 s >= 2.0 s
    mem.observe("FIST", {"a": 4.0}, confidence=0.9)
    assert _disk_count(db) == 4 and not mem.dirty


def test_flush_writes_pending_samples(tmp_path):
    clock = FakeClock()
    db = tmp_path / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.5,
                                save_interval=60.0, clock=clock)
    mem.observe("FIST", {"a": 1.0}, confidence=0.9)
    mem.observe("FIST", {"a": 2.0}, confidence=0.9)
    assert _disk_count(db) == 1 and mem.dirty
    assert mem.flush() is True
    assert _disk_count(db) == 2 and not mem.dirty
    assert mem.flush() is False                        # nada pendiente


def test_flush_disabled_or_clean_does_not_touch_disk(tmp_path):
    db = tmp_path / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=False)
    mem.observe("FIST", {"a": 1.0}, confidence=0.9)
    assert mem.flush() is False
    assert not db.exists()


def test_save_interval_zero_writes_every_observe(tmp_path):
    clock = FakeClock()
    db = tmp_path / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.5,
                                save_interval=0.0, clock=clock)
    for i in range(3):
        mem.observe("FIST", {"a": float(i)}, confidence=0.9)
        assert _disk_count(db) == i + 1


def test_persistence_reload_after_flush_keeps_everything(tmp_path):
    clock = FakeClock()
    db = tmp_path / "adaptive.json"
    mem1 = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.5,
                                 save_interval=30.0, clock=clock)
    for i in range(5):
        mem1.observe("POINT", {"a": float(i)}, confidence=0.9)
    mem1.flush()
    mem2 = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.5)
    assert mem2.counts()["POINT"] == 5


# ---------------------------------------------------------------------------
# samples(): lo que consume RobustKNNRuntime(extra_samples=...)
# ---------------------------------------------------------------------------

def test_samples_returns_label_feature_tuples(tmp_path):
    db = tmp_path / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.5)
    mem.observe("FIST", SAMPLE_FEATURES, confidence=0.9)
    mem.observe("pinch", {"angle_0": 0.5, "ratio_0": 1.0}, confidence=0.9)
    samples = mem.samples()
    assert len(samples) == 2
    labels = sorted(label for label, _ in samples)
    assert labels == ["FIST", "PINCH"]
    fist = next(feat for label, feat in samples if label == "FIST")
    assert "angle_0" in fist and "palm_center_x" not in fist
    assert all(isinstance(v, float) for v in fist.values())


def test_samples_empty_when_disabled_or_new(tmp_path):
    assert AdaptiveGestureMemory(tmp_path / "a.json", enabled=False).samples() == []
    assert AdaptiveGestureMemory(tmp_path / "b.json", enabled=True).samples() == []


def test_samples_skips_malformed_entries(tmp_path):
    db = tmp_path / "adaptive.json"
    db.write_text(json.dumps({
        "gestures": {
            "FIST": [{"a": 1.0}, "basura", {"a": "texto"}, {"a": float("nan")}, 42],
            "POINT": "no es lista",
        }
    }), encoding="utf-8")
    mem = AdaptiveGestureMemory(db, enabled=True)
    assert mem.samples() == [("FIST", {"a": 1.0})]


# ---------------------------------------------------------------------------
# BUG: la ruta por default (~/.tletl/) puede no existir -> FileNotFoundError
# ---------------------------------------------------------------------------

def test_creates_missing_parent_directory(tmp_path):
    db = tmp_path / "no" / "existe" / "adaptive.json"
    mem = AdaptiveGestureMemory(db, enabled=True, min_confidence=0.5)
    assert mem.observe("FIST", {"a": 1.0}, confidence=0.9) is True
    assert db.exists()
