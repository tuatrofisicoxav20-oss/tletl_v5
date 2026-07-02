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
