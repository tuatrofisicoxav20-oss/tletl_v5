from __future__ import annotations

"""Edge cases y robustez: entradas vacías, NaN, overrides de entorno, dos manos.

Estos tests EJECUTAN los límites para destapar crashes que los tests felices no ven.
"""

import math

import pytest

from tletl_core.classifier import RobustKNNRuntime
from tletl_core.config import load_config
from tletl_core.guard import geometric_rule
from tletl_core.critic import strict_critic
from tletl_core.orientation import orientation_bucket
from tletl_core.pipeline import TletlPipeline


@pytest.fixture(scope="module")
def runtime(bank_path):
    return RobustKNNRuntime(bank_path, k=13)


@pytest.fixture(scope="module")
def pipeline(bank_path):
    return TletlPipeline(bank_path, config=load_config(),
                         adaptive_path="/tmp/tletl_adaptive_edge_ignore.json")


# ── Entradas degeneradas ─────────────────────────────────────────────────────

def test_classifier_empty_features_no_crash(runtime):
    pred = runtime.predict({}, strict=True)
    assert pred.label in ("NEUTRAL", *runtime.counts.keys())
    assert math.isfinite(pred.confidence)


def test_classifier_nan_features_no_crash(runtime):
    feats = {"index_tip_wrist": float("nan"), "thumb_tip_index_tip": float("inf"), "palm_width": 1.0}
    pred = runtime.predict(feats, strict=False)
    assert math.isfinite(pred.confidence)


def test_pipeline_empty_features_no_crash(pipeline):
    res = pipeline.process_features({}, hand="edge")
    assert res.stable_gesture  # algún string, sin crash
    assert 0.0 <= res.confidence <= 1.0


def test_geometric_rule_empty_is_neutral_or_fist():
    # sin features, todos los dedos cuentan como no-extendidos → FIST o NEUTRAL, sin crash
    assert geometric_rule({}) in ("NEUTRAL", "FIST")


def test_orientation_bucket_empty_is_unknown():
    assert orientation_bucket({}) == "UNKNOWN"


def test_critic_neutral_always_accepted():
    r = strict_critic("NEUTRAL", 0.1, {}, {})
    assert r.accepted and r.gesture == "NEUTRAL"


def test_critic_unknown_gesture_rejected():
    r = strict_critic("BANANA", 0.99, {}, {})
    assert not r.accepted


# ── Configuración / entorno ──────────────────────────────────────────────────

def test_env_override_k(monkeypatch):
    monkeypatch.setenv("TLETL_AI_V47_K", "7")
    cfg = load_config()
    assert cfg["classifier"]["k"] == 7


def test_env_override_dry_run(monkeypatch):
    monkeypatch.setenv("TLETL_DRY_RUN", "1")
    cfg = load_config()
    assert cfg["fedora"]["dry_run"] is True


def test_config_defaults_without_toml(tmp_path):
    # un toml inexistente → defaults
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg["classifier"]["k"] == 13
    assert cfg["adaptive"]["enabled"] is False


# ── Bus con dos manos ────────────────────────────────────────────────────────

def test_bus_serializes_both_hands(tmp_path, pipeline, bank_rows):
    from tletl_core.bus import TletlStateBus
    from tletl_core.features import get_features, get_label
    from tletl_core.state import TletlFrameState

    fist = [get_features(o) for o in bank_rows if get_label(o) == "FIST"][:8]
    palm = [get_features(o) for o in bank_rows if get_label(o) == "OPEN_PALM"][:8]
    pipeline.reset()
    dr = mr = None
    for i in range(8):
        dr = pipeline.process_features(fist[i % len(fist)], hand="dom")
        mr = pipeline.process_features(palm[i % len(palm)], hand="mod")

    st = TletlFrameState()
    st.dom = pipeline.to_hand_state(dr, side="Right")
    st.mod = pipeline.to_hand_state(mr, side="Left")
    bus = TletlStateBus(tmp_path / "s.json")
    bus.write(st)
    data = bus.read()
    assert data["dom"]["present"] and data["mod"]["present"]
    assert data["dom"]["side"] == "Right" and data["mod"]["side"] == "Left"
