from __future__ import annotations

import math

from tletl_core.features import extract_live_features, get_features, get_label, LABELS
from conftest import make_landmarks


def test_extract_returns_finite_features():
    lm = make_landmarks("OPEN_PALM")
    feats = extract_live_features(lm)
    assert len(feats) >= 30
    for k, v in feats.items():
        assert math.isfinite(v), f"feature no finita: {k}={v}"


def test_orientation_features_present():
    feats = extract_live_features(make_landmarks("OPEN_PALM"))
    for key in ("palm_normal_x", "palm_facing_score", "palm_roll"):
        assert key in feats


def test_pinch_reduces_thumb_index_distance():
    open_feats = extract_live_features(make_landmarks("OPEN_PALM"))
    pinch_feats = extract_live_features(make_landmarks("PINCH"))
    assert pinch_feats["thumb_tip_index_tip"] < open_feats["thumb_tip_index_tip"]


def test_get_label_variants():
    assert get_label({"label": "pinch"}) == "PINCH"
    assert get_label({"gesture": "FIST"}) == "FIST"
    assert get_label({"nope": 1}) is None


def test_get_features_filters_meta():
    obj = {"label": "FIST", "timestamp": 123, "features": {"a": 1.0, "b": 2.0}}
    feats = get_features(obj)
    assert feats == {"a": 1.0, "b": 2.0}


def test_labels_count():
    assert len(LABELS) == 7
    assert "NEUTRAL" in LABELS
