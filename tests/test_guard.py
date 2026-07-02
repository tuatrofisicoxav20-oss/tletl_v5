"""tests/test_guard.py – Tests para tletl_core/guard.py."""

from __future__ import annotations

import sys
from pathlib import Path

import importlib.util


ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tletl_core.classifier import Prediction
from tletl_core.features import extract_live_features
from tletl_core.guard import geometric_rule, rule_guard

# Importamos make_landmarks desde conftest sin depender del fixture de pytest
# (tests/ no es un package, así que se importa por path)
_conftest_spec = importlib.util.spec_from_file_location("_conftest", TESTS_DIR / "conftest.py")
_conftest_mod = importlib.util.module_from_spec(_conftest_spec)
_conftest_spec.loader.exec_module(_conftest_mod)
make_landmarks = _conftest_mod.make_landmarks

# Defaults usados en varios tests
DANGEROUS = ["PINCH", "POINT", "VICTORY", "THREE", "FIST"]
CONF_THRESHOLD = 0.58
MARGIN_THRESHOLD = 0.12


def _make_prediction(raw_label: str, confidence: float, margin: float) -> Prediction:
    """Construye un Prediction mínimo para los tests del guard."""
    return Prediction(
        label="NEUTRAL" if confidence < 0.47 else raw_label,
        raw_label=raw_label,
        confidence=confidence,
        margin=margin,
        orientation="PALM_FRONT",
        reason="TEST",
        ok=confidence >= 0.47,
        votes={raw_label: confidence},
    )


# ---------------------------------------------------------------------------
# Tests de rule_guard
# ---------------------------------------------------------------------------

class TestRuleGuard:
    def test_pinch_low_conf_rule_neutral_is_blocked(self):
        """PINCH con conf baja y rule_raw='NEUTRAL' → bloqueado."""
        pred = _make_prediction("PINCH", confidence=0.40, margin=0.08)
        gesture, reason = rule_guard(
            pred,
            rule_raw="NEUTRAL",
            dangerous=DANGEROUS,
            conf_threshold=CONF_THRESHOLD,
            margin_threshold=MARGIN_THRESHOLD,
        )
        assert gesture == "NEUTRAL"
        assert reason.startswith("GUARD_BLOCK")

    def test_pinch_rule_agrees_passes(self):
        """PINCH con conf baja pero rule_raw='PINCH' → pasa (coinciden)."""
        pred = _make_prediction("PINCH", confidence=0.40, margin=0.08)
        gesture, reason = rule_guard(
            pred,
            rule_raw="PINCH",
            dangerous=DANGEROUS,
            conf_threshold=CONF_THRESHOLD,
            margin_threshold=MARGIN_THRESHOLD,
        )
        assert gesture == "PINCH"
        assert reason == "OK"

    def test_non_dangerous_gesture_always_passes(self):
        """Gesto no peligroso (NEUTRAL) pasa siempre aunque rule difiera."""
        pred = _make_prediction("NEUTRAL", confidence=0.30, margin=0.05)
        gesture, reason = rule_guard(
            pred,
            rule_raw="OPEN_PALM",
            dangerous=DANGEROUS,
            conf_threshold=CONF_THRESHOLD,
            margin_threshold=MARGIN_THRESHOLD,
        )
        assert gesture == "NEUTRAL"
        assert reason == "OK"

    def test_high_conf_dangerous_passes_despite_rule_mismatch(self):
        """IA segura (conf alta, margin alto) se impone aunque rule difiera."""
        pred = _make_prediction("PINCH", confidence=0.85, margin=0.30)
        gesture, reason = rule_guard(
            pred,
            rule_raw="NEUTRAL",
            dangerous=DANGEROUS,
            conf_threshold=CONF_THRESHOLD,
            margin_threshold=MARGIN_THRESHOLD,
        )
        assert gesture == "PINCH"
        assert reason == "OK"

    def test_open_palm_non_dangerous_always_passes(self):
        """OPEN_PALM (no está en dangerous) nunca es bloqueado."""
        pred = _make_prediction("OPEN_PALM", confidence=0.20, margin=0.01)
        gesture, reason = rule_guard(
            pred,
            rule_raw="FIST",
            dangerous=DANGEROUS,
            conf_threshold=CONF_THRESHOLD,
            margin_threshold=MARGIN_THRESHOLD,
        )
        assert gesture == "OPEN_PALM"
        assert reason == "OK"


# ---------------------------------------------------------------------------
# Tests de geometric_rule
# ---------------------------------------------------------------------------

class TestGeometricRule:
    def test_fist_not_open_palm(self):
        """geometric_rule sobre FIST no devuelve OPEN_PALM."""
        lm = make_landmarks("FIST")
        features = extract_live_features(lm)
        result = geometric_rule(features)
        assert result != "OPEN_PALM"

    def test_open_palm_not_fist(self):
        """geometric_rule sobre OPEN_PALM no devuelve FIST."""
        lm = make_landmarks("OPEN_PALM")
        features = extract_live_features(lm)
        result = geometric_rule(features)
        assert result != "FIST"

    def test_result_is_valid_label(self):
        """geometric_rule siempre devuelve un label de LABELS."""
        from tletl_core.features import LABELS
        for gesture in ["OPEN_PALM", "FIST", "POINT", "VICTORY", "THREE"]:
            lm = make_landmarks(gesture)
            features = extract_live_features(lm)
            result = geometric_rule(features)
            assert result in LABELS, f"geometric_rule('{gesture}') -> '{result}' no está en LABELS"
