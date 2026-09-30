"""
Tests del crítico estricto de Tletl v5.

Los landmarks sintéticos de make_landmarks producen nz ≈ 1.0 (normal apuntando
hacia la cámara), lo que se mapea a:
  palm_facing_score ≈ 1.0, back_hand_score ≈ 0.0, side_hand_score ≈ 0.0
→ orientation_bucket devuelve 'PALM_FRONT', coherente con gestos de acción.
"""

from __future__ import annotations

from tletl_core.critic import (
    ACTION_GESTURES,
    MIN_CONF,
    CriticResult,
    strict_critic,
)
from tletl_core.features import extract_live_features
from conftest import make_landmarks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feats(gesture: str = "PINCH") -> dict:
    """Features reales extraídas de landmarks sintéticos."""
    return extract_live_features(make_landmarks(gesture))


def _feats_palm_front() -> dict:
    """Features con palm_facing_score alto explícito (garantiza PALM_FRONT)."""
    feats = _feats("PINCH")
    feats["palm_facing_score"] = 0.90
    feats["back_hand_score"] = 0.05
    feats["side_hand_score"] = 0.05
    return feats


# ---------------------------------------------------------------------------
# Test 1: gesto de acción con confianza muy baja → rechazado
# ---------------------------------------------------------------------------

def test_action_low_confidence_rejected():
    """PINCH con confianza 0.30 (< MIN_CONF 0.72) debe ser rechazado."""
    feats = _feats("PINCH")
    result = strict_critic("PINCH", 0.30, feats)
    assert isinstance(result, CriticResult)
    assert result.accepted is False
    assert result.gesture == result.fallback  # gesture debe ser el fallback
    assert result.gesture == "NEUTRAL"
    assert "confianza baja" in result.reason.lower() or "0.30" in result.reason


# ---------------------------------------------------------------------------
# Test 2: gesto de acción con confianza alta y orientación frontal → aceptado
# ---------------------------------------------------------------------------

def test_action_high_confidence_palm_front_accepted():
    """PINCH con confianza 0.95 y mano de frente (PALM_FRONT) debe ser aceptado."""
    feats = _feats_palm_front()
    result = strict_critic("PINCH", 0.95, feats)
    assert isinstance(result, CriticResult)
    assert result.accepted is True
    assert result.gesture == "PINCH"


# ---------------------------------------------------------------------------
# Test 3: NEUTRAL siempre aceptado (no es gesto de acción)
# ---------------------------------------------------------------------------

def test_neutral_always_accepted():
    """NEUTRAL con confianza mínima suficiente siempre debe ser aceptado."""
    result = strict_critic("NEUTRAL", 0.70, {})
    assert isinstance(result, CriticResult)
    assert result.accepted is True
    assert result.gesture == "NEUTRAL"


def test_neutral_accepted_without_features():
    """NEUTRAL no requiere orientación aunque require_orientation=True."""
    result = strict_critic("NEUTRAL", 0.60, features={}, require_orientation=True)
    assert result.accepted is True
    assert result.gesture == "NEUTRAL"


# ---------------------------------------------------------------------------
# Test 4: CriticResult tiene los campos esperados
# ---------------------------------------------------------------------------

def test_critic_result_fields():
    """CriticResult debe tener los campos: accepted, gesture, reason, fallback."""
    result = strict_critic("OPEN_PALM", 0.80, _feats("OPEN_PALM"))
    assert hasattr(result, "accepted")
    assert hasattr(result, "gesture")
    assert hasattr(result, "reason")
    assert hasattr(result, "fallback")
    assert isinstance(result.accepted, bool)
    assert isinstance(result.gesture, str)
    assert isinstance(result.reason, str)
    assert result.fallback == "NEUTRAL"


# ---------------------------------------------------------------------------
# Tests adicionales de robustez
# ---------------------------------------------------------------------------

def test_invalid_gesture_rejected():
    """Un gesto desconocido debe ser rechazado."""
    result = strict_critic("HELICOPTER", 0.99, {})
    assert result.accepted is False
    assert result.gesture == "NEUTRAL"


def test_action_side_hand_rejected():
    """Gesto de acción con SIDE_HAND (orientación incoherente) debe rechazarse."""
    feats = _feats("PINCH")
    # Forzar side_hand dominante
    feats["palm_facing_score"] = 0.10
    feats["back_hand_score"] = 0.10
    feats["side_hand_score"] = 0.85
    result = strict_critic("PINCH", 0.95, feats, require_orientation=True)
    assert result.accepted is False
    assert "orientación" in result.reason.lower() or "SIDE_HAND" in result.reason


def test_require_orientation_false_skips_orientation_check():
    """Con require_orientation=False no se valida el bucket de orientación."""
    feats = _feats("PINCH")
    feats["palm_facing_score"] = 0.10
    feats["back_hand_score"] = 0.10
    feats["side_hand_score"] = 0.85
    result = strict_critic("PINCH", 0.95, feats, require_orientation=False)
    assert result.accepted is True
    assert result.gesture == "PINCH"


def test_action_unknown_orientation_high_conf_accepted():
    """Gesto de acción sin features de orientación pero conf muy alta → aceptado."""
    # Sin pasar features → no hay claves de orientación → fallback a conf alta
    result = strict_critic("FIST", 0.90, features={}, require_orientation=True)
    assert result.accepted is True
    assert result.gesture == "FIST"


def test_action_unknown_orientation_low_conf_rejected():
    """Gesto de acción sin features de orientación y conf no alta → rechazado."""
    result = strict_critic("FIST", 0.75, features={}, require_orientation=True)
    assert result.accepted is False
    assert result.gesture == "NEUTRAL"


def test_min_conf_dict_completeness():
    """MIN_CONF debe cubrir todos los gestos válidos."""
    from tletl_core.critic import VALID_GESTURES
    assert set(MIN_CONF.keys()) == VALID_GESTURES


def test_action_gestures_subset_of_valid():
    """ACTION_GESTURES debe ser subconjunto de los gestos válidos."""
    from tletl_core.critic import VALID_GESTURES
    assert ACTION_GESTURES.issubset(VALID_GESTURES)


# ---------------------------------------------------------------------------
# Tabla de umbrales inyectable ([critic.min_conf] del toml; antes ignorada)
# ---------------------------------------------------------------------------

def _feats_palm_front_for(gesture: str) -> dict:
    feats = _feats(gesture)
    feats["palm_facing_score"] = 0.90
    feats["back_hand_score"] = 0.05
    feats["side_hand_score"] = 0.05
    return feats


def test_min_conf_table_lowers_pinch_threshold():
    """Con PINCH=0.60 en la tabla, PINCH a 0.65 con PALM_FRONT se acepta."""
    result = strict_critic("PINCH", 0.65, _feats_palm_front(), min_conf={"PINCH": 0.60})
    assert result.accepted is True
    assert result.gesture == "PINCH"


def test_default_table_still_rejects_pinch_065():
    """Sin tabla, manda MIN_CONF del módulo (PINCH 0.72): 0.65 se rechaza."""
    result = strict_critic("PINCH", 0.65, _feats_palm_front())
    assert result.accepted is False
    assert result.gesture == "NEUTRAL"
    assert "0.72" in result.reason


def test_min_conf_missing_gestures_fall_back_to_module_defaults():
    """Una tabla que solo trae PINCH no toca el umbral de FIST (0.68)."""
    table = {"PINCH": 0.60}
    rejected = strict_critic("FIST", 0.67, _feats_palm_front_for("FIST"), min_conf=table)
    assert rejected.accepted is False and "0.68" in rejected.reason
    accepted = strict_critic("FIST", 0.69, _feats_palm_front_for("FIST"), min_conf=table)
    assert accepted.accepted is True and accepted.gesture == "FIST"


def test_min_conf_keys_are_case_insensitive_and_values_coerced():
    result = strict_critic("PINCH", 0.65, _feats_palm_front(), min_conf={"pinch": "0.6"})
    assert result.accepted is True


def test_min_conf_does_not_mutate_module_table():
    before = dict(MIN_CONF)
    strict_critic("PINCH", 0.65, _feats_palm_front(), min_conf={"PINCH": 0.10, "FIST": 0.10})
    assert MIN_CONF == before


def test_min_conf_from_config_helper_accepts_daytime_pinch():
    """critic_min_conf(load_config()) trae PINCH=0.62 (evidencia de validación):
    una pinza diurna a 0.65 ya no muere en el critic."""
    from pathlib import Path
    from tletl_core.config import critic_min_conf, load_config
    table = critic_min_conf(load_config(Path("/no/existe/tletl.toml")))
    assert table["PINCH"] == 0.62
    result = strict_critic("PINCH", 0.65, _feats_palm_front(), min_conf=table)
    assert result.accepted is True and result.gesture == "PINCH"


def test_min_conf_affects_high_conf_fallback_without_orientation():
    """Sin orientación, el umbral 'muy alta' es max(0.82, min_conf+0.08): sube con la tabla."""
    assert strict_critic("FIST", 0.90, features={}, min_conf={"FIST": 0.68}).accepted is True
    assert strict_critic("FIST", 0.90, features={}, min_conf={"FIST": 0.85}).accepted is False
