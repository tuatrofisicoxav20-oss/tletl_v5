"""tests/test_lowlight.py

Tests para tletl_core.lowlight.LowLightEnhancer.
"""

import numpy as np
import pytest

from tletl_core.lowlight import LowLightEnhancer


@pytest.fixture
def enh():
    return LowLightEnhancer()


# ---------------------------------------------------------------------------
# Frame oscuro (luma ≈ 12) → modo distinto de "normal"
# ---------------------------------------------------------------------------

def test_dark_frame_mode_not_normal(enh):
    frame = np.full((48, 64, 3), 12, dtype=np.uint8)
    enh.enhance(frame)
    assert enh.last_mode != "normal", (
        f"Se esperaba modo distinto de 'normal' para frame oscuro, "
        f"pero got '{enh.last_mode}' (last_luma={enh.last_luma:.1f})"
    )


# ---------------------------------------------------------------------------
# Frame brillante (luma ≈ 200) → modo "normal"
# ---------------------------------------------------------------------------

def test_bright_frame_mode_normal(enh):
    frame = np.full((48, 64, 3), 200, dtype=np.uint8)
    enh.enhance(frame)
    assert enh.last_mode == "normal", (
        f"Se esperaba modo 'normal' para frame brillante, "
        f"pero got '{enh.last_mode}' (last_luma={enh.last_luma:.1f})"
    )


# ---------------------------------------------------------------------------
# enhance() devuelve array del mismo shape y dtype uint8
# ---------------------------------------------------------------------------

def test_enhance_returns_same_shape_and_dtype(enh):
    frame = np.full((48, 64, 3), 12, dtype=np.uint8)
    result = enh.enhance(frame)
    assert result.shape == frame.shape, (
        f"Shape cambia: {frame.shape} → {result.shape}"
    )
    assert result.dtype == np.uint8, (
        f"dtype cambia: esperaba uint8, got {result.dtype}"
    )


def test_enhance_bright_returns_same_shape_and_dtype(enh):
    frame = np.full((48, 64, 3), 200, dtype=np.uint8)
    result = enh.enhance(frame)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# last_luma se actualiza
# ---------------------------------------------------------------------------

def test_last_luma_updated(enh):
    frame = np.full((48, 64, 3), 50, dtype=np.uint8)
    enh.enhance(frame)
    assert enh.last_luma > 0.0, "last_luma debería ser > 0 para un frame no negro"


# ---------------------------------------------------------------------------
# Parámetros personalizados
# ---------------------------------------------------------------------------

def test_custom_clip_limit_instantiation():
    enh = LowLightEnhancer(clip_limit=4.0, gamma_dark=1.8)
    assert enh.clip_limit == 4.0
    assert enh.gamma_dark == 1.8
    frame = np.full((48, 64, 3), 30, dtype=np.uint8)
    result = enh.enhance(frame)
    assert result.dtype == np.uint8
