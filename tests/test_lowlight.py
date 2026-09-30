"""tests/test_lowlight.py

Tests para tletl_core.lowlight.LowLightEnhancer.
"""

import cv2
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


# ---------------------------------------------------------------------------
# gamma_dark deriva los tres exponentes (antes se guardaba sin usarse) y las
# LUT se calculan una sola vez. BUG: la gamma estaba doblemente invertida y
# el "realce" OSCURECÍA (128 -> 83 en el canal Y).
# ---------------------------------------------------------------------------

def _textured_frame(mean: float, seed: int = 0) -> np.ndarray:
    """Frame BGR con gradiente + ruido y luma media ~mean (para que CLAHE tenga textura)."""
    rng = np.random.default_rng(seed)
    base = np.linspace(mean - 25, mean + 25, 64, dtype=np.float32)[None, :].repeat(48, 0)
    frame = base[..., None] + rng.normal(0.0, 6.0, (48, 64, 3))
    return np.clip(frame, 0, 255).astype(np.uint8)


def _luma(img: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0].mean())


def _expected_with_old_constant(frame: np.ndarray, exponent: float, clip_limit: float = 2.0) -> np.ndarray:
    """Mismo flujo (CLAHE + LUT + merge) con la constante histórica como EXPONENTE."""
    y, cr, cb = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lut = np.array([(i / 255.0) ** exponent * 255 for i in range(256)], dtype=np.uint8)
    y3 = cv2.LUT(clahe.apply(y), lut)
    return cv2.cvtColor(cv2.merge([y3, cr, cb]), cv2.COLOR_YCrCb2BGR)


def test_gamma_dark_16_reproduces_historical_exponents():
    enh = LowLightEnhancer(gamma_dark=1.6)
    assert enh.exponents["dark"] == pytest.approx(0.62, abs=0.006)     # 1/1.6 = 0.625
    assert enh.exponents["medium"] == pytest.approx(0.76, abs=0.002)   # 1/(1+0.6*0.53)
    assert enh.exponents["low"] == pytest.approx(0.88, abs=0.001)      # 1/(1+0.6*0.227)


@pytest.mark.parametrize("mean,mode,old_exponent", [
    (30.0, "dark", 0.62),
    (75.0, "medium", 0.76),
    (100.0, "low", 0.88),
])
def test_gamma_dark_16_matches_old_constants_output(mean, mode, old_exponent):
    """Salida igual a la de las constantes viejas (±1 nivel por 0.625 vs 0.62)."""
    frame = _textured_frame(mean)
    enh = LowLightEnhancer(gamma_dark=1.6)
    out = enh.enhance(frame)
    assert enh.last_mode == mode
    expected = _expected_with_old_constant(frame, old_exponent)
    diff = np.abs(out.astype(np.int16) - expected.astype(np.int16))
    # 255·(x^0.62 - x^0.625) < 0.76 -> la LUT difiere en <= 1 nivel de Y, que la
    # conversión YCrCb->BGR propaga a <= 1 (2 con redondeo) por canal.
    assert diff.max() <= 2
    assert diff.mean() <= 1.0


def test_enhancer_brightens_dark_frames_and_more_with_bigger_gamma():
    frame = _textured_frame(30.0)
    out16 = LowLightEnhancer(gamma_dark=1.6).enhance(frame)
    out24 = LowLightEnhancer(gamma_dark=2.4).enhance(frame)
    assert _luma(out16) > _luma(frame)
    assert _luma(out24) > _luma(out16)
    # la LUT del modo oscuro ACLARA (con el bug 128 -> 83)
    assert LowLightEnhancer(gamma_dark=1.6)._luts["dark"][128] > 128
    assert LowLightEnhancer(gamma_dark=2.4)._luts["dark"][128] > LowLightEnhancer(gamma_dark=1.6)._luts["dark"][128]


def test_luts_are_cached_not_rebuilt_per_frame(monkeypatch):
    enh = LowLightEnhancer(gamma_dark=1.6)
    assert set(enh._luts) == {"dark", "medium", "low"}
    assert all(lut.shape == (256,) and lut.dtype == np.uint8 for lut in enh._luts.values())

    def boom(*_a, **_k):
        raise AssertionError("la LUT no debe reconstruirse en enhance()")

    monkeypatch.setattr(enh, "_build_lut", boom)
    for mean in (30.0, 75.0, 100.0, 200.0):
        enh.enhance(_textured_frame(mean))


def test_mode_for_thresholds():
    enh = LowLightEnhancer()
    assert enh.mode_for(10.0) == "dark"
    assert enh.mode_for(61.9) == "dark"
    assert enh.mode_for(62.0) == "medium"
    assert enh.mode_for(87.9) == "medium"
    assert enh.mode_for(88.0) == "low"
    assert enh.mode_for(111.9) == "low"
    assert enh.mode_for(112.0) == "normal"


def test_invalid_gamma_dark_rejected():
    with pytest.raises(ValueError):
        LowLightEnhancer(gamma_dark=0.0)
    with pytest.raises(ValueError):
        LowLightEnhancer(gamma_dark=-1.0)
