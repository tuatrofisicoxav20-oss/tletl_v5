from __future__ import annotations

"""Fixtures comunes de los tests de Tletl v5.

Importante: el core trabaja sobre landmarks ya en forma de Point (no necesita
mediapipe). Aquí generamos landmarks sintéticos plausibles para cada gesto y
exponemos el banco real para los tests del clasificador.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tletl_core.geometry import Point  # noqa: E402

BANK_PATH = ROOT / "datasets" / "tletl_gesture_bank_v2_features.jsonl"


# Plantilla de una mano abierta en coordenadas normalizadas estilo MediaPipe.
# 21 landmarks: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
_OPEN_HAND = [
    (0.50, 0.90, 0.00),   # 0 wrist
    (0.42, 0.82, 0.01),   # 1 thumb cmc
    (0.37, 0.74, 0.01),   # 2 thumb mcp
    (0.33, 0.67, 0.02),   # 3 thumb ip
    (0.30, 0.60, 0.02),   # 4 thumb tip
    (0.45, 0.62, 0.00),   # 5 index mcp
    (0.44, 0.50, 0.00),   # 6 index pip
    (0.43, 0.40, 0.00),   # 7 index dip
    (0.42, 0.30, 0.00),   # 8 index tip
    (0.52, 0.60, 0.00),   # 9 middle mcp
    (0.52, 0.47, 0.00),   # 10 middle pip
    (0.52, 0.36, 0.00),   # 11 middle dip
    (0.52, 0.25, 0.00),   # 12 middle tip
    (0.59, 0.62, 0.00),   # 13 ring mcp
    (0.60, 0.50, 0.00),   # 14 ring pip
    (0.60, 0.40, 0.00),   # 15 ring dip
    (0.61, 0.31, 0.00),   # 16 ring tip
    (0.66, 0.66, 0.00),   # 17 pinky mcp
    (0.68, 0.56, 0.00),   # 18 pinky pip
    (0.69, 0.48, 0.00),   # 19 pinky dip
    (0.70, 0.41, 0.00),   # 20 pinky tip
]


def _curl_finger(lm, tip, pip, dip, mcp):
    """Curva un dedo trayendo tip/dip/pip cerca del mcp (simula puño)."""
    mx, my, mz = lm[mcp]
    lm[dip] = (mx + 0.01, my + 0.02, mz + 0.02)
    lm[pip] = (mx, my + 0.01, mz + 0.01)
    lm[tip] = (mx + 0.005, my + 0.03, mz + 0.03)
    return lm


def make_landmarks(gesture: str = "OPEN_PALM"):
    """Devuelve 21 Point sintéticos para el gesto pedido."""
    lm = [tuple(p) for p in _OPEN_HAND]
    g = gesture.upper()
    if g == "FIST":
        for tip, pip, dip, mcp in [(8, 6, 7, 5), (12, 10, 11, 9), (16, 14, 15, 13), (20, 18, 19, 17)]:
            lm = _curl_finger(lm, tip, pip, dip, mcp)
    elif g == "POINT":
        for tip, pip, dip, mcp in [(12, 10, 11, 9), (16, 14, 15, 13), (20, 18, 19, 17)]:
            lm = _curl_finger(lm, tip, pip, dip, mcp)
    elif g == "VICTORY":
        for tip, pip, dip, mcp in [(16, 14, 15, 13), (20, 18, 19, 17)]:
            lm = _curl_finger(lm, tip, pip, dip, mcp)
    elif g == "THREE":
        for tip, pip, dip, mcp in [(20, 18, 19, 17)]:
            lm = _curl_finger(lm, tip, pip, dip, mcp)
    elif g == "PINCH":
        # acercar pulgar (4) a índice (8)
        lm[4] = (0.42, 0.31, 0.00)
        for tip, pip, dip, mcp in [(12, 10, 11, 9), (16, 14, 15, 13), (20, 18, 19, 17)]:
            lm = _curl_finger(lm, tip, pip, dip, mcp)
    return [Point(x, y, z) for (x, y, z) in lm]


@pytest.fixture(scope="session")
def bank_path() -> Path:
    assert BANK_PATH.exists(), f"banco no encontrado: {BANK_PATH}"
    return BANK_PATH


@pytest.fixture(scope="session")
def bank_rows():
    rows = []
    for line in BANK_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


@pytest.fixture
def open_hand():
    return make_landmarks("OPEN_PALM")
