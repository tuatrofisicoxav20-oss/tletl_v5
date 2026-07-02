"""guard.py – Rule guard para Tletl v5.

Porta la lógica de cruce IA-vs-regla geométrica desde ai_runtime_v47_live.py.
No importa cv2, ydotool, bpy ni nada específico del sistema operativo.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# geometric_rule – clasificador geométrico puro (sin banco, sin IA)
# ---------------------------------------------------------------------------

def _finger_extended(features: Dict[str, float], name: str) -> bool:
    """True si el dedo `name` está extendido según sus features."""
    vertical = features.get(f"{name}_vertical", 0.0)
    curl = features.get(f"{name}_curl", 0.0)
    tip_mcp = features.get(f"{name}_tip_mcp", 0.0)
    # Un dedo extendido tiene: pip más alto que tip (vertical > 0), curl positivo
    # (tip más lejos de muñeca que pip) y tip_mcp grande.
    return (vertical > 0.08 and curl > 0.03) or tip_mcp > 0.55


def _pinch_close(features: Dict[str, float]) -> bool:
    """True si pulgar e índice están suficientemente cerca para formar PINCH."""
    d = features.get("thumb_tip_index_tip", 1.0)
    return d < 0.42


def geometric_rule(features: Dict[str, float]) -> str:
    """Segunda opinión por geometría pura (sin banco). Devuelve un label de LABELS o 'NEUTRAL'."""
    index_ext = _finger_extended(features, "index")
    middle_ext = _finger_extended(features, "middle")
    ring_ext = _finger_extended(features, "ring")
    pinky_ext = _finger_extended(features, "pinky")

    long_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])

    # FIST: ningún dedo largo extendido
    all_closed = not index_ext and not middle_ext and not ring_ext and not pinky_ext
    if all_closed:
        return "FIST"

    # Palma abierta: todos los dedos extendidos
    if long_count >= 4:
        return "OPEN_PALM"

    # THREE: índice + medio + anular
    if index_ext and middle_ext and ring_ext and not pinky_ext:
        return "THREE"

    # VICTORY: solo índice + medio
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "VICTORY"

    # POINT: solo índice
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "POINT"

    # PINCH: pulgar cerca del índice y demás dedos no completamente extendidos
    if _pinch_close(features) and long_count <= 1:
        return "PINCH"

    return "NEUTRAL"


# ---------------------------------------------------------------------------
# rule_guard – árbitro entre predicción IA y regla geométrica
# ---------------------------------------------------------------------------

def rule_guard(
    prediction,
    rule_raw: str,
    *,
    dangerous: List[str],
    conf_threshold: float,
    margin_threshold: float,
) -> Tuple[str, str]:
    """Devuelve (gesture_final, reason).

    Lógica: si prediction.raw_label está en `dangerous` y rule_raw != prediction.raw_label
    y (prediction.confidence < conf_threshold o prediction.margin < margin_threshold):
        -> ('NEUTRAL', f'GUARD_BLOCK_{prediction.raw_label}')
    En cualquier otro caso -> (prediction.raw_label, 'OK').
    """
    raw = prediction.raw_label
    conf = prediction.confidence
    margin = prediction.margin

    if (
        raw in dangerous
        and rule_raw != raw
        and (conf < conf_threshold or margin < margin_threshold)
    ):
        return "NEUTRAL", f"GUARD_BLOCK_{raw}"

    return raw, "OK"
