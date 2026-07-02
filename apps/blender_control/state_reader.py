"""
apps/blender_control/state_reader.py
Lector del bus de estado Tletl para el addon de Blender.

No depende de bpy. Puede usarse en tests, scripts externos y dentro del addon.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ── Lectura del bus ──────────────────────────────────────────────────────────

def read_state(path: str | Path = "tletl_state.json") -> Dict[str, Any]:
    """
    Lee tletl_state.json y devuelve el dict crudo.
    Devuelve {} si el archivo no existe o no es JSON válido.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# Alias legacy para compatibilidad con código antiguo
read_tletl_state = read_state


# ── Helpers de acceso ────────────────────────────────────────────────────────

def _hand(state: Dict[str, Any], side: str) -> Dict[str, Any]:
    """Devuelve el sub-dict de la mano indicada ('dom' o 'mod'), o {}."""
    val = state.get(side)
    if isinstance(val, dict):
        return val
    # compatibilidad con formato antiguo {"hands": {"dom": {...}}}
    hands = state.get("hands")
    if isinstance(hands, dict):
        val = hands.get(side)
        if isinstance(val, dict):
            return val
    return {}


def dom_gesture(state: Dict[str, Any]) -> str:
    """Devuelve el gesto de la mano dominante, o 'NO_HAND'."""
    return _hand(state, "dom").get("gesture", "NO_HAND")


def mod_gesture(state: Dict[str, Any]) -> str:
    """Devuelve el gesto de la mano modificadora, o 'NO_HAND'."""
    return _hand(state, "mod").get("gesture", "NO_HAND")


def dom_palm(state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Devuelve la posición normalizada de la palma dominante como (x, y),
    o None si la mano no está presente.
    """
    hand = _hand(state, "dom")
    palm = hand.get("palm")
    if palm is None:
        return None
    # Puede venir como lista [x, y] o como dict {"x":…, "y":…}
    if isinstance(palm, (list, tuple)) and len(palm) >= 2:
        return float(palm[0]), float(palm[1])
    if isinstance(palm, dict):
        x = palm.get("x")
        y = palm.get("y")
        if x is not None and y is not None:
            return float(x), float(y)
    return None


def mod_palm(state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Devuelve la posición normalizada de la palma modificadora como (x, y),
    o None si la mano no está presente.
    """
    hand = _hand(state, "mod")
    palm = hand.get("palm")
    if palm is None:
        return None
    if isinstance(palm, (list, tuple)) and len(palm) >= 2:
        return float(palm[0]), float(palm[1])
    if isinstance(palm, dict):
        x = palm.get("x")
        y = palm.get("y")
        if x is not None and y is not None:
            return float(x), float(y)
    return None


def intent_name(state: Dict[str, Any]) -> str:
    """Devuelve el nombre del intent activo, o 'NONE'."""
    intent = state.get("intent")
    if isinstance(intent, dict):
        return intent.get("name", "NONE")
    return "NONE"


def current_mode(state: Dict[str, Any]) -> str:
    """Devuelve el modo de operación actual."""
    return state.get("mode", "NONE")


def get_transform(state: Dict[str, Any]) -> Dict[str, float]:
    """
    Devuelve el transform del bus.
    Garantiza las claves: x, y, z, rot_x, rot_y, rot_z, scale.
    """
    defaults: Dict[str, float] = {
        "x": 0.0, "y": 0.0, "z": 0.0,
        "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0,
        "scale": 1.0,
    }
    t = state.get("transform")
    if isinstance(t, dict):
        defaults.update({k: float(v) for k, v in t.items() if k in defaults})
    return defaults


# ── Resumen de texto (útil para logs) ────────────────────────────────────────

def summarize_state(state: Dict[str, Any]) -> str:
    if not state:
        return "NO_STATE"
    g    = dom_gesture(state)
    mg   = mod_gesture(state)
    palm = dom_palm(state)
    p_str = f"palm=({palm[0]:.2f},{palm[1]:.2f})" if palm else "palm=None"
    return f"dom={g} mod={mg} {p_str} intent={intent_name(state)} mode={current_mode(state)}"


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "tletl_state.json"
    state = read_state(path)
    print(summarize_state(state))
