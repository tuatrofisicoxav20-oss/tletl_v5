"""
apps/blender_control/state_reader.py
Lector del bus de estado Tletl para el addon de Blender.

No depende de bpy ni de tletl_core: el Python embebido de Blender NO puede
importar el core, así que este archivo tiene que ser autosuficiente. Se
distribuye junto al addon (tools/build_blender_addon.py lo mete en el zip) y el
addon trae además una copia INLINE de estas funciones por si se instala el
archivo suelto (ver tletl_blender_addon.py).

REGLA DE RUTAS (copiada de tletl_core/paths.py; hay un test que verifica que
las dos implementaciones coinciden):

    $TLETL_STATE_PATH  >  $TLETL_HOME/tletl_state.json  >  ~/.tletl/tletl_state.json
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ENV_HOME = "TLETL_HOME"
ENV_STATE_PATH = "TLETL_STATE_PATH"
BUS_FILENAME = "tletl_state.json"


# ── Ruta del bus (réplica exacta de tletl_core.paths) ─────────────────────────

def tletl_home() -> Path:
    """Directorio de runtime del usuario: $TLETL_HOME o ~/.tletl."""
    value = os.environ.get(ENV_HOME, "").strip()
    return Path(value).expanduser() if value else Path.home() / ".tletl"


def default_bus_path() -> Path:
    """Ruta del bus JSON que comparten la app de cámara y Blender.

    Misma regla que `tletl_core.paths.default_bus_path()`; si cambias una,
    cambia la otra (tests/test_state_reader.py las compara).
    """
    value = os.environ.get(ENV_STATE_PATH, "").strip()
    return Path(value).expanduser() if value else tletl_home() / BUS_FILENAME


# ── Lectura del bus ──────────────────────────────────────────────────────────

def read_state(path: str | os.PathLike[str] | None = None) -> Dict[str, Any]:
    """
    Lee tletl_state.json y devuelve el dict crudo.
    Devuelve {} si el archivo no existe, está vacío, no es JSON válido o el JSON
    no es un objeto (una lista rompería a todos los helpers de abajo).
    `path=None` usa `default_bus_path()`.
    """
    p = Path(path).expanduser() if path is not None and str(path).strip() else default_bus_path()
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return {}
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


# Alias legacy para compatibilidad con código antiguo
read_tletl_state = read_state


# ── Helpers de acceso ────────────────────────────────────────────────────────

def _hand(state: Dict[str, Any], side: str) -> Dict[str, Any]:
    """Devuelve el sub-dict de la mano indicada ('dom' o 'mod'), o {}."""
    if not isinstance(state, dict):
        return {}
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


def _palm_of(hand: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    palm = hand.get("palm")
    if palm is None:
        return None
    # Puede venir como lista [x, y] o como dict {"x":…, "y":…}
    if isinstance(palm, (list, tuple)) and len(palm) >= 2:
        try:
            return float(palm[0]), float(palm[1])
        except (TypeError, ValueError):
            return None
    if isinstance(palm, dict):
        x = palm.get("x")
        y = palm.get("y")
        if x is not None and y is not None:
            return float(x), float(y)
    return None


def hand_gesture(state: Dict[str, Any], side: str) -> str:
    """Gesto (estable) de la mano `side` ('dom' | 'mod'), o 'NO_HAND'."""
    gesture = _hand(state, side).get("gesture", "NO_HAND")
    return str(gesture) if gesture else "NO_HAND"


def dom_gesture(state: Dict[str, Any]) -> str:
    """Devuelve el gesto de la mano dominante, o 'NO_HAND'."""
    return hand_gesture(state, "dom")


def mod_gesture(state: Dict[str, Any]) -> str:
    """Devuelve el gesto de la mano modificadora, o 'NO_HAND'."""
    return hand_gesture(state, "mod")


def hand_palm(state: Dict[str, Any], side: str) -> Optional[Tuple[float, float]]:
    """Posición normalizada (x, y) de la palma de `side`, o None si no hay mano."""
    return _palm_of(_hand(state, side))


def dom_palm(state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Devuelve la posición normalizada de la palma dominante como (x, y),
    o None si la mano no está presente.
    """
    return hand_palm(state, "dom")


def mod_palm(state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """
    Devuelve la posición normalizada de la palma modificadora como (x, y),
    o None si la mano no está presente.
    """
    return hand_palm(state, "mod")


def hand_confidence(state: Dict[str, Any], side: str) -> float:
    """Confianza del clasificador para la mano `side` (0.0 si no hay dato)."""
    try:
        return float(_hand(state, side).get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def hand_present(state: Dict[str, Any], side: str) -> bool:
    return bool(_hand(state, side).get("present", False))


def intent_name(state: Dict[str, Any]) -> str:
    """Devuelve el nombre del intent activo, o 'NONE'."""
    intent = state.get("intent") if isinstance(state, dict) else None
    if isinstance(intent, dict):
        return str(intent.get("name", "NONE") or "NONE")
    return "NONE"


def current_mode(state: Dict[str, Any]) -> str:
    """Devuelve el modo de operación de la app de cámara (NAVEGADOR/CURSOR/VENTANAS)."""
    if not isinstance(state, dict):
        return "NONE"
    return str(state.get("mode", "NONE") or "NONE")


def control_enabled(state: Dict[str, Any]) -> bool:
    """Control ON/OFF de la app de cámara: `extra.control` (fallback: `selected`)."""
    if not isinstance(state, dict):
        return False
    extra = state.get("extra")
    if isinstance(extra, dict) and "control" in extra:
        return bool(extra.get("control"))
    return bool(state.get("selected", False))


def app_fps(state: Dict[str, Any]) -> float:
    """FPS que reporta la app de cámara (0.0 si no hay dato)."""
    try:
        return float(state.get("fps", 0.0) or 0.0) if isinstance(state, dict) else 0.0
    except (TypeError, ValueError):
        return 0.0


def state_timestamp(state: Dict[str, Any]) -> Optional[float]:
    """Timestamp unix del frame, o None si falta o no es válido (<= 0)."""
    if not isinstance(state, dict):
        return None
    ts = state.get("timestamp")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return None
    if not math.isfinite(ts) or ts <= 0:
        return None
    return float(ts)


def state_age(state: Dict[str, Any], now: float | None = None) -> float:
    """Segundos desde que la app escribió el frame; inf si no hay timestamp."""
    ts = state_timestamp(state)
    if ts is None:
        return float("inf")
    current = time.time() if now is None else float(now)
    return max(0.0, current - ts)


def is_stale(state: Dict[str, Any], max_age: float, now: float | None = None) -> bool:
    """True si el frame es más viejo que `max_age` segundos (o no tiene timestamp)."""
    return state_age(state, now) > float(max_age)


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
    t = state.get("transform") if isinstance(state, dict) else None
    if isinstance(t, dict):
        for k, v in t.items():
            if k in defaults:
                try:
                    defaults[k] = float(v)
                except (TypeError, ValueError):
                    continue
    return defaults


# ── Resumen de texto (útil para logs) ────────────────────────────────────────

def summarize_state(state: Dict[str, Any]) -> str:
    if not state:
        return "NO_STATE"
    g    = dom_gesture(state)
    mg   = mod_gesture(state)
    palm = dom_palm(state)
    p_str = f"palm=({palm[0]:.2f},{palm[1]:.2f})" if palm else "palm=None"
    return (f"dom={g} mod={mg} {p_str} intent={intent_name(state)} mode={current_mode(state)} "
            f"control={'ON' if control_enabled(state) else 'OFF'} age={state_age(state):.2f}s")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"bus: {Path(path).expanduser() if path else default_bus_path()}")
    print(summarize_state(read_state(path)))
