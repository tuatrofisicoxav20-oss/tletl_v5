"""
apps/blender_control/tletl_blender_addon.py
Addon de Blender para controlar y CREAR objetos 3D con gestos Tletl.

bpy se importa de forma LAZY / guarded para que el módulo sea testeable
fuera de Blender. TODO el código que usa bpy está detrás del guard
`if bpy is not None`.

Capas (de abajo hacia arriba):

  * Lector del bus (state_reader): paquete > módulo vecino > copia INLINE.
    La copia inline existe porque instalar este archivo suelto en Blender
    tronaba con `ModuleNotFoundError: state_reader` (VALIDACION_FISICA §5).
  * `BlenderGestureMapper`  — PURO. Traduce deltas de palma a un transform
    con suavizado exponencial REAL (target acumulado; no se pierde movimiento).
  * `GestureSession`        — PURO. Máquina de estados del addon: modo
    TRANSFORM/CREATE, hold de THREE, flancos de PINCH/VICTORY, primitivas.
    Devuelve COMANDOS abstractos por tick.
  * Capa bpy                — sólo ejecuta comandos: escribe el transform al
    objeto activo (sin ensuciar canales que no cambiaron) o crea primitivas
    con bmesh (nunca bpy.ops: los timers no tienen contexto de operador).

Instalación: ver apps/blender_control/README.md (zip generado por
`./launchers/tletl-build-blender-addon.sh`; sirve como addon legacy y como
extensión de Blender 4.2+).
"""
from __future__ import annotations

# ── Guard de bpy (LAZY) ──────────────────────────────────────────────────────
try:
    import bpy  # type: ignore
except ImportError:
    bpy = None  # type: ignore

import math
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Lector del bus: paquete > módulo vecino > copia inline ───────────────────
# `STATE_READER_SOURCE` dice cuál de las tres rutas se usó (los tests lo miran).

STATE_READER_SOURCE = "package"
try:
    from .state_reader import (  # type: ignore
        app_fps, control_enabled, current_mode, default_bus_path, dom_gesture, dom_palm,
        hand_confidence, intent_name, is_stale, mod_gesture, mod_palm, read_state, state_age,
    )
except ImportError:
    try:
        # Blender puede cargar el addon como módulo top-level con state_reader.py al lado.
        import importlib as _il
        _sr = _il.import_module("state_reader")
        app_fps          = _sr.app_fps           # type: ignore
        control_enabled  = _sr.control_enabled   # type: ignore
        current_mode     = _sr.current_mode      # type: ignore
        default_bus_path = _sr.default_bus_path  # type: ignore
        dom_gesture      = _sr.dom_gesture       # type: ignore
        dom_palm         = _sr.dom_palm          # type: ignore
        hand_confidence  = _sr.hand_confidence   # type: ignore
        intent_name      = _sr.intent_name       # type: ignore
        is_stale         = _sr.is_stale          # type: ignore
        mod_gesture      = _sr.mod_gesture       # type: ignore
        mod_palm         = _sr.mod_palm          # type: ignore
        read_state       = _sr.read_state        # type: ignore
        state_age        = _sr.state_age         # type: ignore
        STATE_READER_SOURCE = "sibling"
    except (ImportError, AttributeError):
        # ── Copia INLINE mínima de state_reader.py (archivo instalado suelto) ──
        # Debe mantenerse en sincronía con apps/blender_control/state_reader.py;
        # tests/test_blender_addon_build.py compara ambas implementaciones.
        STATE_READER_SOURCE = "inline"
        import json as _json
        import os as _os

        def default_bus_path() -> Path:  # type: ignore[misc]
            """$TLETL_STATE_PATH > $TLETL_HOME/tletl_state.json > ~/.tletl/tletl_state.json."""
            value = _os.environ.get("TLETL_STATE_PATH", "").strip()
            if value:
                return Path(value).expanduser()
            home = _os.environ.get("TLETL_HOME", "").strip()
            base = Path(home).expanduser() if home else Path.home() / ".tletl"
            return base / "tletl_state.json"

        def read_state(path: Any = None) -> Dict[str, Any]:  # type: ignore[misc]
            p = Path(path).expanduser() if path is not None and str(path).strip() else default_bus_path()
            try:
                text = p.read_text(encoding="utf-8")
            except (OSError, ValueError):
                return {}
            if not text.strip():
                return {}
            try:
                data = _json.loads(text)
            except ValueError:
                return {}
            return data if isinstance(data, dict) else {}

        def _hand(state: Dict[str, Any], side: str) -> Dict[str, Any]:
            if not isinstance(state, dict):
                return {}
            val = state.get(side)
            if isinstance(val, dict):
                return val
            hands = state.get("hands")
            if isinstance(hands, dict) and isinstance(hands.get(side), dict):
                return hands[side]
            return {}

        def _palm(hand: Dict[str, Any]) -> Optional[Tuple[float, float]]:
            palm = hand.get("palm")
            if isinstance(palm, (list, tuple)) and len(palm) >= 2:
                try:
                    return float(palm[0]), float(palm[1])
                except (TypeError, ValueError):
                    return None
            if isinstance(palm, dict) and palm.get("x") is not None and palm.get("y") is not None:
                return float(palm["x"]), float(palm["y"])
            return None

        def dom_gesture(state: Dict[str, Any]) -> str:  # type: ignore[misc]
            return str(_hand(state, "dom").get("gesture") or "NO_HAND")

        def mod_gesture(state: Dict[str, Any]) -> str:  # type: ignore[misc]
            return str(_hand(state, "mod").get("gesture") or "NO_HAND")

        def dom_palm(state: Dict[str, Any]) -> Optional[Tuple[float, float]]:  # type: ignore[misc]
            return _palm(_hand(state, "dom"))

        def mod_palm(state: Dict[str, Any]) -> Optional[Tuple[float, float]]:  # type: ignore[misc]
            return _palm(_hand(state, "mod"))

        def hand_confidence(state: Dict[str, Any], side: str) -> float:  # type: ignore[misc]
            try:
                return float(_hand(state, side).get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        def intent_name(state: Dict[str, Any]) -> str:  # type: ignore[misc]
            intent = state.get("intent") if isinstance(state, dict) else None
            return str(intent.get("name") or "NONE") if isinstance(intent, dict) else "NONE"

        def current_mode(state: Dict[str, Any]) -> str:  # type: ignore[misc]
            return str(state.get("mode") or "NONE") if isinstance(state, dict) else "NONE"

        def control_enabled(state: Dict[str, Any]) -> bool:  # type: ignore[misc]
            if not isinstance(state, dict):
                return False
            extra = state.get("extra")
            if isinstance(extra, dict) and "control" in extra:
                return bool(extra.get("control"))
            return bool(state.get("selected", False))

        def app_fps(state: Dict[str, Any]) -> float:  # type: ignore[misc]
            try:
                return float(state.get("fps", 0.0) or 0.0) if isinstance(state, dict) else 0.0
            except (TypeError, ValueError):
                return 0.0

        def state_age(state: Dict[str, Any], now: Optional[float] = None) -> float:  # type: ignore[misc]
            ts = state.get("timestamp") if isinstance(state, dict) else None
            if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts) or ts <= 0:
                return float("inf")
            return max(0.0, (time.time() if now is None else float(now)) - float(ts))

        def is_stale(state: Dict[str, Any], max_age: float, now: Optional[float] = None) -> bool:  # type: ignore[misc]
            return state_age(state, now) > float(max_age)


# ── Metadatos del addon ──────────────────────────────────────────────────────

bl_info = {
    "name":        "Tletl Gesture Control",
    "author":      "Tletl Project",
    "version":     (5, 2, 0),
    "blender":     (4, 0, 0),
    "location":    "View3D > N-Panel > Tletl",
    "description": "Controla y crea objetos 3D con gestos de mano en tiempo real (bus Tletl).",
    "category":    "Object",
}

# Identificador con el que Blender registró el addon. Es `__package__` cuando se
# instala como carpeta/zip (legacy) o como extensión (bl_ext.<repo>.<id>) y
# `__name__` cuando se instala el .py suelto. Sirve para bl_idname de las
# preferencias y para `bpy.context.preferences.addons[_ADDON_ID]`.
_ADDON_ID = __package__ or __name__

ADDON_VERSION = ".".join(str(v) for v in bl_info["version"])


# ── Constantes ───────────────────────────────────────────────────────────────

DEFAULT_BUS_PATH = str(default_bus_path())

# Defaults recalibrados. En la validación física (docs/VALIDACION_FISICA_v5.md
# §4) el usuario terminó en "Gain 14 / Smoothing 1.0" porque el "smoothing"
# viejo era LOSSY: multiplicaba el delta de cada frame por `smoothing` y tiraba
# el resto (con 0.35 se perdía el 65 % del movimiento; la ganancia efectiva del
# default 4.0 era 1.4). Ahora el suavizado es exponencial hacia un target
# acumulado y NO pierde movimiento: `gain` vuelve a significar "unidades de
# Blender por ancho de cámara recorrido", así que 10 (≈ el cubo por defecto
# mide 2) equivale al tacto que se validó con 14 lossy, y 0.5 sólo alisa el
# temblor de la mano sin frenar el arrastre.
DEFAULT_GAIN        = 10.0
DEFAULT_SMOOTHING   = 0.5
DEFAULT_ROT_GAIN    = 2.0    # vueltas relativas por unidad de desplazamiento de mod.x
DEFAULT_SCALE_GAIN  = 1.5    # sensibilidad del escalado por distancia entre manos
DEFAULT_Z_GAIN      = 0.0    # elevación en Z con mod PINCH vertical; 0 = apagado (comportamiento validado)
DEFAULT_STALE_AFTER = 1.0    # segundos sin frame nuevo para declarar el bus obsoleto
DEFAULT_INTERVAL    = 0.033  # segundos entre ticks del timer (~30 Hz, el ritmo del bus)
MIN_INTERVAL        = 0.005
MIN_SCALE           = 0.05
MODE_HOLD_SECONDS   = 0.8    # cuánto sostener THREE para cambiar de modo
CHANGE_EPS          = 1e-9   # por debajo de esto un canal se considera "sin cambio" (no se escribe)
RESYNC_EPS          = 1e-6   # tolerancia (relativa, con piso 1) para detectar cambios externos

TRANSFORM_KEYS = ("x", "y", "z", "rot_x", "rot_y", "rot_z", "scale")

ADDON_MODES = ("TRANSFORM", "CREATE")
PRIMITIVES  = ("cube", "sphere", "cylinder", "cone", "plane")

# Primitivas -> operador de bmesh.ops y sus argumentos. Cilindro = cono con
# radios iguales. Tamaños iguales a los de bpy.ops.mesh.primitive_*_add.
PRIMITIVE_SPECS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "cube":     ("create_cube",     {"size": 2.0}),
    "sphere":   ("create_uvsphere", {"u_segments": 32, "v_segments": 16, "radius": 1.0}),
    "cylinder": ("create_cone",     {"cap_ends": True, "cap_tris": False, "segments": 32,
                                     "radius1": 1.0, "radius2": 1.0, "depth": 2.0}),
    "cone":     ("create_cone",     {"cap_ends": True, "cap_tris": False, "segments": 32,
                                     "radius1": 1.0, "radius2": 0.0, "depth": 2.0}),
    "plane":    ("create_grid",     {"x_segments": 1, "y_segments": 1, "size": 1.0}),
}

BUS_OK      = "ok"
BUS_MISSING = "missing"
BUS_INVALID = "invalid"
BUS_STALE   = "stale"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LÓGICA PURA — SIN bpy, TESTEABLE                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def normalize_transform(current: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Devuelve un transform con todas las claves y defaults seguros."""
    src = current or {}
    return {
        "x":     float(src.get("x",     0.0)),
        "y":     float(src.get("y",     0.0)),
        "z":     float(src.get("z",     0.0)),
        "rot_x": float(src.get("rot_x", 0.0)),
        "rot_y": float(src.get("rot_y", 0.0)),
        "rot_z": float(src.get("rot_z", 0.0)),
        "scale": float(src.get("scale", 1.0)),
    }


def changed_channels(old: Dict[str, float], new: Dict[str, float],
                     eps: float = CHANGE_EPS) -> List[str]:
    """Claves de TRANSFORM_KEYS cuyo valor cambió más de `eps` entre old y new.

    La capa bpy sólo escribe esos canales: escribir location/rotation/scale en
    cada tick ensucia el depsgraph y la pila de undo aunque nada se mueva.
    """
    a = normalize_transform(old)
    b = normalize_transform(new)
    return [k for k in TRANSFORM_KEYS if abs(b[k] - a[k]) > eps]


def scale_reference(scale_xyz: Tuple[float, float, float] | List[float]) -> float:
    """Escalar que representa una escala (posiblemente NO uniforme) del objeto.

    Es el mayor valor absoluto de los tres ejes, así un objeto (1, 2, 1) o
    espejado (-1, 1, 1) conserva proporción y signo al multiplicarlo por un
    factor. Si los tres ejes son 0 devuelve 1.0 (no hay nada que escalar).
    """
    ref = max(abs(float(v)) for v in scale_xyz) if len(scale_xyz) else 0.0
    return ref if ref > 1e-9 else 1.0


def scale_factor(old_ref: float, new_ref: float) -> float:
    """Factor multiplicativo que lleva la referencia de escala old -> new."""
    old_ref = float(old_ref)
    if abs(old_ref) <= 1e-9:
        return 1.0
    return float(new_ref) / old_ref


def apply_scale_factor(scale_xyz: Tuple[float, float, float] | List[float],
                       factor: float) -> Tuple[float, ...]:
    """Multiplica cada eje por `factor` (conserva escalas no uniformes y espejos)."""
    return tuple(float(v) * float(factor) for v in scale_xyz)


def palm_to_world_xy(palm: Tuple[float, float],
                     cursor: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     gain: float = DEFAULT_GAIN) -> Tuple[float, float, float]:
    """Palma normalizada (0..1) -> punto del plano XY alrededor del cursor 3D.

    Misma convención que la traslación: x de cámara -> +X, y de cámara
    invertida -> +Y (arriba en pantalla = +Y mundo), z = z del cursor. El centro
    de la cámara (0.5, 0.5) cae exactamente en el cursor.
    """
    px, py = float(palm[0]), float(palm[1])
    return (float(cursor[0]) + (px - 0.5) * float(gain),
            float(cursor[1]) + (0.5 - py) * float(gain),
            float(cursor[2]))


def bus_status(state: Dict[str, Any], path_exists: bool, now: float, stale_after: float,
               path: str = "") -> Tuple[str, str]:
    """Clasifica el bus: (código, mensaje para el panel).

    Códigos: BUS_MISSING (no hay archivo), BUS_INVALID (ilegible / no es JSON),
    BUS_STALE (timestamp más viejo que `stale_after`) y BUS_OK. El addon sólo
    actúa con BUS_OK.
    """
    if not path_exists:
        return BUS_MISSING, f"Sin bus en {path}".rstrip()
    if not state:
        return BUS_INVALID, f"Bus ilegible en {path}".rstrip()
    age = state_age(state, now)
    if age > float(stale_after):
        if math.isinf(age):
            return BUS_STALE, "BUS OBSOLETO (sin timestamp) — ¿corre la app de cámara?"
        return BUS_STALE, f"BUS OBSOLETO ({age:.1f} s) — ¿corre la app de cámara?"
    return BUS_OK, f"Conectado ({app_fps(state):.0f} fps app, {age * 1000.0:.0f} ms)"


def status_lines(state: Dict[str, Any]) -> List[str]:
    """Líneas de estado que el panel muestra a partir del ÚLTIMO frame leído."""
    if not state:
        return ["Sin datos del bus"]
    return [
        f"Control app: {'ON' if control_enabled(state) else 'OFF'}",
        f"Dom: {dom_gesture(state)} ({hand_confidence(state, 'dom'):.2f})",
        f"Mod: {mod_gesture(state)} ({hand_confidence(state, 'mod'):.2f})",
        f"Intent: {intent_name(state)}",
        f"Modo app: {current_mode(state)}",
    ]


@dataclass
class AddonSettings:
    """Parámetros del addon en forma PURA (sin bpy).

    `from_prefs()` los lee de las AddonPreferences si existen y cae a los
    defaults cuando no (p. ej. el addon se registró con otro id): el timer
    nunca debe morir por no encontrar sus preferencias.
    """
    bus_path: str = DEFAULT_BUS_PATH
    gain: float = DEFAULT_GAIN
    smoothing: float = DEFAULT_SMOOTHING
    rot_gain: float = DEFAULT_ROT_GAIN
    scale_gain: float = DEFAULT_SCALE_GAIN
    z_gain: float = DEFAULT_Z_GAIN
    stale_after: float = DEFAULT_STALE_AFTER
    interval: float = DEFAULT_INTERVAL

    @classmethod
    def from_prefs(cls, prefs: Any) -> "AddonSettings":
        settings = cls()
        if prefs is None:
            return settings
        for f in fields(cls):
            value = getattr(prefs, f.name, None)
            if value is None:
                continue
            try:
                setattr(settings, f.name, str(value) if f.name == "bus_path" else float(value))
            except (TypeError, ValueError):
                continue
        settings.interval = max(MIN_INTERVAL, float(settings.interval))
        settings.stale_after = max(0.0, float(settings.stale_after))
        return settings

    def resolved_bus_path(self) -> Path:
        """Ruta del bus: la de las prefs (con ~ expandida) o el default de la regla común.
        Una ruta relativa al .blend de Blender (`//archivo.json`) se resuelve con bpy."""
        text = str(self.bus_path or "").strip()
        if not text:
            return default_bus_path()
        if text.startswith("//") and bpy is not None:
            try:
                text = bpy.path.abspath(text)
            except Exception:  # noqa: BLE001 - sin .blend guardado, bpy puede quejarse
                pass
        return Path(text).expanduser()


class BlenderGestureMapper:
    """Mapea el estado del bus a transformaciones del objeto, por DELTA entre frames.

    Mantiene la palma anterior de cada mano para traducir el MOVIMIENTO (no la
    posición absoluta) en cambios de transform — "agarrar y arrastrar" real.

    Suavizado REAL (exponencial): cada delta de palma se acumula en un TARGET
    (`target += gain * delta`) y cada `update()` acerca `current` al target en
    la fracción `smoothing`. Con smoothing=1.0 el objeto sigue la mano frame a
    frame (igual que antes); con 0.5 alisa el temblor pero NO pierde movimiento:
    si la mano se detiene el objeto converge al target. Con 0.0 no se mueve.

    El target se RESINCRONIZA a `current` (para no aplicar movimiento viejo):
      - por canal, cuando su gesto de control no está activo (x/y sin dom PINCH,
        rot_z/z sin mod PINCH, scale sin dom PINCH + mod OPEN_PALM);
      - completo, con dom FIST (safety) y con `reset()`;
      - completo, cuando el `current` que nos pasan difiere de lo último que
        devolvimos (el usuario movió/deshizo el objeto a mano o cambió de objeto).

    Reglas de gestos:
      - dom FIST                       → safety stop: no toca nada y resetea continuidad.
      - dom OPEN_PALM                  → release: rompe la continuidad (sin salto al volver a PINCH).
      - dom PINCH                      → traslación XY por delta de la palma dominante.
      - mod PINCH                      → rotación Z por delta horizontal; con z_gain>0,
                                         el delta vertical eleva/baja el objeto en Z.
      - dom PINCH + mod OPEN_PALM      → escala uniforme por cambio de la distancia entre palmas.
    """

    def __init__(self, gain: float = DEFAULT_GAIN, smoothing: float = DEFAULT_SMOOTHING,
                 rot_gain: float = DEFAULT_ROT_GAIN, scale_gain: float = DEFAULT_SCALE_GAIN,
                 z_gain: float = DEFAULT_Z_GAIN):
        self.gain = float(gain)
        self.smoothing = float(smoothing)
        self.rot_gain = float(rot_gain)
        self.scale_gain = float(scale_gain)
        self.z_gain = float(z_gain)
        self._prev_dom: Optional[Tuple[float, float]] = None
        self._prev_mod: Optional[Tuple[float, float]] = None
        self._prev_dist: Optional[float] = None
        self._target: Optional[Dict[str, float]] = None
        self._last_out: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    def configure(self, **params: float) -> None:
        """Actualiza ganancias/suavizado desde las prefs sin perder continuidad."""
        for name in ("gain", "smoothing", "rot_gain", "scale_gain", "z_gain"):
            if name in params and params[name] is not None:
                setattr(self, name, float(params[name]))

    def reset(self) -> None:
        """Rompe la continuidad de las palmas y descarta el target pendiente."""
        self._prev_dom = None
        self._prev_mod = None
        self._prev_dist = None
        self._target = None
        self._last_out = None

    @property
    def target(self) -> Optional[Dict[str, float]]:
        """Target acumulado (copia) o None si no hay uno vigente."""
        return dict(self._target) if self._target is not None else None

    @staticmethod
    def _differs(a: Dict[str, float], b: Dict[str, float]) -> bool:
        """Cambio externo: tolerancia relativa con piso 1.0 para que el redondeo a
        float32 de Blender (≈1e-7 relativo) no dispare resincronizaciones espurias."""
        for k in TRANSFORM_KEYS:
            if abs(a[k] - b[k]) > RESYNC_EPS * max(1.0, abs(a[k]), abs(b[k])):
                return True
        return False

    # ------------------------------------------------------------------
    def update(self, state: Dict[str, Any], current: Dict[str, float]) -> Dict[str, float]:
        cur = normalize_transform(current)
        d = dom_gesture(state)
        m = mod_gesture(state)
        dpalm = dom_palm(state)
        mpalm = mod_palm(state)

        # --- Safety stop: soltar todo y olvidar el target ---
        if d == "FIST":
            self.reset()
            return cur

        # --- Resync completo si el objeto cambió por fuera (o es la primera vez) ---
        if self._target is None or self._last_out is None or self._differs(cur, self._last_out):
            self._target = dict(cur)
        tgt = self._target

        # Canales que ningún gesto controla: siempre siguen al objeto.
        tgt["rot_x"] = cur["rot_x"]
        tgt["rot_y"] = cur["rot_y"]

        # --- Traslación XY por delta de la palma dominante (PINCH = agarrar) ---
        if d == "PINCH" and dpalm is not None:
            if self._prev_dom is not None:
                tgt["x"] += (dpalm[0] - self._prev_dom[0]) * self.gain
                tgt["y"] += (self._prev_dom[1] - dpalm[1]) * self.gain  # Y invertido (pantalla→mundo)
            self._prev_dom = dpalm
        else:
            # OPEN_PALM (release), NEUTRAL, sin mano…: romper continuidad y resync del canal
            self._prev_dom = None
            tgt["x"], tgt["y"] = cur["x"], cur["y"]

        # --- Rotación Z (y elevación Z opcional) por delta de la palma modificadora ---
        if m == "PINCH" and mpalm is not None:
            if self._prev_mod is not None:
                tgt["rot_z"] += (mpalm[0] - self._prev_mod[0]) * self.rot_gain * math.pi
                tgt["z"] += (self._prev_mod[1] - mpalm[1]) * self.z_gain
            self._prev_mod = mpalm
        else:
            self._prev_mod = None
            tgt["rot_z"], tgt["z"] = cur["rot_z"], cur["z"]

        # --- Escala por distancia entre manos (dom PINCH + mod OPEN_PALM) ---
        if dpalm is not None and mpalm is not None and d == "PINCH" and m == "OPEN_PALM":
            dist = math.hypot(dpalm[0] - mpalm[0], dpalm[1] - mpalm[1])
            if self._prev_dist is not None and self._prev_dist > 1e-6:
                tgt["scale"] = max(MIN_SCALE, tgt["scale"] + (dist - self._prev_dist) * self.scale_gain)
            self._prev_dist = dist
        else:
            self._prev_dist = None
            tgt["scale"] = cur["scale"]

        # --- Suavizado exponencial hacia el target (sin pérdida) ---
        s = min(1.0, max(0.0, self.smoothing))
        out = {k: cur[k] + (tgt[k] - cur[k]) * s for k in TRANSFORM_KEYS}
        self._last_out = dict(out)
        return out


def map_state_to_transform(
    state: Dict[str, Any],
    current: Dict[str, float],
    *,
    gain: float = DEFAULT_GAIN,
    smoothing: float = DEFAULT_SMOOTHING,
    mapper: Optional[BlenderGestureMapper] = None,
) -> Dict[str, float]:
    """Wrapper sin estado de un solo frame.

    Para control real entre frames usa `BlenderGestureMapper().update()` (mantiene
    la continuidad de las palmas). Esta función crea un mapper efímero si no se le
    pasa uno, por lo que un único frame aislado no produce traslación (no hay palma
    previa para el delta) — es el comportamiento correcto de "agarrar".
    """
    mp = mapper or BlenderGestureMapper(gain=gain, smoothing=smoothing)
    return mp.update(state, current)


Command = Tuple[Any, ...]


class GestureSession:
    """Máquina de estados del addon — PURA, sin bpy.

    Cada `tick(state, current, cursor)` devuelve una lista de comandos:
      ("mode", "CREATE"|"TRANSFORM")        el modo cambió (por THREE sostenido)
      ("primitive", "cube"|…)               VICTORY cambió la primitiva a crear
      ("spawn", "cube"|…, (x, y, z))        crear esa primitiva en esa posición
      ("transform", {x, y, z, rot_*, scale}) aplicar el transform al objeto activo

    Modo TRANSFORM: delega en `BlenderGestureMapper` (PINCH mueve, mod PINCH
    rota, dos manos escalan, FIST frena, OPEN_PALM suelta).
    Modo CREATE: dom PINCH en su flanco de subida crea la primitiva actual en
    la posición de la palma (plano XY alrededor del cursor 3D); VICTORY en su
    flanco de subida cicla la primitiva; OPEN_PALM no hace nada; FIST es
    seguridad (no crea). En ambos modos, dom THREE sostenido `hold_seconds`
    alterna el modo UNA vez por sostén (hay que soltar THREE para repetir).

    Los flancos requieren conocer el gesto del tick ANTERIOR: tras `reset()` o
    un cambio de modo, el primer tick sólo "ceba" el detector (así un PINCH que
    ya estaba sostenido cuando volvió el bus no crea nada por sorpresa).
    """

    def __init__(self, mapper: Optional[BlenderGestureMapper] = None,
                 clock: Callable[[], float] = time.monotonic,
                 hold_seconds: float = MODE_HOLD_SECONDS):
        self.mapper = mapper or BlenderGestureMapper()
        self.clock = clock
        self.hold_seconds = float(hold_seconds)
        self.mode: str = "TRANSFORM"
        self.primitive_index: int = 0
        self.spawn_count: int = 0
        self.last_spawn: Optional[Tuple[str, Tuple[float, float, float]]] = None
        self.last_event: str = ""
        self._prev_dom: Optional[str] = None
        self._three_since: Optional[float] = None
        self._three_fired: bool = False

    # ------------------------------------------------------------------
    @property
    def primitive(self) -> str:
        return PRIMITIVES[self.primitive_index % len(PRIMITIVES)]

    def reset(self) -> None:
        """Bus perdido / addon detenido: olvidar continuidad, flancos y sostén."""
        self.mapper.reset()
        self._prev_dom = None
        self._three_since = None
        self._three_fired = False

    def set_mode(self, mode: str) -> str:
        """Cambia de modo (panel o gesto). Rompe continuidad y re-ceba los flancos,
        pero NO toca el sostén de THREE (si no, un THREE largo alternaría dos veces)."""
        mode = str(mode).upper()
        if mode not in ADDON_MODES:
            raise ValueError(f"modo desconocido: {mode!r} (esperaba {ADDON_MODES})")
        if mode != self.mode:
            self.mode = mode
            self.mapper.reset()
            self._prev_dom = None
            self.last_event = f"Modo {mode}"
        return self.mode

    def toggle_mode(self) -> str:
        return self.set_mode("CREATE" if self.mode == "TRANSFORM" else "TRANSFORM")

    def cycle_primitive(self) -> str:
        self.primitive_index = (self.primitive_index + 1) % len(PRIMITIVES)
        self.last_event = f"Primitiva {self.primitive}"
        return self.primitive

    def set_primitive(self, name: str) -> str:
        name = str(name).lower()
        if name not in PRIMITIVES:
            raise ValueError(f"primitiva desconocida: {name!r} (esperaba {PRIMITIVES})")
        self.primitive_index = PRIMITIVES.index(name)
        return self.primitive

    # ------------------------------------------------------------------
    def _update_mode_hold(self, gesture: str, now: float) -> Optional[Command]:
        """THREE sostenido >= hold_seconds alterna el modo una sola vez por sostén."""
        if gesture != "THREE":
            self._three_since = None
            self._three_fired = False
            return None
        if self._three_since is None:
            self._three_since = now
            self._three_fired = False
            return None
        # 1e-9 de tolerancia: 6.65 - 5.85 da 0.7999999999999998 en coma flotante
        if not self._three_fired and now - self._three_since >= self.hold_seconds - 1e-9:
            self._three_fired = True
            self.toggle_mode()
            return ("mode", self.mode)
        return None

    def tick(self, state: Dict[str, Any], current: Optional[Dict[str, float]] = None,
             cursor: Tuple[float, float, float] = (0.0, 0.0, 0.0),
             now: Optional[float] = None) -> List[Command]:
        """Procesa un frame FRESCO del bus. `current` es el transform del objeto
        activo (None si no hay objeto controlable). Devuelve los comandos a ejecutar."""
        commands: List[Command] = []
        t = self.clock() if now is None else float(now)
        d = dom_gesture(state)
        prev = self._prev_dom
        rising = prev is not None and d != prev

        mode_cmd = self._update_mode_hold(d, t)
        if mode_cmd is not None:
            commands.append(mode_cmd)
            self._prev_dom = d          # set_mode re-cebó los flancos; THREE es el gesto vigente
            return commands

        if self.mode == "CREATE":
            if d == "FIST":
                self.mapper.reset()     # seguridad: nunca crear con el puño
            elif d == "PINCH" and rising:
                palm = dom_palm(state)
                if palm is not None:
                    pos = palm_to_world_xy(palm, cursor, gain=self.mapper.gain)
                    self.spawn_count += 1
                    self.last_spawn = (self.primitive, pos)
                    self.last_event = f"Creado {self.primitive} en ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                    commands.append(("spawn", self.primitive, pos))
            elif d == "VICTORY" and rising:
                commands.append(("primitive", self.cycle_primitive()))
            # OPEN_PALM / NEUTRAL / POINT / sin mano: nada
        else:
            if current is None:
                self.mapper.reset()     # sin objeto: no acumular continuidad para el siguiente
            else:
                commands.append(("transform", self.mapper.update(state, current)))

        self._prev_dom = d
        return commands


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CÓDIGO bpy — sólo se ejecuta dentro de Blender                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Estado que el panel muestra (lo escribe el timer; el panel NO lee el bus).
last_error: str = ""
_last_state: Dict[str, Any] = {}
_last_status: Tuple[str, str] = (BUS_MISSING, "Tletl detenido")
_addon_running = False
_session = GestureSession()

if bpy is not None:

    # Tipos de objeto que tienen location/rotation/scale utilizables.
    _CONTROLLABLE_TYPES = {"MESH", "EMPTY", "CURVE", "SURFACE", "META", "FONT",
                           "ARMATURE", "LATTICE", "LIGHT", "CAMERA", "GPENCIL", "GREASEPENCIL"}
    _EULER_MODES = {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"}

    # ── Acceso robusto a prefs / contexto ────────────────────────────────────

    def _get_prefs() -> Any:
        """Preferencias del addon o None si Blender lo registró con otro id."""
        try:
            return bpy.context.preferences.addons[_ADDON_ID].preferences
        except (KeyError, AttributeError, TypeError):
            return None

    def _current_settings() -> AddonSettings:
        return AddonSettings.from_prefs(_get_prefs())

    def _active_object() -> Any:
        """Objeto activo: view_layer (siempre disponible en timers) y luego context."""
        obj = None
        try:
            obj = bpy.context.view_layer.objects.active
        except AttributeError:
            obj = None
        if obj is None:
            obj = getattr(bpy.context, "active_object", None)
        return obj

    def _cursor_location() -> Tuple[float, float, float]:
        try:
            loc = bpy.context.scene.cursor.location
            return (float(loc.x), float(loc.y), float(loc.z))
        except AttributeError:
            return (0.0, 0.0, 0.0)

    def _redraw_view3d() -> None:
        """Marca todas las vistas 3D de todas las ventanas para redibujar (guarded)."""
        try:
            for window in bpy.context.window_manager.windows:
                screen = window.screen
                if screen is None:
                    continue
                for area in screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()
        except (AttributeError, RuntimeError):
            pass

    # ── Objeto <-> transform ─────────────────────────────────────────────────

    def _read_current_transform(obj: Any) -> Dict[str, float]:
        return {
            "x":     float(obj.location.x),
            "y":     float(obj.location.y),
            "z":     float(obj.location.z),
            "rot_x": float(obj.rotation_euler.x),
            "rot_y": float(obj.rotation_euler.y),
            "rot_z": float(obj.rotation_euler.z),
            "scale": scale_reference(tuple(obj.scale)),
        }

    def _apply_transform_to_object(obj: Any, old: Dict[str, float], new: Dict[str, float]) -> List[str]:
        """Escribe SOLO los canales que cambiaron. La escala se aplica como FACTOR
        sobre los tres ejes para respetar escalas no uniformes. Devuelve los canales escritos."""
        changed = changed_channels(old, new)
        if not changed:
            return changed
        if "x" in changed:
            obj.location.x = new["x"]
        if "y" in changed:
            obj.location.y = new["y"]
        if "z" in changed:
            obj.location.z = new["z"]
        rot_changed = [k for k in ("rot_x", "rot_y", "rot_z") if k in changed]
        if rot_changed:
            if getattr(obj, "rotation_mode", "XYZ") in _EULER_MODES:
                if "rot_x" in changed:
                    obj.rotation_euler.x = new["rot_x"]
                if "rot_y" in changed:
                    obj.rotation_euler.y = new["rot_y"]
                if "rot_z" in changed:
                    obj.rotation_euler.z = new["rot_z"]
            else:
                raise RuntimeError(f"rotation_mode {obj.rotation_mode!r} no soportado (usa Euler XYZ)")
        if "scale" in changed:
            factor = scale_factor(old["scale"], new["scale"])
            obj.scale = apply_scale_factor(tuple(obj.scale), factor)
        return changed

    # ── Primitivas con bmesh (sin bpy.ops) ───────────────────────────────────

    def _spawn_primitive(kind: str, location: Tuple[float, float, float]) -> Any:
        """Crea la primitiva `kind` en `location`, la enlaza a la colección activa
        y la deja activa + seleccionada (así TRANSFORM la mueve de inmediato)."""
        import bmesh  # type: ignore

        op_name, kwargs = PRIMITIVE_SPECS[kind]
        bm = bmesh.new()
        try:
            getattr(bmesh.ops, op_name)(bm, **kwargs)
            mesh = bpy.data.meshes.new(f"Tletl_{kind}")
            bm.to_mesh(mesh)
        finally:
            bm.free()
        mesh.update()
        obj = bpy.data.objects.new(mesh.name, mesh)
        obj.location = tuple(float(v) for v in location)

        collection = getattr(bpy.context, "collection", None)
        if collection is None:
            collection = bpy.context.scene.collection
        collection.objects.link(obj)

        view_layer = bpy.context.view_layer
        try:
            for other in view_layer.objects:
                if other.select_get():
                    other.select_set(False)
            obj.select_set(True)
            view_layer.objects.active = obj
        except RuntimeError as exc:
            # p. ej. la colección activa está excluida del view layer: el objeto ya
            # existe; sólo no pudo quedar activo. Se informa sin perder la creación.
            global last_error
            last_error = f"creado {obj.name} pero no se pudo activar: {exc}"
        return obj

    # ── Ejecución de comandos ────────────────────────────────────────────────

    def _execute_command(cmd: Command, obj: Any, current: Optional[Dict[str, float]]) -> None:
        kind = cmd[0]
        if kind == "transform":
            if obj is not None and current is not None:
                _apply_transform_to_object(obj, current, cmd[1])
        elif kind == "spawn":
            _spawn_primitive(cmd[1], cmd[2])
            print(f"[Tletl] {_session.last_event}")
        elif kind in ("mode", "primitive"):
            print(f"[Tletl] {_session.last_event}")

    def _tick(settings: AddonSettings) -> None:
        global _last_state, _last_status
        path = settings.resolved_bus_path()
        exists = path.exists()
        state = read_state(path) if exists else {}
        code, message = bus_status(state, exists, time.time(), settings.stale_after, path=str(path))
        _last_state = state
        _last_status = (code, message)
        if code != BUS_OK:
            _session.reset()
            return

        _session.mapper.configure(gain=settings.gain, smoothing=settings.smoothing,
                                  rot_gain=settings.rot_gain, scale_gain=settings.scale_gain,
                                  z_gain=settings.z_gain)
        obj = _active_object()
        if obj is None or obj.type not in _CONTROLLABLE_TYPES:
            obj = None
        current = _read_current_transform(obj) if obj is not None else None
        for cmd in _session.tick(state, current, cursor=_cursor_location()):
            _execute_command(cmd, obj, current)

    # ── Timer callback ───────────────────────────────────────────────────────

    def _tletl_timer_callback() -> Optional[float]:
        """Función que Blender llama periódicamente vía bpy.app.timers.
        Nunca muere: cualquier excepción queda en `last_error` (visible en el panel)."""
        global last_error
        if not _addon_running:
            return None  # detiene el timer
        settings = AddonSettings()
        try:
            settings = _current_settings()   # también dentro del try: nada debe matar el timer
            _tick(settings)
            last_error = ""
        except Exception as exc:  # noqa: BLE001 - el timer debe seguir vivo
            last_error = f"{type(exc).__name__}: {exc}"
            _session.reset()
        try:
            _redraw_view3d()
        except Exception:  # noqa: BLE001 - el redibujado nunca es motivo para parar
            pass
        return settings.interval

    # ── Preferencias del addon ───────────────────────────────────────────────

    class TletlAddonPreferences(bpy.types.AddonPreferences):
        bl_idname = _ADDON_ID

        bus_path: bpy.props.StringProperty(  # type: ignore
            name="Bus path",
            description="Ruta al tletl_state.json que escribe la app de cámara (vacío = default común)",
            default=DEFAULT_BUS_PATH,
            subtype="FILE_PATH",
        )
        gain: bpy.props.FloatProperty(  # type: ignore
            name="Gain",
            description="Unidades de Blender por ancho de cámara recorrido con PINCH",
            default=DEFAULT_GAIN, min=0.1, max=50.0,
        )
        smoothing: bpy.props.FloatProperty(  # type: ignore
            name="Smoothing",
            description="Fracción del camino al target por tick (1 = instantáneo; bajo = más suave, sin perder movimiento)",
            default=DEFAULT_SMOOTHING, min=0.01, max=1.0,
        )
        rot_gain: bpy.props.FloatProperty(  # type: ignore
            name="Rot gain",
            description="Medias vueltas en Z por ancho de cámara recorrido con mod PINCH",
            default=DEFAULT_ROT_GAIN, min=0.0, max=20.0,
        )
        scale_gain: bpy.props.FloatProperty(  # type: ignore
            name="Scale gain",
            description="Sensibilidad de la escala al separar/juntar las manos",
            default=DEFAULT_SCALE_GAIN, min=0.0, max=20.0,
        )
        z_gain: bpy.props.FloatProperty(  # type: ignore
            name="Z gain",
            description="Elevación en Z por alto de cámara recorrido verticalmente con mod PINCH (0 = apagado)",
            default=DEFAULT_Z_GAIN, min=0.0, max=50.0,
        )
        stale_after: bpy.props.FloatProperty(  # type: ignore
            name="Stale after (s)",
            description="Segundos sin frame nuevo para declarar el bus obsoleto y no actuar",
            default=DEFAULT_STALE_AFTER, min=0.1, max=30.0,
        )
        interval: bpy.props.FloatProperty(  # type: ignore
            name="Interval (s)",
            description="Segundos entre ticks del timer (0.033 ≈ 30 Hz)",
            default=DEFAULT_INTERVAL, min=MIN_INTERVAL, max=1.0, precision=3,
        )

        def draw(self, context: Any) -> None:
            layout = self.layout
            layout.prop(self, "bus_path")
            row = layout.row(align=True)
            row.prop(self, "gain")
            row.prop(self, "smoothing")
            row = layout.row(align=True)
            row.prop(self, "rot_gain")
            row.prop(self, "scale_gain")
            row.prop(self, "z_gain")
            row = layout.row(align=True)
            row.prop(self, "stale_after")
            row.prop(self, "interval")

    # ── Operadores ───────────────────────────────────────────────────────────

    class TLETL_OT_Start(bpy.types.Operator):
        bl_idname  = "tletl.start"
        bl_label   = "Iniciar Tletl"
        bl_description = "Empieza a leer el bus de gestos y controlar el objeto activo"

        def execute(self, context: Any) -> set:
            global _addon_running, last_error, _last_status
            if _addon_running:
                self.report({"INFO"}, "Tletl ya está corriendo.")
                return {"CANCELLED"}
            _addon_running = True
            last_error = ""
            _last_status = (BUS_MISSING, "Esperando el primer tick…")
            _session.reset()
            if not bpy.app.timers.is_registered(_tletl_timer_callback):
                bpy.app.timers.register(_tletl_timer_callback, persistent=True)
            self.report({"INFO"}, f"Tletl iniciado (bus: {_current_settings().resolved_bus_path()}).")
            return {"FINISHED"}

    class TLETL_OT_Stop(bpy.types.Operator):
        bl_idname  = "tletl.stop"
        bl_label   = "Detener Tletl"
        bl_description = "Detiene la lectura del bus de gestos"

        def execute(self, context: Any) -> set:
            global _addon_running, _last_status
            _addon_running = False
            _last_status = (BUS_MISSING, "Tletl detenido")
            _session.reset()
            if bpy.app.timers.is_registered(_tletl_timer_callback):
                bpy.app.timers.unregister(_tletl_timer_callback)
            self.report({"INFO"}, "Tletl detenido.")
            return {"FINISHED"}

    class TLETL_OT_SetMode(bpy.types.Operator):
        bl_idname  = "tletl.set_mode"
        bl_label   = "Modo del addon"
        bl_description = "TRANSFORM mueve/rota/escala el objeto activo; CREATE crea primitivas con PINCH"

        mode: bpy.props.EnumProperty(  # type: ignore
            name="Modo",
            items=[(m, m, "") for m in ADDON_MODES],
            default="TRANSFORM",
        )

        def execute(self, context: Any) -> set:
            _session.set_mode(self.mode)
            return {"FINISHED"}

    class TLETL_OT_CyclePrimitive(bpy.types.Operator):
        bl_idname  = "tletl.cycle_primitive"
        bl_label   = "Siguiente primitiva"
        bl_description = "Cambia la primitiva que crea PINCH en modo CREATE (cube → sphere → cylinder → cone → plane)"

        def execute(self, context: Any) -> set:
            _session.cycle_primitive()
            return {"FINISHED"}

    class TLETL_OT_SpawnNow(bpy.types.Operator):
        bl_idname  = "tletl.spawn_now"
        bl_label   = "Crear en el cursor"
        bl_description = "Crea la primitiva actual en el cursor 3D (misma rutina bmesh que el gesto)"

        def execute(self, context: Any) -> set:
            global last_error
            try:
                _spawn_primitive(_session.primitive, _cursor_location())
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                self.report({"ERROR"}, last_error)
                return {"CANCELLED"}
            return {"FINISHED"}

    # ── Panel N ──────────────────────────────────────────────────────────────

    _STATUS_ICONS = {BUS_OK: "CHECKMARK", BUS_STALE: "ERROR", BUS_MISSING: "CANCEL", BUS_INVALID: "ERROR"}

    class TLETL_PT_Panel(bpy.types.Panel):
        bl_label      = "Tletl Control"
        bl_idname     = "TLETL_PT_panel"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category   = "Tletl"

        def draw(self, context: Any) -> None:
            layout = self.layout
            prefs = _get_prefs()
            settings = AddonSettings.from_prefs(prefs)

            col = layout.column(align=True)
            if _addon_running:
                col.operator("tletl.stop", icon="PAUSE")
            else:
                col.operator("tletl.start", icon="PLAY")

            box = layout.box()
            box.label(text=f"Modo addon: {_session.mode}", icon="ORIENTATION_GIMBAL")
            row = box.row(align=True)
            for mode in ADDON_MODES:
                op = row.operator("tletl.set_mode", text=mode, depress=(_session.mode == mode))
                op.mode = mode
            if _session.mode == "CREATE":
                row = box.row(align=True)
                row.label(text=f"Primitiva: {_session.primitive}", icon="MESH_DATA")
                row.operator("tletl.cycle_primitive", text="", icon="FILE_REFRESH")
                box.operator("tletl.spawn_now", icon="ADD")
                box.label(text=f"Creados: {_session.spawn_count}")
            if _session.last_event:
                box.label(text=_session.last_event)

            box = layout.box()
            box.label(text="Parámetros", icon="PREFERENCES")
            if prefs is not None:
                for name in ("gain", "smoothing", "rot_gain", "scale_gain", "z_gain",
                             "stale_after", "interval"):
                    box.prop(prefs, name)
                box.prop(prefs, "bus_path")
            else:
                box.label(text="Prefs no encontradas: usando defaults", icon="ERROR")

            box = layout.box()
            box.label(text="Estado del bus", icon="LINKED")
            code, message = _last_status
            box.label(text=message, icon=_STATUS_ICONS.get(code, "QUESTION"))
            box.label(text=f"Bus: {settings.resolved_bus_path()}")
            if _last_state:
                for line in status_lines(_last_state):
                    box.label(text=line)
            if last_error:
                layout.label(text=f"Error: {last_error}", icon="ERROR")

    # ── Registro ─────────────────────────────────────────────────────────────

    _CLASSES = [
        TletlAddonPreferences,
        TLETL_OT_Start,
        TLETL_OT_Stop,
        TLETL_OT_SetMode,
        TLETL_OT_CyclePrimitive,
        TLETL_OT_SpawnNow,
        TLETL_PT_Panel,
    ]

    def register() -> None:
        for cls in _CLASSES:
            bpy.utils.register_class(cls)

    def unregister() -> None:
        global _addon_running
        _addon_running = False
        _session.reset()
        if bpy.app.timers.is_registered(_tletl_timer_callback):
            bpy.app.timers.unregister(_tletl_timer_callback)
        for cls in reversed(_CLASSES):
            bpy.utils.unregister_class(cls)
