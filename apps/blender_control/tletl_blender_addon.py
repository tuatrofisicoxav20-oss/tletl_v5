"""
apps/blender_control/tletl_blender_addon.py
Addon de Blender para controlar objetos 3D con gestos Tletl.

bpy se importa de forma LAZY / guarded para que el módulo sea testeable
fuera de Blender. TODO el código que usa bpy está detrás del guard
`if bpy is not None`.

La inteligencia de mapeo vive en `BlenderGestureMapper` (clase PURA, sin bpy,
con estado para calcular deltas entre frames). El addon mantiene una instancia
global y la alimenta con el estado del bus en cada tick del timer.

Instalación:
    Edit > Preferences > Add-ons > Install… > seleccionar este archivo.
    Activar "Tletl Gesture Control".
    Panel N > pestaña "Tletl" en cualquier vista 3D.
"""
from __future__ import annotations

# ── Guard de bpy (LAZY) ──────────────────────────────────────────────────────
try:
    import bpy  # type: ignore
except ImportError:
    bpy = None  # type: ignore

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Importar state_reader de forma compatible con uso como paquete (tests)
# y como addon standalone de Blender (carga directa de archivo).
try:
    from .state_reader import read_state, dom_gesture, mod_gesture, dom_palm, mod_palm  # type: ignore
except ImportError:
    # Blender carga el addon como módulo top-level; usamos import absoluto
    import importlib as _il
    import os as _os
    import sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    _sr = _il.import_module("state_reader")
    read_state  = _sr.read_state    # type: ignore
    dom_gesture = _sr.dom_gesture   # type: ignore
    mod_gesture = _sr.mod_gesture   # type: ignore
    dom_palm    = _sr.dom_palm      # type: ignore
    mod_palm    = _sr.mod_palm      # type: ignore


# ── Metadatos del addon ──────────────────────────────────────────────────────

bl_info = {
    "name":        "Tletl Gesture Control",
    "author":      "Tletl Project",
    "version":     (5, 0, 0),
    "blender":     (4, 0, 0),
    "location":    "View3D > N-Panel > Tletl",
    "description": "Controla objetos 3D con gestos de mano en tiempo real.",
    "category":    "Object",
}


# ── Constantes ───────────────────────────────────────────────────────────────

DEFAULT_BUS_PATH  = str(Path.home() / "tletl_state.json")
DEFAULT_GAIN      = 4.0
DEFAULT_SMOOTHING = 0.35
DEFAULT_ROT_GAIN  = 2.0   # vueltas relativas por unidad de desplazamiento de mod.x
DEFAULT_SCALE_GAIN = 1.5  # sensibilidad del escalado por distancia entre manos
DEFAULT_INTERVAL  = 0.05  # segundos entre actualizaciones (~20 fps)
MIN_SCALE         = 0.05

TRANSFORM_KEYS = ("x", "y", "z", "rot_x", "rot_y", "rot_z", "scale")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LÓGICA DE MAPEO — PURA, SIN bpy, TESTEABLE                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _lerp(a: float, b: float, t: float) -> float:
    """Interpolación lineal: a + (b - a) * t, t en [0, 1]."""
    return a + (b - a) * t


def normalize_transform(current: Dict[str, float]) -> Dict[str, float]:
    """Devuelve un transform con todas las claves y defaults seguros."""
    return {
        "x":     float(current.get("x",     0.0)),
        "y":     float(current.get("y",     0.0)),
        "z":     float(current.get("z",     0.0)),
        "rot_x": float(current.get("rot_x", 0.0)),
        "rot_y": float(current.get("rot_y", 0.0)),
        "rot_z": float(current.get("rot_z", 0.0)),
        "scale": float(current.get("scale", 1.0)),
    }


class BlenderGestureMapper:
    """Mapea el estado del bus a transformaciones del objeto, por DELTA entre frames.

    Mantiene la palma anterior de cada mano para traducir el MOVIMIENTO (no la
    posición absoluta) en cambios de transform — es decir, "agarrar y arrastrar"
    real, sin la deriva continua del modo joystick.

    Reglas:
      - dom FIST           → safety stop: no toca nada y resetea continuidad.
      - dom OPEN_PALM      → release: rompe la continuidad de la mano dominante
                             (al volver a hacer PINCH no pega un salto).
      - dom PINCH          → traslación XY por delta de la palma dominante.
      - mod PINCH          → rotación Z por delta de la palma modificadora (x).
      - dom PINCH + mod OPEN_PALM (ambas manos) → escala por cambio de la
                             distancia entre las dos palmas (pinch-to-zoom).
    """

    def __init__(self, gain: float = DEFAULT_GAIN, smoothing: float = DEFAULT_SMOOTHING,
                 rot_gain: float = DEFAULT_ROT_GAIN, scale_gain: float = DEFAULT_SCALE_GAIN):
        self.gain = float(gain)
        self.smoothing = float(smoothing)
        self.rot_gain = float(rot_gain)
        self.scale_gain = float(scale_gain)
        self._prev_dom: Optional[Tuple[float, float]] = None
        self._prev_mod: Optional[Tuple[float, float]] = None
        self._prev_dist: Optional[float] = None

    def reset(self) -> None:
        self._prev_dom = None
        self._prev_mod = None
        self._prev_dist = None

    def update(self, state: Dict[str, Any], current: Dict[str, float]) -> Dict[str, float]:
        out = normalize_transform(current)
        d = dom_gesture(state)
        m = mod_gesture(state)
        dpalm = dom_palm(state)
        mpalm = mod_palm(state)

        # --- Safety stop: soltar todo ---
        if d == "FIST":
            self.reset()
            return out

        # --- Release de la mano dominante (no mover, romper continuidad) ---
        if d == "OPEN_PALM":
            self._prev_dom = None

        # --- Traslación XY por delta de la palma dominante (PINCH = agarrar) ---
        if d == "PINCH" and dpalm is not None:
            if self._prev_dom is not None:
                dx = (dpalm[0] - self._prev_dom[0]) * self.gain
                dy = (self._prev_dom[1] - dpalm[1]) * self.gain  # Y invertido (pantalla→mundo)
                out["x"] = _lerp(out["x"], out["x"] + dx, self.smoothing)
                out["y"] = _lerp(out["y"], out["y"] + dy, self.smoothing)
            self._prev_dom = dpalm
        else:
            self._prev_dom = None

        # --- Rotación Z por delta de la palma modificadora ---
        if m == "PINCH" and mpalm is not None:
            if self._prev_mod is not None:
                dz = (mpalm[0] - self._prev_mod[0]) * self.rot_gain * math.pi
                out["rot_z"] = _lerp(out["rot_z"], out["rot_z"] + dz, self.smoothing)
            self._prev_mod = mpalm
        else:
            self._prev_mod = None

        # --- Escala por distancia entre manos (dom PINCH + mod OPEN_PALM) ---
        if dpalm is not None and mpalm is not None and d == "PINCH" and m == "OPEN_PALM":
            dist = math.hypot(dpalm[0] - mpalm[0], dpalm[1] - mpalm[1])
            if self._prev_dist is not None and self._prev_dist > 1e-6:
                delta = (dist - self._prev_dist) * self.scale_gain
                target = out["scale"] + delta
                out["scale"] = max(MIN_SCALE, _lerp(out["scale"], target, self.smoothing))
            self._prev_dist = dist
        else:
            self._prev_dist = None

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


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CÓDIGO bpy — sólo se ejecuta dentro de Blender                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if bpy is not None:

    # ── Estado global del addon ──────────────────────────────────────────────

    _addon_running = False
    _mapper = BlenderGestureMapper()

    # Tipos de objeto que tienen location/rotation/scale utilizables.
    _CONTROLLABLE_TYPES = {"MESH", "EMPTY", "CURVE", "SURFACE", "META", "FONT",
                           "ARMATURE", "LATTICE", "LIGHT", "CAMERA", "GPENCIL"}

    def _get_prefs() -> Any:
        return bpy.context.preferences.addons[__name__].preferences

    def _apply_transform_to_object(obj: Any, t: Dict[str, float]) -> None:
        """Escribe el transform calculado al objeto activo de Blender."""
        obj.location.x     = t["x"]
        obj.location.y     = t["y"]
        obj.location.z     = t["z"]
        obj.rotation_euler = (t["rot_x"], t["rot_y"], t["rot_z"])
        obj.scale          = (t["scale"], t["scale"], t["scale"])

    def _read_current_transform(obj: Any) -> Dict[str, float]:
        return {
            "x":     obj.location.x,
            "y":     obj.location.y,
            "z":     obj.location.z,
            "rot_x": obj.rotation_euler.x,
            "rot_y": obj.rotation_euler.y,
            "rot_z": obj.rotation_euler.z,
            "scale": obj.scale.x,
        }

    # ── Timer callback ───────────────────────────────────────────────────────

    def _tletl_timer_callback() -> Optional[float]:
        """Función que Blender llama periódicamente vía bpy.app.timers."""
        if not _addon_running:
            return None  # detiene el timer

        prefs = _get_prefs()
        _mapper.gain = prefs.gain
        _mapper.smoothing = prefs.smoothing

        state = read_state(prefs.bus_path)
        if not state:
            return DEFAULT_INTERVAL

        obj = bpy.context.active_object
        if obj is None or obj.type not in _CONTROLLABLE_TYPES:
            return DEFAULT_INTERVAL

        current = _read_current_transform(obj)
        new_t = _mapper.update(state, current)
        _apply_transform_to_object(obj, new_t)

        # Forzar redibujado
        if bpy.context.screen is not None:
            for area in bpy.context.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()

        return DEFAULT_INTERVAL

    # ── Preferencias del addon ───────────────────────────────────────────────

    class TletlAddonPreferences(bpy.types.AddonPreferences):
        bl_idname = __name__

        bus_path: bpy.props.StringProperty(  # type: ignore
            name="Bus path",
            description="Ruta al archivo tletl_state.json",
            default=DEFAULT_BUS_PATH,
            subtype="FILE_PATH",
        )
        gain: bpy.props.FloatProperty(  # type: ignore
            name="Gain",
            description="Factor de amplificación del movimiento",
            default=DEFAULT_GAIN,
            min=0.1, max=20.0,
        )
        smoothing: bpy.props.FloatProperty(  # type: ignore
            name="Smoothing",
            description="Factor de suavizado del movimiento (0=rígido, 1=instantáneo)",
            default=DEFAULT_SMOOTHING,
            min=0.01, max=1.0,
        )

        def draw(self, context: Any) -> None:
            layout = self.layout
            layout.prop(self, "bus_path")
            layout.prop(self, "gain")
            layout.prop(self, "smoothing")

    # ── Operadores ───────────────────────────────────────────────────────────

    class TLETL_OT_Start(bpy.types.Operator):
        bl_idname  = "tletl.start"
        bl_label   = "Iniciar Tletl"
        bl_description = "Empieza a leer el bus de gestos y controlar el objeto activo"

        def execute(self, context: Any) -> set:
            global _addon_running
            if _addon_running:
                self.report({"INFO"}, "Tletl ya está corriendo.")
                return {"CANCELLED"}
            _addon_running = True
            _mapper.reset()
            if not bpy.app.timers.is_registered(_tletl_timer_callback):
                bpy.app.timers.register(_tletl_timer_callback, persistent=True)
            self.report({"INFO"}, "Tletl iniciado.")
            return {"FINISHED"}

    class TLETL_OT_Stop(bpy.types.Operator):
        bl_idname  = "tletl.stop"
        bl_label   = "Detener Tletl"
        bl_description = "Detiene la lectura del bus de gestos"

        def execute(self, context: Any) -> set:
            global _addon_running
            _addon_running = False
            if bpy.app.timers.is_registered(_tletl_timer_callback):
                bpy.app.timers.unregister(_tletl_timer_callback)
            self.report({"INFO"}, "Tletl detenido.")
            return {"FINISHED"}

    # ── Panel N ──────────────────────────────────────────────────────────────

    class TLETL_PT_Panel(bpy.types.Panel):
        bl_label      = "Tletl Control"
        bl_idname     = "TLETL_PT_panel"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category   = "Tletl"

        def draw(self, context: Any) -> None:
            layout = self.layout
            prefs  = _get_prefs()

            col = layout.column(align=True)
            if _addon_running:
                col.operator("tletl.stop", icon="PAUSE")
            else:
                col.operator("tletl.start", icon="PLAY")

            layout.separator()
            box = layout.box()
            box.label(text="Parámetros:")
            box.prop(prefs, "gain")
            box.prop(prefs, "smoothing")
            box.prop(prefs, "bus_path")

            layout.separator()
            layout.label(text="Estado:")
            state = read_state(prefs.bus_path)
            if state:
                layout.label(text=f"Dom: {dom_gesture(state)}  Mod: {mod_gesture(state)}")
                palm = dom_palm(state)
                if palm:
                    layout.label(text=f"Palm: ({palm[0]:.2f}, {palm[1]:.2f})")
            else:
                layout.label(text="Bus sin datos", icon="ERROR")

    # ── Registro ─────────────────────────────────────────────────────────────

    _CLASSES = [
        TletlAddonPreferences,
        TLETL_OT_Start,
        TLETL_OT_Stop,
        TLETL_PT_Panel,
    ]

    def register() -> None:
        for cls in _CLASSES:
            bpy.utils.register_class(cls)

    def unregister() -> None:
        global _addon_running
        _addon_running = False
        if bpy.app.timers.is_registered(_tletl_timer_callback):
            bpy.app.timers.unregister(_tletl_timer_callback)
        for cls in reversed(_CLASSES):
            bpy.utils.unregister_class(cls)
