# -*- coding: utf-8 -*-
"""apps/autocad_control/freecad_macro.py — Tletl -> FreeCAD EN VIVO (ruta 3, Fedora).

FreeCAD es libre, nativo en Fedora (`sudo dnf install freecad`) y su unidad
interna es el milímetro. Esta macro:

  1. lee el bus de archivo que escribe la app de cámara (misma regla de ruta que
     tletl_core.paths.default_bus_path: $TLETL_STATE_PATH > $TLETL_HOME/
     tletl_state.json > ~/.tletl/tletl_state.json — replicada aquí porque el
     Python de FreeCAD no puede importar tletl_core salvo que el repo esté en
     sys.path),
  2. lo pasa por el MISMO GestureModeler de las otras rutas si encuentra el repo
     (variable de entorno TLETL_REPO=/ruta/al/repo, o esta macro ejecutada desde
     dentro del repo); si no, usa `MinimalModeler` (mover + crear + tipo + modo,
     sin rotación ni escala),
  3. y crea/actualiza primitivas Part (Part::Box, Part::Cylinder, Part::Sphere,
     Part::Cone y una cuña extruida) con un QTimer de 50 ms.

Uso en FreeCAD:
    export TLETL_REPO=/ruta/tletl_v5     # opcional pero recomendado (rotar/escalar)
    freecad &
    Macro > Macros… > "Ubicación de macros de usuario" > copiar este archivo ahí
    (o elegirlo con el botón de carpeta) > Ejecutar.
    En la consola de Python: `stop()` detiene el timer; `start()` lo reinicia.

Todo import de FreeCAD/Qt está protegido: el módulo se importa y se prueba en
Linux sin FreeCAD (tests/test_autocad_control.py). NO se pudo ejecutar dentro
de FreeCAD desde el entorno de desarrollo: las llamadas a Part/Placement son
las documentadas (Part::Box.Length/Width/Height, Part::Cylinder.Radius/Height,
Part::Sphere.Radius, Part::Cone.Radius1/Radius2/Height, obj.Placement).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # ── guard de FreeCAD ────────────────────────────────────────────────
    import FreeCAD as App  # type: ignore
    import FreeCADGui as Gui  # type: ignore
    import Part  # type: ignore
    FREECAD_AVAILABLE = True
except ImportError:
    App = Gui = Part = None  # type: ignore
    FREECAD_AVAILABLE = False

QtCore: Any = None
if FREECAD_AVAILABLE:  # pragma: no cover - solo dentro de FreeCAD
    for _qt in ("PySide", "PySide2", "PySide6"):
        try:
            QtCore = __import__(_qt, fromlist=["QtCore"]).QtCore
            break
        except ImportError:
            continue

ENV_HOME = "TLETL_HOME"
ENV_STATE_PATH = "TLETL_STATE_PATH"
ENV_REPO = "TLETL_REPO"
BUS_FILENAME = "tletl_state.json"
INTERVAL_MS = 50
DEFAULT_GAIN = 100.0        # mm que recorre el sólido al cruzar el encuadre (FreeCAD trabaja en mm)
DEFAULT_MODE_HOLD = 0.8
DEFAULT_STALE_AFTER = 1.0
MIN_SCALE = 0.05
SOLID_KINDS = ("box", "cylinder", "sphere", "cone", "wedge")


# ── Helpers puros (testeables sin FreeCAD) ──────────────────────────────────

def resolve_bus_path() -> Path:
    """Réplica de tletl_core.paths.default_bus_path() (FreeCAD no importa tletl_core)."""
    value = os.environ.get(ENV_STATE_PATH, "").strip()
    if value:
        return Path(value).expanduser()
    home = os.environ.get(ENV_HOME, "").strip()
    base = Path(home).expanduser() if home else Path.home() / ".tletl"
    return base / BUS_FILENAME


def read_bus(path: Path) -> Dict[str, Any]:
    """Dict del bus o {} (archivo ausente, vacío o corrupto). Nunca lanza."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def repo_candidates() -> List[Path]:
    """Dónde buscar el repo: $TLETL_REPO y la carpeta del repo si la macro vive en él."""
    out: List[Path] = []
    env = os.environ.get(ENV_REPO, "").strip()
    if env:
        out.append(Path(env).expanduser())
    try:
        out.append(Path(__file__).resolve().parents[2])
    except (NameError, IndexError):
        pass
    return out


def load_modeler_class() -> Optional[type]:
    """GestureModeler del repo, o None si no está al alcance (=> MinimalModeler)."""
    for root in repo_candidates():
        if not (root / "apps" / "autocad_control" / "gesture_modeler.py").exists():
            continue
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from apps.autocad_control.gesture_modeler import GestureModeler  # type: ignore
            return GestureModeler
        except ImportError:
            continue
    return None


def local_center(kind: str, dims: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Centro de la caja envolvente en coordenadas LOCALES de la primitiva Part:
    Part::Box nace en la esquina (0,0,0); cilindro/cono con la base en z=0;
    esfera y cuña ya centradas."""
    sx, sy, sz = dims
    if kind == "box":
        return (sx / 2.0, sy / 2.0, sz / 2.0)
    if kind in ("cylinder", "cone"):
        return (0.0, 0.0, sz / 2.0)
    return (0.0, 0.0, 0.0)


def placement_base(center: Tuple[float, float, float], local: Tuple[float, float, float],
                   rotation_z: float) -> Tuple[float, float, float]:
    """Base del Placement para que, tras rotar `rotation_z` alrededor de Z, el
    centro local caiga en `center`: base = center - R·local."""
    c, s = math.cos(rotation_z), math.sin(rotation_z)
    rx = c * local[0] - s * local[1]
    ry = s * local[0] + c * local[1]
    return (center[0] - rx, center[1] - ry, center[2] - local[2])


def _hand_gesture(state: Dict[str, Any], side: str) -> str:
    hand = state.get(side) if isinstance(state, dict) else None
    if not isinstance(hand, dict):
        return "NO_HAND"
    return str(hand.get("gesture") or hand.get("stable_gesture") or "NO_HAND")


def _hand_palm(state: Dict[str, Any], side: str) -> Optional[Tuple[float, float]]:
    hand = state.get(side) if isinstance(state, dict) else None
    palm = hand.get("palm") if isinstance(hand, dict) else None
    if isinstance(palm, (list, tuple)) and len(palm) >= 2:
        return float(palm[0]), float(palm[1])
    return None


class _MiniSolid:
    """Sólido mínimo con la misma interfaz que gesture_modeler.Solid (lo que lee el backend)."""

    __slots__ = ("id", "kind", "center", "size", "rotation_z", "scale")

    def __init__(self, id: int, kind: str, center: Tuple[float, float, float], size: Tuple[float, float, float]):
        self.id = int(id)
        self.kind = kind
        self.center = tuple(float(v) for v in center)
        self.size = tuple(float(v) for v in size)
        self.rotation_z = 0.0
        self.scale = 1.0

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        return (self.size[0] * self.scale, self.size[1] * self.scale, self.size[2] * self.scale)

    def copy(self) -> "_MiniSolid":
        """Instantánea para la op "add" (como hace GestureModeler): los moves
        posteriores no deben alterar lo que ya se entregó al backend."""
        other = _MiniSolid(self.id, self.kind, self.center, self.size)
        other.rotation_z, other.scale = self.rotation_z, self.scale
        return other


class MinimalModeler:
    """Fallback sin dependencias con el subconjunto esencial del GestureModeler:
    THREE 0.8 s alterna modo, CREATE+PINCH crea, VICTORY cambia tipo, TRANSFORM+
    PINCH mueve por delta, OPEN_PALM suelta, FIST rompe continuidad, frame
    obsoleto => nada. Sin rotación/escala/fin de sesión (usa TLETL_REPO)."""

    def __init__(self, gain: float = DEFAULT_GAIN, spawn_size: Optional[float] = None,
                 mode_hold: float = DEFAULT_MODE_HOLD, stale_after: Optional[float] = DEFAULT_STALE_AFTER):
        self.gain = float(gain)
        self.spawn_size = float(spawn_size) if spawn_size is not None else abs(self.gain) * 0.1
        self.mode_hold = float(mode_hold)
        self.stale_after = stale_after
        self.mode = "TRANSFORM"
        self.kind_index = 0
        self.solids: List[_MiniSolid] = []
        self.selected_id: Optional[int] = None
        self.next_id = 1
        self.status = "esperando bus"
        self.ops_emitted = 0
        self._prev_palm: Optional[Tuple[float, float]] = None
        self._prev_gesture = "NO_HAND"
        self._three_since: Optional[float] = None
        self._three_fired = False

    @property
    def kind(self) -> str:
        return SOLID_KINDS[self.kind_index]

    def describe(self) -> str:
        sel = f"#{self.selected_id}" if self.selected_id is not None else "-"
        return f"modo={self.mode} tipo={self.kind} sel={sel} solidos={len(self.solids)} (MinimalModeler)"

    def _selected(self) -> Optional[_MiniSolid]:
        for s in self.solids:
            if s.id == self.selected_id:
                return s
        return None

    def update(self, state: Dict[str, Any], now: Optional[float] = None) -> List[tuple]:
        now = time.time() if now is None else float(now)
        ts = state.get("timestamp") if isinstance(state, dict) else None
        if self.stale_after is not None and isinstance(ts, (int, float)) and ts > 0 and now - ts > self.stale_after:
            self._prev_palm = None
            self._three_since = None
            self.status = "bus obsoleto"
            return []
        d = _hand_gesture(state, "dom")
        palm = _hand_palm(state, "dom")
        rising_pinch = d == "PINCH" and self._prev_gesture != "PINCH"
        rising_victory = d == "VICTORY" and self._prev_gesture != "VICTORY"
        self._prev_gesture = d
        ops: List[tuple] = []

        if d == "THREE":
            if self._three_since is None:
                self._three_since = now
                self._three_fired = False
            elif not self._three_fired and now - self._three_since >= self.mode_hold:
                self._three_fired = True
                self.mode = "CREATE" if self.mode == "TRANSFORM" else "TRANSFORM"
                self._prev_palm = None
                ops.append(("mode", self.mode))
        else:
            self._three_since = None
            self._three_fired = False

        if d == "FIST":
            self._prev_palm = None
            self.status = "FIST: parada de seguridad"
        elif self.mode == "CREATE":
            self._prev_palm = None
            if rising_victory:
                self.kind_index = (self.kind_index + 1) % len(SOLID_KINDS)
                ops.append(("kind", self.kind))
            if rising_pinch and palm is not None:
                s = self.spawn_size
                solid = _MiniSolid(self.next_id, self.kind,
                                   ((palm[0] - 0.5) * self.gain, (0.5 - palm[1]) * self.gain, 0.0), (s, s, s))
                self.next_id += 1
                self.solids.append(solid)
                self.selected_id = solid.id
                ops.append(("add", solid.copy()))
                ops.append(("select", solid.id))
                self.status = f"creado {solid.kind} #{solid.id}"
        else:
            if d == "PINCH" and palm is not None:
                sel = self._selected()
                if self._prev_palm is not None and sel is not None:
                    dx = (palm[0] - self._prev_palm[0]) * self.gain
                    dy = (self._prev_palm[1] - palm[1]) * self.gain
                    if dx != 0.0 or dy != 0.0:
                        sel.center = (sel.center[0] + dx, sel.center[1] + dy, sel.center[2])
                        ops.append(("move", sel.id, dx, dy, 0.0))
                self._prev_palm = palm
                self.status = "PINCH: moviendo" if sel is not None else "PINCH sin sólido (THREE -> CREATE)"
            else:
                self._prev_palm = None
        self.ops_emitted += len(ops)
        return ops


# ── Backend FreeCAD (solo corre dentro de FreeCAD) ──────────────────────────

def wedge_shape(sx: float, sy: float, sz: float) -> Any:  # pragma: no cover - necesita Part
    """Cuña centrada: cara vertical en -x, rampa hacia +x, extruida en Y."""
    V = App.Vector
    profile = Part.makePolygon([V(-sx / 2, -sy / 2, -sz / 2), V(sx / 2, -sy / 2, -sz / 2),
                                V(-sx / 2, -sy / 2, sz / 2), V(-sx / 2, -sy / 2, -sz / 2)])
    return Part.Face(profile).extrude(V(0, sy, 0))


class FreeCadSceneBackend:  # pragma: no cover - necesita FreeCAD
    """Aplica ops a un documento de FreeCAD con primitivas Part."""

    def __init__(self, doc: Any = None):
        if doc is None:
            doc = App.ActiveDocument or App.newDocument("Tletl")
        self.doc = doc
        self.objects: Dict[int, Any] = {}
        self.solids: Dict[int, Dict[str, Any]] = {}
        self.selected_id: Optional[int] = None

    def apply(self, ops: List[tuple]) -> int:
        n = 0
        for op in ops:
            try:
                if self.apply_op(op):
                    n += 1
            except Exception as exc:
                App.Console.PrintError(f"[tletl] error aplicando {op[0]}: {exc}\n")
        if n:
            self.doc.recompute()
        return n

    def apply_op(self, op: tuple) -> bool:
        name = op[0]
        if name == "add":
            self._add(op[1])
            return True
        rec = self.solids.get(op[1]) if len(op) > 1 else None
        if rec is None:
            return False
        if name == "move":
            cx, cy, cz = rec["center"]
            rec["center"] = (cx + op[2], cy + op[3], cz + op[4])
            self._place(op[1])
        elif name == "rotate":
            rec["rotation_z"] += op[2]
            self._place(op[1])
        elif name == "scale":
            rec["scale"] = max(MIN_SCALE, rec["scale"] * op[2])
            self._resize(op[1])
            self._place(op[1])
        elif name == "select":
            self.selected_id = op[1]
            try:
                Gui.Selection.clearSelection()
                Gui.Selection.addSelection(self.objects[op[1]])
            except Exception:
                pass
        else:
            return False
        return True

    def _add(self, solid: Any) -> None:
        kind = solid.kind
        name = f"Tletl{kind.capitalize()}{solid.id}"
        if kind == "box":
            obj = self.doc.addObject("Part::Box", name)
        elif kind == "cylinder":
            obj = self.doc.addObject("Part::Cylinder", name)
        elif kind == "sphere":
            obj = self.doc.addObject("Part::Sphere", name)
        elif kind == "cone":
            obj = self.doc.addObject("Part::Cone", name)
        else:
            obj = self.doc.addObject("Part::Feature", name)
        self.objects[solid.id] = obj
        self.solids[solid.id] = {"kind": kind, "center": tuple(solid.center), "size": tuple(solid.size),
                                 "rotation_z": float(solid.rotation_z), "scale": float(solid.scale)}
        self._resize(solid.id)
        self._place(solid.id)

    def _dims(self, solid_id: int) -> Tuple[float, float, float]:
        rec = self.solids[solid_id]
        s = rec["scale"]
        return (rec["size"][0] * s, rec["size"][1] * s, rec["size"][2] * s)

    def _resize(self, solid_id: int) -> None:
        obj = self.objects[solid_id]
        kind = self.solids[solid_id]["kind"]
        sx, sy, sz = self._dims(solid_id)
        if kind == "box":
            obj.Length, obj.Width, obj.Height = sx, sy, sz
        elif kind == "cylinder":
            obj.Radius, obj.Height = sx / 2.0, sz
        elif kind == "sphere":
            obj.Radius = sx / 2.0
        elif kind == "cone":
            obj.Radius1, obj.Radius2, obj.Height = sx / 2.0, 0.0, sz
        else:
            obj.Shape = wedge_shape(sx, sy, sz)

    def _place(self, solid_id: int) -> None:
        obj = self.objects[solid_id]
        rec = self.solids[solid_id]
        base = placement_base(rec["center"], local_center(rec["kind"], self._dims(solid_id)), rec["rotation_z"])
        rot = App.Rotation(App.Vector(0, 0, 1), math.degrees(rec["rotation_z"]))
        obj.Placement = App.Placement(App.Vector(*base), rot)


class TletlFreeCadBridge:  # pragma: no cover - necesita FreeCAD + Qt
    """QTimer que lee el bus y aplica ops. Guardado en `_BRIDGE` para que el timer viva."""

    def __init__(self, bus_path: Optional[str] = None, gain: float = DEFAULT_GAIN,
                 interval_ms: int = INTERVAL_MS):
        self.bus_path = Path(bus_path).expanduser() if bus_path else resolve_bus_path()
        modeler_cls = load_modeler_class()
        if modeler_cls is not None:
            self.modeler: Any = modeler_cls(gain=gain)
            self.modeler_name = "GestureModeler (repo)"
        else:
            self.modeler = MinimalModeler(gain=gain)
            self.modeler_name = "MinimalModeler (sin TLETL_REPO: sin rotar/escalar)"
        self.backend = FreeCadSceneBackend()
        self._last_text: Optional[str] = None
        self.frames = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(interval_ms))

    def _poll(self) -> Optional[Dict[str, Any]]:
        try:
            text = self.bus_path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return None
        if text == self._last_text:
            return None
        self._last_text = text
        state = read_bus(self.bus_path)
        return state or None

    def tick(self) -> None:
        state = self._poll()
        if state is None:
            return
        self.frames += 1
        ops = self.modeler.update(state, time.time())
        if ops:
            self.backend.apply(ops)
            for op in ops:
                detail = op[1] if len(op) > 1 else ""
                if hasattr(detail, "id"):          # op "add": el sólido
                    detail = f"#{detail.id} {detail.kind}"
                App.Console.PrintMessage(f"[tletl] {op[0]} {detail}\n")

    def stop(self) -> None:
        self.timer.stop()


_BRIDGE: Optional[Any] = None


def start(bus_path: Optional[str] = None, gain: float = DEFAULT_GAIN, interval_ms: int = INTERVAL_MS) -> Any:
    """Arranca (o reinicia) el puente. Llamar desde la consola de Python de FreeCAD."""
    global _BRIDGE
    if not FREECAD_AVAILABLE or QtCore is None:
        raise RuntimeError("esta macro debe ejecutarse dentro de FreeCAD (FreeCADGui + PySide)")
    if _BRIDGE is not None:
        _BRIDGE.stop()
    _BRIDGE = TletlFreeCadBridge(bus_path=bus_path, gain=gain, interval_ms=interval_ms)
    App.Console.PrintMessage(
        f"[tletl] puente activo: bus={_BRIDGE.bus_path} modeler={_BRIDGE.modeler_name} "
        f"gain={gain:g} mm. THREE 0.8 s = modo, CREATE+PINCH = crear, VICTORY = tipo. stop() para parar.\n")
    return _BRIDGE


def stop() -> None:
    global _BRIDGE
    if _BRIDGE is not None:
        _BRIDGE.stop()
        _BRIDGE = None
        if FREECAD_AVAILABLE:
            App.Console.PrintMessage("[tletl] puente detenido.\n")


if __name__ == "__main__":
    if FREECAD_AVAILABLE and QtCore is not None:  # pragma: no cover
        start()
    else:
        print("Esta macro se ejecuta dentro de FreeCAD (Macro > Macros… > Ejecutar). "
              f"Bus que leería: {resolve_bus_path()}")
