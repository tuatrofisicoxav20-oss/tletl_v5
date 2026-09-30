"""apps/autocad_control/autocad_client.py — AutoCAD EN VIVO desde Windows (ruta 1).

AutoCAD no corre en Linux. Este cliente corre en la máquina Windows donde está
AutoCAD abierto: recibe por UDP los frames que la app de cámara de Fedora
publica (`[bus] udp_target = "IP_WINDOWS:5055"` o `TLETL_BUS_UDP`), alimenta el
mismo `GestureModeler` de las otras rutas y aplica cada op a AutoCAD por
automatización COM (pywin32: `win32com.client.Dispatch("AutoCAD.Application")`).

    python -m apps.autocad_control.autocad_client --udp 0.0.0.0:5055
    python -m apps.autocad_control.autocad_client --bus PATH      # cámara y AutoCAD en la misma PC
    python -m apps.autocad_control.autocad_client --dry-run       # solo verifica que llegan frames

Requisitos en Windows: Python 3.12, `pip install pywin32`, el repo clonado
(solo usa tletl_core.bus/paths y este paquete: NO necesita numpy/opencv/
mediapipe), AutoCAD abierto con un dibujo activo, y el firewall permitiendo
UDP 5055 entrante.

PITFALL COM: los puntos deben viajar como VARIANT de doubles
(VT_ARRAY | VT_R8). Una tupla de Python llega como VT_VARIANT y AutoCAD
contesta "Invalid argument". `_point()` lo resuelve, y devuelve una tupla
cuando pythoncom no existe (Linux/tests).

HONESTIDAD: este módulo NO se pudo ejecutar contra AutoCAD real desde el
entorno de desarrollo (Linux, sin COM). La lógica se probó con un ModelSpace
falso que registra las llamadas (tests/test_autocad_control.py). Los nombres y
firmas COM son los de la referencia ActiveX de AutoCAD:
AddBox(Center, L, W, H), AddCylinder(Center, R, H), AddSphere(Center, R),
AddCone(Center, R, H), AddWedge(Center, L, W, H), Move(P1, P2),
Rotate3D(P1, P2, ang_rad), ScaleEntity(Base, factor), Document.Regen(1).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Optional

from .bus_source import open_source
from .dxf_export import load_scene_json
from .gesture_modeler import Op, Scene, Solid, format_op
from .session import (
    DEFAULT_UDP,
    add_modeler_arguments,
    add_source_arguments,
    modeler_from_args,
    run_loop,
)

PROG_ID = "AutoCAD.Application"
# AcRegenType (biblioteca de tipos de AutoCAD, p. ej. comtypes.gen.AutoCAD / win32com):
#   acActiveViewport = 0, acAllViewports = 1. Estaban invertidos: Regen(1) regeneraba TODOS los viewports.
AC_ACTIVE_VIEWPORT = 0   # acActiveViewport (Document.Regen)
AC_ALL_VIEWPORTS = 1     # acAllViewports
PYWIN32_HINT = "pywin32 no está instalado (o esto no es Windows): pip install pywin32"


def _point(x: float, y: float, z: float = 0.0) -> Any:
    """Punto 3D para COM: VARIANT(VT_ARRAY | VT_R8, [x, y, z]) si hay pythoncom,
    tupla de floats si no (Linux/tests)."""
    try:
        import pythoncom  # type: ignore
        from win32com.client import VARIANT  # type: ignore
    except ImportError:
        return (float(x), float(y), float(z))
    return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, [float(x), float(y), float(z)])


def dispatch_autocad(prog_id: str = PROG_ID) -> Any:
    """Se conecta a la instancia de AutoCAD abierta (Dispatch la lanza si no hay)."""
    try:
        import win32com.client  # type: ignore
    except ImportError as exc:
        raise ImportError(PYWIN32_HINT) from exc
    app = win32com.client.Dispatch(prog_id)
    try:
        app.Visible = True
    except Exception:  # pragma: no cover - algunas versiones lo rechazan; no importa
        pass
    return app


class AutoCADBackend:
    """Aplica ops del GestureModeler a AutoCAD. Mantiene `entities` {id: entidad COM}
    y `solids` {id: Solid} (copia local para conocer centro/rotación al rotar y escalar)."""

    def __init__(self, app: Any = None, *, regen_every: int = 25,
                 log: Optional[Callable[[str], Any]] = None):
        self.app = app if app is not None else dispatch_autocad()
        try:
            self.doc = self.app.ActiveDocument
            self.space = self.doc.ModelSpace
        except Exception as exc:
            raise RuntimeError("AutoCAD no tiene un dibujo activo: abre o crea un dibujo (Ctrl+N) "
                               "y vuelve a correr el cliente") from exc
        self.entities: Dict[int, Any] = {}
        self.solids: Dict[int, Solid] = {}
        self.selected_id: Optional[int] = None
        self.regen_every = int(regen_every)
        self.log = log
        self.applied = 0
        self.failed = 0
        self._since_regen = 0

    # -- entrada --
    def apply(self, ops: Iterable[Op]) -> int:
        """Aplica una lista de ops; devuelve cuántas se aplicaron con éxito."""
        return sum(1 for op in ops if self.apply_op(op))

    def apply_op(self, op: Op) -> bool:
        name = op[0]
        try:
            if name == "add":
                self._add(op[1])
            elif name == "move":
                self._move(int(op[1]), float(op[2]), float(op[3]), float(op[4]))
            elif name == "rotate":
                self._rotate(int(op[1]), float(op[2]))
            elif name == "scale":
                self._scale(int(op[1]), float(op[2]))
            elif name == "select":
                self._select(int(op[1]))
            else:
                return False   # mode / kind / end_session: no tocan el dibujo
        except Exception as exc:   # COM lanza pywintypes.com_error; el bucle no debe morir
            self.failed += 1
            if self.log is not None:
                self.log(f"[autocad] error aplicando '{format_op(op)}': {exc}")
            return False
        self.applied += 1
        self._since_regen += 1
        if self.regen_every > 0 and self._since_regen >= self.regen_every:
            self.regen()
        return True

    def load_scene(self, scene: Scene) -> int:
        """Dibuja una escena guardada (JSON) antes de empezar a modelar."""
        n = sum(1 for solid in scene.solids if self.apply_op(("add", solid)))
        if scene.selected_id is not None:
            self.apply_op(("select", scene.selected_id))
        return n

    # -- ops --
    def _entity(self, solid_id: int) -> Any:
        ent = self.entities.get(solid_id)
        if ent is None:
            raise KeyError(f"el sólido #{solid_id} no existe en AutoCAD (¿se creó antes de conectar?)")
        return ent

    def _add(self, solid: Solid) -> None:
        sx, sy, sz = solid.dimensions
        cx, cy, cz = solid.center
        center = _point(cx, cy, cz)
        kind = solid.kind
        if kind == "box":
            ent = self.space.AddBox(center, sx, sy, sz)
        elif kind == "cylinder":
            ent = self.space.AddCylinder(center, sx / 2.0, sz)
        elif kind == "sphere":
            ent = self.space.AddSphere(center, sx / 2.0)
        elif kind == "cone":
            ent = self.space.AddCone(center, sx / 2.0, sz)
        elif kind == "wedge":
            ent = self.space.AddWedge(center, sx, sy, sz)
        else:
            raise ValueError(f"tipo de sólido desconocido {kind!r}")
        # Registrar ANTES de rotar: si Rotate3D falla (dibujo con comando abierto),
        # la entidad ya existe en AutoCAD y no debe quedar huérfana sin id.
        self.entities[solid.id] = ent
        self.solids[solid.id] = replace(solid)
        if solid.rotation_z != 0.0:
            try:
                ent.Rotate3D(center, _point(cx, cy, cz + 1.0), float(solid.rotation_z))
            except Exception as exc:  # COM: pywintypes.com_error
                self.solids[solid.id].rotation_z = 0.0
                if self.log is not None:
                    self.log(f"[autocad] sólido #{solid.id} creado pero no rotado: {exc}")
        self.regen()   # que el sólido nuevo se vea de inmediato

    def _move(self, solid_id: int, dx: float, dy: float, dz: float) -> None:
        ent = self._entity(solid_id)
        ent.Move(_point(0.0, 0.0, 0.0), _point(dx, dy, dz))
        s = self.solids[solid_id]
        s.center = (s.center[0] + dx, s.center[1] + dy, s.center[2] + dz)

    def _rotate(self, solid_id: int, dangle: float) -> None:
        """Rotación alrededor del eje vertical (Z) que pasa por el centro del sólido."""
        ent = self._entity(solid_id)
        s = self.solids[solid_id]
        cx, cy, cz = s.center
        ent.Rotate3D(_point(cx, cy, cz), _point(cx, cy, cz + 1.0), dangle)
        s.rotation_z += dangle

    def _scale(self, solid_id: int, factor: float) -> None:
        ent = self._entity(solid_id)
        s = self.solids[solid_id]
        ent.ScaleEntity(_point(*s.center), factor)
        s.scale *= factor

    def _select(self, solid_id: int) -> None:
        self._entity(solid_id)   # KeyError si no existe: cuenta como fallo, no como aplicado
        self.selected_id = solid_id

    def regen(self) -> None:
        self._since_regen = 0
        try:
            self.doc.Regen(AC_ACTIVE_VIEWPORT)
        except Exception as exc:  # pragma: no cover - un Regen fallido no es fatal
            if self.log is not None:
                self.log(f"[autocad] Regen falló: {exc}")


class DryRunBackend:
    """No toca AutoCAD: sirve para verificar en Windows que los frames UDP llegan
    y que los gestos producen ops (el bucle ya las imprime)."""

    def __init__(self) -> None:
        self.ops: List[Op] = []

    def apply(self, ops: Iterable[Op]) -> int:
        ops = list(ops)
        self.ops.extend(ops)
        return len(ops)


# ── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m apps.autocad_control.autocad_client",
        description="Cliente Windows: recibe el bus de Tletl por UDP y modela en AutoCAD por COM (pywin32).",
        epilog="Sin --udp ni --bus escucha en 0.0.0.0:5055. Usa --dry-run para probar la recepción sin AutoCAD.")
    add_source_arguments(ap)
    add_modeler_arguments(ap)
    ap.add_argument("--dry-run", action="store_true", help="no toca AutoCAD: solo imprime ops")
    ap.add_argument("--regen-every", type=int, default=25,
                    help="Regen del viewport cada N ops (default %(default)s)")
    return ap


def main(argv: Optional[Iterable[str]] = None, *, backend_factory: Optional[Callable[..., Any]] = None,
         clock: Callable[[], float] = time.time, sleep: Callable[[float], Any] = time.sleep,
         printer: Callable[[str], Any] = print) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.udp is None and args.bus is None:
        args.udp = DEFAULT_UDP
    scene = load_scene_json(args.load) if args.load else None
    modeler = modeler_from_args(args, scene=scene)

    if args.dry_run:
        backend: Any = DryRunBackend()
    else:
        factory = backend_factory or (lambda: AutoCADBackend(regen_every=args.regen_every, log=printer))
        try:
            backend = factory()
        except Exception as exc:  # ImportError (pywin32), RuntimeError (sin dibujo), com_error (sin AutoCAD)
            printer(f"[autocad] no pude conectar con AutoCAD: {type(exc).__name__}: {exc}")
            return 2
        if scene is not None:
            printer(f"[autocad] dibujando escena guardada: {backend.load_scene(scene)} sólidos")

    source = open_source(bus=args.bus, udp=args.udp)
    printer(f"[autocad] fuente: {source.describe()}  unidades: {args.units}  gain: {modeler.gain:g}"
            f"  backend: {'dry-run' if args.dry_run else 'AutoCAD COM'}")
    printer("[autocad] THREE 0.8 s cambia de modo; en CREATE, PINCH crea y VICTORY cambia el tipo. Ctrl+C para salir.")
    try:
        result = run_loop(source, modeler, backend.apply, hz=args.hz, clock=clock, sleep=sleep,
                          printer=printer, max_ticks=args.max_ticks)
    finally:
        source.close()
    printer(f"[autocad] fin: {result.frames} frames, {result.ops} ops, {len(modeler.scene)} sólidos"
            + (f", {backend.failed} errores COM" if hasattr(backend, "failed") and backend.failed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
