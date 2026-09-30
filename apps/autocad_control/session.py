"""apps/autocad_control/session.py — sesión de modelado por gestos en Fedora.

Ruta 2 (offline): lee el bus (archivo o UDP), alimenta el `GestureModeler`,
imprime cada op y una línea de estado, y al terminar (Ctrl+C, FIST sostenido
3 s, o fuente agotada) guarda la escena como JSON junto al DXF y exporta el
DXF (ezdxf). Si ezdxf no está, el JSON queda igual y se re-exporta después con
`python -m apps.autocad_control.dxf_export escena.json`.

`run_loop()` es el bucle genérico poll -> update -> sink que también usa el
cliente de AutoCAD (autocad_client.py); reloj y sleep son inyectables para
probarlo sin dormir.

    python -m apps.autocad_control.session --export modelo.dxf            # bus por archivo
    python -m apps.autocad_control.session --udp 0.0.0.0:5055 --export m.dxf
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from .bus_source import open_source
from .dxf_export import UNIT_CODES, export_scene_dxf, load_scene_json, save_scene_json
from .gesture_modeler import (
    DEFAULT_END_HOLD,
    DEFAULT_MODE_HOLD,
    DEFAULT_ROT_GAIN,
    DEFAULT_SCALE_GAIN,
    DEFAULT_STALE_AFTER,
    DEFAULT_Z_GAIN,
    GestureModeler,
    Op,
    Scene,
    format_op,
)

DEFAULT_EXPORT = "tletl_modelo.dxf"
DEFAULT_HZ = 20.0
DEFAULT_UDP = "0.0.0.0:5055"
# Ganancia por default según unidades: el encuadre completo de la cámara mide
# ~10 cm en el dibujo (un sólido nuevo mide gain/10).
DEFAULT_GAIN_BY_UNITS = {"mm": 100.0, "cm": 10.0, "m": 1.0, "in": 4.0}

GESTURE_TABLE = """\
gestos (mano dominante = dom, otra mano = mod):
  THREE 0.8 s            alterna modo TRANSFORM <-> CREATE (un cambio por hold)
  FIST                   parada de seguridad; sostenido 3 s termina la sesión
  TRANSFORM  dom PINCH   mueve el sólido seleccionado (arrastre por delta)
             mod PINCH   rota en Z (delta horizontal); --z-gain: delta vertical = altura
             dom PINCH + mod OPEN_PALM   escala (separar/juntar las manos)
             dom OPEN_PALM               suelta (sin salto al volver a pinzar)
  CREATE     dom PINCH   crea un sólido del tipo actual donde está la palma
             dom VICTORY cambia el tipo: box > cylinder > sphere > cone > wedge
"""


@dataclass
class LoopResult:
    ticks: int = 0
    frames: int = 0
    ops: int = 0
    ended: bool = False        # terminó por op end_session (FIST sostenido)
    interrupted: bool = False  # Ctrl+C


def status_line(modeler: GestureModeler, source: Any, now: float) -> str:
    age = source.age(now)
    age_txt = f"{age:.2f}s" if age != float("inf") else "sin frames"
    return f"[cad] {modeler.describe()} bus={age_txt} ops={modeler.ops_emitted} | {modeler.status}"


def run_loop(source: Any, modeler: GestureModeler, sink: Optional[Callable[[List[Op]], Any]] = None, *,
             hz: float = DEFAULT_HZ, clock: Callable[[], float] = time.time,
             sleep: Callable[[float], Any] = time.sleep, printer: Callable[[str], Any] = print,
             status_every: float = 1.0, max_ticks: Optional[int] = None,
             stop_on_end: bool = True) -> LoopResult:
    """poll -> modeler.update -> imprime ops -> sink(ops). Termina con Ctrl+C,
    con la op end_session, cuando la fuente marca `finished`, o a `max_ticks`."""
    period = 1.0 / max(float(hz), 0.1)
    result = LoopResult()
    last_status = clock()
    try:
        while True:
            now = clock()
            state = source.poll()
            if state is not None:
                result.frames += 1
                ops = modeler.update(state, now)
                for op in ops:
                    printer(f"[op] {format_op(op)}")
                if ops and sink is not None:
                    sink(ops)
                result.ops += len(ops)
                if stop_on_end and any(op[0] == "end_session" for op in ops):
                    result.ended = True
                    break
            elif getattr(source, "finished", False):
                break
            result.ticks += 1
            if now - last_status >= status_every:
                printer(status_line(modeler, source, now))
                last_status = now
            if max_ticks is not None and result.ticks >= max_ticks:
                break
            elapsed = clock() - now
            sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        result.interrupted = True
        printer("\n[cad] Interrumpido (Ctrl+C).")
    return result


class CadSession:
    """Bucle + cierre: guarda `escena.json` junto al DXF y exporta el DXF."""

    def __init__(self, source: Any, modeler: GestureModeler, export_path: str | Path, *,
                 units: str = "mm", hz: float = DEFAULT_HZ, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], Any] = time.sleep, printer: Callable[[str], Any] = print,
                 status_every: float = 1.0, max_ticks: Optional[int] = None):
        self.source = source
        self.modeler = modeler
        self.export_path = Path(export_path).expanduser()
        self.json_path = self.export_path.with_suffix(".json")
        self.units = units
        self.hz = hz
        self.clock = clock
        self.sleep = sleep
        self.printer = printer
        self.status_every = status_every
        self.max_ticks = max_ticks
        self.result: Optional[LoopResult] = None
        self.dxf_written: Optional[Path] = None

    def run(self) -> int:
        p = self.printer
        p(f"[cad] fuente: {self.source.describe()}  unidades: {self.units}  gain: {self.modeler.gain:g}")
        p(f"[cad] export: {self.export_path}  (escena: {self.json_path})")
        p(f"[cad] modo inicial {self.modeler.mode} con {len(self.modeler.scene)} sólidos. Ctrl+C o FIST 3 s para terminar.")
        self.result = run_loop(self.source, self.modeler, None, hz=self.hz, clock=self.clock,
                               sleep=self.sleep, printer=p, status_every=self.status_every,
                               max_ticks=self.max_ticks)
        return self.finish()

    def finish(self) -> int:
        scene = self.modeler.scene
        scene.units = self.units
        p = self.printer
        save_scene_json(scene, self.json_path)
        p(f"[cad] escena guardada: {self.json_path} ({len(scene)} sólidos)")
        if len(scene) == 0:
            p("[cad] 0 sólidos: en TRANSFORM no se crea nada; THREE 0.8 s -> CREATE y PINCH crea.")
        try:
            self.dxf_written = export_scene_dxf(scene, self.export_path, units=self.units)
        except ImportError as exc:
            p(f"[cad] NO se exportó el DXF: {exc}")
            p(f"      re-exporta luego con: python -m apps.autocad_control.dxf_export {self.json_path}")
            return 2
        p(f"[cad] DXF exportado: {self.dxf_written}  (ábrelo en AutoCAD: OPEN, o arrástralo a la ventana)")
        return 0


# ── argparse compartido con autocad_client ──────────────────────────────────

def add_source_arguments(ap: argparse.ArgumentParser) -> None:
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--bus", metavar="PATH", default=None,
                     help="archivo del bus (default: ~/.tletl/tletl_state.json o $TLETL_STATE_PATH)")
    src.add_argument("--udp", metavar="HOST:PORT", nargs="?", const=DEFAULT_UDP, default=None,
                     help=f"escuchar el bus por UDP (sin valor: {DEFAULT_UDP})")


def add_modeler_arguments(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("modelado")
    g.add_argument("--units", choices=sorted(UNIT_CODES), default="mm",
                   help="unidades del dibujo/DXF (default %(default)s)")
    g.add_argument("--gain", type=float, default=None,
                   help="unidades que recorre el sólido al cruzar todo el encuadre "
                        "(default por unidades: mm=100, cm=10, m=1, in=4)")
    g.add_argument("--spawn-size", type=float, default=None, help="tamaño del sólido nuevo (default gain/10)")
    g.add_argument("--rot-gain", type=float, default=DEFAULT_ROT_GAIN,
                   help="vueltas por unidad de desplazamiento horizontal de la mano mod (default %(default)s)")
    g.add_argument("--scale-gain", type=float, default=DEFAULT_SCALE_GAIN,
                   help="sensibilidad de la escala por distancia entre manos (default %(default)s)")
    g.add_argument("--z-gain", type=float, default=DEFAULT_Z_GAIN,
                   help="unidades en Z por delta vertical de la mano mod con PINCH (default 0 = off)")
    g.add_argument("--mode-hold", type=float, default=DEFAULT_MODE_HOLD,
                   help="segundos de THREE para cambiar de modo (default %(default)s)")
    g.add_argument("--end-hold", type=float, default=DEFAULT_END_HOLD,
                   help="segundos de FIST para terminar (0 = desactivado; default %(default)s)")
    g.add_argument("--stale", type=float, default=DEFAULT_STALE_AFTER,
                   help="segundos tras los que un frame se considera obsoleto (default %(default)s)")
    g.add_argument("--deadzone", type=float, default=0.0,
                   help="delta normalizado mínimo de la palma para mover (default 0)")
    g.add_argument("--hz", type=float, default=DEFAULT_HZ, help="frecuencia del bucle (default %(default)s)")
    g.add_argument("--load", metavar="escena.json", default=None, help="continuar una escena guardada")
    g.add_argument("--max-ticks", type=int, default=None, help=argparse.SUPPRESS)


def modeler_from_args(args: argparse.Namespace, *, scene: Optional[Scene] = None) -> GestureModeler:
    gain = args.gain if args.gain is not None else DEFAULT_GAIN_BY_UNITS[args.units]
    end_hold = args.end_hold if args.end_hold and args.end_hold > 0 else None
    stale = args.stale if args.stale and args.stale > 0 else None
    return GestureModeler(gain=gain, rot_gain=args.rot_gain, scale_gain=args.scale_gain,
                          z_gain=args.z_gain, spawn_size=args.spawn_size, mode_hold=args.mode_hold,
                          end_hold=end_hold, stale_after=stale, deadzone=args.deadzone, scene=scene)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m apps.autocad_control.session",
        description="Sesión de modelado por gestos (Fedora): lee el bus de Tletl, modela sólidos y exporta un DXF.",
        epilog=GESTURE_TABLE, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_arguments(ap)
    ap.add_argument("--export", metavar="modelo.dxf", default=DEFAULT_EXPORT,
                    help="DXF de salida (default %(default)s); la escena JSON va al lado")
    add_modeler_arguments(ap)
    return ap


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    scene = load_scene_json(args.load) if args.load else None
    modeler = modeler_from_args(args, scene=scene)
    source = open_source(bus=args.bus, udp=args.udp)
    session = CadSession(source, modeler, args.export, units=args.units, hz=args.hz,
                         max_ticks=args.max_ticks)
    try:
        return session.run()
    finally:
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
