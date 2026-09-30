"""apps/autocad_control/gesture_modeler.py — de gestos a operaciones de modelado.

Máquina de estados PURA (sin bpy, sin COM, sin ezdxf, sin FreeCAD) que recibe
un frame del bus por tick (`update(state, now) -> list[Op]`) y produce
operaciones sobre una `Scene` de sólidos. Los backends (AutoCAD por COM,
export DXF, macro de FreeCAD) solo APLICAN esas operaciones: toda la decisión
está aquí y se prueba sin ningún CAD instalado.

Semántica (espejo del BlenderGestureMapper de apps/blender_control, validado
físicamente en docs/VALIDACION_FISICA_v5.md §4):

  Modo TRANSFORM (inicial)
    dom PINCH                 -> mover el sólido seleccionado por DELTA de la palma
                                 (x pantalla -> +X; y pantalla invertido -> +Y)
    mod PINCH                 -> rotar en Z por delta horizontal de la palma mod
                                 (opcional: delta vertical -> mover en Z, z_gain)
    dom PINCH + mod OPEN_PALM -> escalar por cambio de distancia entre palmas
    dom OPEN_PALM             -> soltar (rompe la continuidad, no mueve)
    dom FIST                  -> parada de seguridad (resetea continuidad, sin op)
    NEUTRAL / NO_HAND         -> nada
  Modo CREATE
    dom PINCH (flanco)        -> crear un sólido del tipo actual en la palma
                                 proyectada al plano XY (z=0); queda seleccionado
    dom VICTORY (flanco)      -> ciclar tipo: box -> cylinder -> sphere -> cone -> wedge
  Ambos modos
    dom THREE sostenido >= mode_hold (0.8 s) -> alternar modo (un disparo por hold)
    dom FIST  sostenido >= end_hold  (3 s)   -> op ("end_session",) (opcional)
    frame obsoleto (el timestamp deja de avanzar > stale_after) -> reset de continuidad, sin ops

Los deltas se calculan entre frames consecutivos: un frame aislado NUNCA mueve
nada (es "agarrar y arrastrar", no joystick). El reloj es inyectable.

Ops (tuplas):
    ("mode", "CREATE"|"TRANSFORM")   ("kind", "cylinder")
    ("add", Solid)                   ("select", id)
    ("move", id, dx, dy, dz)         ("rotate", id, dangle_rad)
    ("scale", id, factor)            ("end_session",)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

Vec3 = Tuple[float, float, float]
Op = Tuple[Any, ...]

SOLID_KINDS: Tuple[str, ...] = ("box", "cylinder", "sphere", "cone", "wedge")
MODES: Tuple[str, ...] = ("TRANSFORM", "CREATE")
OP_NAMES: Tuple[str, ...] = ("mode", "kind", "add", "select", "move", "rotate", "scale", "end_session")

DEFAULT_GAIN = 1.0          # unidades de dibujo que recorre el sólido al cruzar todo el encuadre
DEFAULT_ROT_GAIN = 2.0      # vueltas relativas por unidad de desplazamiento horizontal (como Blender)
DEFAULT_SCALE_GAIN = 1.5    # sensibilidad del escalado por distancia entre manos (como Blender)
DEFAULT_Z_GAIN = 0.0        # 0 = el delta vertical de la mano mod NO mueve en Z
DEFAULT_MODE_HOLD = 0.8     # segundos de THREE para alternar modo
DEFAULT_END_HOLD = 3.0      # segundos de FIST para terminar la sesión
DEFAULT_STALE_AFTER = 1.0   # segundos: frame más viejo que esto => bus obsoleto
MIN_SCALE = 0.05
SCENE_FORMAT = "tletl-cad-scene"
SCENE_VERSION = 1


# ── Lectura tolerante del frame del bus ─────────────────────────────────────

def _hand(state: Dict[str, Any], side: str) -> Dict[str, Any]:
    """Sub-dict de la mano ('dom'/'mod') o {}. Acepta el formato legacy {"hands": {...}}."""
    if not isinstance(state, dict):
        return {}
    val = state.get(side)
    if isinstance(val, dict):
        return val
    hands = state.get("hands")
    if isinstance(hands, dict) and isinstance(hands.get(side), dict):
        return hands[side]
    return {}


def hand_gesture(state: Dict[str, Any], side: str) -> str:
    """Gesto de la mano (`gesture`, o `stable_gesture` si falta), 'NO_HAND' si no hay."""
    hand = _hand(state, side)
    g = hand.get("gesture") or hand.get("stable_gesture") or "NO_HAND"
    return str(g)


def hand_palm(state: Dict[str, Any], side: str) -> Optional[Tuple[float, float]]:
    """Palma normalizada (x, y) en 0..1 (origen arriba-izquierda) o None."""
    palm = _hand(state, side).get("palm")
    if isinstance(palm, (list, tuple)) and len(palm) >= 2:
        try:
            return float(palm[0]), float(palm[1])
        except (TypeError, ValueError):
            return None
    if isinstance(palm, dict) and palm.get("x") is not None and palm.get("y") is not None:
        return float(palm["x"]), float(palm["y"])
    return None


def frame_timestamp(state: Dict[str, Any]) -> Optional[float]:
    ts = state.get("timestamp") if isinstance(state, dict) else None
    if isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts > 0:
        return float(ts)
    return None


# ── Modelo de escena ────────────────────────────────────────────────────────

def _vec3(value: Any, default: Vec3) -> Vec3:
    if value is None:
        return default
    seq = list(value)
    if len(seq) != 3:
        raise ValueError(f"se esperaba un vector de 3 componentes, recibí {value!r}")
    return (float(seq[0]), float(seq[1]), float(seq[2]))


@dataclass
class Solid:
    """Un sólido primitivo. `center` es el centro de su caja envolvente (igual que
    AddBox/AddCylinder/... de AutoCAD). `size` es el tamaño base; `scale` lo
    multiplica; `rotation_z` en radianes alrededor del eje vertical que pasa por
    el centro."""
    id: int
    kind: str
    center: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3 = (1.0, 1.0, 1.0)
    rotation_z: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in SOLID_KINDS:
            raise ValueError(f"tipo de sólido desconocido {self.kind!r}; válidos: {SOLID_KINDS}")
        self.id = int(self.id)
        self.center = _vec3(self.center, (0.0, 0.0, 0.0))
        self.size = _vec3(self.size, (1.0, 1.0, 1.0))
        self.rotation_z = float(self.rotation_z)
        self.scale = float(self.scale)

    @property
    def dimensions(self) -> Vec3:
        """Tamaño efectivo (size × scale) — lo que un backend debe dibujar."""
        s = self.scale
        return (self.size[0] * s, self.size[1] * s, self.size[2] * s)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind,
            "center": list(self.center), "size": list(self.size),
            "rotation_z": self.rotation_z, "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Solid":
        return cls(
            id=int(data["id"]), kind=str(data["kind"]),
            center=_vec3(data.get("center"), (0.0, 0.0, 0.0)),
            size=_vec3(data.get("size"), (1.0, 1.0, 1.0)),
            rotation_z=float(data.get("rotation_z", 0.0)),
            scale=float(data.get("scale", 1.0)),
        )


@dataclass
class Scene:
    """Lista de sólidos + selección. `apply(op)` es el ÚNICO reductor de ops:
    lo usan el modeler (para su propia escena) y los backends que quieran
    llevar una copia fiel."""
    solids: List[Solid] = field(default_factory=list)
    selected_id: Optional[int] = None
    next_id: int = 1
    units: str = "mm"

    # -- consultas --
    def get(self, solid_id: Optional[int]) -> Optional[Solid]:
        if solid_id is None:
            return None
        for s in self.solids:
            if s.id == solid_id:
                return s
        return None

    @property
    def selected(self) -> Optional[Solid]:
        return self.get(self.selected_id)

    def __len__(self) -> int:
        return len(self.solids)

    # -- mutación --
    def add(self, kind: str, center: Vec3 = (0.0, 0.0, 0.0), size: Vec3 = (1.0, 1.0, 1.0),
            *, rotation_z: float = 0.0, scale: float = 1.0, select: bool = True) -> Solid:
        """Crea un sólido con el siguiente id (y lo selecciona). Devuelve el objeto guardado."""
        solid = Solid(id=self.next_id, kind=kind, center=center, size=size,
                      rotation_z=rotation_z, scale=scale)
        self.apply(("add", solid))
        if select:
            self.apply(("select", solid.id))
        return self.get(solid.id)  # type: ignore[return-value]

    def apply(self, op: Op) -> bool:
        """Aplica una op. Devuelve False si no cambió nada (id inexistente, op no
        de escena como "mode"/"kind"/"end_session", ...)."""
        name = op[0]
        if name == "add":
            solid: Solid = op[1]
            if self.get(solid.id) is not None:
                return False
            self.solids.append(replace(solid))
            self.next_id = max(self.next_id, solid.id + 1)
            return True
        if name == "select":
            target = self.get(op[1])
            self.selected_id = target.id if target is not None else None
            return target is not None
        solid = self.get(op[1]) if len(op) > 1 else None
        if solid is None:
            return False
        if name == "move":
            _, _, dx, dy, dz = op
            cx, cy, cz = solid.center
            solid.center = (cx + float(dx), cy + float(dy), cz + float(dz))
            return True
        if name == "rotate":
            solid.rotation_z += float(op[2])
            return True
        if name == "scale":
            solid.scale = max(MIN_SCALE, solid.scale * float(op[2]))
            return True
        return False

    # -- serialización --
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": SCENE_FORMAT, "version": SCENE_VERSION, "units": self.units,
            "next_id": self.next_id, "selected_id": self.selected_id,
            "solids": [s.to_dict() for s in self.solids],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scene":
        if not isinstance(data, dict) or not isinstance(data.get("solids"), list):
            raise ValueError("escena inválida: se esperaba un dict con la lista 'solids'")
        solids = [Solid.from_dict(d) for d in data["solids"]]
        max_id = max((s.id for s in solids), default=0)
        scene = cls(solids=solids, selected_id=data.get("selected_id"),
                    next_id=max(int(data.get("next_id", 1)), max_id + 1),
                    units=str(data.get("units", "mm")))
        if scene.get(scene.selected_id) is None:
            scene.selected_id = solids[-1].id if solids else None
        return scene


# ── Modeler ─────────────────────────────────────────────────────────────────

class GestureModeler:
    """Convierte frames del bus en ops de modelado y las aplica a `self.scene`.

    `update()` devuelve las ops del tick (ya aplicadas a la escena propia) para
    que el llamador las imprima y las pase a su backend.
    """

    def __init__(self, *, gain: float = DEFAULT_GAIN, rot_gain: float = DEFAULT_ROT_GAIN,
                 scale_gain: float = DEFAULT_SCALE_GAIN, z_gain: float = DEFAULT_Z_GAIN,
                 spawn_size: Optional[float] = None, mode_hold: float = DEFAULT_MODE_HOLD,
                 end_hold: Optional[float] = DEFAULT_END_HOLD,
                 stale_after: Optional[float] = DEFAULT_STALE_AFTER, deadzone: float = 0.0,
                 scene: Optional[Scene] = None, clock: Callable[[], float] = time.time):
        self.gain = float(gain)
        self.rot_gain = float(rot_gain)
        self.scale_gain = float(scale_gain)
        self.z_gain = float(z_gain)
        self.spawn_size = float(spawn_size) if spawn_size is not None else abs(self.gain) * 0.1
        self.mode_hold = float(mode_hold)
        self.end_hold = float(end_hold) if end_hold is not None else None
        self.stale_after = float(stale_after) if stale_after is not None else None
        self.deadzone = float(deadzone)
        self.scene = scene if scene is not None else Scene()
        self.clock = clock

        self.mode: str = "TRANSFORM"
        self.kind_index: int = 0
        self.ticks = 0
        self.stale_frames = 0
        self.ops_emitted = 0
        self.status = "esperando bus"
        self.last_dom = "NO_HAND"
        self.last_mod = "NO_HAND"

        self._prev_dom: Optional[Tuple[float, float]] = None
        self._prev_mod: Optional[Tuple[float, float]] = None
        self._prev_dist: Optional[float] = None
        self._prev_dom_gesture = "NO_HAND"
        self._three_since: Optional[float] = None
        self._three_fired = False
        self._fist_since: Optional[float] = None
        self._fist_fired = False

        # Seguimiento del flujo de frames para juzgar obsolescencia SIN comparar
        # el reloj del emisor con el del receptor (ver _is_stale).
        self._last_ts: Optional[float] = None
        self._last_advance_at: Optional[float] = None

    # -- propiedades --
    @property
    def kind(self) -> str:
        return SOLID_KINDS[self.kind_index]

    def set_kind(self, kind: str) -> None:
        self.kind_index = SOLID_KINDS.index(kind)

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"modo desconocido {mode!r}")
        self.mode = mode
        self._release_all()

    # -- continuidad --
    def _release_all(self) -> None:
        self._prev_dom = None
        self._prev_mod = None
        self._prev_dist = None

    def _reset_holds(self) -> None:
        self._three_since = None
        self._three_fired = False
        self._fist_since = None
        self._fist_fired = False

    def reset(self) -> None:
        """Rompe continuidad, holds y flancos. NO toca modo, tipo ni escena."""
        self._release_all()
        self._reset_holds()
        self._prev_dom_gesture = "NO_HAND"

    # -- mapeo de coordenadas --
    def palm_to_world(self, palm: Tuple[float, float]) -> Vec3:
        """Palma normalizada (0..1, y hacia abajo) -> punto del plano XY centrado en el origen.
        El encuadre completo mide `gain` unidades de ancho y de alto."""
        return ((palm[0] - 0.5) * self.gain, (0.5 - palm[1]) * self.gain, 0.0)

    # -- tick --
    def update(self, state: Dict[str, Any], now: Optional[float] = None) -> List[Op]:
        now = self.clock() if now is None else float(now)
        self.ticks += 1
        if not isinstance(state, dict):
            state = {}

        if self._is_stale(state, now):
            # Continuidad y holds fuera; el flanco de PINCH se conserva a propósito:
            # si el bus vuelve con la pinza aún cerrada NO debe crear un sólido.
            self.stale_frames += 1
            self._release_all()
            self._reset_holds()
            self.status = "bus obsoleto (frame viejo)"
            self.last_dom = self.last_mod = "NO_HAND"
            return []

        d = hand_gesture(state, "dom")
        m = hand_gesture(state, "mod")
        dpalm = hand_palm(state, "dom")
        mpalm = hand_palm(state, "mod")
        self.last_dom, self.last_mod = d, m

        rising_pinch = d == "PINCH" and self._prev_dom_gesture != "PINCH"
        rising_victory = d == "VICTORY" and self._prev_dom_gesture != "VICTORY"
        self._prev_dom_gesture = d

        ops: List[Op] = []
        ops.extend(self._update_holds(d, now))

        if d == "FIST":
            self._release_all()
            self.status = "FIST: parada de seguridad"
        elif self.mode == "CREATE":
            ops.extend(self._create_mode(d, dpalm, rising_pinch, rising_victory))
        else:
            ops.extend(self._transform_mode(d, m, dpalm, mpalm))

        for op in ops:
            self.scene.apply(op)
        self.ops_emitted += len(ops)
        return ops

    # -- reglas --
    # Un timestamp que retrocede más que esto se trata como un flujo nuevo (la app
    # se reinició o el reloj del emisor se ajustó), no como un frame viejo.
    STREAM_RESTART_TOLERANCE = 5.0

    def _is_stale(self, state: Dict[str, Any], now: float) -> bool:
        """Un frame es obsoleto cuando el flujo NO avanza: mismo timestamp que el
        anterior y ya pasó `stale_after` desde el último avance.

        Solo el PRIMER frame se juzga contra el reloj local (`now - ts`): protege
        contra un bus viejo que quedó en disco. Después se mira únicamente si el
        timestamp avanza, así un cliente UDP en otra máquina con el reloj
        desfasado (el caso real de AutoCAD en Windows) no ve todos los frames
        como obsoletos: como mucho pierde el primero.
        """
        if self.stale_after is None:
            return False
        ts = frame_timestamp(state)
        if ts is None:
            return False

        if self._last_ts is None:
            self._last_ts = ts
            self._last_advance_at = now
            return (now - ts) > self.stale_after

        advanced = ts > self._last_ts or ts < self._last_ts - self.STREAM_RESTART_TOLERANCE
        if advanced:
            self._last_ts = ts
            self._last_advance_at = now
            return False
        since = now - (self._last_advance_at if self._last_advance_at is not None else now)
        return since > self.stale_after

    def _update_holds(self, d: str, now: float) -> List[Op]:
        ops: List[Op] = []
        # THREE sostenido -> alternar modo (un disparo por hold)
        if d == "THREE":
            if self._three_since is None:
                self._three_since = now
                self._three_fired = False
            elif not self._three_fired and now - self._three_since >= self.mode_hold:
                self._three_fired = True
                self.mode = "CREATE" if self.mode == "TRANSFORM" else "TRANSFORM"
                self._release_all()
                ops.append(("mode", self.mode))
                self.status = f"modo {self.mode}"
        else:
            self._three_since = None
            self._three_fired = False
        # FIST sostenido -> fin de sesión (opcional)
        if d == "FIST" and self.end_hold is not None:
            if self._fist_since is None:
                self._fist_since = now
                self._fist_fired = False
            elif not self._fist_fired and now - self._fist_since >= self.end_hold:
                self._fist_fired = True
                ops.append(("end_session",))
                self.status = "FIST sostenido: fin de sesión"
        else:
            self._fist_since = None
            self._fist_fired = False
        return ops

    def _create_mode(self, d: str, dpalm: Optional[Tuple[float, float]],
                     rising_pinch: bool, rising_victory: bool) -> List[Op]:
        ops: List[Op] = []
        self._release_all()
        if rising_victory:
            self.kind_index = (self.kind_index + 1) % len(SOLID_KINDS)
            ops.append(("kind", self.kind))
            self.status = f"tipo {self.kind}"
        if rising_pinch and dpalm is not None:
            s = self.spawn_size
            solid = Solid(id=self.scene.next_id, kind=self.kind,
                          center=self.palm_to_world(dpalm), size=(s, s, s))
            ops.append(("add", solid))
            ops.append(("select", solid.id))
            self.status = f"creado {solid.kind} #{solid.id}"
        elif not ops:
            self.status = f"CREATE: PINCH crea {self.kind}, VICTORY cambia tipo"
        return ops

    def _transform_mode(self, d: str, m: str, dpalm: Optional[Tuple[float, float]],
                        mpalm: Optional[Tuple[float, float]]) -> List[Op]:
        ops: List[Op] = []
        sel = self.scene.selected
        sid = sel.id if sel is not None else None

        if d == "OPEN_PALM":
            self._prev_dom = None
            self.status = "OPEN_PALM: suelto"

        # Traslación XY por delta de la palma dominante (PINCH = agarrar)
        if d == "PINCH" and dpalm is not None:
            if self._prev_dom is not None:
                ddx = dpalm[0] - self._prev_dom[0]
                ddy = dpalm[1] - self._prev_dom[1]
                if math.hypot(ddx, ddy) >= self.deadzone:
                    if sid is not None and (ddx != 0.0 or ddy != 0.0):
                        ops.append(("move", sid, ddx * self.gain, -ddy * self.gain, 0.0))
                    self._prev_dom = dpalm
            else:
                self._prev_dom = dpalm
            self.status = f"PINCH: moviendo #{sid}" if sid is not None else "PINCH sin sólido seleccionado (THREE 0.8 s -> CREATE)"
        else:
            self._prev_dom = None

        # Rotación Z (y Z opcional) por delta de la palma modificadora
        if m == "PINCH" and mpalm is not None:
            if self._prev_mod is not None and sid is not None:
                dang = (mpalm[0] - self._prev_mod[0]) * self.rot_gain * math.pi
                if dang != 0.0:
                    ops.append(("rotate", sid, dang))
                if self.z_gain != 0.0:
                    dz = (self._prev_mod[1] - mpalm[1]) * self.z_gain
                    if dz != 0.0:
                        ops.append(("move", sid, 0.0, 0.0, dz))
            self._prev_mod = mpalm
        else:
            self._prev_mod = None

        # Escala por distancia entre manos (dom PINCH + mod OPEN_PALM)
        if dpalm is not None and mpalm is not None and d == "PINCH" and m == "OPEN_PALM":
            dist = math.hypot(dpalm[0] - mpalm[0], dpalm[1] - mpalm[1])
            if self._prev_dist is not None and self._prev_dist > 1e-6 and sel is not None:
                delta = (dist - self._prev_dist) * self.scale_gain
                new_scale = max(MIN_SCALE, sel.scale + delta)
                factor = new_scale / sel.scale
                if abs(factor - 1.0) > 1e-12:
                    ops.append(("scale", sid, factor))
            self._prev_dist = dist
        else:
            self._prev_dist = None

        return ops

    # -- texto --
    def describe(self) -> str:
        sel = self.scene.selected_id
        return (f"modo={self.mode} tipo={self.kind} sel={'#' + str(sel) if sel is not None else '-'} "
                f"solidos={len(self.scene)} dom={self.last_dom} mod={self.last_mod}")


# ── Formato legible de ops (lo usan los CLIs) ───────────────────────────────

def format_op(op: Op) -> str:
    name = op[0]
    if name == "add":
        s: Solid = op[1]
        cx, cy, cz = s.center
        return f"add #{s.id} {s.kind} center=({cx:.3f}, {cy:.3f}, {cz:.3f}) size={s.size[0]:.3f}"
    if name == "move":
        return f"move #{op[1]} dx={op[2]:+.3f} dy={op[3]:+.3f} dz={op[4]:+.3f}"
    if name == "rotate":
        return f"rotate #{op[1]} dz={math.degrees(op[2]):+.2f}°"
    if name == "scale":
        return f"scale #{op[1]} x{op[2]:.4f}"
    if name == "select":
        return f"select #{op[1]}"
    if name in ("mode", "kind"):
        return f"{name} {op[1]}"
    if name == "end_session":
        return "end_session"
    return " ".join(str(part) for part in op)
