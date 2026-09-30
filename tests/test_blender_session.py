"""tests/test_blender_session.py — máquina de estados del addon (GestureSession),
estado del bus (bus_status), settings y mapeo de palma a mundo. Todo sin bpy.
"""
from __future__ import annotations

import importlib
import math
import sys
import types
import unittest.mock as mock

import pytest


def _import_addon():
    for key in list(sys.modules):
        if key.endswith("tletl_blender_addon"):
            del sys.modules[key]
    with mock.patch.dict(sys.modules, {"bpy": None}):
        return importlib.import_module("apps.blender_control.tletl_blender_addon")


A = _import_addon()


class Clock:
    """Reloj inyectable: el test decide cuánto tiempo pasa entre ticks."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


def _state(dom="NEUTRAL", palm=(0.5, 0.5), mod="NO_HAND", mod_palm=None, ts=100.0, **extra):
    return {
        "timestamp": ts, "fps": 30.0, "mode": "NAVEGADOR",
        "dom": {"present": dom != "NO_HAND", "gesture": dom, "confidence": 0.9,
                "palm": list(palm) if (palm is not None and dom != "NO_HAND") else None},
        "mod": {"present": mod != "NO_HAND", "gesture": mod, "confidence": 0.8,
                "palm": list(mod_palm) if mod_palm else None},
        "intent": {"name": "NONE", "active": False},
        "extra": {"control": True, **extra},
    }


def _zero():
    return A.normalize_transform({})


def _session(**kw):
    clock = Clock()
    s = A.GestureSession(mapper=A.BlenderGestureMapper(gain=10.0, smoothing=1.0), clock=clock, **kw)
    return s, clock


# ── palm_to_world_xy ─────────────────────────────────────────────────────────

def test_palm_center_lands_on_cursor():
    assert A.palm_to_world_xy((0.5, 0.5), (1.0, 2.0, 3.0), gain=10.0) == (1.0, 2.0, 3.0)


def test_palm_mapping_axes_and_inversion():
    x, y, z = A.palm_to_world_xy((1.0, 0.0), (0.0, 0.0, 0.7), gain=4.0)
    assert x == pytest.approx(2.0)      # derecha de cámara -> +X
    assert y == pytest.approx(2.0)      # arriba de pantalla (y=0) -> +Y
    assert z == pytest.approx(0.7)      # z del cursor
    x2, y2, _ = A.palm_to_world_xy((0.0, 1.0), (0.0, 0.0, 0.0), gain=4.0)
    assert x2 == pytest.approx(-2.0) and y2 == pytest.approx(-2.0)


# ── cambio de modo con THREE sostenido ───────────────────────────────────────

def test_three_hold_toggles_mode_once_per_hold():
    s, clock = _session()
    assert s.mode == "TRANSFORM"
    assert s.tick(_state("NEUTRAL"), _zero()) == [("transform", _zero())]
    assert s.tick(_state("THREE"), _zero()) == [("transform", _zero())]   # empieza el sostén
    clock.advance(0.5)
    assert s.tick(_state("THREE"), _zero()) == [("transform", _zero())]   # aún no
    clock.advance(0.35)                                                    # 0.85 s
    assert s.tick(_state("THREE"), _zero()) == [("mode", "CREATE")]
    assert s.mode == "CREATE"
    clock.advance(5.0)                                                     # sigue sostenido
    assert s.tick(_state("THREE"), _zero()) == []                          # NO vuelve a alternar
    assert s.mode == "CREATE"
    # soltar y volver a sostener -> regresa a TRANSFORM
    s.tick(_state("NEUTRAL"), _zero())
    s.tick(_state("THREE"), _zero())
    clock.advance(0.8)
    assert s.tick(_state("THREE"), _zero()) == [("mode", "TRANSFORM")]
    assert s.mode == "TRANSFORM"


def test_three_released_before_hold_does_not_toggle():
    s, clock = _session()
    s.tick(_state("THREE"), _zero())
    clock.advance(0.5)
    s.tick(_state("NEUTRAL"), _zero())
    clock.advance(0.5)
    s.tick(_state("THREE"), _zero())      # nuevo sostén: cuenta desde cero
    clock.advance(0.5)
    assert s.tick(_state("THREE"), _zero()) == [("transform", _zero())]
    assert s.mode == "TRANSFORM"


def test_hold_seconds_is_configurable_and_uses_now_override():
    s, _ = _session(hold_seconds=0.2)
    s.tick(_state("THREE"), _zero(), now=10.0)
    assert s.tick(_state("THREE"), _zero(), now=10.19) == [("transform", _zero())]
    assert s.tick(_state("THREE"), _zero(), now=10.2) == [("mode", "CREATE")]


def test_mode_switch_from_panel_resets_mapper_continuity():
    s, _ = _session()
    s.tick(_state("PINCH", (0.5, 0.5)), _zero())
    s.set_mode("CREATE")
    assert s.mode == "CREATE"
    s.set_mode("TRANSFORM")
    # tras volver, el primer PINCH sólo ceba (no hay salto)
    out = s.tick(_state("PINCH", (0.9, 0.9)), _zero())
    assert out == [("transform", _zero())]
    with pytest.raises(ValueError):
        s.set_mode("FLY")
    assert s.toggle_mode() == "CREATE" and s.toggle_mode() == "TRANSFORM"


# ── modo CREATE ──────────────────────────────────────────────────────────────

def test_create_pinch_rising_edge_spawns_at_palm():
    s, _ = _session()
    s.set_mode("CREATE")
    cursor = (1.0, 2.0, 3.0)
    assert s.tick(_state("NEUTRAL"), None, cursor=cursor) == []          # ceba el flanco
    cmds = s.tick(_state("PINCH", (0.75, 0.25)), None, cursor=cursor)
    assert len(cmds) == 1
    kind, prim, pos = cmds[0]
    assert (kind, prim) == ("spawn", "cube")
    assert pos == pytest.approx((1.0 + 0.25 * 10.0, 2.0 + 0.25 * 10.0, 3.0))
    assert s.spawn_count == 1 and s.last_spawn[0] == "cube"
    # PINCH sostenido: NO crea otro
    assert s.tick(_state("PINCH", (0.7, 0.3)), None, cursor=cursor) == []
    assert s.tick(_state("PINCH", (0.6, 0.4)), None, cursor=cursor) == []
    # soltar y volver a pellizcar: crea otro
    s.tick(_state("OPEN_PALM"), None, cursor=cursor)
    cmds = s.tick(_state("PINCH", (0.5, 0.5)), None, cursor=cursor)
    assert cmds == [("spawn", "cube", (1.0, 2.0, 3.0))]
    assert s.spawn_count == 2


def test_create_spawn_uses_mapper_gain():
    s, _ = _session()
    s.set_mode("CREATE")
    s.mapper.configure(gain=4.0)
    s.tick(_state("NEUTRAL"), None)
    cmds = s.tick(_state("PINCH", (1.0, 0.5)), None)
    assert cmds[0][2] == pytest.approx((2.0, 0.0, 0.0))


def test_create_victory_cycles_primitives_on_rising_edge():
    s, _ = _session()
    s.set_mode("CREATE")
    s.tick(_state("NEUTRAL"), None)
    seen = [s.primitive]
    for _ in range(5):
        cmds = s.tick(_state("VICTORY"), None)
        assert cmds == [("primitive", s.primitive)]
        assert s.tick(_state("VICTORY"), None) == []      # sostenido: no cicla
        s.tick(_state("NEUTRAL"), None)
        seen.append(s.primitive)
    assert seen == ["cube", "sphere", "cylinder", "cone", "plane", "cube"]
    assert list(A.PRIMITIVES) == ["cube", "sphere", "cylinder", "cone", "plane"]


def test_create_open_palm_and_fist_do_nothing():
    s, _ = _session()
    s.set_mode("CREATE")
    s.tick(_state("NEUTRAL"), None)
    assert s.tick(_state("OPEN_PALM", (0.2, 0.2)), None) == []
    assert s.tick(_state("FIST", (0.2, 0.2)), None) == []
    assert s.tick(_state("FIST", (0.3, 0.3)), None) == []
    assert s.spawn_count == 0
    # sin palma no hay dónde crear
    st = _state("PINCH"); st["dom"]["palm"] = None
    assert s.tick(st, None) == []


def test_create_after_reset_requires_priming_tick():
    """Si el bus volvió con PINCH ya sostenido, NO se crea nada por sorpresa."""
    s, _ = _session()
    s.set_mode("CREATE")
    s.reset()
    assert s.tick(_state("PINCH", (0.5, 0.5)), None) == []
    assert s.tick(_state("PINCH", (0.5, 0.5)), None) == []
    s.tick(_state("NEUTRAL"), None)
    assert s.tick(_state("PINCH", (0.5, 0.5)), None)[0][0] == "spawn"


def test_set_primitive_by_name():
    s, _ = _session()
    assert s.set_primitive("cone") == "cone"
    assert s.cycle_primitive() == "plane"
    assert s.cycle_primitive() == "cube"
    with pytest.raises(ValueError):
        s.set_primitive("torus")


def test_primitive_specs_cover_all_primitives():
    assert set(A.PRIMITIVE_SPECS) == set(A.PRIMITIVES)
    for name, (op, kwargs) in A.PRIMITIVE_SPECS.items():
        assert op.startswith("create_") and isinstance(kwargs, dict)
    assert A.PRIMITIVE_SPECS["cylinder"][1]["radius1"] == A.PRIMITIVE_SPECS["cylinder"][1]["radius2"]
    assert A.PRIMITIVE_SPECS["cone"][1]["radius2"] == 0.0


# ── modo TRANSFORM ───────────────────────────────────────────────────────────

def test_transform_mode_moves_via_mapper():
    s, _ = _session()
    s.tick(_state("PINCH", (0.5, 0.5)), _zero())
    cmds = s.tick(_state("PINCH", (0.6, 0.5)), _zero())
    assert cmds[0][0] == "transform"
    assert cmds[0][1]["x"] == pytest.approx(1.0)        # 0.1 * gain 10


def test_transform_mode_without_object_emits_nothing_and_resets():
    s, _ = _session()
    s.tick(_state("PINCH", (0.5, 0.5)), _zero())
    assert s.tick(_state("PINCH", (0.6, 0.5)), None) == []
    # continuidad rota: el siguiente PINCH con objeto no salta
    assert s.tick(_state("PINCH", (0.9, 0.9)), _zero()) == [("transform", _zero())]


def test_transform_mode_ignores_victory_and_pinch_edges():
    s, _ = _session()
    s.tick(_state("NEUTRAL"), _zero())
    assert s.tick(_state("VICTORY"), _zero()) == [("transform", _zero())]
    assert s.primitive == "cube" and s.spawn_count == 0


def test_mode_toggle_by_gesture_in_create_returns_to_transform_and_moves():
    """Flujo completo: crear con PINCH, THREE -> TRANSFORM, mover lo creado."""
    s, clock = _session()
    s.set_mode("CREATE")
    s.tick(_state("NEUTRAL"), None)
    assert s.tick(_state("PINCH", (0.5, 0.5)), None)[0][0] == "spawn"
    s.tick(_state("NEUTRAL"), _zero())
    s.tick(_state("THREE"), _zero()); clock.advance(0.8)
    assert s.tick(_state("THREE"), _zero()) == [("mode", "TRANSFORM")]
    s.tick(_state("NEUTRAL"), _zero())
    s.tick(_state("PINCH", (0.5, 0.5)), _zero())
    cmds = s.tick(_state("PINCH", (0.5, 0.4)), _zero())
    assert cmds[0][1]["y"] == pytest.approx(1.0)


# ── bus_status ───────────────────────────────────────────────────────────────

def test_bus_status_missing_invalid_stale_ok():
    now = 1_000.0
    code, msg = A.bus_status({}, False, now, 1.0, path="/tmp/x.json")
    assert code == A.BUS_MISSING and msg == "Sin bus en /tmp/x.json"

    code, msg = A.bus_status({}, True, now, 1.0, path="/tmp/x.json")
    assert code == A.BUS_INVALID and "ilegible" in msg

    code, msg = A.bus_status(_state(ts=now - 2.5), True, now, 1.0)
    assert code == A.BUS_STALE
    assert msg == "BUS OBSOLETO (2.5 s) — ¿corre la app de cámara?"

    code, msg = A.bus_status(_state(ts=None), True, now, 1.0)
    assert code == A.BUS_STALE and "sin timestamp" in msg

    code, msg = A.bus_status(_state(ts=now - 0.05), True, now, 1.0)
    assert code == A.BUS_OK
    assert msg == "Conectado (30 fps app, 50 ms)"


def test_bus_status_boundary_uses_stale_after():
    now = 50.0
    assert A.bus_status(_state(ts=now - 0.99), True, now, 1.0)[0] == A.BUS_OK
    assert A.bus_status(_state(ts=now - 1.01), True, now, 1.0)[0] == A.BUS_STALE
    assert A.bus_status(_state(ts=now - 1.01), True, now, 2.0)[0] == A.BUS_OK


# ── status_lines ─────────────────────────────────────────────────────────────

def test_status_lines_format():
    st = _state("PINCH", mod="OPEN_PALM", mod_palm=(0.1, 0.1))
    st["intent"]["name"] = "GRAB_OR_SELECT"
    lines = A.status_lines(st)
    assert lines == ["Control app: ON", "Dom: PINCH (0.90)", "Mod: OPEN_PALM (0.80)",
                     "Intent: GRAB_OR_SELECT", "Modo app: NAVEGADOR"]
    assert A.status_lines({}) == ["Sin datos del bus"]


# ── AddonSettings ────────────────────────────────────────────────────────────

def test_settings_defaults_when_prefs_missing(monkeypatch):
    s = A.AddonSettings.from_prefs(None)
    assert s.gain == A.DEFAULT_GAIN == 10.0
    assert s.smoothing == A.DEFAULT_SMOOTHING == 0.5
    assert s.z_gain == 0.0 and s.stale_after == 1.0 and s.interval == pytest.approx(0.033)
    assert s.bus_path == A.DEFAULT_BUS_PATH
    monkeypatch.setenv("TLETL_STATE_PATH", "/tmp/otro.json")
    s.bus_path = "  "
    assert str(s.resolved_bus_path()) == "/tmp/otro.json"      # vacío -> regla común


def test_settings_from_prefs_object():
    prefs = types.SimpleNamespace(bus_path="~/b.json", gain=14, smoothing="0.25", rot_gain=1.0,
                                  scale_gain=2.0, z_gain=3.0, stale_after=-1.0, interval=0.0001,
                                  extra_attr="ignored")
    s = A.AddonSettings.from_prefs(prefs)
    assert s.gain == 14.0 and s.smoothing == 0.25 and s.z_gain == 3.0
    assert s.stale_after == 0.0                      # nunca negativo
    assert s.interval == A.MIN_INTERVAL              # piso del timer
    assert s.resolved_bus_path().is_absolute() and s.resolved_bus_path().name == "b.json"


def test_settings_ignore_bad_values():
    prefs = types.SimpleNamespace(gain="mucho", smoothing=None)
    s = A.AddonSettings.from_prefs(prefs)
    assert s.gain == A.DEFAULT_GAIN and s.smoothing == A.DEFAULT_SMOOTHING


def test_default_bus_path_constant_comes_from_reader():
    sr = importlib.import_module("apps.blender_control.state_reader")
    assert A.DEFAULT_BUS_PATH == str(sr.default_bus_path())
    assert A.default_bus_path() == sr.default_bus_path()


def test_module_level_state_exists_without_bpy():
    assert A.bpy is None
    assert A.last_error == ""
    assert isinstance(A._session, A.GestureSession)
    assert A._ADDON_ID == "apps.blender_control"
    assert not hasattr(A, "register")
    assert math.isclose(A.MODE_HOLD_SECONDS, 0.8)
