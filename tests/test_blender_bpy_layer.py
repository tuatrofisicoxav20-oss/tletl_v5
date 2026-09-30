"""tests/test_blender_bpy_layer.py — la capa bpy del addon con un bpy/bmesh FALSOS.

Blender no está instalado en CI, pero la capa que habla con bpy (registro,
timer, escritura de transform, spawn con bmesh, panel, operadores) es donde
más duele un error de API. Este harness la ejecuta entera con módulos falsos
que registran las llamadas. Adaptado del harness del revisor independiente.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
import types
import unittest.mock as mock

import pytest

ADDON_MODULE = "apps.blender_control.tletl_blender_addon"
ADDON_ID = "apps.blender_control"


# ── mathutils falso ──────────────────────────────────────────────────────────

class Vec:
    """Registra las escrituras por componente (obj.location.x = v) en el log del dueño."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        object.__setattr__(self, "_log", None)
        object.__setattr__(self, "_tag", "")
        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))
        object.__setattr__(self, "z", float(z))

    def __setattr__(self, name, value):
        if name in ("x", "y", "z"):
            value = float(value)
            if self._log is not None:
                self._log.append(f"{self._tag}.{name}")
        object.__setattr__(self, name, value)

    def __iter__(self):
        return iter((self.x, self.y, self.z))


class FakeObj:
    def __init__(self, name, data=None, type_="MESH"):
        self.name = name
        self.data = data
        self.type = type_
        self.rotation_mode = "XYZ"
        self.writes: list = []
        self._selected = False
        self.location, self.rotation_euler, self.scale = Vec(), Vec(), Vec(1, 1, 1)
        self.writes.clear()

    def _mk(attr):  # noqa: N805 - fábrica de properties
        def get(self):
            return getattr(self, "_" + attr)

        def set_(self, value):
            self.writes.append(attr)
            v = value if isinstance(value, Vec) else Vec(*value)
            object.__setattr__(v, "_log", self.writes)
            object.__setattr__(v, "_tag", attr)
            setattr(self, "_" + attr, v)
        return property(get, set_)

    location = _mk("loc")
    rotation_euler = _mk("rot")
    scale = _mk("scale")

    def select_get(self, view_layer=None):
        return self._selected

    def select_set(self, state, view_layer=None):
        self._selected = bool(state)


class FakeMesh:
    def __init__(self, name):
        self.name = name
        self.updated = 0
        self.from_bm = None

    def update(self):
        self.updated += 1


class Coll:
    def __init__(self):
        self.objs: list = []
        self.objects = types.SimpleNamespace(link=self.objs.append)


class Objects(list):
    active = None


def make_fake_bpy(addon_id):
    bpy = types.ModuleType("bpy")
    bpy.types = types.SimpleNamespace(
        AddonPreferences=type("AddonPreferences", (), {}),
        Operator=type("Operator", (), {"report": lambda self, *a, **k: None}),
        Panel=type("Panel", (), {}),
    )
    bpy.props = types.SimpleNamespace(
        StringProperty=lambda **k: ("StringProperty", k),
        FloatProperty=lambda **k: ("FloatProperty", k),
        EnumProperty=lambda **k: ("EnumProperty", k),
    )
    registered: list = []
    bpy.utils = types.SimpleNamespace(register_class=registered.append,
                                      unregister_class=registered.remove)
    timers: dict = {}
    bpy.app = types.SimpleNamespace(timers=types.SimpleNamespace(
        register=lambda fn, first_interval=0.0, persistent=False: timers.__setitem__(fn, persistent),
        unregister=lambda fn: timers.pop(fn),
        is_registered=lambda fn: fn in timers,
    ))
    bpy.path = types.SimpleNamespace(abspath=lambda p: "/blend/dir/" + p[2:] if p.startswith("//") else p)
    meshes: list = []
    objects: list = []

    def new_mesh(name):
        m = FakeMesh(name)
        meshes.append(m)
        return m

    def new_obj(name, data):
        o = FakeObj(name, data)
        objects.append(o)
        return o

    bpy.data = types.SimpleNamespace(meshes=types.SimpleNamespace(new=new_mesh),
                                     objects=types.SimpleNamespace(new=new_obj))
    prefs = types.SimpleNamespace(bus_path="", gain=10.0, smoothing=1.0, rot_gain=2.0,
                                  scale_gain=1.5, z_gain=0.0, stale_after=1.0, interval=0.033)
    view_objs = Objects()
    coll = Coll()
    bpy.context = types.SimpleNamespace(
        preferences=types.SimpleNamespace(addons={addon_id: types.SimpleNamespace(preferences=prefs)}),
        view_layer=types.SimpleNamespace(objects=view_objs),
        scene=types.SimpleNamespace(cursor=types.SimpleNamespace(location=Vec(1, 2, 3)), collection=Coll()),
        window_manager=types.SimpleNamespace(windows=[types.SimpleNamespace(
            screen=types.SimpleNamespace(areas=[types.SimpleNamespace(type="VIEW_3D", tag_redraw=lambda: None)]))]),
        collection=coll,
    )
    # enlazar también al view layer, como haría Blender con una colección visible
    coll.objects.link = lambda o: (coll.objs.append(o), view_objs.append(o))
    bpy._fake = types.SimpleNamespace(registered=registered, timers=timers, meshes=meshes,
                                      objects=objects, prefs=prefs, view_objs=view_objs, coll=coll)
    return bpy


def make_fake_bmesh():
    bmesh = types.ModuleType("bmesh")
    calls: list = []
    bms: list = []

    class BM:
        def __init__(self):
            self.freed = False
            self.op = None

        def to_mesh(self, mesh):
            mesh.from_bm = self.op

        def free(self):
            self.freed = True

    def new():
        b = BM()
        bms.append(b)
        return b

    def op(name):
        def f(bm, **kw):
            bm.op = (name, kw)
            calls.append((name, kw))
        return f

    bmesh.new = new
    bmesh.ops = types.SimpleNamespace(create_cube=op("create_cube"), create_uvsphere=op("create_uvsphere"),
                                      create_cone=op("create_cone"), create_grid=op("create_grid"))
    bmesh._fake = types.SimpleNamespace(calls=calls, bms=bms)
    return bmesh


def frame(dom="NEUTRAL", palm=(0.5, 0.5), mod="NO_HAND", mpalm=None, ts=None):
    return {"timestamp": time.time() if ts is None else ts, "fps": 30.0, "mode": "NAVEGADOR",
            "dom": {"present": True, "gesture": dom, "confidence": 0.9, "palm": list(palm)},
            "mod": {"present": mod != "NO_HAND", "gesture": mod, "confidence": 0.8,
                    "palm": list(mpalm) if mpalm else None},
            "intent": {"name": "NONE"}, "extra": {"control": True}}


@pytest.fixture
def addon(tmp_path):
    """Addon importado con bpy/bmesh falsos, aislado: al salir se restaura sys.modules
    y se descarta el módulo para que otros tests lo reimporten limpio (con bpy=None)."""
    bpy = make_fake_bpy(ADDON_ID)
    bmesh = make_fake_bmesh()
    for key in list(sys.modules):
        if key.endswith("tletl_blender_addon"):
            del sys.modules[key]
    with mock.patch.dict(sys.modules, {"bpy": bpy, "bmesh": bmesh}):
        module = importlib.import_module(ADDON_MODULE)
        bus = tmp_path / "bus.json"
        bpy._fake.prefs.bus_path = str(bus)
        module.register()
        op = module.TLETL_OT_Start()
        assert op.execute(None) == {"FINISHED"}
        yield types.SimpleNamespace(A=module, bpy=bpy, bmesh=bmesh, bus=bus,
                                    write=lambda st: bus.write_text(json.dumps(st), encoding="utf-8"))
        module.TLETL_OT_Stop().execute(None)
        module.unregister()
    for key in list(sys.modules):
        if key.endswith("tletl_blender_addon"):
            del sys.modules[key]


def _add_cube(ctx, scale=(1, 1, 1)):
    cube = FakeObj("Cube")
    cube.scale = scale
    cube.writes.clear()
    ctx.bpy._fake.view_objs.append(cube)
    Objects.active = cube
    return cube


# ── Tests ────────────────────────────────────────────────────────────────────

def test_register_and_start(addon):
    A, bpy = addon.A, addon.bpy
    assert A.bpy is bpy and A._ADDON_ID == ADDON_ID
    assert len(bpy._fake.registered) == 7
    assert A.TletlAddonPreferences.bl_idname == ADDON_ID
    assert A._addon_running and bpy._fake.timers[A._tletl_timer_callback] is True
    assert A.TLETL_OT_Start().execute(None) == {"CANCELLED"}      # ya corría


def test_missing_bus_keeps_timer_alive(addon):
    A = addon.A
    addon.bus.unlink(missing_ok=True)
    assert A._tletl_timer_callback() == 0.033
    assert A._last_status[0] == A.BUS_MISSING and A.last_error == ""


def test_pinch_drag_writes_only_location_xy(addon):
    A = addon.A
    cube = _add_cube(addon, scale=(1, 2, 1))
    addon.write(frame("PINCH", (0.5, 0.5)))
    A._tletl_timer_callback()
    assert A._last_status[0] == A.BUS_OK
    assert cube.writes == []                                     # tick de cebado: nada escrito
    addon.write(frame("PINCH", (0.6, 0.4)))
    assert A._tletl_timer_callback() == 0.033 and A.last_error == ""
    assert abs(cube.location.x - 1.0) < 1e-9 and abs(cube.location.y - 1.0) < 1e-9
    assert cube.writes == ["loc.x", "loc.y"]                     # ni z, ni rot, ni scale
    assert tuple(cube.scale) == (1.0, 2.0, 1.0)


def test_two_hand_scale_keeps_non_uniform_proportions(addon):
    A = addon.A
    cube = _add_cube(addon, scale=(1, 2, 1))
    addon.write(frame("PINCH", (0.4, 0.5), "OPEN_PALM", (0.6, 0.5)))
    A._tletl_timer_callback()
    addon.write(frame("PINCH", (0.2, 0.5), "OPEN_PALM", (0.8, 0.5)))
    A._tletl_timer_callback()
    assert A.last_error == ""
    sx, sy, sz = cube.scale
    assert sx > 1.0 and abs(sy - 2 * sx) < 1e-9 and abs(sz - sx) < 1e-9
    ref = A.scale_reference(tuple(cube.scale))
    assert abs(A._session.mapper._last_out["scale"] - ref) < 1e-9   # sin resync espurio


def test_mirrored_object_keeps_sign_and_min_scale(addon):
    A = addon.A
    cube = _add_cube(addon, scale=(-1, 1, 1))
    addon.write(frame("PINCH", (0.05, 0.5), "OPEN_PALM", (0.95, 0.5)))
    A._tletl_timer_callback()
    addon.write(frame("PINCH", (0.5, 0.5), "OPEN_PALM", (0.5, 0.5)))
    A._tletl_timer_callback()
    assert A.last_error == ""
    sx, sy, sz = cube.scale
    assert sx < 0 < sy and abs(sx) == sy == sz and sy >= A.MIN_SCALE - 1e-12


def test_quaternion_rotation_mode_reports_error_and_timer_survives(addon):
    A = addon.A
    cube = _add_cube(addon)
    cube.rotation_mode = "QUATERNION"
    addon.write(frame("NEUTRAL", mod="PINCH", mpalm=(0.4, 0.5)))
    A._tletl_timer_callback()
    addon.write(frame("NEUTRAL", mod="PINCH", mpalm=(0.6, 0.5)))
    assert A._tletl_timer_callback() == 0.033 and "rotation_mode" in A.last_error
    cube.rotation_mode = "XYZ"
    addon.write(frame("NEUTRAL", mod="PINCH", mpalm=(0.4, 0.5)))
    A._tletl_timer_callback()
    addon.write(frame("NEUTRAL", mod="PINCH", mpalm=(0.6, 0.5)))
    A._tletl_timer_callback()
    assert A.last_error == ""
    assert abs(cube.rotation_euler.z - 0.2 * 2.0 * 3.141592653589793) < 1e-9


def test_stale_bus_freezes_object(addon):
    A = addon.A
    cube = _add_cube(addon)
    addon.write(frame("PINCH", (0.5, 0.5)))
    A._tletl_timer_callback()
    before = tuple(cube.location)
    addon.write(frame("PINCH", (0.9, 0.9), ts=time.time() - 5))
    assert A._tletl_timer_callback() == 0.033
    assert A._last_status[0] == A.BUS_STALE and tuple(cube.location) == before
    addon.write(frame("PINCH", (0.9, 0.9)))                      # fresco otra vez: solo ceba
    A._tletl_timer_callback()
    assert tuple(cube.location) == before


def test_create_mode_spawns_with_bmesh_linked_active_selected(addon):
    A, bpy, bmesh = addon.A, addon.bpy, addon.bmesh
    cube = _add_cube(addon)
    A._session.reset()
    addon.write(frame("THREE"))
    A._tletl_timer_callback()
    time.sleep(0.05)
    A._session.hold_seconds = 0.01
    addon.write(frame("THREE"))
    A._tletl_timer_callback()
    assert A._session.mode == "CREATE"
    addon.write(frame("NEUTRAL"))
    A._tletl_timer_callback()
    addon.write(frame("PINCH", (0.75, 0.25)))
    assert A._tletl_timer_callback() == 0.033 and A.last_error == ""
    assert [o.name for o in bpy._fake.objects] == ["Tletl_cube"]
    new = bpy._fake.objects[0]
    assert tuple(new.location) == (1 + 2.5, 2 + 2.5, 3.0)        # palma -> XY alrededor del cursor
    assert new in bpy._fake.coll.objs                             # enlazado a la colección activa
    assert bpy._fake.view_objs.active is new                     # activo vía view_layer
    assert new._selected and not cube._selected                  # selección exclusiva
    assert bmesh._fake.calls[-1] == ("create_cube", {"size": 2.0}) and bmesh._fake.bms[-1].freed
    assert bpy._fake.meshes[-1].updated == 1
    # VICTORY cambia el tipo; el siguiente PINCH crea una esfera
    addon.write(frame("VICTORY"))
    A._tletl_timer_callback()
    addon.write(frame("PINCH", (0.5, 0.5)))
    A._tletl_timer_callback()
    assert bpy._fake.objects[-1].name == "Tletl_sphere" and bmesh._fake.calls[-1][0] == "create_uvsphere"
    for prim in A.PRIMITIVES:
        A._session.set_primitive(prim)
        assert A.TLETL_OT_SpawnNow().execute(None) == {"FINISHED"}
    assert [c[0] for c in bmesh._fake.calls[-5:]] == [
        "create_cube", "create_uvsphere", "create_cone", "create_cone", "create_grid"]
    assert A.last_error == ""


def test_prefs_lookup_failure_falls_back_to_defaults(addon):
    A, bpy = addon.A, addon.bpy
    bpy.context.preferences.addons.clear()
    addon.write(frame("NEUTRAL"))
    assert A._tletl_timer_callback() == A.DEFAULT_INTERVAL and A.last_error == ""


def test_missing_view_layer_is_captured_not_fatal(addon):
    A, bpy = addon.A, addon.bpy
    saved = bpy.context.view_layer
    del bpy.context.view_layer
    bpy.context.active_object = None
    addon.write(frame("PINCH"))
    assert A._tletl_timer_callback() == 0.033
    bpy.context.view_layer = saved


def test_prefs_exception_of_any_type_keeps_timer_alive(addon):
    """_current_settings() corre dentro del try del timer: ni un error raro de prefs lo mata."""
    A, bpy = addon.A, addon.bpy

    class Boom:
        @property
        def preferences(self):
            raise ValueError("prefs rotas")

    bpy.context.preferences.addons[ADDON_ID] = Boom()
    addon.write(frame("NEUTRAL"))
    assert A._tletl_timer_callback() == A.DEFAULT_INTERVAL
    assert "prefs rotas" in A.last_error


def test_blend_relative_bus_path_is_resolved_with_bpy(addon):
    A = addon.A
    s = A.AddonSettings(bus_path="//bus/tletl_state.json")
    assert str(s.resolved_bus_path()) == "/blend/dir/bus/tletl_state.json"
    assert A.AddonSettings(bus_path="").resolved_bus_path() == A.default_bus_path()


def test_panel_draw_and_stop(addon):
    A = addon.A

    class Layout:
        def __getattr__(self, name):
            return lambda *a, **k: Layout()

    panel = A.TLETL_PT_Panel()
    panel.layout = Layout()
    panel.draw(None)
    assert A.TLETL_OT_Stop().execute(None) == {"FINISHED"} and not A._addon_running
    assert A._tletl_timer_callback() is None                      # detenido: el timer termina
