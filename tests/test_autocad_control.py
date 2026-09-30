"""tests/test_autocad_control.py — fuentes del bus, export DXF, cliente AutoCAD
(COM simulado), sesión con reloj falso, macro de FreeCAD (helpers puros),
launcher y packaging.

Nada de esto necesita AutoCAD, Windows, pywin32 ni FreeCAD: el backend COM se
prueba con un ModelSpace falso que registra las llamadas.
"""

from __future__ import annotations

import json
import math
import os
import py_compile
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from tletl_core import paths as core_paths
from tletl_core.bus import TletlStateBus
from tletl_core.state import TletlFrameState, TletlHandState

from apps.autocad_control import autocad_client as ac
from apps.autocad_control import freecad_macro as fm
from apps.autocad_control import session as ses
from apps.autocad_control.bus_source import FileBusSource, ScriptedBusSource, UdpBusSource, open_source
from apps.autocad_control.dxf_export import (
    EZDXF_HINT,
    UNIT_CODES,
    export_scene_dxf,
    layer_name,
    load_scene_json,
    save_scene_json,
    scene_from_json,
    scene_to_json,
    unit_mesh,
)
from apps.autocad_control.gesture_modeler import SOLID_KINDS, GestureModeler, Scene, Solid

ROOT = Path(__file__).resolve().parent.parent
ezdxf = pytest.importorskip("ezdxf")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _state(dom="NO_HAND", dpalm=None, mod="NO_HAND", mpalm=None, ts=None) -> dict:
    st = {"dom": {"gesture": dom, "palm": list(dpalm) if dpalm else None},
          "mod": {"gesture": mod, "palm": list(mpalm) if mpalm else None}}
    if ts is not None:
        st["timestamp"] = ts
    return st


def _write_frame(bus: TletlStateBus, gesture: str, palm, *, timestamp=0.0) -> None:
    st = TletlFrameState(timestamp=timestamp)
    st.dom = TletlHandState(present=True, gesture=gesture, stable_gesture=gesture, palm=palm, critic_ok=True)
    bus.write(st)


class FakeClock:
    """Reloj + sleep falsos: sleep avanza el tiempo en lugar de dormir."""

    def __init__(self, start: float = 1000.0):
        self.t = start
        self.slept = 0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept += 1
        self.t += 0.5


def _full_scene() -> Scene:
    sc = Scene(units="mm")
    sc.add("box", (10, 20, 5), (4, 6, 8), rotation_z=math.pi / 2, scale=2.0)
    sc.add("box", (-30, 0, 0), (5, 5, 5))
    for kind in ("cylinder", "sphere", "cone", "wedge"):
        sc.add(kind, (0, 0, 0), (2, 2, 2))
    return sc


# ── bus_source ───────────────────────────────────────────────────────────────

def test_file_source_default_path_follows_core(monkeypatch, tmp_path):
    monkeypatch.setenv(core_paths.ENV_STATE_PATH, str(tmp_path / "custom.json"))
    assert FileBusSource().path == core_paths.default_bus_path() == tmp_path / "custom.json"
    monkeypatch.delenv(core_paths.ENV_STATE_PATH)
    monkeypatch.setenv(core_paths.ENV_HOME, str(tmp_path / "home"))
    assert FileBusSource().path == tmp_path / "home" / "tletl_state.json"


def test_file_source_read_poll_dedupe_and_age(tmp_path):
    path = tmp_path / "bus.json"
    src = FileBusSource(path)
    assert src.read() == {} and src.poll() is None and src.age() == float("inf")
    bus = TletlStateBus(path)
    _write_frame(bus, "PINCH", (0.4, 0.6), timestamp=100.0)
    first = src.poll()
    assert first is not None and first["dom"]["gesture"] == "PINCH"
    assert src.poll() is None                      # mismo contenido: no se re-entrega
    assert src.read()["dom"]["palm"] == [0.4, 0.6]  # read() siempre devuelve el dict
    _write_frame(bus, "FIST", (0.4, 0.6), timestamp=101.0)
    assert src.poll()["dom"]["gesture"] == "FIST"
    assert src.age(now=101.5) == pytest.approx(0.5) and src.reads == 2
    assert "bus.json" in src.describe()
    src.close()


def test_file_source_corrupt_or_non_dict(tmp_path):
    path = tmp_path / "bus.json"
    src = FileBusSource(path)
    path.write_text("{not json", encoding="utf-8")
    assert src.poll() is None and src.read() == {}
    path.write_text("[1, 2]", encoding="utf-8")
    assert src.poll() is None
    path.write_text("", encoding="utf-8")
    assert src.poll() is None


def test_udp_source_loopback_with_real_bus(tmp_path):
    src = UdpBusSource("127.0.0.1", 0)
    try:
        host, port = src.address
        bus = TletlStateBus(tmp_path / "s.json", udp_target=f"{host}:{port}")
        _write_frame(bus, "PINCH", (0.3, 0.7), timestamp=50.0)
        got = None
        for _ in range(100):
            got = src.poll()
            if got:
                break
            time.sleep(0.01)
        assert got is not None, "no llegó el datagrama por loopback"
        assert got["dom"]["gesture"] == "PINCH" and got["dom"]["palm"] == [0.3, 0.7]
        assert src.received >= 1 and src.age(now=50.25) == pytest.approx(0.25)
        assert src.describe().startswith("udp 127.0.0.1:")
        bus.close()
    finally:
        src.close()


def test_udp_source_feeds_modeler_end_to_end(tmp_path):
    """Dos frames PINCH por UDP -> el modeler produce un move (delta entre frames)."""
    src = UdpBusSource("127.0.0.1", 0)
    try:
        bus = TletlStateBus(tmp_path / "s.json", udp_target=f"127.0.0.1:{src.address[1]}")
        modeler = GestureModeler(gain=10.0, stale_after=None)
        modeler.scene.add("box")
        ops_all = []
        for palm in ((0.5, 0.5), (0.7, 0.5)):
            _write_frame(bus, "PINCH", palm, timestamp=1.0)
            for _ in range(100):
                st = src.poll()
                if st:
                    ops_all.extend(modeler.update(st, 1.0))
                    break
                time.sleep(0.01)
        assert ops_all == [("move", 1, pytest.approx(2.0), pytest.approx(0.0), 0.0)]
        bus.close()
    finally:
        src.close()


def test_udp_source_from_target_and_open_source():
    src = UdpBusSource.from_target("127.0.0.1:0")
    try:
        assert src.address[0] == "127.0.0.1"
    finally:
        src.close()
    with pytest.raises(ValueError):
        UdpBusSource.from_target("sin-puerto")
    file_src = open_source(bus=None, udp=None)
    assert isinstance(file_src, FileBusSource)
    udp_src = open_source(bus="ignorado.json", udp="127.0.0.1:0")
    try:
        assert isinstance(udp_src, UdpBusSource)
    finally:
        udp_src.close()


def test_scripted_source_finishes():
    src = ScriptedBusSource([_state("PINCH", (0.5, 0.5), ts=3.0), None])
    assert not src.finished
    assert src.poll()["dom"]["gesture"] == "PINCH" and not src.finished
    assert src.poll() is None and src.finished
    assert src.poll() is None and src.age(now=4.0) == pytest.approx(1.0)
    assert ScriptedBusSource([]).finished


# ── dxf_export ───────────────────────────────────────────────────────────────

def test_export_all_kinds_layers_audit_and_units(tmp_path):
    out = export_scene_dxf(_full_scene(), tmp_path / "sub" / "escena.dxf", units="mm")
    assert out.exists()
    doc = ezdxf.readfile(out)
    assert doc.dxfversion == "AC1024"                 # R2010
    assert doc.header["$INSUNITS"] == UNIT_CODES["mm"] == 4
    msp = doc.modelspace()
    counts = {kind: len(msp.query(f"MESH[layer=='{layer_name(kind)}']")) for kind in SOLID_KINDS}
    assert counts == {"box": 2, "cylinder": 1, "sphere": 1, "cone": 1, "wedge": 1}
    assert len(msp) == 6
    for kind in SOLID_KINDS:
        assert doc.layers.has_entry(layer_name(kind))
    auditor = doc.audit()
    assert not auditor.has_errors, [str(e) for e in auditor.errors]
    tags = msp.query("MESH[layer=='TLETL_BOX']")[0].get_xdata("TLETL")
    assert tags[0].value == "tletl:1:box"
    assert json.loads(tags[1].value)["kind"] == "box"


def test_export_units_m_and_invalid(tmp_path):
    sc = Scene()
    sc.add("sphere")
    doc = ezdxf.readfile(export_scene_dxf(sc, tmp_path / "m.dxf", units="m"))
    assert doc.header["$INSUNITS"] == 6
    sc.units = "in"
    doc = ezdxf.readfile(export_scene_dxf(sc, tmp_path / "in.dxf"))      # default: scene.units
    assert doc.header["$INSUNITS"] == 1 and doc.header["$MEASUREMENT"] == 0
    with pytest.raises(ValueError):
        export_scene_dxf(sc, tmp_path / "x.dxf", units="parsecs")


def test_transformed_box_bbox_matches_solid(tmp_path):
    from ezdxf.math import BoundingBox
    sc = _full_scene()
    doc = ezdxf.readfile(export_scene_dxf(sc, tmp_path / "bb.dxf"))
    box = doc.modelspace().query("MESH[layer=='TLETL_BOX']")[0]
    bb = BoundingBox(box.vertices)
    # tamaño (4,6,8) x escala 2 = (8,12,16), rotado 90° en Z -> extensión (12, 8, 16) centrada en (10,20,5)
    assert tuple(bb.extmin) == pytest.approx((4.0, 16.0, -3.0))
    assert tuple(bb.extmax) == pytest.approx((16.0, 24.0, 13.0))
    assert tuple(bb.center) == pytest.approx((10.0, 20.0, 5.0))


def test_unit_meshes_are_centered_unit_cubes():
    from ezdxf.math import BoundingBox
    for kind in SOLID_KINDS:
        mesh = unit_mesh(kind, 16)
        bb = BoundingBox(mesh.vertices)
        assert tuple(bb.extmin) == pytest.approx((-0.5, -0.5, -0.5)), kind
        assert tuple(bb.extmax) == pytest.approx((0.5, 0.5, 0.5)), kind
        assert len(mesh.faces) >= 5
    with pytest.raises(ValueError):
        unit_mesh("torus")


def test_unit_meshes_are_closed_manifold_with_outward_normals(tmp_path):
    """CONVTOSOLID de AutoCAD exige una malla CERRADA. Regresión: la cuña se escribía
    como 5 caras sueltas (18 vértices, superficie abierta) porque MeshTransformer.add_face
    no comparte vértices; MeshVertexMerger sí."""
    for kind in SOLID_KINDS:
        diag = unit_mesh(kind, 16).diagnose()
        assert diag.is_closed_surface and diag.is_manifold, kind
        assert diag.euler_characteristic == 2, kind                    # V - E + F de una esfera topológica
        assert diag.estimate_face_normals_direction() > 0.5, kind      # normales hacia afuera
    wedge = unit_mesh("wedge")
    assert (len(wedge.vertices), len(wedge.faces)) == (6, 5)
    # y así llega al DXF: la entidad MESH conserva los vértices compartidos
    sc = Scene()
    sc.add("wedge", (1, 2, 3), (2, 4, 6))
    ent = ezdxf.readfile(export_scene_dxf(sc, tmp_path / "w.dxf")).modelspace().query("MESH")[0]
    assert (len(ent.vertices), len(ent.faces)) == (6, 5)
    from ezdxf.render.mesh import MeshBuilder
    assert MeshBuilder.from_mesh(ent).diagnose().is_closed_surface


def test_scene_json_roundtrip_file(tmp_path):
    sc = _full_scene()
    p = save_scene_json(sc, tmp_path / "d" / "escena.json")
    back = load_scene_json(p)
    assert back.to_dict() == sc.to_dict()
    assert scene_from_json(scene_to_json(sc, indent=None)).selected_id == sc.selected_id
    with pytest.raises(ValueError):
        scene_from_json('{"format": "otro", "solids": []}')


def test_export_without_ezdxf_raises_clear_importerror(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ezdxf", None)
    sc = Scene()
    sc.add("box")
    with pytest.raises(ImportError) as info:
        export_scene_dxf(sc, tmp_path / "x.dxf")
    assert "pip install ezdxf" in str(info.value) and str(info.value) == EZDXF_HINT
    # el JSON no necesita ezdxf
    assert save_scene_json(sc, tmp_path / "ok.json").exists()


def test_dxf_export_cli_reexports(tmp_path, capsys):
    from apps.autocad_control import dxf_export
    p = save_scene_json(_full_scene(), tmp_path / "escena.json")
    assert dxf_export.main([str(p)]) == 0
    assert (tmp_path / "escena.dxf").exists()
    assert dxf_export.main([str(p), str(tmp_path / "otro.dxf"), "--units", "m", "--segments", "12"]) == 0
    assert ezdxf.readfile(tmp_path / "otro.dxf").header["$INSUNITS"] == 6
    assert "6 sólidos" in capsys.readouterr().out


# ── autocad_client: COM simulado ─────────────────────────────────────────────

class FakeEntity:
    def __init__(self, method: str, args: tuple, *, fail: bool = False):
        self.method = method
        self.args = args
        self.calls: list = []
        self.fail = fail

    def _record(self, *call):
        if self.fail:
            raise RuntimeError("com_error simulado")
        self.calls.append(call)

    def Move(self, p1, p2):
        self._record("Move", p1, p2)

    def Rotate3D(self, p1, p2, angle):
        self._record("Rotate3D", p1, p2, angle)

    def ScaleEntity(self, base, factor):
        self._record("ScaleEntity", base, factor)


class FakeModelSpace:
    def __init__(self, *, fail_entities: bool = False):
        self.calls: list = []
        self.entities: list = []
        self.fail_entities = fail_entities

    def _make(self, method, *args):
        self.calls.append((method, *args))
        ent = FakeEntity(method, args, fail=self.fail_entities)
        self.entities.append(ent)
        return ent

    def AddBox(self, center, length, width, height):
        return self._make("AddBox", center, length, width, height)

    def AddCylinder(self, center, radius, height):
        return self._make("AddCylinder", center, radius, height)

    def AddSphere(self, center, radius):
        return self._make("AddSphere", center, radius)

    def AddCone(self, center, radius, height):
        return self._make("AddCone", center, radius, height)

    def AddWedge(self, center, length, width, height):
        return self._make("AddWedge", center, length, width, height)


class FakeDocument:
    def __init__(self, **kw):
        self.ModelSpace = FakeModelSpace(**kw)
        self.regens: list = []

    def Regen(self, mode):
        self.regens.append(mode)


class FakeApp:
    def __init__(self, **kw):
        self.ActiveDocument = FakeDocument(**kw)
        self.Visible = False


class FakeAppNoDoc:
    @property
    def ActiveDocument(self):
        raise RuntimeError("(-2147352567, 'Exception occurred.')")


def _backend(**kw):
    app = FakeApp(**kw.pop("app_kw", {}))
    return ac.AutoCADBackend(app, **kw), app.ActiveDocument.ModelSpace, app.ActiveDocument


def test_point_without_pythoncom_returns_tuple(monkeypatch):
    monkeypatch.setitem(sys.modules, "pythoncom", None)
    assert ac._point(1, 2, 3) == (1.0, 2.0, 3.0)
    assert ac._point(4, 5) == (4.0, 5.0, 0.0)


def test_point_with_fake_pythoncom_builds_variant_of_doubles(monkeypatch):
    class FakeVariant:
        def __init__(self, vt, value):
            self.vt, self.value = vt, value

    pythoncom = types.ModuleType("pythoncom")
    pythoncom.VT_ARRAY, pythoncom.VT_R8 = 0x2000, 5
    client = types.ModuleType("win32com.client")
    client.VARIANT = FakeVariant
    win32com = types.ModuleType("win32com")
    win32com.client = client
    monkeypatch.setitem(sys.modules, "pythoncom", pythoncom)
    monkeypatch.setitem(sys.modules, "win32com", win32com)
    monkeypatch.setitem(sys.modules, "win32com.client", client)
    p = ac._point(1, 2, 3)
    assert isinstance(p, FakeVariant) and p.vt == 0x2000 | 5 and p.value == [1.0, 2.0, 3.0]


def test_dispatch_autocad_uses_prog_id_and_backend_connects(monkeypatch):
    calls = []
    client = types.ModuleType("win32com.client")

    def Dispatch(prog_id):
        calls.append(prog_id)
        return FakeApp()

    client.Dispatch = Dispatch
    win32com = types.ModuleType("win32com")
    win32com.client = client
    monkeypatch.setitem(sys.modules, "win32com", win32com)
    monkeypatch.setitem(sys.modules, "win32com.client", client)
    backend = ac.AutoCADBackend()          # app=None -> Dispatch("AutoCAD.Application")
    assert calls == ["AutoCAD.Application"] and backend.app.Visible is True


def test_backend_without_pywin32_raises_importerror(monkeypatch):
    monkeypatch.setitem(sys.modules, "win32com", None)
    monkeypatch.setitem(sys.modules, "win32com.client", None)
    with pytest.raises(ImportError) as info:
        ac.AutoCADBackend()
    assert "pip install pywin32" in str(info.value)


def test_backend_requires_active_document():
    with pytest.raises(RuntimeError) as info:
        ac.AutoCADBackend(FakeAppNoDoc())
    assert "dibujo activo" in str(info.value)


def test_backend_add_calls_per_kind():
    backend, space, doc = _backend()
    solids = [
        Solid(id=1, kind="box", center=(1, 2, 3), size=(4, 5, 6)),
        Solid(id=2, kind="cylinder", center=(0, 0, 0), size=(2, 2, 7)),
        Solid(id=3, kind="sphere", center=(1, 1, 1), size=(3, 3, 3)),
        Solid(id=4, kind="cone", center=(0, 0, 0), size=(4, 4, 9)),
        Solid(id=5, kind="wedge", center=(0, 0, 0), size=(1, 2, 3), scale=2.0),
    ]
    assert backend.apply([("add", s) for s in solids]) == 5
    assert space.calls == [
        ("AddBox", (1.0, 2.0, 3.0), 4.0, 5.0, 6.0),
        ("AddCylinder", (0.0, 0.0, 0.0), 1.0, 7.0),
        ("AddSphere", (1.0, 1.0, 1.0), 1.5),
        ("AddCone", (0.0, 0.0, 0.0), 2.0, 9.0),
        ("AddWedge", (0.0, 0.0, 0.0), 2.0, 4.0, 6.0),        # escala ya aplicada en las dimensiones
    ]
    assert set(backend.entities) == {1, 2, 3, 4, 5}
    assert len(doc.regens) == 5 and doc.regens[0] == ac.AC_ACTIVE_VIEWPORT   # Regen al crear
    assert backend.solids[5] is not solids[4] and backend.solids[5] == solids[4]  # copia local


def test_backend_add_with_rotation_calls_rotate3d_around_vertical_axis():
    backend, space, _ = _backend()
    backend.apply_op(("add", Solid(id=1, kind="box", center=(2, 3, 4), rotation_z=0.5)))
    ent = space.entities[0]
    assert ent.calls == [("Rotate3D", (2.0, 3.0, 4.0), (2.0, 3.0, 5.0), 0.5)]


def test_backend_move_rotate_scale_select_and_ignored_ops():
    backend, space, doc = _backend(regen_every=0)
    backend.apply_op(("add", Solid(id=1, kind="cylinder", center=(1, 1, 0), size=(2, 2, 2))))
    ent = space.entities[0]
    ops = [("select", 1), ("move", 1, 1.0, 2.0, 0.5), ("rotate", 1, math.pi / 2), ("scale", 1, 2.0),
           ("mode", "CREATE"), ("kind", "box"), ("end_session",), ("select", 99), ("move", 99, 1, 1, 1)]
    assert backend.apply(ops) == 4            # select 1, move, rotate, scale
    assert backend.failed == 2                # select 99 y move 99: ids inexistentes
    assert ent.calls == [
        ("Move", (0.0, 0.0, 0.0), (1.0, 2.0, 0.5)),
        ("Rotate3D", (2.0, 3.0, 0.5), (2.0, 3.0, 1.5), math.pi / 2),   # eje vertical por el centro YA movido
        ("ScaleEntity", (2.0, 3.0, 0.5), 2.0),
    ]
    assert backend.selected_id == 1
    s = backend.solids[1]
    assert s.center == (2.0, 3.0, 0.5) and s.rotation_z == pytest.approx(math.pi / 2) and s.scale == 2.0
    assert backend.failed == 2 and backend.applied == 5      # add + 4 ops; select #99 y move #99 fallan (KeyError)
    assert len(doc.regens) == 1                              # regen_every=0: solo el Regen del add


def test_backend_regen_cadence():
    backend, _, doc = _backend(regen_every=3)
    backend.apply_op(("add", Solid(id=1, kind="box")))       # regen del add
    for _ in range(6):
        backend.apply_op(("move", 1, 0.1, 0.0, 0.0))
    assert len(doc.regens) == 3                              # add + cada 3 ops (2 veces)


def test_regen_uses_acregentype_values_from_type_library():
    """AcRegenType en la biblioteca de tipos de AutoCAD (comtypes.gen.AutoCAD, win32com):
    acActiveViewport = 0, acAllViewports = 1. Regresión: estaban invertidos."""
    assert ac.AC_ACTIVE_VIEWPORT == 0 and ac.AC_ALL_VIEWPORTS == 1
    backend, _, doc = _backend(regen_every=0)
    backend.apply_op(("add", Solid(id=1, kind="sphere")))
    assert doc.regens == [0]


def test_backend_com_error_does_not_raise_and_is_logged():
    logs = []
    backend, _, _ = _backend(app_kw={"fail_entities": True}, log=logs.append)
    backend.apply_op(("add", Solid(id=1, kind="box")))
    assert backend.apply([("move", 1, 1, 0, 0), ("scale", 1, 2.0)]) == 0
    assert backend.failed == 2 and len(logs) == 2 and "com_error simulado" in logs[0]
    assert backend.solids[1].center == (0.0, 0.0, 0.0)       # el estado local no se desincroniza


def test_backend_load_scene():
    backend, space, _ = _backend()
    sc = _full_scene()
    assert backend.load_scene(sc) == 6
    assert [c[0] for c in space.calls] == ["AddBox", "AddBox", "AddCylinder", "AddSphere", "AddCone", "AddWedge"]
    assert backend.selected_id == sc.selected_id


def test_dry_run_backend_records():
    b = ac.DryRunBackend()
    assert b.apply([("mode", "CREATE"), ("select", 1)]) == 2 and len(b.ops) == 2


def test_client_parser_defaults_and_udp_const():
    ns = ac.build_parser().parse_args([])
    assert ns.udp is None and ns.bus is None and ns.units == "mm" and ns.gain is None
    assert ac.build_parser().parse_args(["--udp"]).udp == "0.0.0.0:5055"
    assert ac.build_parser().parse_args(["--udp", "192.168.1.5:6000"]).udp == "192.168.1.5:6000"
    with pytest.raises(SystemExit):
        ac.build_parser().parse_args(["--udp", "--bus", "x.json"])


def test_client_main_dry_run_over_bus_file(tmp_path, capsys):
    bus_path = tmp_path / "bus.json"
    TletlStateBus(bus_path).write(_state("THREE", (0.5, 0.5), ts=1.0))
    rc = ac.main(["--bus", str(bus_path), "--dry-run", "--max-ticks", "2", "--stale", "0"], sleep=lambda s: None)
    out = capsys.readouterr().out
    assert rc == 0 and "backend: dry-run" in out and "fin: 1 frames" in out


def test_client_main_with_fake_backend_loads_scene_and_runs(tmp_path, capsys):
    bus_path = tmp_path / "bus.json"
    TletlStateBus(bus_path).write(_state("NEUTRAL", (0.5, 0.5), ts=1.0))
    scene_path = save_scene_json(_full_scene(), tmp_path / "escena.json")
    backends = []

    def factory():
        b = ac.AutoCADBackend(FakeApp())
        backends.append(b)
        return b

    rc = ac.main(["--bus", str(bus_path), "--load", str(scene_path), "--max-ticks", "2", "--stale", "0",
                  "--units", "m"], backend_factory=factory, sleep=lambda s: None)
    assert rc == 0 and len(backends) == 1 and len(backends[0].entities) == 6
    assert "dibujando escena guardada: 6" in capsys.readouterr().out


def test_client_main_without_pywin32_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "win32com", None)
    monkeypatch.setitem(sys.modules, "win32com.client", None)
    rc = ac.main(["--bus", str(tmp_path / "nope.json"), "--max-ticks", "1"], sleep=lambda s: None)
    assert rc == 2 and "pywin32" in capsys.readouterr().out


# ── session: bucle con reloj falso ───────────────────────────────────────────

def _create_and_move_frames():
    three = _state("THREE", (0.5, 0.5))
    return [three, three, three,                     # 1.0 s de THREE -> CREATE (reloj avanza 0.5 s/tick)
            _state("PINCH", (0.6, 0.4)),             # crea box #1 en (10, 10, 0) con gain 100
            _state("OPEN_PALM", (0.6, 0.4)),
            three, three, three,                     # -> TRANSFORM
            _state("PINCH", (0.5, 0.5)), None,       # None = tick sin frame
            _state("PINCH", (0.6, 0.5))]             # move +10 en x


def test_run_loop_scripted_with_fake_clock():
    clock = FakeClock()
    printed, sunk = [], []
    modeler = GestureModeler(gain=100.0, stale_after=None)
    result = ses.run_loop(ScriptedBusSource(_create_and_move_frames()), modeler, sunk.extend,
                          clock=clock, sleep=clock.sleep, printer=printed.append, status_every=1.0)
    # 11 entradas (10 frames + 1 tick sin frame); el poll que agota la fuente rompe el bucle sin contar tick
    assert result.frames == 10 and result.ticks == 11 and result.ops == 5
    assert not result.ended and not result.interrupted and clock.slept == 11
    assert [op[0] for op in sunk] == ["mode", "add", "select", "mode", "move"]
    assert sunk[1][1].center == pytest.approx((10.0, 10.0, 0.0))
    assert sunk[-1] == ("move", 1, pytest.approx(10.0), pytest.approx(0.0), 0.0)
    assert modeler.scene.get(1).center == pytest.approx((20.0, 10.0, 0.0))   # ops aplicadas a la escena
    assert any(line.startswith("[op] add #1 box") for line in printed)
    assert any(line.startswith("[cad] modo=") for line in printed)          # línea de estado


def test_run_loop_stops_on_end_session_and_max_ticks():
    clock = FakeClock()
    fist = _state("FIST", (0.5, 0.5))
    modeler = GestureModeler(gain=1.0, stale_after=None, end_hold=3.0)
    result = ses.run_loop(ScriptedBusSource([fist] * 20), modeler, None, clock=clock, sleep=clock.sleep,
                          printer=lambda s: None)
    assert result.ended and result.frames == 7           # 0, .5, ... 3.0 s -> end_session en el 7º frame
    result2 = ses.run_loop(ScriptedBusSource([fist] * 20), GestureModeler(stale_after=None), None,
                           clock=FakeClock(), sleep=lambda s: None, printer=lambda s: None, max_ticks=4)
    assert result2.ticks == 4 and not result2.ended


def test_session_exports_json_and_dxf_and_ends_by_fist(tmp_path):
    clock = FakeClock()
    printed = []
    frames = _create_and_move_frames() + [_state("FIST", (0.5, 0.5))] * 8
    modeler = GestureModeler(gain=100.0, stale_after=None)
    session = ses.CadSession(ScriptedBusSource(frames), modeler, tmp_path / "out" / "modelo.dxf",
                             units="mm", clock=clock, sleep=clock.sleep, printer=printed.append)
    assert session.run() == 0
    assert session.result is not None and session.result.ended
    json_path = tmp_path / "out" / "modelo.json"
    assert json_path.exists() and session.dxf_written == tmp_path / "out" / "modelo.dxf"
    saved = load_scene_json(json_path)
    assert len(saved) == 1 and saved.units == "mm" and saved.get(1).center == pytest.approx((20.0, 10.0, 0.0))
    doc = ezdxf.readfile(session.dxf_written)
    assert len(doc.modelspace().query("MESH[layer=='TLETL_BOX']")) == 1
    assert not doc.audit().has_errors
    assert any("DXF exportado" in line for line in printed)


def test_session_keyboard_interrupt_still_exports(tmp_path):
    class InterruptingSource(ScriptedBusSource):
        def poll(self):
            if self.finished:
                raise KeyboardInterrupt
            return super().poll()

    clock = FakeClock()
    src = InterruptingSource(_create_and_move_frames())
    session = ses.CadSession(src, GestureModeler(gain=100.0, stale_after=None), tmp_path / "m.dxf",
                             clock=clock, sleep=clock.sleep, printer=lambda s: None)
    assert session.run() == 0
    assert session.result.interrupted and (tmp_path / "m.json").exists() and (tmp_path / "m.dxf").exists()


def test_session_without_ezdxf_saves_json_and_returns_2(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ezdxf", None)
    printed = []
    modeler = GestureModeler(stale_after=None)
    modeler.scene.add("cone")
    session = ses.CadSession(ScriptedBusSource([]), modeler, tmp_path / "m.dxf",
                             clock=FakeClock(), sleep=lambda s: None, printer=printed.append)
    assert session.run() == 2
    assert (tmp_path / "m.json").exists() and not (tmp_path / "m.dxf").exists()
    assert any("pip install ezdxf" in line for line in printed) and any("dxf_export" in line for line in printed)


def test_session_main_load_resumes_scene_from_bus_file(tmp_path, capsys):
    bus_path = tmp_path / "bus.json"
    TletlStateBus(bus_path).write(_state("NEUTRAL", (0.5, 0.5), ts=1.0))
    scene_path = save_scene_json(_full_scene(), tmp_path / "previa.json")
    export = tmp_path / "salida" / "modelo.dxf"
    rc = ses.main(["--bus", str(bus_path), "--load", str(scene_path), "--export", str(export),
                   "--max-ticks", "2", "--stale", "0", "--hz", "1000"])
    assert rc == 0 and export.exists()
    assert len(load_scene_json(export.with_suffix(".json"))) == 6
    assert "escena guardada" in capsys.readouterr().out


def test_modeler_from_args_defaults_by_units():
    p = ses.build_parser()
    m = ses.modeler_from_args(p.parse_args([]))
    assert m.gain == 100.0 and m.spawn_size == 10.0 and m.end_hold == 3.0 and m.stale_after == 1.0
    m2 = ses.modeler_from_args(p.parse_args(["--units", "m", "--end-hold", "0", "--stale", "0", "--z-gain", "2"]))
    assert m2.gain == 1.0 and m2.end_hold is None and m2.stale_after is None and m2.z_gain == 2.0
    m3 = ses.modeler_from_args(p.parse_args(["--gain", "42", "--spawn-size", "3"]))
    assert m3.gain == 42.0 and m3.spawn_size == 3.0
    assert "THREE" in ses.GESTURE_TABLE and "VICTORY" in ses.GESTURE_TABLE


def test_status_line_mentions_age_and_status():
    src = ScriptedBusSource([_state("PINCH", (0.5, 0.5), ts=10.0)])
    modeler = GestureModeler(stale_after=None)
    modeler.update(src.poll(), 10.5)
    line = ses.status_line(modeler, src, 10.5)
    assert "bus=0.50s" in line and "modo=TRANSFORM" in line and "ops=0" in line


# ── freecad_macro: helpers puros ─────────────────────────────────────────────

def test_macro_imports_without_freecad_and_py_compiles():
    assert fm.FREECAD_AVAILABLE is False and fm.App is None and fm.QtCore is None
    py_compile.compile(str(ROOT / "apps" / "autocad_control" / "freecad_macro.py"), doraise=True)
    with pytest.raises(RuntimeError):
        fm.start()


def test_macro_bus_path_matches_core_rule(monkeypatch, tmp_path):
    monkeypatch.setenv(core_paths.ENV_STATE_PATH, str(tmp_path / "s.json"))
    assert fm.resolve_bus_path() == core_paths.default_bus_path()
    monkeypatch.delenv(core_paths.ENV_STATE_PATH)
    monkeypatch.setenv(core_paths.ENV_HOME, str(tmp_path / "home"))
    assert fm.resolve_bus_path() == core_paths.default_bus_path() == tmp_path / "home" / "tletl_state.json"
    monkeypatch.delenv(core_paths.ENV_HOME)
    assert fm.resolve_bus_path() == core_paths.default_bus_path() == Path.home() / ".tletl" / "tletl_state.json"


def test_macro_read_bus_tolerant(tmp_path):
    p = tmp_path / "bus.json"
    assert fm.read_bus(p) == {}
    p.write_text("{oops", encoding="utf-8")
    assert fm.read_bus(p) == {}
    TletlStateBus(p).write(_state("PINCH", (0.1, 0.2), ts=1.0))
    assert fm.read_bus(p)["dom"]["gesture"] == "PINCH"


def test_macro_local_center_and_placement_base():
    assert fm.local_center("box", (2, 4, 6)) == (1.0, 2.0, 3.0)
    assert fm.local_center("cylinder", (2, 2, 6)) == (0.0, 0.0, 3.0)
    assert fm.local_center("cone", (2, 2, 6)) == (0.0, 0.0, 3.0)
    assert fm.local_center("sphere", (2, 2, 2)) == (0.0, 0.0, 0.0)
    assert fm.local_center("wedge", (2, 2, 2)) == (0.0, 0.0, 0.0)
    # sin rotación: base = centro - centro local
    assert fm.placement_base((10, 10, 10), (1, 2, 3), 0.0) == pytest.approx((9.0, 8.0, 7.0))
    # 90°: el centro local (1,0,0) rota a (0,1,0)
    assert fm.placement_base((0, 0, 0), (1, 0, 0), math.pi / 2) == pytest.approx((0.0, -1.0, 0.0))


def test_macro_load_modeler_class(monkeypatch, tmp_path):
    monkeypatch.setenv(fm.ENV_REPO, str(ROOT))
    assert fm.load_modeler_class() is GestureModeler
    monkeypatch.setattr(fm, "repo_candidates", lambda: [tmp_path])
    assert fm.load_modeler_class() is None


def test_minimal_modeler_matches_gesture_modeler_on_move_and_spawn():
    frames = _create_and_move_frames()
    frames = [f for f in frames if f is not None]
    full = GestureModeler(gain=100.0, stale_after=None)
    mini = fm.MinimalModeler(gain=100.0, stale_after=None)
    ops_full, ops_mini = [], []
    t = 0.0
    for fr in frames:
        ops_full.extend(full.update(fr, t))
        ops_mini.extend(mini.update(fr, t))
        t += 0.5
    assert [op[0] for op in ops_mini] == [op[0] for op in ops_full] == ["mode", "add", "select", "mode", "move"]
    assert ops_mini[1][1].center == pytest.approx(ops_full[1][1].center)
    assert ops_mini[1][1].dimensions == pytest.approx(ops_full[1][1].dimensions)
    assert ops_mini[-1][1:] == pytest.approx(ops_full[-1][1:])
    assert mini.solids[0].center == pytest.approx(full.scene.get(1).center)
    assert "MinimalModeler" in mini.describe() and mini.mode == "TRANSFORM"


def test_minimal_modeler_stale_fist_and_victory():
    mini = fm.MinimalModeler(gain=10.0, stale_after=1.0)
    mini.mode = "CREATE"
    assert mini.update(_state("PINCH", (0.5, 0.5), ts=1.0), 5.0) == []       # obsoleto
    assert mini.update(_state("VICTORY", (0.5, 0.5), ts=5.1), 5.1) == [("kind", "cylinder")]
    assert mini.update(_state("FIST", (0.5, 0.5), ts=5.2), 5.2) == []
    assert "seguridad" in mini.status


# ── launcher / packaging / CLIs ──────────────────────────────────────────────

def test_launcher_exists_and_is_executable():
    launcher = ROOT / "launchers" / "tletl-cad-session.sh"
    assert launcher.exists()
    assert os.access(launcher, os.X_OK)
    text = launcher.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text and "apps.autocad_control.session" in text and '"$@"' in text


def test_pyproject_declares_cad_extra_and_package():
    import tomllib
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert cfg["project"]["optional-dependencies"]["cad"] == ["ezdxf>=1.1"]
    assert "apps.autocad_control" in cfg["tool"]["setuptools"]["packages"]


@pytest.mark.parametrize("module", ["apps.autocad_control.session", "apps.autocad_control.autocad_client",
                                    "apps.autocad_control.dxf_export"])
def test_cli_help_runs_without_pywin32(module):
    proc = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT, capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout


def test_readme_documents_three_routes():
    text = (ROOT / "apps" / "autocad_control" / "README.md").read_text(encoding="utf-8")
    for needle in ("pywin32", "ezdxf", "FreeCAD", "5055", "TLETL_BUS_UDP", "udp_target", "THREE", "VICTORY"):
        assert needle in text, needle


# ── Revisión: entidad registrada antes de rotar, select inválido, factory ────

class _RotateFailsEntity(FakeEntity):
    def Rotate3D(self, p1, p2, angle):
        raise RuntimeError("com_error simulado en Rotate3D")


class _RotateFailsSpace(FakeModelSpace):
    def _make(self, method, *args):
        self.calls.append((method, *args))
        ent = _RotateFailsEntity(method, args)
        self.entities.append(ent)
        return ent


def test_add_registers_entity_even_if_initial_rotation_fails():
    """Si Rotate3D falla tras AddBox, la entidad existe en AutoCAD: debe quedar
    registrada (con rotación 0 local) en vez de huérfana sin id."""
    app = FakeApp()
    app.ActiveDocument.ModelSpace = _RotateFailsSpace()
    log: list = []
    backend = ac.AutoCADBackend(app, log=log.append)
    solid = Solid(id=7, kind="box", center=(1.0, 2.0, 3.0), size=(1.0, 1.0, 1.0), rotation_z=0.5)
    assert backend.apply_op(("add", solid)) is True
    assert 7 in backend.entities and backend.solids[7].rotation_z == 0.0
    assert any("no rotado" in line for line in log)
    # ops posteriores sobre ese id funcionan (no hay KeyError)
    assert backend.apply_op(("move", 7, 1.0, 0.0, 0.0)) is True
    assert backend.solids[7].center == (2.0, 2.0, 3.0)


def test_select_unknown_id_counts_as_failure():
    backend, space, doc = _backend()
    assert backend.apply_op(("select", 99)) is False
    assert backend.failed == 1 and backend.applied == 0 and backend.selected_id is None


def test_main_reports_any_connection_error_and_exits_2(tmp_path):
    class ComErrorLike(Exception):
        pass

    def factory():
        raise ComErrorLike("(-2147221005, 'Cadena de clase no válida', None, None)")

    out: list = []
    rc = ac.main(["--bus", str(tmp_path / "nope.json"), "--max-ticks", "1"],
                 backend_factory=factory, printer=out.append)
    assert rc == 2
    assert any("no pude conectar con AutoCAD" in line and "ComErrorLike" in line for line in out)
