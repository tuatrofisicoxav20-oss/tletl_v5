from __future__ import annotations

"""Integración end-to-end del contrato del bus: app de cámara -> bus -> Blender.

Verifica que el JSON que escribe la app (vía TletlFrameState/TletlHandState y el
TletlStateBus REAL) es exactamente lo que state_reader, el mapper y la sesión de
Blender esperan leer. Este es el único canal entre las dos apps, así que su
contrato debe estar blindado: rutas por default, timestamps reales (bus
obsoleto), palmas, control ON/OFF y el modo CREATE.
"""

import importlib
import sys
import time
import unittest.mock as mock

from tletl_core.bus import TletlStateBus
from tletl_core.state import TletlFrameState, TletlHandState
from tletl_core.intent import gesture_to_common_intent


def _load_blender_addon():
    for key in list(sys.modules.keys()):
        if "tletl_blender_addon" in key:
            del sys.modules[key]
    with mock.patch.dict(sys.modules, {"bpy": None}):
        mod = importlib.import_module("apps.blender_control.tletl_blender_addon")
    return mod


def _reader():
    return importlib.import_module("apps.blender_control.state_reader")


def _frame(dom_gesture, palm, *, side="Right", timestamp=0.0, control=False,
           mod_gesture=None, mod_palm=None):
    st = TletlFrameState(version=5, timestamp=timestamp, fps=30.0)
    st.dom = TletlHandState(present=True, side=side, gesture=dom_gesture,
                            stable_gesture=dom_gesture, critic_ok=True,
                            confidence=0.9, palm=palm)
    if mod_gesture:
        st.mod = TletlHandState(present=True, side="Left", gesture=mod_gesture,
                                stable_gesture=mod_gesture, critic_ok=True,
                                confidence=0.8, palm=mod_palm)
    st.extra = {"swipe": "-", "control": control, "lowlight": "off"}
    st.intent = gesture_to_common_intent(st)
    return st


def _write_frame(bus, dom_gesture, palm, **kw):
    bus.write(_frame(dom_gesture, palm, **kw))


def test_app_to_bus_to_blender_translation(tmp_path):
    """Un PINCH que se mueve a la derecha entre dos frames del bus debe trasladar +x."""
    addon = _load_blender_addon()
    mapper = addon.BlenderGestureMapper(gain=4.0, smoothing=1.0)
    sr = _reader()

    bus = TletlStateBus(tmp_path / "tletl_state.json")
    current = addon.normalize_transform({})

    # frame 1: PINCH en el centro -> fija continuidad, no mueve
    _write_frame(bus, "PINCH", (0.5, 0.5))
    state1 = sr.read_state(tmp_path / "tletl_state.json")
    current = mapper.update(state1, current)
    assert current["x"] == 0.0

    # frame 2: PINCH desplazado a la derecha -> objeto se traslada +x
    _write_frame(bus, "PINCH", (0.7, 0.5))
    state2 = sr.read_state(tmp_path / "tletl_state.json")
    current = mapper.update(state2, current)
    assert current["x"] > 0.0


def test_app_to_bus_fist_is_safety(tmp_path):
    """FIST escrito por la app debe leerse como safety stop en Blender."""
    addon = _load_blender_addon()
    sr = _reader()
    mapper = addon.BlenderGestureMapper()
    bus = TletlStateBus(tmp_path / "s.json")

    _write_frame(bus, "FIST", (0.8, 0.2))
    state = sr.read_state(tmp_path / "s.json")
    assert sr.dom_gesture(state) == "FIST"
    assert sr.intent_name(state) == "SAFETY_STOP"
    out = mapper.update(state, {"x": 5.0, "y": 5.0, "z": 0.0,
                                "rot_x": 0, "rot_y": 0, "rot_z": 0, "scale": 1.0})
    assert out["x"] == 5.0 and out["y"] == 5.0  # no se movió


def test_bus_palm_survives_json_roundtrip(tmp_path):
    """La palma (tuple en el dataclass) debe sobrevivir como [x,y] legible por state_reader."""
    sr = _reader()
    bus = TletlStateBus(tmp_path / "s.json")
    _write_frame(bus, "PINCH", (0.33, 0.66))
    state = sr.read_state(tmp_path / "s.json")
    palm = sr.dom_palm(state)
    assert palm is not None
    assert abs(palm[0] - 0.33) < 1e-9 and abs(palm[1] - 0.66) < 1e-9


def test_default_bus_path_is_shared_by_app_and_addon(monkeypatch, tmp_path):
    """El bug de §5: con defaults, la app y Blender deben ver el MISMO archivo."""
    monkeypatch.delenv("TLETL_STATE_PATH", raising=False)
    monkeypatch.setenv("TLETL_HOME", str(tmp_path / "runtime"))
    addon = _load_blender_addon()
    sr = _reader()

    bus = TletlStateBus()                 # sin ruta explícita: default del core
    _write_frame(bus, "POINT", (0.4, 0.4))
    assert bus.path == sr.default_bus_path() == addon.default_bus_path()
    assert sr.dom_gesture(sr.read_state(None)) == "POINT"   # default del lector
    settings = addon.AddonSettings.from_prefs(None)
    settings.bus_path = ""
    assert settings.resolved_bus_path() == bus.path


def test_fresh_bus_is_connected_and_stale_bus_blocks_action(tmp_path):
    """Timestamps reales: un frame recién escrito conecta; uno viejo es BUS OBSOLETO."""
    addon = _load_blender_addon()
    sr = _reader()
    path = tmp_path / "s.json"
    bus = TletlStateBus(path)

    _write_frame(bus, "PINCH", (0.5, 0.5))            # timestamp=0.0 -> el bus pone time.time()
    state = sr.read_state(path)
    assert sr.state_age(state) < 5.0
    code, msg = addon.bus_status(state, path.exists(), time.time(), 1.0, path=str(path))
    assert code == addon.BUS_OK and msg.startswith("Conectado (30 fps app")

    # la app se cayó hace 5 s (frame con timestamp explícito viejo)
    _write_frame(bus, "PINCH", (0.9, 0.5), timestamp=time.time() - 5.0)
    old = sr.read_state(path)
    assert sr.is_stale(old, 1.0)
    code, msg = addon.bus_status(old, True, time.time(), 1.0, path=str(path))
    assert code == addon.BUS_STALE and msg.startswith("BUS OBSOLETO (5.")

    # el addon resetea la sesión y NO aplica ese PINCH desplazado cuando el bus vuelve
    session = addon.GestureSession(mapper=addon.BlenderGestureMapper(gain=10.0, smoothing=1.0))
    session.tick(state, addon.normalize_transform({}))       # ceba con el frame fresco (0.5, 0.5)
    session.reset()                                          # lo que hace el timer ante BUS_STALE
    _write_frame(bus, "PINCH", (0.9, 0.5))
    cmds = session.tick(sr.read_state(path), addon.normalize_transform({}))
    assert cmds == [("transform", addon.normalize_transform({}))]   # sin salto de 0.5 -> 0.9

    # bus ausente
    code, msg = addon.bus_status({}, False, time.time(), 1.0, path=str(tmp_path / "nope.json"))
    assert code == addon.BUS_MISSING and msg == f"Sin bus en {tmp_path / 'nope.json'}"


def test_control_flag_and_confidence_reach_the_panel(tmp_path):
    addon = _load_blender_addon()
    sr = _reader()
    bus = TletlStateBus(tmp_path / "s.json")
    _write_frame(bus, "PINCH", (0.5, 0.5), control=True, mod_gesture="OPEN_PALM", mod_palm=(0.2, 0.2))
    state = sr.read_state(tmp_path / "s.json")
    assert sr.control_enabled(state) is True
    assert sr.hand_confidence(state, "dom") == 0.9
    assert addon.status_lines(state) == ["Control app: ON", "Dom: PINCH (0.90)", "Mod: OPEN_PALM (0.80)",
                                         "Intent: GRAB_OR_SELECT", "Modo app: NONE"]
    _write_frame(bus, "NEUTRAL", (0.5, 0.5), control=False)
    assert sr.control_enabled(sr.read_state(tmp_path / "s.json")) is False


def test_create_mode_spawn_through_real_bus(tmp_path):
    """Modo CREATE: NEUTRAL -> PINCH escrito por la app crea una primitiva en la palma."""
    addon = _load_blender_addon()
    sr = _reader()
    path = tmp_path / "s.json"
    bus = TletlStateBus(path)
    session = addon.GestureSession(mapper=addon.BlenderGestureMapper(gain=10.0, smoothing=1.0))
    session.set_mode("CREATE")
    cursor = (1.0, -1.0, 0.5)

    _write_frame(bus, "NEUTRAL", (0.5, 0.5))
    assert session.tick(sr.read_state(path), None, cursor=cursor) == []
    _write_frame(bus, "PINCH", (0.6, 0.3))
    cmds = session.tick(sr.read_state(path), None, cursor=cursor)
    assert len(cmds) == 1 and cmds[0][:2] == ("spawn", "cube")
    x, y, z = cmds[0][2]
    assert abs(x - 2.0) < 1e-9 and abs(y - 1.0) < 1e-9 and z == 0.5

    # PINCH sostenido en frames sucesivos: no crea más
    _write_frame(bus, "PINCH", (0.65, 0.3))
    assert session.tick(sr.read_state(path), None, cursor=cursor) == []

    # VICTORY escrito por la app cicla la primitiva; el siguiente PINCH crea una esfera
    _write_frame(bus, "VICTORY", (0.5, 0.5))
    assert session.tick(sr.read_state(path), None, cursor=cursor) == [("primitive", "sphere")]
    _write_frame(bus, "PINCH", (0.5, 0.5))
    assert session.tick(sr.read_state(path), None, cursor=cursor) == [("spawn", "sphere", cursor)]
    assert session.spawn_count == 2
