from __future__ import annotations

"""Integración end-to-end del contrato del bus: app Fedora -> bus -> Blender.

Verifica que el JSON que escribe la app (vía TletlFrameState/TletlHandState) es
exactamente lo que el state_reader y el mapper de Blender esperan leer. Este es
el único canal entre las dos apps, así que su contrato debe estar blindado.
"""

import importlib
import sys
import unittest.mock as mock

from tletl_core.bus import TletlStateBus
from tletl_core.state import TletlFrameState, TletlHandState
from tletl_core.intent import gesture_to_common_intent


def _load_blender_mapper():
    for key in list(sys.modules.keys()):
        if "tletl_blender_addon" in key:
            del sys.modules[key]
    with mock.patch.dict(sys.modules, {"bpy": None}):
        mod = importlib.import_module("apps.blender_control.tletl_blender_addon")
    return mod


def _write_frame(bus, dom_gesture, palm, *, side="Right"):
    st = TletlFrameState(version=5)
    st.dom = TletlHandState(present=True, side=side, gesture=dom_gesture,
                            stable_gesture=dom_gesture, critic_ok=True,
                            confidence=0.9, palm=palm)
    st.intent = gesture_to_common_intent(st)
    bus.write(st)


def test_app_to_bus_to_blender_translation(tmp_path):
    """Un PINCH que se mueve a la derecha entre dos frames del bus debe trasladar +x."""
    addon = _load_blender_mapper()
    mapper = addon.BlenderGestureMapper(gain=4.0, smoothing=1.0)
    sr = importlib.import_module("apps.blender_control.state_reader")

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
    addon = _load_blender_mapper()
    sr = importlib.import_module("apps.blender_control.state_reader")
    mapper = addon.BlenderGestureMapper()
    bus = TletlStateBus(tmp_path / "s.json")

    _write_frame(bus, "FIST", (0.8, 0.2))
    state = sr.read_state(tmp_path / "s.json")
    assert sr.dom_gesture(state) == "FIST"
    out = mapper.update(state, {"x": 5.0, "y": 5.0, "z": 0.0,
                                "rot_x": 0, "rot_y": 0, "rot_z": 0, "scale": 1.0})
    assert out["x"] == 5.0 and out["y"] == 5.0  # no se movió


def test_bus_palm_survives_json_roundtrip(tmp_path):
    """La palma (tuple en el dataclass) debe sobrevivir como [x,y] legible por state_reader."""
    sr = importlib.import_module("apps.blender_control.state_reader")
    bus = TletlStateBus(tmp_path / "s.json")
    _write_frame(bus, "PINCH", (0.33, 0.66))
    state = sr.read_state(tmp_path / "s.json")
    palm = sr.dom_palm(state)
    assert palm is not None
    assert abs(palm[0] - 0.33) < 1e-9 and abs(palm[1] - 0.66) < 1e-9
