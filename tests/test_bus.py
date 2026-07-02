from __future__ import annotations

from tletl_core.bus import TletlStateBus
from tletl_core.state import TletlFrameState, TletlHandState


def test_bus_roundtrip(tmp_path):
    path = tmp_path / "tletl_state.json"
    bus = TletlStateBus(path)
    st = TletlFrameState()
    st.dom = TletlHandState(present=True, gesture="PINCH")
    st.mode = "CURSOR"
    bus.write(st)
    data = bus.read()
    assert data["mode"] == "CURSOR"
    assert data["dom"]["gesture"] == "PINCH"
    assert "timestamp" in data
    assert "transform" in data


def test_bus_read_missing_returns_empty(tmp_path):
    bus = TletlStateBus(tmp_path / "nope.json")
    assert bus.read() == {}


def test_bus_atomic_write_leaves_no_tmp(tmp_path):
    path = tmp_path / "tletl_state.json"
    bus = TletlStateBus(path)
    bus.write(TletlFrameState())
    assert path.exists()
    assert not (tmp_path / "tletl_state.json.tmp").exists()
