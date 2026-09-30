from __future__ import annotations

import json
import time

import pytest

from tletl_core.bus import (
    MAX_UDP_PAYLOAD,
    TletlStateBus,
    UdpStatePublisher,
    UdpStateReceiver,
    parse_udp_target,
)
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
    assert bus.age() == float("inf")


def test_bus_atomic_write_leaves_no_tmp(tmp_path):
    path = tmp_path / "tletl_state.json"
    bus = TletlStateBus(path)
    bus.write(TletlFrameState())
    assert path.exists()
    assert not (tmp_path / "tletl_state.json.tmp").exists()


def test_bus_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "dir" / "tletl_state.json"
    bus = TletlStateBus(path)
    bus.write(TletlFrameState())
    assert path.exists()


def test_bus_timestamp_is_real_even_if_state_has_zero(tmp_path):
    bus = TletlStateBus(tmp_path / "s.json")
    before = time.time()
    bus.write(TletlFrameState())          # timestamp=0.0 por default en el dataclass
    ts = bus.read()["timestamp"]
    assert ts >= before
    assert bus.age() < 5.0


def test_bus_keeps_explicit_timestamp(tmp_path):
    bus = TletlStateBus(tmp_path / "s.json")
    bus.write(TletlFrameState(timestamp=123.5))
    assert bus.read()["timestamp"] == 123.5


def test_bus_compact_by_default_and_strips_features(tmp_path):
    bus = TletlStateBus(tmp_path / "s.json")
    st = TletlFrameState()
    st.dom = TletlHandState(present=True, gesture="PINCH", features={"a": 1.0, "b": 2.0})
    bus.write(st)
    raw = (tmp_path / "s.json").read_text(encoding="utf-8")
    assert "\n" not in raw.strip()
    assert bus.read()["dom"]["features"] == {}
    # el dataclass original no se toca
    assert st.dom.features == {"a": 1.0, "b": 2.0}


def test_bus_include_features_and_pretty(tmp_path):
    bus = TletlStateBus(tmp_path / "s.json", compact=False, include_features=True)
    st = TletlFrameState()
    st.dom = TletlHandState(present=True, gesture="PINCH", features={"a": 1.0})
    bus.write(st)
    raw = (tmp_path / "s.json").read_text(encoding="utf-8")
    assert "\n" in raw
    assert bus.read()["dom"]["features"] == {"a": 1.0}


def test_bus_write_accepts_dict_without_mutating_it(tmp_path):
    bus = TletlStateBus(tmp_path / "s.json")
    src = {"dom": {"gesture": "FIST", "features": {"x": 1}}, "mode": "NONE"}
    bus.write(src)
    assert "timestamp" not in src
    assert src["dom"]["features"] == {"x": 1}
    assert bus.read()["dom"]["gesture"] == "FIST"


def test_bus_read_corrupt_or_empty_returns_empty(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{not json", encoding="utf-8")
    assert TletlStateBus(path).read() == {}
    path.write_text("", encoding="utf-8")
    assert TletlStateBus(path).read() == {}
    path.write_text("[1,2,3]", encoding="utf-8")
    assert TletlStateBus(path).read() == {}


def test_bus_rejects_unknown_types(tmp_path):
    bus = TletlStateBus(tmp_path / "s.json")
    with pytest.raises(TypeError):
        bus.write(42)  # type: ignore[arg-type]


# ── UDP ──────────────────────────────────────────────────────────────────────

def test_parse_udp_target():
    assert parse_udp_target("") is None
    assert parse_udp_target(None) is None
    assert parse_udp_target("192.168.1.7:5055") == ("192.168.1.7", 5055)
    with pytest.raises(ValueError):
        parse_udp_target("sin-puerto")
    with pytest.raises(ValueError):
        parse_udp_target("host:abc")


def test_udp_publisher_to_receiver_loopback(tmp_path):
    rx = UdpStateReceiver("127.0.0.1", 0)          # puerto libre
    try:
        host, port = rx.address
        bus = TletlStateBus(tmp_path / "s.json", udp_target=f"{host}:{port}")
        st = TletlFrameState()
        st.dom = TletlHandState(present=True, gesture="PINCH", palm=(0.4, 0.6))
        bus.write(st)
        bus.write(st)
        got = None
        for _ in range(50):
            got = rx.poll()
            if got:
                break
            time.sleep(0.02)
        assert got is not None, "no llegó ningún datagrama por loopback"
        assert got["dom"]["gesture"] == "PINCH"
        assert got["dom"]["palm"] == [0.4, 0.6]
        assert rx.received >= 1
        assert bus.udp is not None and bus.udp.sent == 2
        bus.close()
        assert bus.udp is None
    finally:
        rx.close()


def test_udp_receiver_ignores_garbage():
    rx = UdpStateReceiver("127.0.0.1", 0)
    try:
        pub = UdpStatePublisher(rx.address)
        pub.send(b"\xff\xfe no es json")
        pub.send(json.dumps([1, 2]).encode())
        time.sleep(0.05)
        assert rx.poll() is None
        assert rx.malformed >= 1
        pub.close()
    finally:
        rx.close()


def test_udp_publisher_drops_oversized_payload():
    rx = UdpStateReceiver("127.0.0.1", 0)
    try:
        pub = UdpStatePublisher(rx.address)
        assert pub.send("x" * (MAX_UDP_PAYLOAD + 1)) is False
        assert pub.dropped == 1 and pub.last_error
        pub.close()
    finally:
        rx.close()
