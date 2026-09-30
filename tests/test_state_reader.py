"""tests/test_state_reader.py — lector del bus para Blender (sin bpy ni tletl_core).

El bug que motiva la parte de rutas (docs/VALIDACION_FISICA_v5.md §5): la app
escribía el bus en un sitio y el addon lo leía en otro. Blender no puede importar
tletl_core, así que state_reader REPLICA la regla y aquí se verifica que ambas
implementaciones coinciden bajo todas las combinaciones de variables de entorno.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from apps.blender_control import state_reader as sr
from tletl_core import paths as core_paths


# ── default_bus_path == tletl_core.paths.default_bus_path ────────────────────

def _clear_env(monkeypatch):
    monkeypatch.delenv("TLETL_STATE_PATH", raising=False)
    monkeypatch.delenv("TLETL_HOME", raising=False)


def test_default_bus_path_matches_core_without_env(monkeypatch):
    _clear_env(monkeypatch)
    assert sr.default_bus_path() == core_paths.default_bus_path()
    assert sr.default_bus_path() == Path.home() / ".tletl" / "tletl_state.json"


def test_default_bus_path_matches_core_with_tletl_home(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TLETL_HOME", str(tmp_path / "rt"))
    assert sr.default_bus_path() == core_paths.default_bus_path()
    assert sr.default_bus_path() == tmp_path / "rt" / "tletl_state.json"


def test_default_bus_path_matches_core_with_state_path_and_tilde(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TLETL_STATE_PATH", "~/custom/dir/bus.json")
    assert sr.default_bus_path() == core_paths.default_bus_path()
    assert sr.default_bus_path() == Path.home() / "custom" / "dir" / "bus.json"


def test_state_path_wins_over_home(monkeypatch, tmp_path):
    monkeypatch.setenv("TLETL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TLETL_STATE_PATH", str(tmp_path / "explicit.json"))
    assert sr.default_bus_path() == core_paths.default_bus_path() == tmp_path / "explicit.json"


def test_blank_env_values_are_ignored(monkeypatch):
    monkeypatch.setenv("TLETL_HOME", "   ")
    monkeypatch.setenv("TLETL_STATE_PATH", "")
    assert sr.default_bus_path() == core_paths.default_bus_path()
    assert sr.tletl_home() == Path.home() / ".tletl"


# ── read_state ───────────────────────────────────────────────────────────────

def test_read_state_missing_file(tmp_path):
    assert sr.read_state(tmp_path / "nope.json") == {}


def test_read_state_corrupt_empty_and_non_dict(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json", encoding="utf-8")
    assert sr.read_state(p) == {}
    p.write_text("", encoding="utf-8")
    assert sr.read_state(p) == {}
    p.write_text("[1, 2, 3]", encoding="utf-8")   # JSON válido pero NO es un objeto
    assert sr.read_state(p) == {}
    # los helpers deben seguir funcionando con lo que devuelve read_state
    assert sr.dom_gesture(sr.read_state(p)) == "NO_HAND"


def test_read_state_directory_returns_empty(tmp_path):
    assert sr.read_state(tmp_path) == {}


def test_read_state_none_uses_default_path(monkeypatch, tmp_path):
    bus = tmp_path / "bus.json"
    bus.write_text(json.dumps({"mode": "CURSOR"}), encoding="utf-8")
    monkeypatch.setenv("TLETL_STATE_PATH", str(bus))
    assert sr.read_state(None)["mode"] == "CURSOR"
    assert sr.read_state("")["mode"] == "CURSOR"
    assert sr.read_tletl_state(None)["mode"] == "CURSOR"   # alias legacy


def test_read_state_expands_tilde(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "b.json").write_text('{"mode": "VENTANAS"}', encoding="utf-8")
    assert sr.read_state("~/b.json")["mode"] == "VENTANAS"


# ── helpers de acceso ────────────────────────────────────────────────────────

def _sample(ts=None, **extra):
    return {
        "timestamp": ts,
        "fps": 27.5,
        "dom": {"present": True, "gesture": "PINCH", "confidence": 0.87, "palm": [0.3, 0.6]},
        "mod": {"present": True, "gesture": "OPEN_PALM", "confidence": 0.71, "palm": {"x": 0.8, "y": 0.2}},
        "intent": {"name": "GRAB_OR_SELECT", "active": True},
        "mode": "NAVEGADOR",
        "selected": False,
        "extra": {"swipe": "-", "control": True, "lowlight": "off", **extra},
    }


def test_gestures_palms_and_confidence():
    st = _sample()
    assert sr.dom_gesture(st) == "PINCH"
    assert sr.mod_gesture(st) == "OPEN_PALM"
    assert sr.dom_palm(st) == (0.3, 0.6)
    assert sr.mod_palm(st) == (0.8, 0.2)          # formato dict también
    assert sr.hand_confidence(st, "dom") == pytest.approx(0.87)
    assert sr.hand_confidence(st, "mod") == pytest.approx(0.71)
    assert sr.hand_confidence({}, "dom") == 0.0
    assert sr.hand_present(st, "dom") is True
    assert sr.hand_present({}, "mod") is False


def test_palm_none_or_malformed():
    assert sr.dom_palm({"dom": {"gesture": "FIST", "palm": None}}) is None
    assert sr.dom_palm({"dom": {"palm": [0.1]}}) is None
    assert sr.dom_palm({"dom": {"palm": ["a", "b"]}}) is None
    assert sr.dom_palm({"dom": {"palm": {"x": 0.5}}}) is None


def test_legacy_hands_format():
    st = {"hands": {"dom": {"gesture": "POINT", "palm": [0.1, 0.9]}}}
    assert sr.dom_gesture(st) == "POINT"
    assert sr.dom_palm(st) == (0.1, 0.9)
    assert sr.mod_gesture(st) == "NO_HAND"


def test_intent_mode_control_fps():
    st = _sample()
    assert sr.intent_name(st) == "GRAB_OR_SELECT"
    assert sr.current_mode(st) == "NAVEGADOR"
    assert sr.control_enabled(st) is True
    assert sr.app_fps(st) == pytest.approx(27.5)
    assert sr.intent_name({}) == "NONE"
    assert sr.current_mode({}) == "NONE"
    assert sr.app_fps({}) == 0.0
    assert sr.app_fps({"fps": "x"}) == 0.0


def test_control_enabled_fallbacks():
    assert sr.control_enabled({"extra": {"control": False}, "selected": True}) is False  # extra manda
    assert sr.control_enabled({"selected": True}) is True                                # fallback
    assert sr.control_enabled({}) is False
    assert sr.control_enabled([]) is False  # type: ignore[arg-type]


def test_state_age_and_is_stale():
    now = 1_000_000.0
    fresh = _sample(ts=now - 0.2)
    old = _sample(ts=now - 5.0)
    assert sr.state_age(fresh, now) == pytest.approx(0.2)
    assert sr.state_age(old, now) == pytest.approx(5.0)
    assert sr.is_stale(fresh, 1.0, now) is False
    assert sr.is_stale(old, 1.0, now) is True
    # timestamp del futuro (relojes distintos) nunca da edad negativa
    assert sr.state_age(_sample(ts=now + 3.0), now) == 0.0


def test_state_age_inf_without_timestamp():
    for st in ({}, _sample(ts=None), _sample(ts=0), _sample(ts="ayer"), _sample(ts=True),
               _sample(ts=float("nan"))):
        assert math.isinf(sr.state_age(st, 10.0))
        assert sr.is_stale(st, 1e9, 10.0) is True
    assert sr.state_timestamp(_sample(ts=12.5)) == 12.5


def test_state_age_uses_wall_clock_by_default():
    st = _sample(ts=time.time() - 0.5)
    assert 0.4 <= sr.state_age(st) < 5.0


def test_get_transform_defaults_and_bad_values():
    t = sr.get_transform({"transform": {"x": "1.5", "scale": "no", "otro": 3}})
    assert t["x"] == 1.5 and t["scale"] == 1.0 and "otro" not in t
    assert set(sr.get_transform({}).keys()) == {"x", "y", "z", "rot_x", "rot_y", "rot_z", "scale"}


def test_summarize_state():
    assert sr.summarize_state({}) == "NO_STATE"
    txt = sr.summarize_state(_sample(ts=time.time()))
    assert "dom=PINCH" in txt and "mod=OPEN_PALM" in txt and "control=ON" in txt
