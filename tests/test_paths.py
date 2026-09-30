"""Tests de tletl_core/paths.py y de los helpers de rutas de config.py.

El bug que motiva estos tests: la app escribía el bus en <cwd>/tletl_state.json
y el addon de Blender leía ~/tletl_state.json. Ahora hay UNA regla.
"""

from __future__ import annotations

from pathlib import Path

from tletl_core import paths
from tletl_core.config import (
    adaptive_path_from_config,
    bank_path_from_config,
    bus_path_from_config,
    load_config,
)


def test_default_bus_lives_in_tletl_home(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ENV_STATE_PATH, raising=False)
    monkeypatch.setenv(paths.ENV_HOME, str(tmp_path / "home"))
    assert paths.default_bus_path() == tmp_path / "home" / "tletl_state.json"
    assert paths.default_adaptive_path() == tmp_path / "home" / "tletl_adaptive_runtime.json"


def test_default_home_is_dot_tletl(monkeypatch):
    monkeypatch.delenv(paths.ENV_HOME, raising=False)
    monkeypatch.delenv(paths.ENV_STATE_PATH, raising=False)
    assert paths.tletl_home() == Path.home() / ".tletl"
    assert paths.default_bus_path().parent == Path.home() / ".tletl"


def test_env_state_path_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_STATE_PATH, str(tmp_path / "custom.json"))
    assert paths.default_bus_path() == tmp_path / "custom.json"
    cfg = load_config(tmp_path / "nope.toml")
    assert bus_path_from_config(cfg) == tmp_path / "custom.json"


def test_bus_path_from_config_prefers_bus_section(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ENV_STATE_PATH, raising=False)
    cfg = load_config(tmp_path / "nope.toml")
    cfg["bus"]["path"] = "~/mi_bus.json"
    cfg["blender"]["bus_path"] = "otro.json"
    assert bus_path_from_config(cfg) == Path("~/mi_bus.json").expanduser()


def test_bus_path_legacy_blender_alias(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ENV_STATE_PATH, raising=False)
    cfg = load_config(tmp_path / "nope.toml")
    cfg["bus"]["path"] = ""
    cfg["blender"]["bus_path"] = "runtime/legacy.json"   # relativo -> raíz del repo
    assert bus_path_from_config(cfg) == paths.repo_root() / "runtime" / "legacy.json"


def test_adaptive_never_next_to_bank(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ENV_ADAPTIVE_PATH, raising=False)
    monkeypatch.setenv(paths.ENV_HOME, str(tmp_path))
    cfg = load_config(tmp_path / "nope.toml")
    p = adaptive_path_from_config(cfg)
    assert p.parent != paths.repo_root() / "datasets"
    assert p == tmp_path / "tletl_adaptive_runtime.json"


def test_bank_default_is_repo_dataset(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ENV_BANK, raising=False)
    cfg = load_config(tmp_path / "nope.toml")
    assert bank_path_from_config(cfg) == paths.repo_root() / "datasets" / "tletl_gesture_bank_v2_features.jsonl"
    assert bank_path_from_config(cfg).exists()


def test_resolve_path_relative_and_empty(tmp_path):
    assert paths.resolve_path("", tmp_path / "d.json") == tmp_path / "d.json"
    assert paths.resolve_path(None, tmp_path / "d.json") == tmp_path / "d.json"
    assert paths.resolve_path("x.json", tmp_path / "d.json", relative_to=tmp_path) == tmp_path / "x.json"
    assert paths.resolve_path("/abs/x.json", tmp_path / "d.json", relative_to=tmp_path) == Path("/abs/x.json")


def test_ensure_parent_creates_dir(tmp_path):
    p = tmp_path / "a" / "b" / "c.json"
    assert paths.ensure_parent(p) == p
    assert p.parent.is_dir()
