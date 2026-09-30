"""tests/test_blender_addon_build.py — empaquetado del addon y auto-suficiencia.

Cubre los dos bugs de instalación de docs/VALIDACION_FISICA_v5.md §5:
  * el .py suelto tronaba con ModuleNotFoundError: state_reader -> copia inline;
  * el zip debe traer ambos archivos y funcionar como paquete (legacy y extensión).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import py_compile
import subprocess
import sys
import tomllib
import unittest.mock as mock
import zipfile
from pathlib import Path

import pytest

from tools import build_blender_addon as build

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "launchers" / "tletl-build-blender-addon.sh"


def _import_addon(blocked: dict):
    """Importa el addon del repo forzando que los módulos en `blocked` fallen."""
    for key in list(sys.modules):
        if key.endswith("tletl_blender_addon"):
            del sys.modules[key]
    with mock.patch.dict(sys.modules, blocked):
        return importlib.import_module("apps.blender_control.tletl_blender_addon")


# ── bl_info / versión ────────────────────────────────────────────────────────

def test_bl_info_parsed_without_import():
    info = build.read_bl_info()
    assert info["name"] == "Tletl Gesture Control"
    assert isinstance(info["version"], tuple) and len(info["version"]) == 3


def test_version_matches_addon_constant():
    addon = _import_addon({"bpy": None})
    assert build.addon_version() == addon.ADDON_VERSION
    assert build.addon_version().count(".") == 2


def test_bad_version_rejected(tmp_path):
    src = tmp_path / "a.py"
    src.write_text('bl_info = {"version": (1, 2)}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        build.addon_version(src)
    src.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build.read_bl_info(src)


# ── manifiesto ───────────────────────────────────────────────────────────────

def test_manifest_is_valid_extension_manifest():
    data = tomllib.loads(build.render_manifest("5.2.0"))
    assert data["schema_version"] == "1.0.0"
    assert data["id"] == "tletl_blender_addon"
    assert data["version"] == "5.2.0"
    assert data["type"] == "add-on"
    assert data["blender_version_min"] == "4.2.0"
    assert data["license"] == ["SPDX:MIT"]
    assert data["tags"] == ["Object", "3D View"]
    assert data["name"] and data["maintainer"]
    assert len(data["tagline"]) <= 64 and not data["tagline"].endswith(".")


def test_manifest_rejects_long_tagline():
    with pytest.raises(ValueError):
        build.render_manifest("1.0.0", tagline="x" * 65)


# ── build del zip ────────────────────────────────────────────────────────────

def test_build_zip_members_and_compile(tmp_path):
    target = build.build(tmp_path / "dist")
    assert target.exists()
    assert target.name == f"tletl_blender_addon-{build.addon_version()}.zip"
    assert not target.with_suffix(".zip.tmp").exists()

    with zipfile.ZipFile(target) as zf:
        names = sorted(zf.namelist())
        assert names == sorted(build.ZIP_MEMBERS)
        assert zf.testzip() is None
        manifest = tomllib.loads(zf.read("tletl_blender_addon/blender_manifest.toml").decode("utf-8"))
        assert manifest["version"] == build.addon_version()
        # el __init__.py es el addon completo (con bl_info para el instalador legacy)
        init_src = zf.read("tletl_blender_addon/__init__.py").decode("utf-8")
        assert "bl_info" in init_src and "def register()" in init_src
        zf.extractall(tmp_path / "x")

    init_path = tmp_path / "x" / "tletl_blender_addon" / "__init__.py"
    py_compile.compile(str(init_path), doraise=True)
    py_compile.compile(str(tmp_path / "x" / "tletl_blender_addon" / "state_reader.py"), doraise=True)


def test_extracted_package_imports_as_blender_would(tmp_path):
    """El zip instalado es un PAQUETE: el import relativo de state_reader debe
    funcionar y el id del addon debe ser el nombre del paquete."""
    target = build.build(tmp_path / "dist")
    with zipfile.ZipFile(target) as zf:
        zf.extractall(tmp_path / "addons")
    pkg_dir = tmp_path / "addons" / "tletl_blender_addon"

    name = "tletl_blender_addon"
    spec = importlib.util.spec_from_file_location(
        name, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)])
    module = importlib.util.module_from_spec(spec)
    saved = {k: sys.modules.get(k) for k in (name, f"{name}.state_reader", "bpy")}
    try:
        sys.modules["bpy"] = None            # type: ignore[assignment]  # fuera de Blender
        sys.modules[name] = module
        spec.loader.exec_module(module)      # type: ignore[union-attr]
        assert module.STATE_READER_SOURCE == "package"
        assert module._ADDON_ID == name
        assert module.bpy is None
        assert not hasattr(module, "register")   # register sólo existe con bpy
        assert module.read_state(tmp_path / "nope.json") == {}
        assert module.GestureSession().mode == "TRANSFORM"
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_build_overwrites_existing_zip(tmp_path):
    out = tmp_path / "dist"
    first = build.build(out)
    first.write_bytes(b"basura")
    second = build.build(out)
    assert second == first and zipfile.is_zipfile(second)


def test_build_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build.build(tmp_path, reader_source=tmp_path / "missing.py")


def test_cli_script_mode(tmp_path):
    """`python tools/build_blender_addon.py --out DIR` funciona como script suelto."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_blender_addon.py"), "--out", str(tmp_path / "d")],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[ok]" in proc.stdout
    assert (tmp_path / "d" / f"tletl_blender_addon-{build.addon_version()}.zip").exists()


def test_cli_print_manifest(capsys):
    assert build.main(["--print-manifest"]) == 0
    out = capsys.readouterr().out
    assert tomllib.loads(out)["id"] == "tletl_blender_addon"


def test_launcher_exists_and_calls_module():
    assert LAUNCHER.exists()
    assert os.access(LAUNCHER, os.X_OK)
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "tools.build_blender_addon" in text
    assert text.startswith("#!/usr/bin/env bash")


# ── archivo suelto: copia inline de state_reader ─────────────────────────────

_INLINE_BLOCKED = {"bpy": None, "state_reader": None, "apps.blender_control.state_reader": None}


def _sample_state():
    return {
        "timestamp": 1_000.0, "fps": 29.0, "mode": "CURSOR", "selected": False,
        "dom": {"present": True, "gesture": "PINCH", "confidence": 0.9, "palm": [0.25, 0.75]},
        "mod": {"present": True, "gesture": "OPEN_PALM", "confidence": 0.8, "palm": {"x": 0.6, "y": 0.4}},
        "intent": {"name": "GRAB_OR_SELECT", "active": True},
        "extra": {"control": True},
    }


def test_addon_imports_without_state_reader_module():
    """Simula el .py suelto en Blender: sin state_reader disponible en ninguna forma."""
    addon = _import_addon(_INLINE_BLOCKED)
    assert addon.STATE_READER_SOURCE == "inline"
    st = _sample_state()
    assert addon.dom_gesture(st) == "PINCH"
    assert addon.mod_gesture(st) == "OPEN_PALM"
    assert addon.dom_palm(st) == (0.25, 0.75)
    assert addon.mod_palm(st) == (0.6, 0.4)
    assert addon.intent_name(st) == "GRAB_OR_SELECT"
    assert addon.hand_confidence(st, "dom") == pytest.approx(0.9)
    assert addon.control_enabled(st) is True
    assert addon.app_fps(st) == pytest.approx(29.0)
    assert addon.state_age(st, 1_000.5) == pytest.approx(0.5)
    assert addon.is_stale(st, 0.2, 1_000.5) is True
    assert addon.is_stale(st, 1.0, 1_000.5) is False
    assert addon.state_age({}, 5.0) == float("inf")
    # el mapper de siempre funciona sobre la copia inline
    m = addon.BlenderGestureMapper(gain=4.0, smoothing=1.0)
    m.update(st, addon.normalize_transform({}))
    st2 = _sample_state(); st2["dom"]["palm"] = [0.45, 0.75]
    assert m.update(st2, addon.normalize_transform({}))["x"] == pytest.approx(0.8)


def test_inline_reader_matches_package_reader(tmp_path, monkeypatch):
    """La copia inline y state_reader.py deben dar lo mismo (incluida la ruta del bus)."""
    addon = _import_addon(_INLINE_BLOCKED)
    sr = importlib.import_module("apps.blender_control.state_reader")

    samples = [_sample_state(), {}, {"dom": {"gesture": "FIST", "palm": None}, "timestamp": 0},
               {"hands": {"dom": {"gesture": "POINT", "palm": [0.1, 0.2]}}, "selected": True}]
    for st in samples:
        for fn in ("dom_gesture", "mod_gesture", "dom_palm", "mod_palm", "intent_name",
                   "control_enabled", "app_fps", "current_mode"):
            assert getattr(addon, fn)(st) == getattr(sr, fn)(st), fn
        for side in ("dom", "mod"):
            assert addon.hand_confidence(st, side) == sr.hand_confidence(st, side)
        assert addon.state_age(st, 2_000.0) == sr.state_age(st, 2_000.0)
        assert addon.is_stale(st, 1.0, 2_000.0) == sr.is_stale(st, 1.0, 2_000.0)

    for env in ({}, {"TLETL_HOME": str(tmp_path / "h")}, {"TLETL_STATE_PATH": "~/x/bus.json"}):
        monkeypatch.delenv("TLETL_HOME", raising=False)
        monkeypatch.delenv("TLETL_STATE_PATH", raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        assert addon.default_bus_path() == sr.default_bus_path()

    p = tmp_path / "s.json"
    p.write_text("[1,2]", encoding="utf-8")
    assert addon.read_state(p) == sr.read_state(p) == {}
    p.write_text('{"mode": "X"}', encoding="utf-8")
    assert addon.read_state(p) == sr.read_state(p) == {"mode": "X"}
    assert addon.read_state(tmp_path / "nope.json") == {}


def test_sibling_state_reader_is_used_when_available(tmp_path):
    """Dos archivos copiados a mano en la carpeta de addons: se usa el vecino."""
    sr_path = ROOT / "apps" / "blender_control" / "state_reader.py"
    spec = importlib.util.spec_from_file_location("state_reader", sr_path)
    sibling = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sibling)  # type: ignore[union-attr]
    addon = _import_addon({"bpy": None, "apps.blender_control.state_reader": None,
                           "state_reader": sibling})
    assert addon.STATE_READER_SOURCE == "sibling"
    assert addon.read_state is sibling.read_state
