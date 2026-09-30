"""tools/build_blender_addon.py — empaqueta el addon de Blender como zip instalable.

Genera `dist/tletl_blender_addon-<version>.zip` con esta estructura:

    tletl_blender_addon/
        __init__.py             copia de apps/blender_control/tletl_blender_addon.py
        state_reader.py         lector del bus (sin bpy ni tletl_core)
        blender_manifest.toml   manifiesto de extensión (Blender 4.2+)

El MISMO zip se instala de dos maneras:
  * addon legacy:  Edit > Preferences > Add-ons > (˅) Install from Disk…
  * extensión:     Edit > Preferences > Get Extensions > (˅) Install from Disk…

Motivo (docs/VALIDACION_FISICA_v5.md §5): instalar el .py suelto tronaba con
`ModuleNotFoundError: state_reader`. Ahora el zip lleva ambos archivos y el
addon usa `__package__` como id, así funciona en los dos formatos.

Sólo stdlib: se puede correr como script (`python tools/build_blender_addon.py`)
o como módulo (`python -m tools.build_blender_addon`). `dist/` está en .gitignore.
"""
from __future__ import annotations

import argparse
import ast
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
ADDON_DIR = ROOT / "apps" / "blender_control"
ADDON_SOURCE = ADDON_DIR / "tletl_blender_addon.py"
READER_SOURCE = ADDON_DIR / "state_reader.py"
DEFAULT_OUT_DIR = ROOT / "dist"

PACKAGE_NAME = "tletl_blender_addon"
MANIFEST_NAME = "blender_manifest.toml"
ADDON_NAME = "Tletl Gesture Control"
# Blender exige tagline <= 64 caracteres y sin punto final.
ADDON_TAGLINE = "Controla y crea objetos 3D con gestos de mano (bus Tletl)"
ADDON_MAINTAINER = "Tletl Project"
BLENDER_VERSION_MIN = "4.2.0"
ADDON_LICENSE = "SPDX:MIT"
ADDON_TAGS = ("Object", "3D View")

ZIP_MEMBERS = (
    f"{PACKAGE_NAME}/__init__.py",
    f"{PACKAGE_NAME}/state_reader.py",
    f"{PACKAGE_NAME}/{MANIFEST_NAME}",
)


def read_bl_info(source: Path = ADDON_SOURCE) -> Dict[str, Any]:
    """Extrae `bl_info` del addon SIN importarlo (nada de bpy ni side effects)."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "bl_info" in targets:
                info = ast.literal_eval(node.value)
                if not isinstance(info, dict):
                    break
                return info
    raise ValueError(f"no encontré `bl_info = {{...}}` en {source}")


def addon_version(source: Path = ADDON_SOURCE) -> str:
    """'5.2.0' a partir de bl_info['version'] == (5, 2, 0)."""
    version = read_bl_info(source).get("version")
    if not isinstance(version, (tuple, list)) or len(version) != 3 \
            or not all(isinstance(v, int) and v >= 0 for v in version):
        raise ValueError(f"bl_info['version'] debe ser (major, minor, patch), recibí {version!r}")
    return ".".join(str(v) for v in version)


def _toml_str(text: str) -> str:
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_manifest(version: str, *, addon_id: str = PACKAGE_NAME, name: str = ADDON_NAME,
                    tagline: str = ADDON_TAGLINE, maintainer: str = ADDON_MAINTAINER,
                    blender_version_min: str = BLENDER_VERSION_MIN,
                    license_id: str = ADDON_LICENSE, tags: tuple = ADDON_TAGS) -> str:
    """Contenido de blender_manifest.toml (esquema 1.0.0 de extensiones de Blender)."""
    if len(tagline) > 64 or tagline.endswith("."):
        raise ValueError("tagline: máximo 64 caracteres y sin punto final (regla de Blender)")
    tag_list = ", ".join(_toml_str(t) for t in tags)
    return (
        "schema_version = \"1.0.0\"\n"
        f"id = {_toml_str(addon_id)}\n"
        f"version = {_toml_str(version)}\n"
        f"name = {_toml_str(name)}\n"
        f"tagline = {_toml_str(tagline)}\n"
        f"maintainer = {_toml_str(maintainer)}\n"
        "type = \"add-on\"\n"
        f"blender_version_min = {_toml_str(blender_version_min)}\n"
        f"license = [{_toml_str(license_id)}]\n"
        f"tags = [{tag_list}]\n"
    )


def zip_name(version: str) -> str:
    return f"{PACKAGE_NAME}-{version}.zip"


def build(out_dir: Path | str = DEFAULT_OUT_DIR, *, version: Optional[str] = None,
          addon_source: Path = ADDON_SOURCE, reader_source: Path = READER_SOURCE) -> Path:
    """Construye el zip y devuelve su ruta. Sobrescribe si ya existe."""
    for src in (addon_source, reader_source):
        if not src.is_file():
            raise FileNotFoundError(f"falta {src}")
    version = version or addon_version(addon_source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / zip_name(version)
    tmp = target.with_suffix(".zip.tmp")

    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(addon_source, f"{PACKAGE_NAME}/__init__.py")
        zf.write(reader_source, f"{PACKAGE_NAME}/state_reader.py")
        zf.writestr(f"{PACKAGE_NAME}/{MANIFEST_NAME}", render_manifest(version))
    tmp.replace(target)
    return target


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Empaqueta el addon de Blender de Tletl como zip.")
    p.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="directorio de salida (default: dist/)")
    p.add_argument("--print-manifest", action="store_true", help="imprime el manifiesto y sale")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    version = addon_version()
    if args.print_manifest:
        sys.stdout.write(render_manifest(version))
        return 0
    target = build(args.out, version=version)
    print(f"[ok] addon v{version} -> {target}")
    print("     instala el zip en Blender: Preferences > Add-ons > Install from Disk… "
          "(o Get Extensions > Install from Disk…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
