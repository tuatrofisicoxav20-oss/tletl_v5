"""apps/autocad_control/dxf_export.py — escena de sólidos -> DXF (ruta 2: offline).

Cada `Solid` se escribe como una entidad MESH (DXF R2010, AC1024) en una capa
por tipo (`TLETL_BOX`, `TLETL_CYLINDER`, ...). Los MESH se generan con
`ezdxf.render.forms` a partir de una malla unitaria centrada en el origen
(caja envolvente [-0.5, 0.5]^3) transformada con una `Matrix44`:
escala(size × scale) -> rotación Z -> traslación(center). Así la caja
envolvente del MESH coincide EXACTAMENTE con `Solid.center`/`dimensions`,
igual que AddBox/AddCylinder/... de AutoCAD.

`$INSUNITS` se fija según `units` para que AutoCAD (y FreeCAD, LibreCAD,
QCAD, ...) abran el archivo en la escala correcta.

ezdxf se importa de forma LAZY: importar este módulo no lo exige; solo
exportar. Sin ezdxf, `export_scene_dxf` lanza ImportError con la instrucción
de instalación. El round trip JSON (`scene_to_json`/`scene_from_json`) no
necesita ezdxf: una sesión se guarda siempre y se re-exporta después.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .gesture_modeler import SOLID_KINDS, Scene, Solid

DEFAULT_DXF_VERSION = "R2010"
DEFAULT_SEGMENTS = 32
# Códigos de $INSUNITS (ezdxf.units): 1=in, 4=mm, 5=cm, 6=m
UNIT_CODES: Dict[str, int] = {"in": 1, "mm": 4, "cm": 5, "m": 6}
LAYER_PREFIX = "TLETL_"
LAYER_COLORS: Dict[str, int] = {"box": 1, "cylinder": 3, "sphere": 5, "cone": 2, "wedge": 6}
APPID = "TLETL"
EZDXF_HINT = "ezdxf no está instalado: pip install ezdxf   (o: pip install -e '.[cad]')"


def require_ezdxf():
    """Importa ezdxf o lanza ImportError con la instrucción de instalación."""
    try:
        import ezdxf  # noqa: WPS433 (import lazy a propósito)
    except ImportError as exc:
        raise ImportError(EZDXF_HINT) from exc
    return ezdxf


def layer_name(kind: str) -> str:
    return LAYER_PREFIX + kind.upper()


# ── Mallas unitarias ────────────────────────────────────────────────────────

def wedge_mesh():
    """Cuña unitaria centrada: base 1×1 en z=-0.5, cara vertical en x=-0.5 y
    rampa que baja hacia +x (misma convención que AddWedge de AutoCAD).
    Caras con normales hacia afuera. Se construye con MeshVertexMerger para que
    las caras COMPARTAN vértices (6 vértices, 5 caras): con MeshTransformer.add_face
    cada cara traía sus propios vértices (18) y la malla quedaba abierta, así que
    CONVTOSOLID de AutoCAD la rechazaba."""
    from ezdxf.render.mesh import MeshTransformer, MeshVertexMerger

    a = (-0.5, -0.5, -0.5)
    b = (0.5, -0.5, -0.5)
    c = (0.5, 0.5, -0.5)
    d = (-0.5, 0.5, -0.5)
    e = (-0.5, -0.5, 0.5)
    f = (-0.5, 0.5, 0.5)
    mesh = MeshVertexMerger()
    mesh.add_face([a, d, c, b])   # base (-z)
    mesh.add_face([a, b, e])      # lado y=-0.5
    mesh.add_face([d, f, c])      # lado y=+0.5
    mesh.add_face([a, e, f, d])   # cara vertical x=-0.5
    mesh.add_face([b, c, f, e])   # rampa
    return MeshTransformer.from_builder(mesh)


def unit_mesh(kind: str, segments: int = DEFAULT_SEGMENTS):
    """Malla del tipo pedido con caja envolvente [-0.5, 0.5]^3."""
    from ezdxf.math import Matrix44
    from ezdxf.render import forms

    segments = max(8, int(segments))
    if kind == "box":
        return forms.cube(center=True)
    if kind == "cylinder":
        mesh = forms.cylinder(count=segments, radius=0.5, top_center=(0, 0, 1))
        return mesh.transform(Matrix44.translate(0, 0, -0.5))
    if kind == "sphere":
        return forms.sphere(count=segments, stacks=max(4, segments // 2), radius=0.5)
    if kind == "cone":
        mesh = forms.cone(count=segments, radius=0.5, apex=(0, 0, 1))
        return mesh.transform(Matrix44.translate(0, 0, -0.5))
    if kind == "wedge":
        return wedge_mesh()
    raise ValueError(f"tipo de sólido desconocido {kind!r}; válidos: {SOLID_KINDS}")


def solid_matrix(solid: Solid):
    """Matrix44 que lleva la malla unitaria al sólido: escala -> rotación Z -> traslación."""
    from ezdxf.math import Matrix44

    sx, sy, sz = solid.dimensions
    return Matrix44.chain(
        Matrix44.scale(sx, sy, sz),
        Matrix44.z_rotate(solid.rotation_z),
        Matrix44.translate(*solid.center),
    )


def solid_mesh(solid: Solid, segments: int = DEFAULT_SEGMENTS):
    return unit_mesh(solid.kind, segments).transform(solid_matrix(solid))


# ── Export ──────────────────────────────────────────────────────────────────

def export_scene_dxf(scene: Scene, path: str | Path, *, units: Optional[str] = None,
                     dxfversion: str = DEFAULT_DXF_VERSION, segments: int = DEFAULT_SEGMENTS) -> Path:
    """Escribe la escena como DXF y devuelve la ruta. `units` en {"mm","cm","m","in"}
    (default: `scene.units`). Lanza ImportError (con instrucción) si falta ezdxf."""
    ezdxf = require_ezdxf()
    unit_key = str(units or scene.units or "mm").lower()
    if unit_key not in UNIT_CODES:
        raise ValueError(f"unidades desconocidas {units!r}; válidas: {sorted(UNIT_CODES)}")

    doc = ezdxf.new(dxfversion, setup=True)
    doc.header["$INSUNITS"] = UNIT_CODES[unit_key]
    doc.header["$MEASUREMENT"] = 0 if unit_key == "in" else 1   # 1 = métrico
    doc.appids.add(APPID)
    for kind in SOLID_KINDS:
        doc.layers.add(layer_name(kind), color=LAYER_COLORS[kind])

    msp = doc.modelspace()
    for solid in scene.solids:
        mesh = solid_mesh(solid, segments)
        entity = mesh.render_mesh(msp, dxfattribs={"layer": layer_name(solid.kind)})
        # Metadatos mínimos para reconstruir el sólido desde el DXF (XDATA).
        entity.set_xdata(APPID, [(1000, f"tletl:{solid.id}:{solid.kind}"),
                                 (1000, json.dumps(solid.to_dict(), separators=(",", ":")))])

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(out)
    return out


# ── JSON de escena (no necesita ezdxf) ──────────────────────────────────────

def scene_to_json(scene: Scene, *, indent: Optional[int] = 2) -> str:
    return json.dumps(scene.to_dict(), indent=indent, ensure_ascii=False)


def scene_from_json(text: str) -> Scene:
    data: Any = json.loads(text)
    if isinstance(data, dict) and data.get("format") not in (None, "tletl-cad-scene"):
        raise ValueError(f"formato de escena desconocido: {data.get('format')!r}")
    return Scene.from_dict(data)


def save_scene_json(scene: Scene, path: str | Path) -> Path:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(scene_to_json(scene) + "\n", encoding="utf-8")
    return out


def load_scene_json(path: str | Path) -> Scene:
    return scene_from_json(Path(path).expanduser().read_text(encoding="utf-8"))


# ── CLI: re-exportar una escena guardada ────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m apps.autocad_control.dxf_export",
        description="Re-exporta a DXF una escena guardada por la sesión (escena.json).")
    ap.add_argument("scene", metavar="escena.json")
    ap.add_argument("dxf", metavar="salida.dxf", nargs="?", default=None,
                    help="default: mismo nombre que el JSON con extensión .dxf")
    ap.add_argument("--units", choices=sorted(UNIT_CODES), default=None,
                    help="default: las unidades guardadas en la escena")
    ap.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS,
                    help="segmentos de cilindros/esferas/conos (default %(default)s)")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    scene = load_scene_json(args.scene)
    target = Path(args.dxf) if args.dxf else Path(args.scene).with_suffix(".dxf")
    try:
        out = export_scene_dxf(scene, target, units=args.units, segments=args.segments)
    except ImportError as exc:
        print(f"[dxf] {exc}")
        return 2
    print(f"[dxf] {len(scene)} sólidos -> {out} (unidades: {args.units or scene.units})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
