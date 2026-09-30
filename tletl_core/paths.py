"""tletl_core/paths.py — rutas de runtime compartidas por TODOS los clientes.

Problema que resuelve (bug anotado en docs/VALIDACION_FISICA_v5.md §5): la app
escribía el bus en `<cwd>/tletl_state.json` y el addon de Blender leía
`~/tletl_state.json`, así que con defaults jamás se veían. Además la memoria
adaptativa se escribía en `datasets/`, junto al banco sagrado.

Regla nueva: todo artefacto de runtime vive en un directorio propio, fuera del
repo, y se resuelve con UNA sola función por artefacto:

    bus       -> $TLETL_STATE_PATH  o  $TLETL_HOME/tletl_state.json
    adaptive  -> $TLETL_ADAPTIVE_PATH o $TLETL_HOME/tletl_adaptive_runtime.json
    TLETL_HOME por default = ~/.tletl

Se eligió el HOME (y no XDG_RUNTIME_DIR, que es tmpfs) porque Blender instalado
como Flatpak no ve /run/user/<uid>, y este bug ya costó una validación. Quien
quiera tmpfs exporta `TLETL_STATE_PATH=$XDG_RUNTIME_DIR/tletl_state.json`.

IMPORTANTE: `apps/blender_control/state_reader.py` replica `default_bus_path()`
porque el Python de Blender no puede importar tletl_core. Si cambias la regla
aquí, cámbiala allá (hay un test que verifica que coinciden).
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_HOME = "TLETL_HOME"
ENV_STATE_PATH = "TLETL_STATE_PATH"
ENV_ADAPTIVE_PATH = "TLETL_ADAPTIVE_PATH"
ENV_BANK = "TLETL_GESTURE_BANK"

BUS_FILENAME = "tletl_state.json"
ADAPTIVE_FILENAME = "tletl_adaptive_runtime.json"
BANK_RELATIVE = Path("datasets") / "tletl_gesture_bank_v2_features.jsonl"


def repo_root() -> Path:
    """Raíz del repositorio (carpeta que contiene tletl_core/)."""
    return Path(__file__).resolve().parent.parent


def tletl_home() -> Path:
    """Directorio de runtime del usuario. No se crea aquí (ver ensure_parent)."""
    value = os.environ.get(ENV_HOME, "").strip()
    return Path(value).expanduser() if value else Path.home() / ".tletl"


def default_bus_path() -> Path:
    """Ruta del bus JSON que comparten la app de cámara, Blender y CAD."""
    value = os.environ.get(ENV_STATE_PATH, "").strip()
    return Path(value).expanduser() if value else tletl_home() / BUS_FILENAME


def default_adaptive_path() -> Path:
    """Ruta de la memoria adaptativa. NUNCA junto al banco."""
    value = os.environ.get(ENV_ADAPTIVE_PATH, "").strip()
    return Path(value).expanduser() if value else tletl_home() / ADAPTIVE_FILENAME


def default_bank_path() -> Path:
    """Banco de gestos: env > datasets/ del repo."""
    value = os.environ.get(ENV_BANK, "").strip()
    return Path(value).expanduser() if value else repo_root() / BANK_RELATIVE


def resolve_path(configured: str | os.PathLike[str] | None, default: Path,
                 *, relative_to: Path | None = None) -> Path:
    """Devuelve `configured` (expandido, y relativo a `relative_to` si no es absoluto)
    o `default` si viene vacío/None."""
    if configured is None or str(configured).strip() == "":
        return default
    p = Path(str(configured)).expanduser()
    if not p.is_absolute() and relative_to is not None:
        p = relative_to / p
    return p


def ensure_parent(path: Path) -> Path:
    """Crea el directorio padre si no existe y devuelve la misma ruta."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
