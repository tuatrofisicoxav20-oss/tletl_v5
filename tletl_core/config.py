from __future__ import annotations

"""Carga de configuración de Tletl v5.

Lee config/tletl.toml y aplica overrides por variable de entorno (compatibilidad
con las costumbres del runtime viejo: TLETL_AI_V47_K, etc.). El core no impone
rutas: si no hay toml, devuelve los defaults de abajo.
"""

import os
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


DEFAULTS: Dict[str, Dict[str, Any]] = {
    "classifier": {"k": 13, "orientation_weight": 0.45, "strict": True,
                    "min_confidence": 0.47, "min_margin": 0.18},
    "guard": {"enabled": True, "conf_threshold": 0.55, "margin_threshold": 0.20,
               "dangerous": ["PINCH", "POINT", "VICTORY", "THREE", "FIST"]},
    "critic": {"enabled": True, "require_orientation": True},
    "temporal": {"size": 7, "min_count": 4},
    "lowlight": {"enabled": True, "clip_limit": 2.0, "gamma_dark": 1.6},
    "adaptive": {"enabled": False, "min_confidence": 0.78, "max_samples_per_gesture": 120},
    "fedora": {"dominant_hand": "Right", "dry_run": False},
    "blender": {"bus_path": "tletl_state.json", "move_gain": 4.0, "smoothing": 0.35},
    "camera": {"index": 0, "width": 1280, "height": 720, "max_num_hands": 2},
}

# Overrides por env var -> (sección, clave, conversor)
ENV_MAP = {
    "TLETL_AI_V47_K": ("classifier", "k", int),
    "TLETL_AI_V47_ORIENTATION_WEIGHT": ("classifier", "orientation_weight", float),
    "TLETL_TEMPORAL_SIZE": ("temporal", "size", int),
    "TLETL_TEMPORAL_MIN": ("temporal", "min_count", int),
    "TLETL_DRY_RUN": ("fedora", "dry_run", lambda v: v not in ("", "0", "false", "False")),
    "TLETL_LOWLIGHT": ("lowlight", "enabled", lambda v: v not in ("", "0", "false", "False")),
    "TLETL_ADAPTIVE": ("adaptive", "enabled", lambda v: v not in ("", "0", "false", "False")),
}


def _default_toml_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "tletl.toml"


def load_config(path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    cfg: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in DEFAULTS.items()}

    toml_path = Path(path) if path else _default_toml_path()
    if toml_path.exists():
        with open(toml_path, "rb") as fh:
            loaded = tomllib.load(fh)
        for section, values in loaded.items():
            cfg.setdefault(section, {})
            if isinstance(values, dict):
                cfg[section].update(values)

    for env_key, (section, key, conv) in ENV_MAP.items():
        if env_key in os.environ:
            try:
                cfg.setdefault(section, {})[key] = conv(os.environ[env_key])
            except Exception:
                pass

    return cfg
