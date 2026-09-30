"""Carga de configuración de Tletl v5.

Lee config/tletl.toml y aplica overrides por variable de entorno (compatibilidad
con las costumbres del runtime viejo: TLETL_AI_V47_K, etc.). El core no impone
rutas: si no hay toml, devuelve los DEFAULTS de abajo.

Contrato:
  - `load_config()` SIEMPRE devuelve todas las secciones y claves de DEFAULTS
    (el toml y las env vars solo sobreescriben). Las apps pueden indexar sin
    `.get()` defensivo.
  - Los sub-diccionarios (p.ej. `[critic.min_conf]`) se fusionan clave a clave:
    un toml que solo cambie PINCH conserva los demás umbrales.
  - Las rutas vacías ("") significan "usa el default de tletl_core.paths".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from . import paths

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _flag(v: str) -> bool:
    return v.strip().lower() not in ("", "0", "false", "no", "off")


# Umbrales de confianza mínima por gesto para el critic. PINCH bajó de 0.72 a
# 0.62 por la evidencia de docs/VALIDACION_FISICA_v5.md: a luz diurna el KNN
# acierta PINCH 68/80 con confianza 0.62–0.70 y el critic lo mataba (0/10).
# El guard (conf/margen/regla geométrica) y el filtro temporal siguen activos.
CRITIC_MIN_CONF_DEFAULTS: Dict[str, float] = {
    "OPEN_PALM": 0.70,
    "FIST": 0.68,
    "POINT": 0.69,
    "VICTORY": 0.72,
    "PINCH": 0.62,
    "THREE": 0.72,
    "NEUTRAL": 0.55,
}

DEFAULTS: Dict[str, Dict[str, Any]] = {
    "classifier": {"k": 13, "orientation_weight": 0.45, "strict": True,
                    "min_confidence": 0.47, "min_margin": 0.18},
    "guard": {"enabled": True, "conf_threshold": 0.55, "margin_threshold": 0.20,
               "dangerous": ["PINCH", "POINT", "VICTORY", "THREE", "FIST"]},
    "critic": {"enabled": True, "require_orientation": True,
                "min_conf": dict(CRITIC_MIN_CONF_DEFAULTS)},
    "temporal": {"size": 7, "min_count": 4},
    "lowlight": {"enabled": True, "clip_limit": 2.0, "gamma_dark": 1.6},
    "adaptive": {"enabled": False, "min_confidence": 0.78, "max_samples_per_gesture": 120,
                  "path": "", "save_interval": 2.0},
    "fedora": {"dominant_hand": "Right", "dry_run": False,
                "safety_hold": 0.35, "tap_max": 0.35, "drag_min": 0.40,
                "action_conf": 0.52},
    "blender": {"bus_path": ""},   # alias legacy de [bus].path (se conserva por compatibilidad)
    "bus": {"path": "", "include_features": False, "udp_target": ""},
    "camera": {"index": 0, "width": 1280, "height": 720, "fps": 30, "max_num_hands": 2,
                "proc_width": 640, "headless": False, "fourcc": "MJPG"},
    # backend "auto" = MediaPipe Hand Landmarker (Tasks API) como principal
    # (aprobado por el usuario 2026-09-30); "legacy" (mp.solutions) solo como
    # respaldo si Tasks no está disponible. auto_download baja el .task si falta.
    "tracker": {"backend": "auto", "model_path": "", "det_conf": 0.72, "track_conf": 0.72,
                 "model_complexity": 1, "auto_download": True,
                 "gesture_hint": False, "gesture_model_path": "", "gesture_hint_min_score": 0.5},
    "paths": {"bank": ""},
}

# Overrides por env var -> (sección, clave, conversor)
ENV_MAP: Dict[str, Tuple[str, str, Callable[[str], Any]]] = {
    "TLETL_AI_V47_K": ("classifier", "k", int),
    "TLETL_AI_V47_ORIENTATION_WEIGHT": ("classifier", "orientation_weight", float),
    "TLETL_TEMPORAL_SIZE": ("temporal", "size", int),
    "TLETL_TEMPORAL_MIN": ("temporal", "min_count", int),
    "TLETL_DRY_RUN": ("fedora", "dry_run", _flag),
    "TLETL_DOMINANT_HAND": ("fedora", "dominant_hand", str),
    "TLETL_LOWLIGHT": ("lowlight", "enabled", _flag),
    "TLETL_ADAPTIVE": ("adaptive", "enabled", _flag),
    "TLETL_ADAPTIVE_PATH": ("adaptive", "path", str),
    "TLETL_STATE_PATH": ("bus", "path", str),
    "TLETL_BUS_UDP": ("bus", "udp_target", str),
    "TLETL_CAMERA": ("camera", "index", int),
    "TLETL_HEADLESS": ("camera", "headless", _flag),
    "TLETL_TRACKER_BACKEND": ("tracker", "backend", str),
    "TLETL_TRACKER_MODEL": ("tracker", "model_path", str),
    "TLETL_TRACKER_AUTO_DOWNLOAD": ("tracker", "auto_download", _flag),
    "TLETL_GESTURE_HINT": ("tracker", "gesture_hint", _flag),
    "TLETL_GESTURE_MODEL": ("tracker", "gesture_model_path", str),
    "TLETL_CAMERA_FOURCC": ("camera", "fourcc", str),
    "TLETL_GESTURE_BANK": ("paths", "bank", str),
}


def _default_toml_path() -> Path:
    return paths.repo_root() / "config" / "tletl.toml"


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Fusiona src en dst un nivel más profundo que dict.update (para tablas anidadas)."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            merged = dict(dst[key])
            merged.update(value)
            dst[key] = merged
        else:
            dst[key] = value


def _copy_defaults() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for section, values in DEFAULTS.items():
        out[section] = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
                        for k, v in values.items()}
    return out


def load_config(path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    cfg = _copy_defaults()

    toml_path = Path(path) if path else _default_toml_path()
    if toml_path.exists():
        with open(toml_path, "rb") as fh:
            loaded = tomllib.load(fh)
        for section, values in loaded.items():
            cfg.setdefault(section, {})
            if isinstance(values, dict):
                _deep_update(cfg[section], values)

    for env_key, (section, key, conv) in ENV_MAP.items():
        if env_key in os.environ:
            try:
                cfg.setdefault(section, {})[key] = conv(os.environ[env_key])
            except Exception:
                pass

    return cfg


# ---------------------------------------------------------------------------
# Helpers de rutas derivadas de la config (UNA sola regla para todas las apps)
# ---------------------------------------------------------------------------

def bus_path_from_config(cfg: Dict[str, Dict[str, Any]]) -> Path:
    """Ruta del bus: env TLETL_STATE_PATH (ya aplicada por ENV_MAP) > [bus].path
    > [blender].bus_path (legacy) > ~/.tletl/tletl_state.json."""
    configured = cfg.get("bus", {}).get("path") or cfg.get("blender", {}).get("bus_path") or ""
    return paths.resolve_path(configured, paths.default_bus_path(), relative_to=paths.repo_root())


def adaptive_path_from_config(cfg: Dict[str, Dict[str, Any]]) -> Path:
    configured = cfg.get("adaptive", {}).get("path") or ""
    return paths.resolve_path(configured, paths.default_adaptive_path(), relative_to=paths.repo_root())


def bank_path_from_config(cfg: Dict[str, Dict[str, Any]]) -> Path:
    configured = cfg.get("paths", {}).get("bank") or ""
    return paths.resolve_path(configured, paths.default_bank_path(), relative_to=paths.repo_root())


def critic_min_conf(cfg: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Tabla de umbrales del critic garantizando los 7 gestos (faltantes -> default)."""
    table = dict(CRITIC_MIN_CONF_DEFAULTS)
    custom = cfg.get("critic", {}).get("min_conf")
    if isinstance(custom, dict):
        for gesture, value in custom.items():
            try:
                table[str(gesture).upper()] = float(value)
            except (TypeError, ValueError):
                continue
    return table
