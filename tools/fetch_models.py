"""tools/fetch_models.py — descarga los modelos .task de MediaPipe que usa Tletl.

    python -m tools.fetch_models                 # ambos modelos a ~/.tletl/models/
    python -m tools.fetch_models --dir /ruta     # otro directorio
    python -m tools.fetch_models --force         # re-descarga aunque existan
    python -m tools.fetch_models --check         # solo informa qué falta (exit 1 si falta algo)
    python -m tools.fetch_models hand_landmarker.task   # solo uno

Modelos (float16, "latest" oficial de Google):
  - hand_landmarker.task     -> backend "tasks" de apps/common/hand_tracker.py (obligatorio
                                con mediapipe >= 0.10.31 / 1.x, donde ya no existe mp.solutions)
  - gesture_recognizer.task  -> apps/common/gesture_hint.py (opcional, [tracker] gesture_hint)

El directorio destino es el mismo que usa el tracker por default
(`TLETL_HOME/models`, es decir ~/.tletl/models). Se descarga a un archivo .part y
se renombra al final (nunca queda un .task a medias); se exige > 1 MB porque un
proxy o un portal cautivo devuelven HTML de unos KB con código 200.

Sin dependencias fuera de la librería estándar (urllib respeta HTTPS_PROXY).
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from apps.common.hand_tracker import GESTURE_MODEL_NAME, HAND_MODEL_NAME, default_models_dir

MODELS: Dict[str, str] = {
    HAND_MODEL_NAME: ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                      "hand_landmarker/float16/latest/hand_landmarker.task"),
    GESTURE_MODEL_NAME: ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
                         "gesture_recognizer/float16/latest/gesture_recognizer.task"),
}
MIN_BYTES = 1_000_000          # un .task real pesa ~8 MB; menos que esto es un error disfrazado
CHUNK_SIZE = 1 << 16
TIMEOUT_S = 60.0

Opener = Callable[[str], Any]   # url -> objeto con .read(n) usable como context manager


class FetchError(RuntimeError):
    """Descarga fallida, cortada o demasiado pequeña."""


def human_size(n_bytes: int) -> str:
    n = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} GB"  # pragma: no cover


def _default_opener(url: str) -> Any:
    return urllib.request.urlopen(url, timeout=TIMEOUT_S)


def download(url: str, dest: str | Path, *, opener: Optional[Opener] = None, force: bool = False,
             chunk_size: int = CHUNK_SIZE, min_bytes: int = MIN_BYTES) -> Tuple[str, int]:
    """Descarga `url` en `dest`. Devuelve ("skipped" | "downloaded", tamaño en bytes).

    - Si `dest` ya existe con tamaño >= min_bytes y no `force`: no toca la red.
    - Escribe en dest.part y renombra al final; cualquier fallo borra el .part.
    - Lanza FetchError si la red falla o el archivo queda por debajo de min_bytes.
    """
    dest = Path(dest)
    if not force and dest.is_file() and dest.stat().st_size >= min_bytes:
        return "skipped", dest.stat().st_size

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    open_url = opener or _default_opener
    try:
        with open_url(url) as response, open(part, "wb") as fh:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise FetchError(f"{dest.name}: {exc} ({url})") from exc

    size = part.stat().st_size
    if size < min_bytes:
        part.unlink(missing_ok=True)
        raise FetchError(f"{dest.name}: descarga demasiado pequeña ({human_size(size)} < {human_size(min_bytes)}); "
                         "¿proxy, portal cautivo o URL cambiada?")
    os.replace(part, dest)
    return "downloaded", size


def fetch_models(names: Iterable[str], dest_dir: str | Path, *, opener: Optional[Opener] = None,
                 force: bool = False, min_bytes: int = MIN_BYTES, log: Callable[[str], None] = print) -> int:
    """Descarga cada modelo de `names` a `dest_dir`. Devuelve 0 si todo bien, 1 si algo falló."""
    dest_dir = Path(dest_dir).expanduser()
    failures = 0
    for name in names:
        url = MODELS.get(name)
        if url is None:
            log(f"[FAIL] modelo desconocido: {name} (conozco {', '.join(MODELS)})")
            failures += 1
            continue
        dest = dest_dir / name
        try:
            status, size = download(url, dest, opener=opener, force=force, min_bytes=min_bytes)
        except FetchError as exc:
            log(f"[FAIL] {exc}")
            failures += 1
            continue
        verb = "ya estaba" if status == "skipped" else "descargado"
        log(f"[ok] {name}: {verb} ({human_size(size)}) -> {dest}")
    return 0 if failures == 0 else 1


def check_models(names: Iterable[str], dest_dir: str | Path, *, min_bytes: int = MIN_BYTES,
                 log: Callable[[str], None] = print) -> int:
    """Informa qué modelos están y cuáles faltan. 0 si están todos, 1 si falta alguno."""
    dest_dir = Path(dest_dir).expanduser()
    missing = 0
    for name in names:
        dest = dest_dir / name
        if dest.is_file() and dest.stat().st_size >= min_bytes:
            log(f"[ok] {name}: {human_size(dest.stat().st_size)} -> {dest}")
        else:
            log(f"[MISSING] {name}: no está en {dest_dir} (ejecuta ./launchers/tletl-fetch-models.sh)")
            missing += 1
    return 0 if missing == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Descarga los modelos .task de MediaPipe para Tletl v5.")
    # Sin `choices`: main() valida los nombres y devuelve un mensaje propio + exit 1
    # (hay un test que lo exige), en vez del error genérico de argparse.
    ap.add_argument("names", nargs="*", default=None,
                    help=f"modelos a descargar (default: todos: {', '.join(MODELS)})")
    ap.add_argument("--dir", default=None, help="directorio destino (default: $TLETL_HOME/models = ~/.tletl/models)")
    ap.add_argument("--force", action="store_true", help="re-descarga aunque el archivo exista")
    ap.add_argument("--check", action="store_true", help="solo comprueba qué modelos están; no descarga")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    names = list(args.names) if args.names else list(MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(f"[FAIL] modelo(s) desconocido(s): {', '.join(unknown)}. Conozco: {', '.join(MODELS)}")
        return 2
    dest_dir = Path(args.dir).expanduser() if args.dir else default_models_dir()
    print(f"[TLETL] modelos en {dest_dir}")
    if args.check:
        return check_models(names, dest_dir)
    return fetch_models(names, dest_dir, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
