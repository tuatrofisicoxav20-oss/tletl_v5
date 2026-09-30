"""apps/common/gesture_hint.py — segunda opinión con el Gesture Recognizer de MediaPipe.

El modelo `gesture_recognizer.task` clasifica por sí solo Closed_Fist, Open_Palm,
Pointing_Up, Victory, Thumb_Up, Thumb_Down e ILoveYou. Los cuatro primeros
mapean 1:1 al vocabulario de Tletl (FIST, OPEN_PALM, POINT, VICTORY); el resto
(y "None") no dicen nada -> `None`. Está entrenado con miles de manos ajenas,
así que es una opinión INDEPENDIENTE del banco KNN: la app la pasa como
`hint=` a `pipeline.process_features` y el core decide qué hacer con ella.

Es OPCIONAL y viene apagado: se activa con `[tracker] gesture_hint = true` o
`TLETL_GESTURE_HINT=1`. `GestureHint.create()` devuelve None (nunca lanza) si
está apagado, si falta el modelo o si mediapipe no lo puede cargar.

Costo: es un segundo modelo por frame (~10 ms en CPU, corre su propio
landmarker por dentro). Si la laptop va justa, déjalo apagado: mejora
precisión, no fps.

Lateralidad: misma convención que hand_tracker (frame espejado antes de
detectar -> etiqueta correcta sin intercambio).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from apps.common.hand_tracker import (
    GESTURE_MODEL_NAME,
    HAND_MODEL_NAME,
    Downloader,
    Logger,
    MonotonicTimestamps,
    TrackerModelMissing,
    _flag,
    _import_mediapipe,
    _mp_image_types,
    ensure_model_file,
    normalize_handedness,
    resolve_model_path,
)

ENV_ENABLED = "TLETL_GESTURE_HINT"
ENV_MODEL = "TLETL_GESTURE_MODEL"
CFG_ENABLED_KEY = "gesture_hint"            # [tracker] gesture_hint = false
CFG_MODEL_KEY = "gesture_model_path"        # [tracker] gesture_model_path = ""
CFG_MIN_SCORE_KEY = "gesture_hint_min_score"  # [tracker] gesture_hint_min_score = 0.5

# Nombre MediaPipe -> gesto Tletl. Lo que no está aquí no opina.
MP_TO_TLETL: Dict[str, str] = {
    "Closed_Fist": "FIST",
    "Open_Palm": "OPEN_PALM",
    "Pointing_Up": "POINT",
    "Victory": "VICTORY",
}

Hint = Tuple[str, Optional[str]]  # (handedness 'Left'|'Right', gesto Tletl o None)


def map_mp_gesture(name: Optional[str]) -> Optional[str]:
    """'Closed_Fist' -> 'FIST', 'Open_Palm' -> 'OPEN_PALM', 'Pointing_Up' -> 'POINT',
    'Victory' -> 'VICTORY'; None/'None'/Thumb_Up/ILoveYou/desconocidos -> None."""
    if not name:
        return None
    return MP_TO_TLETL.get(str(name).strip())


def hint_enabled(cfg_tracker: Optional[Dict[str, Any]]) -> bool:
    """Lee [tracker].gesture_hint del dict de config YA fusionado.

    La precedencia CLI > env > toml la resuelven `tletl_core.config.load_config`
    (env TLETL_GESTURE_HINT vía ENV_MAP) y `main.apply_cli_overrides` (--gesture-hint).
    Aquí NO se vuelve a leer la env var: si se hiciera, `TLETL_GESTURE_HINT=0`
    anularía un `--gesture-hint` explícito.
    """
    value = (cfg_tracker or {}).get(CFG_ENABLED_KEY, False)
    return _flag(value) if isinstance(value, str) else bool(value)


def resolve_gesture_model_path(cfg_tracker: Optional[Dict[str, Any]]) -> Path:
    """env TLETL_GESTURE_MODEL > [tracker].gesture_model_path > mismo directorio que
    el modelo de landmarks (respeta TLETL_TRACKER_MODEL / [tracker].model_path)."""
    cfg = cfg_tracker or {}
    env = os.environ.get(ENV_MODEL, "").strip()
    configured = env or str(cfg.get(CFG_MODEL_KEY) or "").strip()
    if configured:
        p = Path(configured).expanduser()
        if configured.endswith(("/", os.sep)) or p.is_dir():
            return p / GESTURE_MODEL_NAME
        return p
    return resolve_model_path(cfg.get("model_path"), name=HAND_MODEL_NAME).parent / GESTURE_MODEL_NAME


def hints_from_result(result: Any, min_score: float = 0.0) -> List[Hint]:
    """Convierte un GestureRecognizerResult en [(handedness, gesto_tletl|None), ...] (puro)."""
    gestures = getattr(result, "gestures", None) or []
    handed = getattr(result, "handedness", None) or []
    out: List[Hint] = []
    for i, categories in enumerate(gestures):
        label = "Right"
        if i < len(handed) and handed[i]:
            top_hand = handed[i][0]
            label = normalize_handedness(getattr(top_hand, "category_name", None)
                                         or getattr(top_hand, "display_name", None))
        tletl: Optional[str] = None
        if categories:
            top = categories[0]
            try:
                score = float(getattr(top, "score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            if score >= min_score:
                tletl = map_mp_gesture(getattr(top, "category_name", None))
        out.append((label, tletl))
    return out


def hint_for(hints: List[Hint], handedness: str) -> Optional[str]:
    """Hint de la mano con esa lateralidad (la primera que coincida), o None."""
    for label, hint in hints:
        if label == handedness:
            return hint
    return None


class GestureHint:
    """Envuelve un GestureRecognizer en modo VIDEO.

    `recognizer` es cualquier objeto con `recognize_for_video(image, ts_ms)` y `close()`;
    `image_factory` convierte el ndarray RGB en lo que el recognizer espera (los
    tests inyectan la identidad para no depender de mediapipe).
    """

    def __init__(self, recognizer: Any, model_path: Optional[Path] = None, *,
                 min_score: float = 0.5,
                 image_factory: Optional[Callable[[np.ndarray], Any]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._recognizer = recognizer
        self.model_path = Path(model_path) if model_path else None
        self.min_score = float(min_score)
        self._image_factory = image_factory or _default_image_factory()
        self._stamps = MonotonicTimestamps(clock)
        self.frames = 0
        self.closed = False

    @classmethod
    def create(cls, cfg_tracker: Optional[Dict[str, Any]] = None, num_hands: int = 2, *,
               downloader: Optional[Downloader] = None, log: Logger = print) -> Optional["GestureHint"]:
        """Devuelve un GestureHint listo, o None si está apagado / falta el modelo (y no se
        pudo descargar con [tracker] auto_download) / mediapipe no lo carga. Nunca lanza."""
        cfg = dict(cfg_tracker or {})
        if not hint_enabled(cfg):
            return None
        try:
            model = ensure_model_file(resolve_gesture_model_path(cfg), GESTURE_MODEL_NAME,
                                      auto_download=_flag(cfg.get("auto_download", True)),
                                      downloader=downloader, log=log)
        except TrackerModelMissing as exc:
            log(f"[TLETL] gesture_hint pedido pero sin modelo: {exc} Sigo sin hint.")
            return None
        det_conf = float(cfg.get("det_conf", 0.72))
        track_conf = float(cfg.get("track_conf", 0.72))
        try:
            mp = _import_mediapipe()
            from mediapipe.tasks.python import BaseOptions, vision  # noqa: PLC0415
            options = vision.GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=str(model)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max(1, int(num_hands)),
                min_hand_detection_confidence=det_conf,
                min_hand_presence_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
            recognizer = vision.GestureRecognizer.create_from_options(options)
            image_cls, image_fmt = _mp_image_types(mp)
        except Exception as exc:  # ImportError, OSError (.so), ValueError (modelo corrupto)...
            log(f"[TLETL] gesture_hint desactivado: no pude crear el GestureRecognizer ({exc}).")
            return None

        def make_image(rgb: np.ndarray) -> Any:
            return image_cls(image_format=image_fmt.SRGB, data=np.ascontiguousarray(rgb))

        return cls(recognizer, model, min_score=float(cfg.get(CFG_MIN_SCORE_KEY, 0.5)),
                   image_factory=make_image)

    def process(self, rgb: np.ndarray, timestamp_ms: Optional[int] = None) -> List[Hint]:
        ts = self._stamps.next(timestamp_ms)
        self.frames += 1
        result = self._recognizer.recognize_for_video(self._image_factory(rgb), ts)
        return hints_from_result(result, self.min_score)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        close = getattr(self._recognizer, "close", None)
        if callable(close):
            close()


def _default_image_factory() -> Callable[[np.ndarray], Any]:
    """Fábrica lazy de mp.Image (solo se importa mediapipe al primer frame)."""
    cache: Dict[str, Any] = {}

    def make(rgb: np.ndarray) -> Any:
        if "cls" not in cache:
            cache["cls"], cache["fmt"] = _mp_image_types(_import_mediapipe())
        return cache["cls"](image_format=cache["fmt"].SRGB, data=np.ascontiguousarray(rgb))

    return make
