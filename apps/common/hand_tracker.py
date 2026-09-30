"""apps/common/hand_tracker.py — abstracción del detector de manos de Tletl.

La app de Fedora y la herramienta del banco ya no hablan con MediaPipe
directamente: le piden a `HandTracker` una lista de `HandDetection` (21 `Point`
normalizados + lateralidad + score) y el resto del sistema no sabe qué modelo
la produjo. Cambiar de detector (o de versión de MediaPipe) es cambiar UN archivo.

Backends:
  - "tasks":  `mediapipe.tasks.python.vision.HandLandmarker` en modo VIDEO. ES EL
    DETECTOR PRINCIPAL (decisión del usuario, docs/RECOMENDACIONES_DETECCION_MANOS.md).
    Necesita el modelo `hand_landmarker.task`; si falta y `[tracker] auto_download`
    está activo, se descarga una vez a ~/.tletl/models/ (tools/fetch_models.py).
  - "legacy": `mp.solutions.hands.Hands` (mediapipe <= 0.10.2x). Solo de respaldo:
    esa API ya no existe en mediapipe >= 0.10.31 / 1.x, que fue lo que rompió la
    app durante la validación (docs/VALIDACION_FISICA_v5.md §5).
  - "auto":   tasks si la Tasks API se puede importar; legacy SOLO si Tasks no
    está disponible (mediapipe muy viejo) o si crear el landmarker falla
    (librería nativa ausente, modelo imposible de obtener). Se imprime UNA línea
    diciendo qué backend se eligió y por qué. "tasks" y "legacy" fuerzan.

Ambos devuelven los MISMOS 21 landmarks normalizados (x, y, z) con los mismos
índices (0 muñeca, 1-4 pulgar, 5-8 índice, 9-12 medio, 13-16 anular, 17-20
meñique), así que el banco de gestos y `extract_live_features` no cambian.

LATERALIDAD (Left/Right) Y ESPEJO — leer antes de "corregir" nada:
MediaPipe calcula la lateralidad ASUMIENDO que la imagen ya está espejada
(cámara selfie: tu mano derecha aparece a la derecha de la imagen). Las apps
de Tletl hacen `cv2.flip(frame, 1)` ANTES de detectar, así que la imagen que
ve el detector ya es un espejo y la etiqueta sale correcta tal cual: NO hay
que intercambiar Left/Right en ningún backend. Si algún día se deja de
espejar el frame, hay que invertir la etiqueta aquí (y solo aquí).

Timestamps: el modo VIDEO de Tasks exige timestamps en ms estrictamente
crecientes; `HandTracker` los genera desde `time.monotonic()` (o sanea los que
le pase el llamador) para que nunca se repitan ni retrocedan.

REGLA DURA del proyecto: este módulo vive FUERA de tletl_core precisamente
porque importa mediapipe (lazy, solo al crear un backend), cv2 (lazy, solo
para dibujar) y tools.fetch_models (lazy, solo para descargar un modelo).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from tletl_core.geometry import Point

ENV_HOME = "TLETL_HOME"
ENV_MODEL = "TLETL_TRACKER_MODEL"
MODELS_SUBDIR = "models"
HAND_MODEL_NAME = "hand_landmarker.task"
GESTURE_MODEL_NAME = "gesture_recognizer.task"
FETCH_HINT = "./launchers/tletl-fetch-models.sh"
BACKENDS = ("auto", "tasks", "legacy")
NUM_LANDMARKS = 21

Downloader = Callable[[str, Path], Any]   # (nombre del modelo, destino) -> descarga o lanza
Logger = Callable[[str], None]

# Mismas 21 conexiones que mp.solutions.hands.HAND_CONNECTIONS (palma + 5 dedos).
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),                    # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),                    # índice
    (5, 9), (9, 10), (10, 11), (11, 12),               # medio
    (9, 13), (13, 14), (14, 15), (15, 16),             # anular
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),   # meñique + base de la palma
)


@dataclass
class HandDetection:
    """Una mano detectada: 21 landmarks normalizados (x, y en [0,1], z relativo)."""
    landmarks: List[Point]
    handedness: str = "Right"   # 'Left' | 'Right' (ver docstring del módulo)
    score: float = 0.0          # confianza de la lateralidad

    @property
    def is_complete(self) -> bool:
        return len(self.landmarks) == NUM_LANDMARKS


class TrackerModelMissing(FileNotFoundError):
    """Falta el archivo .task del backend tasks (y no se pudo descargar)."""


class TrackerUnavailable(RuntimeError):
    """mediapipe no está instalado, no carga, o no trae el backend pedido."""


def _flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


# ---------------------------------------------------------------------------
# Rutas de modelos
# ---------------------------------------------------------------------------

def default_models_dir() -> Path:
    """$TLETL_HOME/models (default ~/.tletl/models). Misma raíz que el bus."""
    home = os.environ.get(ENV_HOME, "").strip()
    base = Path(home).expanduser() if home else Path.home() / ".tletl"
    return base / MODELS_SUBDIR


def resolve_model_path(configured: str | os.PathLike[str] | None,
                       name: str = HAND_MODEL_NAME) -> Path:
    """Ruta del modelo: env TLETL_TRACKER_MODEL > [tracker].model_path > models_dir/name.

    Si el valor apunta a un directorio (o termina en '/'), se le añade `name`.
    """
    env = os.environ.get(ENV_MODEL, "").strip()
    raw = env or (str(configured).strip() if configured else "")
    if not raw:
        return default_models_dir() / name
    p = Path(raw).expanduser()
    if raw.endswith(("/", os.sep)) or p.is_dir():
        return p / name
    return p


def _default_downloader(name: str, dest: Path, log: Logger = print) -> None:
    """Descarga `name` con tools/fetch_models.py (import lazy: es una herramienta, no una dependencia)."""
    from tools.fetch_models import MODELS, download, human_size  # noqa: PLC0415

    url = MODELS.get(name)
    if not url:
        raise TrackerModelMissing(f"no conozco la URL de descarga de {name}")
    log(f"[TLETL] falta {name}: descargando (~8 MB) a {dest} ...")
    status, size = download(url, dest)
    log(f"[TLETL] {name} listo ({human_size(size)}).")


def ensure_model_file(model_path: Path, name: str, *, auto_download: bool,
                      downloader: Optional[Downloader] = None, log: Logger = print) -> Path:
    """Devuelve `model_path` si existe. Si falta: con auto_download lo descarga
    (una vez, al directorio configurado); si no, o si la descarga falla, lanza
    TrackerModelMissing con la instrucción de ejecutar tletl-fetch-models.sh."""
    model_path = Path(model_path)
    if model_path.is_file():
        return model_path
    hint = (f"Ejecuta {FETCH_HINT} (o apunta [tracker] model_path / TLETL_TRACKER_MODEL al archivo .task).")
    if not auto_download:
        raise TrackerModelMissing(f"No encuentro el modelo {model_path} y [tracker] auto_download está apagado. {hint}")
    try:
        if downloader is None:
            _default_downloader(name, model_path, log)
        else:
            downloader(name, model_path)
    except TrackerModelMissing:
        raise
    except Exception as exc:
        raise TrackerModelMissing(f"No encuentro el modelo {model_path} y la descarga automática falló: {exc}. {hint}") from exc
    if not model_path.is_file():
        raise TrackerModelMissing(f"La descarga de {name} no dejó el archivo en {model_path}. {hint}")
    return model_path


# ---------------------------------------------------------------------------
# Disponibilidad de backends
# ---------------------------------------------------------------------------

def _import_mediapipe():
    # Baja el ruido de glog (versiones 0.10.x lo respetan; en 1.x los "W0000
    # inference_feedback_manager" salen igual y son inofensivos). Solo si el
    # usuario no lo fijó él mismo.
    os.environ.setdefault("GLOG_minloglevel", "2")
    import mediapipe as mp  # noqa: PLC0415 - lazy a propósito
    return mp


def mediapipe_version() -> Optional[str]:
    """'1.0.1', '0.10.21'... o None si mediapipe no importa."""
    try:
        return str(getattr(_import_mediapipe(), "__version__", "?"))
    except Exception:
        return None


def legacy_available() -> bool:
    """True si mediapipe trae la API legacy `mp.solutions.hands` (<= 0.10.2x)."""
    try:
        mp = _import_mediapipe()
    except Exception:  # ImportError, OSError (.so ausente), etc.
        return False
    solutions = getattr(mp, "solutions", None)
    return solutions is not None and hasattr(solutions, "hands")


def tasks_available() -> bool:
    """True si mediapipe trae la Tasks API con HandLandmarker."""
    try:
        _import_mediapipe()
        from mediapipe.tasks.python import vision  # noqa: PLC0415
    except Exception:
        return False
    return hasattr(vision, "HandLandmarker")


def normalize_backend_name(requested: str | None) -> str:
    name = (requested or "auto").strip().lower()
    if name not in BACKENDS:
        raise ValueError(f"backend desconocido {requested!r}; usa uno de {BACKENDS}")
    return name


def select_backend(requested: str | None) -> str:
    """'auto' | 'tasks' | 'legacy' -> 'tasks' | 'legacy' (sin crear ningún modelo).

    'auto' = tasks si la Tasks API se importa; legacy solo si no. Un nombre
    desconocido es ValueError; sin ningún backend, TrackerUnavailable.
    """
    name = normalize_backend_name(requested)
    if name != "auto":
        return name
    if tasks_available():
        return "tasks"
    if legacy_available():
        return "legacy"
    raise TrackerUnavailable(
        "mediapipe no está instalado o no carga (ni la Tasks API ni mp.solutions). "
        "Instala con: pip install 'mediapipe>=0.10.21' (Python 3.12)."
    )


def describe_backend(cfg_tracker: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resumen PURO del detector (no crea modelos ni descarga nada). Para el
    healthcheck, el arranque de la app y los tests."""
    cfg = dict(cfg_tracker or {})
    requested = str(cfg.get("backend", "auto") or "auto")
    tasks_ok = tasks_available()
    legacy_ok = legacy_available()
    model_path = resolve_model_path(cfg.get("model_path"))
    try:
        from apps.common.gesture_hint import resolve_gesture_model_path  # noqa: PLC0415 (import lazy: evita ciclo)
        gesture_model = resolve_gesture_model_path(cfg)
    except Exception:  # pragma: no cover - defensivo
        gesture_model = model_path.parent / GESTURE_MODEL_NAME
    try:
        name = normalize_backend_name(requested)
    except ValueError:
        name = "auto"
    if name == "tasks" or (name == "auto" and tasks_ok):
        backend, reason = "tasks", ("forzado por config" if name == "tasks" else "Tasks API disponible")
    elif name == "legacy" or legacy_ok:
        backend, reason = "legacy", ("forzado por config" if name == "legacy" else "Tasks API no disponible; mp.solutions sí")
    else:
        backend, reason = "none", "mediapipe no importa"
    return {
        "requested": requested,
        "backend": backend,
        "reason": reason,
        "mediapipe_version": mediapipe_version(),
        "tasks_available": tasks_ok,
        "legacy_available": legacy_ok,
        "model_path": str(model_path),
        "model_present": model_path.is_file(),
        "gesture_model_path": str(gesture_model),
        "gesture_model_present": gesture_model.is_file(),
        "auto_download": _flag(cfg.get("auto_download", True)),
        "gesture_hint": _flag(cfg.get("gesture_hint", False)),
    }


def format_backend_summary(info: Dict[str, Any]) -> str:
    """Una línea legible del dict de describe_backend()."""
    present = "presente" if info.get("model_present") else "FALTA"
    return (f"detector backend={info.get('backend')} ({info.get('reason')}) | mediapipe "
            f"{info.get('mediapipe_version') or 'no instalado'} | tasks:{'sí' if info.get('tasks_available') else 'no'} "
            f"legacy:{'sí' if info.get('legacy_available') else 'no'} | modelo {info.get('model_path')} ({present}) "
            f"| auto_download:{'on' if info.get('auto_download') else 'off'} "
            f"| gesture_hint:{'on' if info.get('gesture_hint') else 'off'}")


# ---------------------------------------------------------------------------
# Conversión de resultados crudos -> HandDetection (funciones puras, testeables)
# ---------------------------------------------------------------------------

def normalize_handedness(label: Any) -> str:
    """'left'/'LEFT'/'Left' -> 'Left'; cualquier otra cosa -> 'Right'."""
    text = str(label or "").strip().lower()
    return "Left" if text.startswith("l") else "Right"


def _label_and_score(category: Any) -> Tuple[str, float]:
    # legacy: .label/.score ; tasks: .category_name/.display_name/.score
    label = (getattr(category, "category_name", None)
             or getattr(category, "label", None)
             or getattr(category, "display_name", None))
    try:
        score = float(getattr(category, "score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return normalize_handedness(label), score


def _points(raw_landmarks: Any) -> List[Point]:
    return [Point(float(p.x), float(p.y), float(getattr(p, "z", 0.0) or 0.0)) for p in raw_landmarks]


def detections_from_legacy(result: Any) -> List[HandDetection]:
    """Convierte el resultado de `mp.solutions.hands.Hands.process()`."""
    hands = getattr(result, "multi_hand_landmarks", None) or []
    handed = getattr(result, "multi_handedness", None) or []
    out: List[HandDetection] = []
    for i, hand in enumerate(hands):
        label, score = "Right", 0.0
        if i < len(handed):
            classes = getattr(handed[i], "classification", None) or []
            if classes:
                label, score = _label_and_score(classes[0])
        out.append(HandDetection(_points(hand.landmark), label, score))
    return out


def detections_from_tasks(result: Any) -> List[HandDetection]:
    """Convierte un `HandLandmarkerResult` (o `GestureRecognizerResult`) de Tasks."""
    hands = getattr(result, "hand_landmarks", None) or []
    handed = getattr(result, "handedness", None) or []
    out: List[HandDetection] = []
    for i, hand in enumerate(hands):
        label, score = "Right", 0.0
        if i < len(handed) and handed[i]:
            label, score = _label_and_score(handed[i][0])
        out.append(HandDetection(_points(hand), label, score))
    return out


# ---------------------------------------------------------------------------
# Timestamps para el modo VIDEO
# ---------------------------------------------------------------------------

class MonotonicTimestamps:
    """Genera timestamps en ms estrictamente crecientes (exigencia de Tasks VIDEO).

    Dos frames en el mismo milisegundo, un reloj que se queda quieto o un
    timestamp externo repetido se resuelven sumando 1 ms: MediaPipe lanza
    ValueError si el timestamp no crece.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._t0 = clock()
        self.last = 0

    def next(self, timestamp_ms: Optional[int] = None) -> int:
        if timestamp_ms is None:
            ts = int((self._clock() - self._t0) * 1000.0) + 1
        else:
            ts = int(timestamp_ms)
        if ts <= self.last:
            ts = self.last + 1
        self.last = ts
        return ts


# ---------------------------------------------------------------------------
# Backends concretos (solo aquí se toca mediapipe)
# ---------------------------------------------------------------------------

def _mp_image_types(mp: Any) -> Tuple[Any, Any]:
    """(Image, ImageFormat) tanto en 0.10.x (mp.Image) como en 1.x."""
    image = getattr(mp, "Image", None)
    fmt = getattr(mp, "ImageFormat", None)
    if image is not None and fmt is not None:
        return image, fmt
    from mediapipe.tasks.python.vision.core import image as image_lib  # noqa: PLC0415
    return image_lib.Image, image_lib.ImageFormat


class _LegacyBackend:
    name = "legacy"

    def __init__(self, num_hands: int, det_conf: float, track_conf: float, model_complexity: int = 1):
        try:
            mp = _import_mediapipe()
        except Exception as exc:
            raise TrackerUnavailable(f"no puedo importar mediapipe: {exc}") from exc
        solutions = getattr(mp, "solutions", None)
        if solutions is None or not hasattr(solutions, "hands"):
            raise TrackerUnavailable(
                f"backend 'legacy' pedido pero mediapipe {getattr(mp, '__version__', '?')} ya no trae "
                "mp.solutions. Usa [tracker] backend = \"tasks\" (o \"auto\") y descarga el modelo con "
                f"{FETCH_HINT}"
            )
        self._hands = solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=int(num_hands),
            model_complexity=int(model_complexity),
            min_detection_confidence=float(det_conf),
            min_tracking_confidence=float(track_conf),
        )

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> List[HandDetection]:
        return detections_from_legacy(self._hands.process(rgb))

    def close(self) -> None:
        self._hands.close()


class _TasksBackend:
    name = "tasks"

    def __init__(self, model_path: Path, num_hands: int, det_conf: float, track_conf: float):
        self.model_path = Path(model_path)
        # La comprobación del modelo va ANTES de importar mediapipe: el mensaje
        # útil ("descarga el modelo") no debe depender de que mediapipe cargue.
        if not self.model_path.is_file():
            raise TrackerModelMissing(f"No encuentro el modelo {self.model_path}. Ejecuta {FETCH_HINT}.")
        try:
            mp = _import_mediapipe()
            from mediapipe.tasks.python import BaseOptions, vision  # noqa: PLC0415
        except Exception as exc:
            raise TrackerUnavailable(f"no puedo importar la Tasks API de mediapipe: {exc}") from exc
        self._image_cls, self._image_fmt = _mp_image_types(mp)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=int(num_hands),
            min_hand_detection_confidence=float(det_conf),
            min_hand_presence_confidence=float(det_conf),
            min_tracking_confidence=float(track_conf),
        )
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except OSError as exc:  # librería nativa ausente (libEGL/libGLES)
            raise TrackerUnavailable(
                f"MediaPipe Tasks no pudo cargar su librería nativa: {exc}. "
                "Fedora: sudo dnf install mesa-libEGL mesa-libGLES ; Debian/Ubuntu: apt install libegl1 libgles2"
            ) from exc
        except Exception as exc:  # RuntimeError "Unable to open zip archive": .task corrupto/truncado
            # Sin esto el error crudo de MediaPipe salía tal cual, saltándose el
            # respaldo legacy de `create(auto)` y sin decir cómo arreglarlo.
            raise TrackerUnavailable(
                f"MediaPipe Tasks no pudo cargar el modelo {self.model_path} ({exc}). "
                f"Si el archivo está corrupto o a medias, bórralo o ejecuta {FETCH_HINT} --force"
            ) from exc

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> List[HandDetection]:
        image = self._image_cls(image_format=self._image_fmt.SRGB, data=np.ascontiguousarray(rgb))
        return detections_from_tasks(self._landmarker.detect_for_video(image, int(timestamp_ms)))

    def close(self) -> None:
        self._landmarker.close()


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

class HandTracker:
    """Detector de manos con backend intercambiable.

    Uso:
        tracker = HandTracker.create(cfg["tracker"], num_hands=2)
        dets = tracker.process(rgb)          # rgb: HxWx3 uint8, ya espejado
        tracker.close()

    `backend` es cualquier objeto con `detect(rgb, timestamp_ms) -> List[HandDetection]`
    y `close()`; los tests inyectan uno falso.
    """

    def __init__(self, backend: Any, name: Optional[str] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._backend = backend
        self._name = name or getattr(backend, "name", type(backend).__name__)
        self._stamps = MonotonicTimestamps(clock)
        self.frames = 0
        self.last_timestamp_ms = 0
        self.closed = False

    @property
    def backend_name(self) -> str:
        return self._name

    @classmethod
    def create(cls, cfg_tracker: Optional[Dict[str, Any]] = None, num_hands: int = 2, *,
               downloader: Optional[Downloader] = None, log: Logger = print) -> "HandTracker":
        """Construye el backend según `[tracker]` (backend, det_conf, track_conf, model_path,
        model_complexity, auto_download).

        auto: intenta tasks (descargando el modelo si hace falta y auto_download lo permite);
        si eso falla y mp.solutions existe, cae a legacy avisando por qué. tasks/legacy fuerzan
        y propagan el error (TrackerModelMissing / TrackerUnavailable) con mensaje claro.
        """
        cfg = dict(cfg_tracker or {})
        requested = normalize_backend_name(cfg.get("backend", "auto"))
        det_conf = float(cfg.get("det_conf", 0.72))
        track_conf = float(cfg.get("track_conf", 0.72))
        hands = max(1, int(num_hands))
        auto_download = _flag(cfg.get("auto_download", True))

        def make_legacy() -> "HandTracker":
            return cls(_LegacyBackend(hands, det_conf, track_conf, int(cfg.get("model_complexity", 1))), "legacy")

        def make_tasks() -> "HandTracker":
            model = ensure_model_file(resolve_model_path(cfg.get("model_path")), HAND_MODEL_NAME,
                                      auto_download=auto_download, downloader=downloader, log=log)
            return cls(_TasksBackend(model, hands, det_conf, track_conf), "tasks")

        if requested == "tasks":
            return make_tasks()
        if requested == "legacy":
            return make_legacy()

        # auto
        if tasks_available():
            try:
                tracker = make_tasks()
                log("[TLETL] detector: backend tasks (MediaPipe HandLandmarker).")
                return tracker
            except (TrackerModelMissing, TrackerUnavailable) as exc:
                if not legacy_available():
                    raise
                log(f"[TLETL] detector: backend tasks no disponible ({exc}); uso legacy (mp.solutions).")
                return make_legacy()
        if legacy_available():
            log("[TLETL] detector: backend legacy (mp.solutions); este mediapipe no trae la Tasks API.")
            return make_legacy()
        raise TrackerUnavailable(
            "mediapipe no está instalado o no carga (ni la Tasks API ni mp.solutions). "
            "Instala con: pip install 'mediapipe>=0.10.21' (Python 3.12)."
        )

    def process(self, rgb: np.ndarray, timestamp_ms: Optional[int] = None) -> List[HandDetection]:
        """Detecta manos en un frame RGB (HxWx3 uint8). Devuelve [] si no hay manos."""
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"process() espera un frame RGB HxWx3, recibí shape {getattr(rgb, 'shape', None)}")
        ts = self._stamps.next(timestamp_ms)
        self.last_timestamp_ms = ts
        self.frames += 1
        detections = self._backend.detect(rgb, ts)
        # Una mano incompleta (recorte del frame, resultado a medias) rompería
        # extract_live_features; se descarta aquí y no en cada app.
        return [d for d in detections if d.is_complete]

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Dibujo (sustituye a mp.solutions.drawing_utils)
# ---------------------------------------------------------------------------

def landmarks_to_pixels(detection: HandDetection, width: int, height: int) -> List[Tuple[int, int]]:
    return [(int(round(p.x * width)), int(round(p.y * height))) for p in detection.landmarks]


def draw_landmarks(frame_bgr: np.ndarray, detection: HandDetection, *,
                   color: Tuple[int, int, int] = (0, 200, 0),
                   point_color: Tuple[int, int, int] = (0, 0, 255),
                   thickness: int = 2, radius: int = 3) -> None:
    """Dibuja el esqueleto de la mano (21 puntos + HAND_CONNECTIONS) sobre un frame BGR."""
    import cv2  # noqa: PLC0415 - lazy: el módulo debe importarse sin cv2

    h, w = frame_bgr.shape[:2]
    pts = landmarks_to_pixels(detection, w, h)
    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame_bgr, pts[a], pts[b], color, thickness, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame_bgr, (x, y), radius, point_color, -1, cv2.LINE_AA)
