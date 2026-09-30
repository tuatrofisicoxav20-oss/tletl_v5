"""
tools/gesture_bank.py
Herramienta de captura de gestos para el banco de entrenamiento de Tletl v5.

Uso:
    python3 -m tools.gesture_bank --dataset gestures.jsonl
    python3 -m tools.gesture_bank --dataset gestures.jsonl --dual-hand
    python3 -m tools.gesture_bank --backend tasks     # fuerza el HandLandmarker (Tasks API)

Teclas:
    1-7  seleccionan el gesto activo
    SPACE guarda una muestra
    A    activa/desactiva autosave
    Q    sale

El detector es apps.common.hand_tracker.HandTracker (backend auto/legacy/tasks,
default [tracker].backend de config/tletl.toml), así que funciona igual con
mediapipe 0.10.21 (API legacy) y con 1.x (Tasks). Los landmarks son los mismos
21 puntos: las muestras nuevas son compatibles con el banco existente.

cv2, mediapipe y el tracker se importan de forma LAZY dentro de main() para que
el módulo pueda importarse sin esas dependencias (tests/test_blender_map.py).

Bug corregido en v5.2: antes se releía y parseaba el JSONL COMPLETO en cada
frame solo para mostrar los conteos (con 2 600 muestras, varios ms por frame y
creciendo). Ahora `SampleCounter` carga una vez y se actualiza al guardar.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ── Constantes públicas ──────────────────────────────────────────────────────

LABELS = ["OPEN_PALM", "FIST", "POINT", "VICTORY", "PINCH", "THREE", "NEUTRAL"]

GESTURE_HELP = {
    "OPEN_PALM": "Palma abierta natural, dedos separados.",
    "FIST":      "Puño cerrado normal.",
    "POINT":     "Solo índice extendido.",
    "VICTORY":   "Índice y medio extendidos.",
    "PINCH":     "Pulgar e índice juntos, sin cerrar toda la mano.",
    "THREE":     "Tres dedos extendidos.",
    "NEUTRAL":   "Mano relajada que NO debe hacer acciones.",
}

# ord('1')..ord('7') → LABELS[0..6]
KEY_TO_GESTURE: Dict[int, str] = {ord(str(i + 1)): lbl for i, lbl in enumerate(LABELS)}

BACKEND_CHOICES = ("auto", "legacy", "tasks")


# ── Funciones puras testeables ───────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Devuelve el ArgumentParser configurado sin ejecutar nada.

    Los defaults None (índice de cámara y claves de [tracker]) se resuelven en
    main() contra config/tletl.toml; así el toml manda y la CLI sobreescribe.
    Ancho/alto/fps son fijos (640x480@30): la captura no necesita más.
    """
    p = argparse.ArgumentParser(
        description="Tletl v5 – herramienta de captura de gestos al banco JSONL",
    )
    p.add_argument(
        "--dataset", "-d",
        default="gesture_bank.jsonl",
        help="Ruta al archivo JSONL donde se guardan las muestras.",
    )
    p.add_argument("--config", default=None, help="ruta a tletl.toml (default: config/tletl.toml)")
    p.add_argument(
        "--camera", "-c",
        type=int, default=None,
        help="Índice de cámara (default: [camera].index / TLETL_CAMERA).",
    )
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps",    type=int, default=30)
    p.add_argument(
        "--backend", choices=BACKEND_CHOICES, default=None,
        help="Detector: auto | legacy (mp.solutions) | tasks (HandLandmarker). Default: [tracker].backend.",
    )
    p.add_argument(
        "--model-complexity", type=int, default=None,
        help="Complejidad del modelo (solo backend legacy: 0, 1 o 2). Default: [tracker].model_complexity.",
    )
    p.add_argument("--det-conf",   type=float, default=None, help="Default: [tracker].det_conf")
    p.add_argument("--track-conf", type=float, default=None, help="Default: [tracker].track_conf")
    p.add_argument(
        "--autosave-interval", type=float, default=0.8,
        help="Segundos mínimos entre guardados automáticos.",
    )
    p.add_argument(
        "--dual-hand", action="store_true",
        help="Activar detección de dos manos y etiquetar cada muestra con Left/Right.",
    )
    return p


def append_sample(
    path: str | Path,
    label: str,
    features: Dict[str, float],
    handedness: str = "Right",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Añade una muestra al banco JSONL.

    Cada línea tiene el formato:
        {"label": "...", "features": {...}, "meta": {"handedness": "..."}, "timestamp": ...}

    Es una función pura (sin estado global) y testeable sin cv2 ni mediapipe.
    """
    record: Dict[str, Any] = {
        "label":    label,
        "features": features,
        "meta":     {"handedness": handedness, **(extra_meta or {})},
        "timestamp": time.time(),
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_samples(path: str | Path) -> list[Dict[str, Any]]:
    """Lee todas las muestras del banco y las devuelve como lista (líneas rotas se ignoran)."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def count_by_label(rows: list[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        lbl = r.get("label", "?") if isinstance(r, dict) else "?"
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def format_counts(counts: Dict[str, int]) -> str:
    """'OPEN_PALM:12  FIST:3  ...' en el orden de LABELS (los 7 siempre aparecen)."""
    return "  ".join(f"{lbl}:{counts.get(lbl, 0)}" for lbl in LABELS)


def short_counts(rows: list[Dict[str, Any]]) -> str:
    return format_counts(count_by_label(rows))


class SampleCounter:
    """Conteo por etiqueta en memoria: se carga UNA vez y se actualiza al guardar.

    Sustituye al `load_samples()` por frame que releía todo el JSONL.
    """

    def __init__(self, counts: Optional[Dict[str, int]] = None):
        self.counts: Dict[str, int] = dict(counts or {})

    @classmethod
    def from_file(cls, path: str | Path) -> "SampleCounter":
        return cls(count_by_label(load_samples(path)))

    def add(self, label: str, n: int = 1) -> None:
        self.counts[label] = self.counts.get(label, 0) + int(n)

    def get(self, label: str) -> int:
        return int(self.counts.get(label, 0))

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))

    def short(self) -> str:
        return format_counts(self.counts)


# ── UI helpers ───────────────────────────────────────────────────────────────

def _draw_text_box(frame: Any, lines: list[str], font_scale: float = 0.48) -> None:
    """Dibuja un cuadro de texto semitransparente sobre el frame (requiere cv2)."""
    import cv2  # noqa: PLC0415 – lazy import intencional

    font      = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    pad       = 8
    line_h    = int(font_scale * 30) + 6
    box_h     = line_h * len(lines) + pad * 2
    box_w     = 520

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        y = pad + (i + 1) * line_h
        cv2.putText(frame, line, (pad, y), font, font_scale, (200, 255, 200), thickness, cv2.LINE_AA)


def tracker_config(args: argparse.Namespace, cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """[tracker] de la config con los overrides de la CLI (solo los no-None)."""
    trk = dict(cfg.get("tracker", {}))
    for attr, key in (("backend", "backend"), ("det_conf", "det_conf"),
                      ("track_conf", "track_conf"), ("model_complexity", "model_complexity")):
        value = getattr(args, attr, None)
        if value is not None:
            trk[key] = value
    return trk


# ── Punto de entrada principal ───────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    # Todos los imports pesados van aquí para que el módulo sea importable sin ellos
    import cv2                                                     # noqa: PLC0415
    from apps.common.hand_tracker import (                         # noqa: PLC0415
        HandTracker, describe_backend, draw_landmarks, format_backend_summary,
    )
    from tletl_core.config import load_config                      # noqa: PLC0415
    from tletl_core.features import extract_live_features          # noqa: PLC0415

    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    camera_index = args.camera if args.camera is not None else int(cfg["camera"]["index"])

    dataset_path    = Path(args.dataset).expanduser().resolve()
    max_num_hands   = 2 if args.dual_hand else 1
    selected        = "OPEN_PALM"
    autosave        = False
    last_saved      = 0.0
    saved_flash     = ""
    counter         = SampleCounter.from_file(dataset_path)   # UNA lectura, no una por frame

    print("[TLETL v5] Banco de gestos")
    print("1 palma | 2 puño | 3 índice | 4 victoria | 5 pinza | 6 tres | 7 neutral")
    print("SPACE guarda | A autosave | Q salir")
    print(f"[DATASET] {dataset_path} ({counter.total} muestras) — {counter.short()}")
    if args.dual_hand:
        print("[DUAL-HAND] Detectando hasta 2 manos; cada muestra se etiqueta con Left/Right")

    trk = tracker_config(args, cfg)
    print(f"[TRACKER] {format_backend_summary(describe_backend(trk))}")
    tracker = HandTracker.create(trk, num_hands=max_num_hands)
    print(f"[TRACKER] backend en uso: {tracker.backend_name}")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        tracker.close()
        raise SystemExit(f"[ERROR] No pude abrir la cámara {camera_index}. Prueba --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] No pude leer cámara.")
                break

            frame = cv2.flip(frame, 1)   # espejo ANTES de detectar (lateralidad correcta, ver hand_tracker)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            detected: list[tuple[str, dict]] = []  # (handedness, features)
            for det in tracker.process(rgb):
                detected.append((det.handedness, extract_live_features(det.landmarks)))
                draw_landmarks(frame, det)

            lines = [
                "TLETL V5 – BANCO DE GESTOS",
                f"Gesto seleccionado: {selected}",
                f"Cómo hacerlo: {GESTURE_HELP.get(selected, '')}",
                f"Autosave: {'ON' if autosave else 'OFF'} | Dataset: {dataset_path.name} | backend: {tracker.backend_name}",
                counter.short(),
                "1 palma | 2 puño | 3 índice | 4 victoria | 5 pinza | 6 tres | 7 neutral",
                "SPACE=guardar muestra | A=autosave | Q=salir",
            ]
            if not detected:
                lines.append("No veo mano. Luz de frente, mano dentro del cuadro.")
            else:
                labels_str = ", ".join(f"{s}" for s, _ in detected)
                lines.append(f"Mano(s) detectada(s): {labels_str}")
            if saved_flash:
                lines.append(saved_flash)

            _draw_text_box(frame, lines, font_scale=0.48)
            cv2.imshow("Tletl v5 – Banco de Gestos", frame)

            key = cv2.waitKey(1) & 0xFF
            now = cv2.getTickCount() / cv2.getTickFrequency()

            if key == ord("q"):
                break
            if key in KEY_TO_GESTURE:
                selected    = KEY_TO_GESTURE[key]
                saved_flash = ""
            if key == ord("a"):
                autosave    = not autosave
                saved_flash = f"Autosave {'ON' if autosave else 'OFF'}"

            should_save = key == 32  # SPACE
            if autosave and detected and now - last_saved > args.autosave_interval:
                should_save = True

            if should_save and detected:
                for side, features in detected:
                    append_sample(dataset_path, selected, features, handedness=side)
                counter.add(selected, len(detected))
                last_saved  = now
                n           = len(detected)
                saved_flash = f"[OK] guardada(s) {n} muestra(s) → {selected} (total {counter.get(selected)})"
                print(saved_flash)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
