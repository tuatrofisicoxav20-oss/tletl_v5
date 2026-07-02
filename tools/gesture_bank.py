"""
tools/gesture_bank.py
Herramienta de captura de gestos para el banco de entrenamiento de Tletl v5.

Uso:
    python3 -m tools.gesture_bank --dataset gestures.jsonl
    python3 -m tools.gesture_bank --dataset gestures.jsonl --dual-hand

Teclas:
    1-7  seleccionan el gesto activo
    SPACE guarda una muestra
    A    activa/desactiva autosave
    Q    sale

cv2 y mediapipe se importan de forma LAZY dentro de main() para que el módulo
pueda importarse sin esas dependencias.
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


# ── Funciones puras testeables ───────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Devuelve el ArgumentParser configurado sin ejecutar nada."""
    p = argparse.ArgumentParser(
        description="Tletl v5 – herramienta de captura de gestos al banco JSONL",
    )
    p.add_argument(
        "--dataset", "-d",
        default="gesture_bank.jsonl",
        help="Ruta al archivo JSONL donde se guardan las muestras.",
    )
    p.add_argument(
        "--camera", "-c",
        type=int, default=0,
        help="Índice de cámara (default: 0).",
    )
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps",    type=int, default=30)
    p.add_argument(
        "--model-complexity", type=int, default=1,
        help="Complejidad del modelo MediaPipe (0, 1 o 2).",
    )
    p.add_argument("--det-conf",   type=float, default=0.7)
    p.add_argument("--track-conf", type=float, default=0.6)
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
    """Lee todas las muestras del banco y las devuelve como lista."""
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
        lbl = r.get("label", "?")
        counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def short_counts(rows: list[Dict[str, Any]]) -> str:
    counts = count_by_label(rows)
    parts = [f"{lbl}:{counts.get(lbl, 0)}" for lbl in LABELS]
    return "  ".join(parts)


# ── UI helpers ───────────────────────────────────────────────────────────────

def _draw_text_box(frame: Any, lines: list[str], font_scale: float = 0.48) -> None:
    """Dibuja un cuadro de texto semitransparente sobre el frame (requiere cv2)."""
    import cv2  # noqa: PLC0415 – lazy import intencional

    font      = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    pad       = 8
    line_h    = int(font_scale * 30) + 6
    box_h     = line_h * len(lines) + pad * 2
    box_w     = 480

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        y = pad + (i + 1) * line_h
        cv2.putText(frame, line, (pad, y), font, font_scale, (200, 255, 200), thickness, cv2.LINE_AA)


# ── Punto de entrada principal ───────────────────────────────────────────────

def main() -> None:
    # Todos los imports pesados van aquí para que el módulo sea importable sin ellos
    import cv2                              # noqa: PLC0415
    import mediapipe as mp                  # noqa: PLC0415

    # core
    from tletl_core.geometry import points_from_mediapipe  # noqa: PLC0415
    from tletl_core.features import extract_live_features  # noqa: PLC0415

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    args = build_parser().parse_args()

    dataset_path    = Path(args.dataset).expanduser().resolve()
    max_num_hands   = 2 if args.dual_hand else 1
    selected        = "OPEN_PALM"
    autosave        = False
    last_saved      = 0.0
    saved_flash     = ""

    print("[TLETL v5] Banco de gestos")
    print("1 palma | 2 puño | 3 índice | 4 victoria | 5 pinza | 6 tres | 7 neutral")
    print("SPACE guarda | A autosave | Q salir")
    if args.dual_hand:
        print("[DUAL-HAND] Detectando hasta 2 manos; cada muestra se etiqueta con Left/Right")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_num_hands,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.det_conf,
        min_tracking_confidence=args.track_conf,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] No pude leer cámara.")
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            detected: list[tuple[str, dict]] = []  # (handedness, features)

            if result.multi_hand_landmarks and result.multi_handedness:
                for hl, hc in zip(result.multi_hand_landmarks, result.multi_handedness, strict=False):
                    side     = hc.classification[0].label  # "Left" o "Right"
                    points   = points_from_mediapipe(hl)
                    features = extract_live_features(points)
                    detected.append((side, features))
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            rows   = load_samples(dataset_path)
            lines  = [
                "TLETL V5 – BANCO DE GESTOS",
                f"Gesto seleccionado: {selected}",
                f"Cómo hacerlo: {GESTURE_HELP.get(selected, '')}",
                f"Autosave: {'ON' if autosave else 'OFF'} | Dataset: {dataset_path.name}",
                short_counts(rows),
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
                last_saved  = now
                n           = len(detected)
                saved_flash = f"[OK] guardada(s) {n} muestra(s) → {selected}"
                print(saved_flash)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
