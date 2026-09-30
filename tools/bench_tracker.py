"""tools/bench_tracker.py — mide ms/frame de HandTracker.process por backend.

    python -m tools.bench_tracker --synthetic                 # sin cámara: frames sintéticos
    python -m tools.bench_tracker --synthetic --backend tasks --frames 200
    python -m tools.bench_tracker --camera 0 --frames 150     # con cámara real (mano delante)

Sirve para responder "¿de dónde salen mis 15 fps?": si el detector tarda ~12 ms
y el bucle completo tarda 65 ms, el cuello no es MediaPipe. El KNN se mide con
`tests/test_classifier.py` (equivalencia + tiempo) o a mano:
`python -c "from tletl_core.classifier import RobustKNNRuntime; ..."`.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np


def synthetic_frames(n: int, width: int = 640, height: int = 360, seed: int = 0) -> Iterator[np.ndarray]:
    """Frames RGB con un blob claro que se mueve (evita que el tracker cachee un frame idéntico)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 40, size=(height, width, 3), dtype=np.uint8)
    for i in range(n):
        frame = base.copy()
        cx = int((0.2 + 0.6 * (i / max(n - 1, 1))) * width)
        cy = height // 2
        r = min(width, height) // 6
        frame[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = 200
        yield frame


def bench(tracker: Any, frames: Iterable[np.ndarray], warmup: int = 5) -> Dict[str, float]:
    """Cronometra tracker.process sobre `frames` (descarta `warmup` frames iniciales)."""
    times: List[float] = []
    hands = 0
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        dets = tracker.process(frame)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            times.append(dt)
            hands += len(dets)
    if not times:
        return {"frames": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "hands": 0}
    arr = np.asarray(times)
    return {
        "frames": len(times),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
        "hands": hands,
    }


def format_result(backend: str, res: Dict[str, float]) -> str:
    return (f"{backend:>7}: {res['frames']} frames | media {res['mean_ms']:.1f} ms | p50 {res['p50_ms']:.1f} | "
            f"p95 {res['p95_ms']:.1f} | max {res['max_ms']:.1f} | manos vistas {int(res['hands'])} "
            f"| ~{1000.0 / max(res['mean_ms'], 1e-6):.0f} fps solo detector")


def camera_frames(index: int, n: int, width: int, height: int) -> Iterator[np.ndarray]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir la cámara {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        for _ in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Benchmark de HandTracker.process (ms/frame por backend).")
    ap.add_argument("--synthetic", action="store_true", help="usa frames sintéticos (sin cámara; es el default si no das --camera)")
    ap.add_argument("--camera", type=int, default=None, help="índice de cámara real (p.ej. 0); mide con tu mano delante")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--backend", choices=["auto", "legacy", "tasks", "all"], default="auto",
                    help="backend a medir; 'all' prueba legacy y tasks si están disponibles")
    ap.add_argument("--num-hands", type=int, default=2)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    from apps.common.hand_tracker import HandTracker, TrackerModelMissing, TrackerUnavailable  # noqa: PLC0415
    from tletl_core.config import load_config  # noqa: PLC0415

    args = build_parser().parse_args(argv)
    cfg = load_config()
    backends = ["legacy", "tasks"] if args.backend == "all" else [args.backend]
    exit_code = 0
    for backend in backends:
        trk = dict(cfg["tracker"])
        trk["backend"] = backend
        try:
            tracker = HandTracker.create(trk, num_hands=args.num_hands)
        except (TrackerModelMissing, TrackerUnavailable, ValueError) as exc:
            print(f"{backend:>7}: no disponible ({exc})")
            exit_code = 1
            continue
        try:
            # Sintético salvo que se pida una cámara explícita (--camera N).
            if args.synthetic or args.camera is None:
                frames: Iterable[np.ndarray] = synthetic_frames(args.frames, args.width, args.height)
            else:
                frames = camera_frames(int(args.camera), args.frames, args.width, args.height)
            print(format_result(tracker.backend_name, bench(tracker, frames)))
        finally:
            tracker.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
