from __future__ import annotations

"""Duel lab — compara configuraciones del clasificador KNN con holdout honesto.

Separa el banco en train/test (sin solape), entrena el RobustKNNRuntime solo con
train y mide accuracy sobre test. Permite comparar distintos k y pesos de
orientación para elegir la mejor configuración sin auto-engaño (no se evalúa sobre
las mismas muestras con las que se entrena).

Uso:  python -m tools.duel_lab [--bank ruta.jsonl]
"""

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tletl_core.classifier import RobustKNNRuntime
from tletl_core.features import get_features, get_label

DEFAULT_BANK = Path(__file__).resolve().parent.parent / "datasets" / "tletl_gesture_bank_v2_features.jsonl"


def _split_rows(bank_path: str | Path, holdout_frac: float, seed: int
                ) -> Tuple[List[dict], List[dict]]:
    rows = []
    for line in Path(bank_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if get_label(obj):
            rows.append(obj)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_test = max(1, int(len(rows) * holdout_frac))
    return rows[n_test:], rows[:n_test]   # (train, test)


def evaluate_knn(bank_path: str | Path, *, k: int = 13, orientation_weight: float = 0.45,
                 holdout_frac: float = 0.2, seed: int = 7,
                 max_test: Optional[int] = None) -> Dict[str, Any]:
    """Entrena con train y evalúa accuracy sobre test (holdout). Devuelve métricas."""
    train, test = _split_rows(bank_path, holdout_frac, seed)
    if max_test is not None:
        test = test[:max_test]

    # Escribir train a un jsonl temporal para alimentar el runtime.
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        for obj in train:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        tmp_path = fh.name

    try:
        runtime = RobustKNNRuntime(tmp_path, k=k, orientation_weight=orientation_weight)
        hits = 0
        total = 0
        per_label_hits: Dict[str, int] = {}
        per_label_total: Dict[str, int] = {}
        for obj in test:
            label = get_label(obj)
            feats = get_features(obj)
            if not label or len(feats) < 12:
                continue
            total += 1
            per_label_total[label] = per_label_total.get(label, 0) + 1
            pred = runtime.predict(feats, strict=False)
            if pred.raw_label == label:
                hits += 1
                per_label_hits[label] = per_label_hits.get(label, 0) + 1
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    accuracy = hits / total if total else 0.0
    per_label_acc = {
        g: round(per_label_hits.get(g, 0) / n, 3)
        for g, n in sorted(per_label_total.items())
    }
    return {
        "k": k, "orientation_weight": orientation_weight,
        "train": len(train), "test": total,
        "accuracy": round(accuracy, 4),
        "per_label_accuracy": per_label_acc,
    }


def duel(bank_path: str | Path, configs: List[Dict[str, float]], *,
         holdout_frac: float = 0.2, seed: int = 7,
         max_test: Optional[int] = None) -> List[Dict[str, Any]]:
    """Evalúa varias configuraciones y devuelve los resultados ordenados por accuracy."""
    results = []
    for cfg in configs:
        results.append(evaluate_knn(
            bank_path,
            k=int(cfg.get("k", 13)),
            orientation_weight=float(cfg.get("orientation_weight", 0.45)),
            holdout_frac=holdout_frac, seed=seed, max_test=max_test,
        ))
    results.sort(key=lambda r: r["accuracy"], reverse=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Comparar configuraciones del KNN de Tletl v5")
    ap.add_argument("--bank", default=str(DEFAULT_BANK))
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    configs = [
        {"k": 9, "orientation_weight": 0.45},
        {"k": 13, "orientation_weight": 0.45},
        {"k": 13, "orientation_weight": 0.25},
        {"k": 21, "orientation_weight": 0.45},
    ]
    print("== Tletl v5 — duel lab (holdout) ==")
    for r in duel(args.bank, configs, holdout_frac=args.holdout, seed=args.seed):
        print(f"  k={r['k']:>2} ow={r['orientation_weight']:.2f} "
              f"train={r['train']} test={r['test']} -> acc={r['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
