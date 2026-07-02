from __future__ import annotations

"""Estadísticas del banco de gestos de Tletl v5.

Reporta: total de muestras, distribución por gesto, balance, features compartidas
útiles (las que el clasificador realmente usa) y posibles problemas (labels
faltantes, muestras con pocas features, claves de orientación ausentes).

Uso:  python -m tools.bank_probe [--bank ruta.jsonl]
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from tletl_core.features import LABELS, META_KEYS, get_features, get_label
from tletl_core.orientation import ORIENTATION_KEYS

DEFAULT_BANK = Path(__file__).resolve().parent.parent / "datasets" / "tletl_gesture_bank_v2_features.jsonl"


def probe_bank(path: str | Path) -> Dict[str, Any]:
    """Analiza el banco y devuelve un dict de estadísticas (sin imprimir)."""
    p = Path(path).expanduser().resolve()
    rows: List[dict] = []
    malformed = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1

    per_label: Counter[str] = Counter()
    feat_counts: Counter[str] = Counter()
    low_feature_rows = 0
    no_orientation_rows = 0
    unknown_labels: Counter[str] = Counter()

    for obj in rows:
        label = get_label(obj)
        feats = get_features(obj)
        if not label or label not in LABELS:
            unknown_labels[str(label)] += 1
            continue
        per_label[label] += 1
        if len(feats) < 12:
            low_feature_rows += 1
        for k in feats:
            if k not in META_KEYS:
                feat_counts[k] += 1
        if not any(k in feats for k in ORIENTATION_KEYS):
            no_orientation_rows += 1

    valid = sum(per_label.values())
    shared = sorted(k for k, c in feat_counts.items() if c >= max(10, int(valid * 0.25)))
    counts = dict(per_label)
    balance = (min(counts.values()) / max(counts.values())) if counts else 0.0

    return {
        "path": str(p),
        "total_lines": len(rows) + malformed,
        "valid_samples": valid,
        "malformed_lines": malformed,
        "per_label": counts,
        "missing_labels": [g for g in LABELS if g not in counts],
        "balance_ratio": round(balance, 3),       # 1.0 = perfectamente balanceado
        "shared_features": len(shared),
        "low_feature_rows": low_feature_rows,
        "rows_without_orientation": no_orientation_rows,
        "unknown_labels": dict(unknown_labels),
    }


def format_report(stats: Dict[str, Any]) -> str:
    lines = [
        "== Tletl v5 — bank probe ==",
        f"banco: {stats['path']}",
        f"muestras válidas: {stats['valid_samples']} / {stats['total_lines']} líneas",
        f"features compartidas útiles: {stats['shared_features']}",
        f"balance (min/max): {stats['balance_ratio']}",
        "distribución por gesto:",
    ]
    for g in LABELS:
        lines.append(f"  {g:10s}: {stats['per_label'].get(g, 0)}")
    if stats["missing_labels"]:
        lines.append(f"⚠ labels faltantes: {stats['missing_labels']}")
    if stats["malformed_lines"]:
        lines.append(f"⚠ líneas malformadas: {stats['malformed_lines']}")
    if stats["low_feature_rows"]:
        lines.append(f"⚠ muestras con <12 features: {stats['low_feature_rows']}")
    if stats["rows_without_orientation"]:
        lines.append(f"⚠ muestras sin orientación: {stats['rows_without_orientation']}")
    if stats["unknown_labels"]:
        lines.append(f"⚠ labels desconocidos: {stats['unknown_labels']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Estadísticas del banco de gestos Tletl v5")
    ap.add_argument("--bank", default=str(DEFAULT_BANK))
    args = ap.parse_args()
    print(format_report(probe_bank(args.bank)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
