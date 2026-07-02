from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .features import LABELS, META_KEYS, get_features, get_label
from .orientation import ORIENTATION_KEYS, orientation_bucket


@dataclass
class Prediction:
    label: str
    raw_label: str
    confidence: float
    margin: float
    orientation: str
    reason: str
    ok: bool
    votes: Dict[str, float]


class RobustKNNRuntime:
    """Runtime KNN robusto de Tletl.

    Este módulo es parte del núcleo común: no sabe nada de Fedora, Blender, ydotool
    ni de ventanas OpenCV. Solo lee un banco y predice gesto + confianza.
    """

    def __init__(self, bank_path: str | Path, k: int = 13, orientation_weight: float = 0.45):
        self.bank_path = Path(bank_path).expanduser().resolve()
        self.k = int(k)
        self.orientation_weight = float(orientation_weight)
        self.samples: List[Tuple[str, Dict[str, float]]] = []
        self.keys: List[str] = []
        self.mean: Dict[str, float] = {}
        self.std: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}
        self.counts: Counter[str] = Counter()
        self.load()

    def load(self) -> None:
        rows: List[Tuple[str, Dict[str, float]]] = []
        for line in self.bank_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            label = get_label(obj)
            feat = get_features(obj)
            if not label or label not in LABELS or len(feat) < 12:
                continue
            rows.append((label, feat))

        if len(rows) < 20:
            raise RuntimeError(f"Banco insuficiente o ilegible: {self.bank_path} ({len(rows)} muestras válidas)")

        feat_counts = Counter()
        for _, feat in rows:
            for key in feat:
                if key not in META_KEYS and isinstance(feat[key], (int, float)):
                    feat_counts[key] += 1

        min_count = max(10, int(len(rows) * 0.25))
        keys = sorted(k for k, count in feat_counts.items() if count >= min_count)
        if len(keys) < 15:
            raise RuntimeError(f"Muy pocas features útiles compartidas: {len(keys)}")

        by_label: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for label, feat in rows:
            by_label[label].append(feat)

        # Balancea para que una clase enorme no aplaste a las demás.
        cap = max(220, int(np.median([len(v) for v in by_label.values()])))
        balanced: List[Tuple[str, Dict[str, float]]] = []
        for label in LABELS:
            feats = by_label.get(label, [])
            if len(feats) > cap:
                step = len(feats) / cap
                feats = [feats[int(i * step)] for i in range(cap)]
            balanced.extend((label, feat) for feat in feats)

        vals_by_key: Dict[str, List[float]] = {k: [] for k in keys}
        for _, feat in balanced:
            for k in keys:
                if k in feat and math.isfinite(float(feat[k])):
                    vals_by_key[k].append(float(feat[k]))

        for k in keys:
            vals = vals_by_key[k]
            m = float(np.mean(vals)) if vals else 0.0
            s = float(np.std(vals)) if len(vals) > 1 else 1.0
            self.mean[k] = m
            self.std[k] = max(s, 0.055)

        self.weights = {}
        for k in keys:
            w = 1.0
            if k in ORIENTATION_KEYS or k.startswith("palm_normal") or k.startswith("palm_") or "depth" in k or "roll" in k or "yaw" in k or "pitch" in k:
                w = self.orientation_weight
            if "thumb_tip_index_tip" in k or "pinch" in k:
                w *= 1.25
            if "curl" in k or "vertical" in k or "tip_mcp" in k or "tip_wrist" in k:
                w *= 1.10
            self.weights[k] = w

        self.samples = balanced
        self.keys = keys
        self.counts = Counter(label for label, _ in balanced)
        print(
            f"[TLETL Core Runtime] Banco cargado: {len(self.samples)} muestras, "
            f"{len(self.keys)} features, labels={dict(self.counts)}"
        )

    def distance(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        total = 0.0
        used = 0
        for k in self.keys:
            if k not in a or k not in b:
                continue
            av = float(a[k])
            bv = float(b[k])
            if not math.isfinite(av) or not math.isfinite(bv):
                continue
            z = (av - bv) / self.std[k]
            total += self.weights[k] * z * z
            used += 1
        if used < max(10, int(len(self.keys) * 0.35)):
            return 1e9
        return math.sqrt(total / max(used, 1))

    def predict(self, feat: Dict[str, float], strict: bool = True) -> Prediction:
        dists: List[Tuple[float, str]] = []
        for label, sample in self.samples:
            d = self.distance(feat, sample)
            if math.isfinite(d):
                dists.append((d, label))
        dists.sort(key=lambda item: item[0])
        neighbors = dists[: self.k]

        votes: Dict[str, float] = defaultdict(float)
        for d, label in neighbors:
            votes[label] += 1.0 / (d + 1e-6)

        if not votes:
            return Prediction("NEUTRAL", "UNKNOWN", 0.0, 0.0, orientation_bucket(feat), "NO_VOTES", False, {})

        ordered = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        raw = ordered[0][0]
        top = ordered[0][1]
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        total = sum(votes.values())
        conf = float(top / max(total, 1e-9))
        margin = float((top - second) / max(total, 1e-9))

        reason = "OK"
        ok = True
        if strict:
            if conf < 0.47:
                ok = False
                reason = "LOW_CONFIDENCE"
            elif margin < 0.18:
                ok = False
                reason = "LOW_MARGIN"
            elif len(neighbors) >= 5 and sum(1 for _, label in neighbors[:5] if label == raw) < 3:
                ok = False
                reason = "SPLIT_NEIGHBORS"

        final = raw if ok else "NEUTRAL"
        return Prediction(final, raw, conf, margin, orientation_bucket(feat), reason, ok, dict(votes))
