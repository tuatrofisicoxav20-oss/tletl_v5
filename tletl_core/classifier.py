"""tletl_core/classifier.py — KNN robusto de Tletl v5 (vectorizado).

El clasificador es un KNN ponderado sobre features z-normalizadas del banco de
gestos. Desde v5.2 la predicción es vectorizada con numpy: el banco se guarda
como una matriz F×N (features × muestras, NaN donde la muestra no tiene la
clave) y la distancia a las N muestras se calcula de un golpe. Antes se
recorrían ~2363 muestras × ~67 features en Python puro por predicción, por mano
y por frame (≈58 ms → la app en vivo se quedaba en ~15 fps).

Garantía de identidad con la implementación anterior (`_predict_reference`):
  - La fórmula es la misma: dist = sqrt(sum(w·z²)/used), z = (q - s)/std,
    inválida (inf) cuando used < max(10, int(F·0.35)).
  - La suma sobre features se acumula en el MISMO orden secuencial (fila a fila
    de la matriz F×N), no con la suma pairwise de numpy, así los flotantes son
    bit a bit iguales a los del bucle viejo.
  - Los k vecinos se eligen como un sort estable por distancia: los empates se
    resuelven por índice del banco, igual que `list.sort` en el código viejo.

REGLA DURA: este módulo no sabe nada de Fedora, Blender, ydotool, mediapipe ni
de ventanas OpenCV. Solo lee un banco y predice gesto + confianza.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import LABELS, META_KEYS, get_features, get_label
from .orientation import ORIENTATION_KEYS, orientation_bucket

# (label, features) — el mismo formato que devuelve AdaptiveGestureMemory.samples()
Sample = Tuple[str, Dict[str, float]]

# Una muestra con menos features que esto no aporta nada al KNN (mismo umbral
# para el banco y para las muestras extra de la memoria adaptativa).
MIN_FEATURES_PER_SAMPLE = 12


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

    Parámetros
    ----------
    bank_path : ruta al banco JSONL (una muestra por línea).
    k : número de vecinos.
    orientation_weight : peso de las features de orientación/profundidad.
    min_confidence, min_margin : umbrales del modo `strict` (antes estaban
        hardcodeados en 0.47/0.18 y `[classifier]` del toml se ignoraba).
    extra_samples : muestras adicionales `(label, features)` — típicamente la
        memoria adaptativa (`AdaptiveGestureMemory.samples()`). Se agregan a las
        filas del banco ANTES del balanceo por clase y ANTES de calcular
        media/desviación, de modo que la normalización (std por feature), el
        conteo de features útiles y el cap por clase las incluyen como si fueran
        parte del banco. Se validan igual que una fila del banco (label en
        LABELS, ≥ MIN_FEATURES_PER_SAMPLE features numéricas finitas).
    verbose : imprime "[TLETL Core Runtime] Banco cargado ..." al cargar.
    """

    def __init__(self, bank_path: str | Path, k: int = 13, orientation_weight: float = 0.45,
                 min_confidence: float = 0.47, min_margin: float = 0.18,
                 extra_samples: Optional[Sequence[Sample]] = None, verbose: bool = True):
        self.bank_path = Path(bank_path).expanduser().resolve()
        self.k = int(k)
        self.orientation_weight = float(orientation_weight)
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self.verbose = bool(verbose)
        self.extra_samples: List[Sample] = self._clean_extra_samples(extra_samples)
        self.extra_used = 0            # cuántas muestras extra pasaron la validación
        self.samples: List[Sample] = []
        self.keys: List[str] = []
        self.mean: Dict[str, float] = {}
        self.std: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}
        self.counts: Counter[str] = Counter()
        # Estructuras vectorizadas (las llena load()).
        self._matrix = np.zeros((0, 0), dtype=np.float64)   # F×N, NaN = clave ausente
        self._valid = np.zeros((0, 0), dtype=bool)          # F×N, True donde hay valor
        self._labels: List[str] = []                         # label de cada columna
        self._std_vec = np.zeros(0, dtype=np.float64)
        self._weight_vec = np.zeros(0, dtype=np.float64)
        self._min_used = 0                                   # mínimo de features compartidas
        self.load()

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_extra_samples(extra: Optional[Sequence[Sample]]) -> List[Sample]:
        """Valida muestras extra sin mutar las originales. Descarta silenciosamente
        lo que no cumpla el contrato (label desconocido, features no numéricas)."""
        out: List[Sample] = []
        for item in extra or ():
            try:
                label, feat = item
            except (TypeError, ValueError):
                continue
            if not isinstance(label, str) or not isinstance(feat, dict):
                continue
            label = label.strip().upper()
            if label not in LABELS:
                continue
            clean: Dict[str, float] = {}
            for key, value in feat.items():
                if key in META_KEYS or not isinstance(value, (int, float)):
                    continue
                fv = float(value)
                if math.isfinite(fv):
                    clean[str(key)] = fv
            if len(clean) < MIN_FEATURES_PER_SAMPLE:
                continue
            out.append((label, clean))
        return out

    def _read_bank(self) -> List[Sample]:
        rows: List[Sample] = []
        for line in self.bank_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            label = get_label(obj)
            feat = get_features(obj)
            if not label or label not in LABELS or len(feat) < MIN_FEATURES_PER_SAMPLE:
                continue
            rows.append((label, feat))
        return rows

    def load(self) -> None:
        rows = self._read_bank()
        # Muestras extra (memoria adaptativa): entran ANTES del balanceo y de la
        # normalización, ver docstring de la clase.
        rows.extend(self.extra_samples)
        self.extra_used = len(self.extra_samples)

        if len(rows) < 20:
            raise RuntimeError(f"Banco insuficiente o ilegible: {self.bank_path} ({len(rows)} muestras válidas)")

        feat_counts: Counter[str] = Counter()
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
        balanced: List[Sample] = []
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

        self.mean = {}
        self.std = {}
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
        self._build_vectors()

        if self.verbose:
            extra_txt = f" (+{self.extra_used} adaptativas)" if self.extra_used else ""
            print(
                f"[TLETL Core Runtime] Banco cargado: {len(self.samples)} muestras{extra_txt}, "
                f"{len(self.keys)} features, labels={dict(self.counts)}"
            )

    def _build_vectors(self) -> None:
        """Construye la matriz F×N (features × muestras) y los vectores std/peso.

        Layout feature-major a propósito: la reducción sobre el eje 0 recorre las
        features en el mismo orden secuencial que el bucle de `distance()`, lo que
        mantiene los resultados bit a bit idénticos (ver docstring del módulo).
        """
        keys = self.keys
        n_feat, n_samp = len(keys), len(self.samples)
        matrix = np.full((n_feat, n_samp), np.nan, dtype=np.float64)
        nan = float("nan")
        for col, (_, feat) in enumerate(self.samples):
            matrix[:, col] = [feat.get(k, nan) for k in keys]
        # Cualquier valor no finito cuenta como "clave ausente", igual que en distance().
        matrix[~np.isfinite(matrix)] = np.nan
        self._matrix = matrix
        self._valid = ~np.isnan(matrix)
        self._labels = [label for label, _ in self.samples]
        self._std_vec = np.array([self.std[k] for k in keys], dtype=np.float64)
        self._weight_vec = np.array([self.weights[k] for k in keys], dtype=np.float64)
        self._min_used = max(10, int(n_feat * 0.35))

    # ------------------------------------------------------------------
    # Distancia (referencia por muestra, API pública)
    # ------------------------------------------------------------------
    def distance(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        """Distancia z-ponderada entre dos dicts de features (implementación de
        referencia, por muestra). Devuelve `math.inf` cuando comparten muy pocas
        features: antes devolvía el centinela finito 1e9, que colaba esas muestras
        como vecinos válidos (con features vacías ganaban las primeras k del banco
        y el clasificador devolvía OPEN_PALM con confianza 1.0)."""
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
            return math.inf
        return math.sqrt(total / max(used, 1))

    # ------------------------------------------------------------------
    # Predicción vectorizada
    # ------------------------------------------------------------------
    def _query_vector(self, feat: Dict[str, float]) -> np.ndarray:
        """Vector F del query; NaN donde la clave falta o no es un número finito."""
        q = np.full(len(self.keys), np.nan, dtype=np.float64)
        for i, key in enumerate(self.keys):
            value = feat.get(key)
            if value is None:
                continue
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                q[i] = fv
        return q

    def _distances(self, q: np.ndarray) -> np.ndarray:
        """Distancia del query a las N muestras (inf donde comparten pocas features)."""
        valid = self._valid & np.isfinite(q)[:, None]          # F×N
        z = q[:, None] - self._matrix                           # NaN donde no hay valor
        z /= self._std_vec[:, None]
        d2 = self._weight_vec[:, None] * z                      # (w·z)·z, mismo orden que distance()
        d2 *= z
        d2[~valid] = 0.0
        # Acumulación secuencial feature a feature (NO pairwise): bit a bit igual
        # que `total += w*z*z` del bucle de referencia.
        total = np.zeros(d2.shape[1], dtype=np.float64)
        for row in d2:
            total += row
        used = valid.sum(axis=0)
        dist = np.sqrt(total / np.maximum(used, 1))
        dist[used < self._min_used] = np.inf
        return dist

    def _nearest(self, dist: np.ndarray) -> List[Tuple[float, str]]:
        """k vecinos más cercanos con distancia finita, ordenados por (distancia,
        índice del banco): equivale al sort estable + slice [:k] del código viejo."""
        cand = np.flatnonzero(np.isfinite(dist))
        n = min(self.k, int(cand.size))
        if n <= 0:
            return []
        dc = dist[cand]
        if n < cand.size:
            part = np.argpartition(dc, n - 1)[:n]
            # Incluir TODOS los empates en la frontera del k-ésimo para que el
            # desempate sea por índice y no por el orden interno de argpartition.
            keep = cand[dc <= dc[part].max()]
        else:
            keep = cand
        order = np.lexsort((keep, dist[keep]))      # primario: distancia; secundario: índice
        chosen = keep[order][:n]
        return [(float(dist[i]), self._labels[i]) for i in chosen]

    def _decide(self, neighbors: List[Tuple[float, str]], feat: Dict[str, float],
                strict: bool) -> Prediction:
        """Voto ponderado 1/d + umbrales estrictos. Compartido por `predict` y
        `_predict_reference` para que la única diferencia sea cómo se buscan vecinos."""
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
            if conf < self.min_confidence:
                ok = False
                reason = "LOW_CONFIDENCE"
            elif margin < self.min_margin:
                ok = False
                reason = "LOW_MARGIN"
            elif len(neighbors) >= 5 and sum(1 for _, label in neighbors[:5] if label == raw) < 3:
                ok = False
                reason = "SPLIT_NEIGHBORS"

        final = raw if ok else "NEUTRAL"
        return Prediction(final, raw, conf, margin, orientation_bucket(feat), reason, ok, dict(votes))

    def predict(self, feat: Dict[str, float], strict: bool = True) -> Prediction:
        """Predice el gesto de un dict de features (vectorizado).

        Sin features compartidas suficientes con ninguna muestra (p.ej. dict
        vacío) no hay vecinos válidos y devuelve NEUTRAL/UNKNOWN con reason
        "NO_VOTES" y ok=False.
        """
        if not self._labels:
            return Prediction("NEUTRAL", "UNKNOWN", 0.0, 0.0, orientation_bucket(feat), "NO_VOTES", False, {})
        q = self._query_vector(feat)
        neighbors = self._nearest(self._distances(q))
        return self._decide(neighbors, feat, strict)

    def _predict_reference(self, feat: Dict[str, float], strict: bool = True) -> Prediction:
        """Implementación de referencia (bucle Python por muestra, la de v5.0/5.1).
        Solo para tests de equivalencia; es ~40× más lenta que `predict`."""
        dists: List[Tuple[float, str]] = []
        for label, sample in self.samples:
            d = self.distance(feat, sample)
            if math.isfinite(d):
                dists.append((d, label))
        dists.sort(key=lambda item: item[0])
        return self._decide(dists[: self.k], feat, strict)
