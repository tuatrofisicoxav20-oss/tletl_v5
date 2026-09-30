"""tests/test_core_adaptive_learning.py

La memoria adaptativa SÍ alimenta al clasificador (antes grababa muestras que
nadie leía). Se usa un banco sintético en tmp + un JSON adaptativo en tmp.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import pytest

from tletl_core.adaptive import AdaptiveGestureMemory
from tletl_core.classifier import RobustKNNRuntime
from tletl_core.config import load_config
from tletl_core.features import LABELS
from tletl_core.pipeline import TletlPipeline

DIM = 20
N_PER_LABEL = 40
FAR_POINT = 3.0        # región del espacio de features donde NO hay banco


def _feat(values: List[float]) -> Dict[str, float]:
    return {f"f{i:02d}": float(v) for i, v in enumerate(values)}


def _write_synthetic_bank(path: Path, seed: int = 5) -> Dict[str, List[float]]:
    """7 clusters gaussianos (uno por label) en [-1, 1]^DIM. Devuelve centroides."""
    rng = random.Random(seed)
    centroids = {label: [rng.uniform(-1.0, 1.0) for _ in range(DIM)] for label in LABELS}
    with path.open("w", encoding="utf-8") as fh:
        for label, centroid in centroids.items():
            for _ in range(N_PER_LABEL):
                feat = _feat([c + rng.gauss(0.0, 0.05) for c in centroid])
                fh.write(json.dumps({"label": label, "features": feat}) + "\n")
    return centroids


def _far_samples(n: int = 30, seed: int = 9) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    return [_feat([FAR_POINT + rng.gauss(0.0, 0.05) for _ in range(DIM)]) for _ in range(n)]


@pytest.fixture
def synthetic_bank(tmp_path):
    bank = tmp_path / "bank.jsonl"
    _write_synthetic_bank(bank)
    return bank


def _pick_new_label(base_runtime: RobustKNNRuntime, query: Dict[str, float]) -> str:
    """Un label distinto del que el banco base asigna a la región lejana."""
    base = base_runtime.predict(query, strict=False).raw_label
    return next(label for label in LABELS if label != base)


# ---------------------------------------------------------------------------
# Nivel clasificador: extra_samples cambian la predicción y la normalización
# ---------------------------------------------------------------------------

def test_extra_samples_change_prediction_and_normalization(synthetic_bank):
    base = RobustKNNRuntime(synthetic_bank, k=5, verbose=False)
    query = _feat([FAR_POINT] * DIM)
    new_label = _pick_new_label(base, query)
    assert base.predict(query, strict=False).raw_label != new_label

    extra = [(new_label, feat) for feat in _far_samples()]
    learned = RobustKNNRuntime(synthetic_bank, k=5, extra_samples=extra, verbose=False)

    assert learned.extra_used == len(extra)
    assert len(learned.samples) == len(base.samples) + len(extra)
    assert learned.counts[new_label] == base.counts[new_label] + len(extra)
    # Se agregan ANTES de calcular media/std: la normalización las incluye.
    assert learned.mean["f00"] != base.mean["f00"]
    assert learned.std["f00"] > base.std["f00"]

    pred = learned.predict(query, strict=True)
    assert pred.raw_label == new_label and pred.ok


def test_extra_samples_invalid_entries_are_dropped(synthetic_bank):
    valid = _far_samples(12)
    extra = [
        ("PINCH", valid[0]),
        ("pinch", valid[1]),                      # label en minúsculas -> normalizado
        ("HELICOPTER", valid[2]),                 # label desconocido
        ("PINCH", {"f00": 1.0}),                  # muy pocas features
        ("PINCH", {**valid[3], "f00": float("nan")}),  # NaN se limpia, el resto queda
        "basura",                                 # no es tupla
        ("PINCH", "no es dict"),
    ]
    rt = RobustKNNRuntime(synthetic_bank, k=5, extra_samples=extra, verbose=False)
    assert rt.extra_used == 3
    # el dict original no se muta
    assert extra[4][1]["f00"] != extra[4][1]["f00"]  # sigue siendo NaN


def test_extra_samples_do_not_mutate_input(synthetic_bank):
    feat = _far_samples(1)[0]
    snapshot = dict(feat)
    RobustKNNRuntime(synthetic_bank, k=5, extra_samples=[("FIST", feat)], verbose=False)
    assert feat == snapshot


# ---------------------------------------------------------------------------
# Nivel memoria: samples() devuelve el formato que espera el clasificador
# ---------------------------------------------------------------------------

def test_memory_samples_roundtrip_into_classifier(synthetic_bank, tmp_path):
    base = RobustKNNRuntime(synthetic_bank, k=5, verbose=False)
    query = _feat([FAR_POINT] * DIM)
    new_label = _pick_new_label(base, query)

    mem = AdaptiveGestureMemory(tmp_path / "adaptive.json", enabled=True,
                                min_confidence=0.5, save_interval=0.0)
    for feat in _far_samples(25):
        assert mem.observe(new_label, {**feat, "palm_center_x": 0.3}, confidence=0.95)
    samples = mem.samples()
    assert len(samples) == 25
    assert all(label == new_label for label, _ in samples)
    assert all("palm_center_x" not in feat for _, feat in samples)

    learned = RobustKNNRuntime(synthetic_bank, k=5, extra_samples=samples, verbose=False)
    assert learned.extra_used == 25
    assert learned.predict(query, strict=False).raw_label == new_label


# ---------------------------------------------------------------------------
# Nivel pipeline: [adaptive].enabled + path -> el KNN carga la memoria
# ---------------------------------------------------------------------------

def _config(tmp_path, adaptive_path: Path, enabled: bool):
    cfg = load_config(tmp_path / "no_existe.toml")      # defaults puros
    cfg["adaptive"]["enabled"] = enabled
    cfg["adaptive"]["path"] = str(adaptive_path)
    cfg["adaptive"]["min_confidence"] = 0.5
    cfg["classifier"]["k"] = 5
    cfg["critic"]["require_orientation"] = False        # el banco sintético no trae orientación
    return cfg


def test_pipeline_loads_adaptive_memory_when_enabled(synthetic_bank, tmp_path):
    base = RobustKNNRuntime(synthetic_bank, k=5, verbose=False)
    query = _feat([FAR_POINT] * DIM)
    new_label = _pick_new_label(base, query)

    adaptive_json = tmp_path / "runtime" / "adaptive.json"
    mem = AdaptiveGestureMemory(adaptive_json, enabled=True, min_confidence=0.5, save_interval=0.0)
    for feat in _far_samples(30):
        mem.observe(new_label, feat, confidence=0.95)
    assert adaptive_json.exists()

    off = TletlPipeline(synthetic_bank, config=_config(tmp_path, adaptive_json, False), verbose=False)
    assert off.classifier.extra_used == 0
    assert off.classifier.predict(query, strict=False).raw_label != new_label

    on = TletlPipeline(synthetic_bank, config=_config(tmp_path, adaptive_json, True), verbose=False)
    assert on.adaptive.enabled and on.adaptive.path == adaptive_json
    assert on.classifier.extra_used == 30
    assert on.classifier.predict(query, strict=False).raw_label == new_label

    # y de punta a punta: tras estabilizar, el gesto estable es el aprendido
    res = None
    for _ in range(8):
        res = on.process_features(query, hand="dom")
    assert res is not None and res.raw_gesture == new_label
    assert res.stable_gesture == new_label and res.ok
    on.close()


def test_pipeline_adaptive_disabled_ignores_existing_file(synthetic_bank, tmp_path):
    adaptive_json = tmp_path / "adaptive.json"
    adaptive_json.write_text(json.dumps({"gestures": {"FIST": _far_samples(5)}}), encoding="utf-8")
    pipeline = TletlPipeline(synthetic_bank, config=_config(tmp_path, adaptive_json, False), verbose=False)
    assert pipeline.classifier.extra_used == 0
    assert pipeline.adaptive.samples() == []      # apagada: ni siquiera carga el archivo
