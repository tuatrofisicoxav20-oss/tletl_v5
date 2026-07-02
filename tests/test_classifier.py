from __future__ import annotations

import random

import pytest

from tletl_core.classifier import Prediction, RobustKNNRuntime
from tletl_core.features import get_features, get_label


@pytest.fixture(scope="module")
def runtime(bank_path):
    return RobustKNNRuntime(bank_path, k=13, orientation_weight=0.45)


def test_bank_loads(runtime):
    assert len(runtime.samples) > 200
    assert len(runtime.keys) >= 15


def test_predict_returns_prediction(runtime, bank_rows):
    obj = bank_rows[0]
    pred = runtime.predict(get_features(obj), strict=False)
    assert isinstance(pred, Prediction)
    assert pred.label in (*runtime.counts.keys(), "NEUTRAL")


def test_predict_recovers_own_samples(runtime, bank_rows):
    """El clasificador debe acertar la mayoría de muestras del propio banco."""
    rng = random.Random(7)
    sample = rng.sample(bank_rows, min(120, len(bank_rows)))
    hits = 0
    total = 0
    for obj in sample:
        label = get_label(obj)
        feats = get_features(obj)
        if not label or len(feats) < 12:
            continue
        total += 1
        pred = runtime.predict(feats, strict=False)
        if pred.raw_label == label:
            hits += 1
    assert total > 50
    acc = hits / total
    assert acc >= 0.70, f"accuracy demasiado baja sobre el banco: {acc:.2%}"
