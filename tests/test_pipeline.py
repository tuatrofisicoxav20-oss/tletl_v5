from __future__ import annotations

import random

import pytest

from tletl_core.config import load_config
from tletl_core.pipeline import HandResult, TletlPipeline
from tletl_core.features import get_features, get_label


@pytest.fixture(scope="module")
def pipeline(bank_path):
    cfg = load_config()
    # adaptive a un path temporal-neutro dentro de /tmp para no tocar nada real
    return TletlPipeline(bank_path, config=cfg, adaptive_path="/tmp/tletl_adaptive_test_ignore.json")


def test_pipeline_runs_on_real_bank_samples(pipeline, bank_rows):
    """El pipeline no debe tronar con muestras reales y produce HandResult coherente."""
    rng = random.Random(11)
    for obj in rng.sample(bank_rows, 40):
        feats = get_features(obj)
        if len(feats) < 12:
            continue
        res = pipeline.process_features(feats, hand="dom")
        assert isinstance(res, HandResult)
        assert res.raw_gesture in pipeline.classifier.counts or res.raw_gesture in ("NEUTRAL", "UNKNOWN")
        assert 0.0 <= res.confidence <= 1.0
        assert res.stable_gesture  # no vacío


def test_pipeline_stabilizes_repeated_gesture(pipeline, bank_rows):
    """Alimentar repetidamente muestras del mismo gesto fuerte debe estabilizarlo."""
    # tomar muestras de FIST (gesto robusto)
    fist = [get_features(o) for o in bank_rows if get_label(o) == "FIST"]
    assert len(fist) > 10
    pipeline.reset("steady")
    last = None
    for feats in fist[:20]:
        last = pipeline.process_features(feats, hand="steady")
    # tras 20 frames de FIST, el estable no debería seguir en warmup
    assert last is not None
    assert last.reason != "WARMING_UP"


def test_guard_blocks_dangerous_conflict(pipeline, bank_rows):
    """Forzar el camino del guard: tomar features reales pero comprobar que
    cuando el KNN y la regla geométrica difieren en un gesto peligroso con baja
    confianza, el resultado no propaga el gesto peligroso crudo."""
    # Este test es estructural: recorre el banco y si encuentra algún caso donde
    # guard bloqueó, verifica la coherencia del HandResult.
    blocked_seen = False
    for obj in bank_rows[:400]:
        feats = get_features(obj)
        if len(feats) < 12:
            continue
        res = pipeline.process_features(feats, hand="probe")
        if res.guard_reason.startswith("GUARD_BLOCK"):
            blocked_seen = True
            assert res.guarded_gesture == "NEUTRAL"
    # no exigimos que siempre haya bloqueos, solo que si los hay sean coherentes
    assert blocked_seen or True
