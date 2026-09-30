from __future__ import annotations

import math
import random

import pytest

from tletl_core.classifier import Prediction, RobustKNNRuntime
from tletl_core.features import get_features, get_label


@pytest.fixture(scope="module")
def runtime(bank_path):
    return RobustKNNRuntime(bank_path, k=13, orientation_weight=0.45, verbose=False)


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


# ---------------------------------------------------------------------------
# Vectorizado vs referencia (bucle Python por muestra)
# ---------------------------------------------------------------------------

def _edge_rows(runtime: RobustKNNRuntime, feats):
    """Casos borde sintéticos: vacío, NaN/inf, pocas claves, cero, enormes, copia exacta."""
    exact = dict(runtime.samples[0][1])
    return [
        {},
        {"index_tip_wrist": float("nan"), "thumb_tip_index_tip": float("inf"), "palm_width": 1.0},
        dict(list(feats[0].items())[:5]),                        # muy pocas compartidas
        dict(list(feats[1].items())[:runtime._min_used]),        # justo en el mínimo
        dict(list(feats[1].items())[:runtime._min_used - 1]),    # justo por debajo
        {k: 0.0 for k in runtime.keys},
        {k: v * 1e6 for k, v in feats[2].items()},
        exact,                                                   # distancia 0 a una muestra
        {**feats[3], "clave_que_no_existe": 1.0},
    ]


def _assert_same_prediction(fast: Prediction, ref: Prediction) -> None:
    assert fast.raw_label == ref.raw_label
    assert fast.label == ref.label
    assert fast.ok == ref.ok
    assert fast.reason == ref.reason
    assert fast.orientation == ref.orientation
    assert math.isclose(fast.confidence, ref.confidence, abs_tol=1e-9)
    assert math.isclose(fast.margin, ref.margin, abs_tol=1e-9)
    assert set(fast.votes) == set(ref.votes)
    for label, value in ref.votes.items():
        assert math.isclose(fast.votes[label], value, rel_tol=1e-9)


def test_vectorized_predict_matches_reference(runtime, bank_rows):
    """predict (numpy) == _predict_reference (bucle) en ~60 filas reales + bordes."""
    rng = random.Random(21)
    feats = [get_features(o) for o in rng.sample(bank_rows, 60)]
    for feat in feats + _edge_rows(runtime, feats):
        for strict in (True, False):
            _assert_same_prediction(runtime.predict(feat, strict=strict),
                                    runtime._predict_reference(feat, strict=strict))


@pytest.mark.parametrize("k", [1, 5, 40])
def test_vectorized_matches_reference_for_other_k(bank_path, bank_rows, k):
    rt = RobustKNNRuntime(bank_path, k=k, verbose=False)
    rng = random.Random(k)
    for obj in rng.sample(bank_rows, 20):
        feat = get_features(obj)
        _assert_same_prediction(rt.predict(feat), rt._predict_reference(feat))


def test_distance_reference_still_works(runtime):
    label, sample = runtime.samples[0]
    assert runtime.distance(sample, sample) == 0.0
    other = runtime.samples[-1][1]
    d = runtime.distance(sample, other)
    assert math.isfinite(d) and d > 0.0


# ---------------------------------------------------------------------------
# BUG: features vacías devolvían OPEN_PALM con confianza 1.0
# ---------------------------------------------------------------------------

def test_distance_too_few_shared_features_is_inf(runtime):
    """Antes devolvía el centinela finito 1e9 y esas muestras colaban como vecinos."""
    _, sample = runtime.samples[0]
    assert runtime.distance({}, sample) == math.inf
    assert math.isinf(runtime.distance(dict(list(sample.items())[:5]), sample))


def test_predict_empty_features_is_no_votes(runtime):
    pred = runtime.predict({})
    assert pred.label == "NEUTRAL"
    assert pred.raw_label == "UNKNOWN"
    assert pred.reason == "NO_VOTES"
    assert pred.ok is False
    assert pred.confidence == 0.0 and pred.margin == 0.0
    assert pred.votes == {}
    # también en modo no estricto
    assert runtime.predict({}, strict=False).reason == "NO_VOTES"


def test_predict_too_few_shared_keys_is_no_votes(runtime, bank_rows):
    feat = dict(list(get_features(bank_rows[0]).items())[:5])
    pred = runtime.predict(feat)
    assert pred.raw_label == "UNKNOWN" and pred.reason == "NO_VOTES" and not pred.ok


# ---------------------------------------------------------------------------
# Umbrales estrictos configurables (antes 0.47/0.18 hardcodeados)
# ---------------------------------------------------------------------------

def test_default_thresholds_match_legacy_constants(runtime):
    assert runtime.min_confidence == 0.47
    assert runtime.min_margin == 0.18


def test_strict_thresholds_come_from_constructor(bank_path, bank_rows):
    rt = RobustKNNRuntime(bank_path, k=13, verbose=False)
    rng = random.Random(3)
    ok_case = None
    for obj in rng.sample(bank_rows, 40):
        feat = get_features(obj)
        pred = rt.predict(feat)
        if pred.ok and pred.confidence < 1.0 and pred.margin < 1.0:
            ok_case = (feat, pred)
            break
    assert ok_case is not None, "no encontré una predicción OK con conf/margen < 1"
    feat, pred = ok_case

    # Subir min_confidence por encima de la confianza real -> LOW_CONFIDENCE
    rt.min_confidence = pred.confidence + 1e-6
    strict = rt.predict(feat)
    assert not strict.ok and strict.reason == "LOW_CONFIDENCE" and strict.label == "NEUTRAL"

    # Confianza permisiva pero margen imposible -> LOW_MARGIN
    rt.min_confidence = 0.0
    rt.min_margin = pred.margin + 1e-6
    strict = rt.predict(feat)
    assert not strict.ok and strict.reason == "LOW_MARGIN"

    # Umbrales en cero -> vuelve a estar OK (misma etiqueta cruda)
    rt.min_margin = 0.0
    again = rt.predict(feat)
    assert again.ok and again.raw_label == pred.raw_label

    # El constructor los acepta explícitamente (umbral justo encima de la conf real)
    rt2 = RobustKNNRuntime(bank_path, min_confidence=pred.confidence + 1e-6, min_margin=0.5,
                           verbose=False)
    assert rt2.min_confidence == pred.confidence + 1e-6 and rt2.min_margin == 0.5
    strict2 = rt2.predict(feat)
    assert not strict2.ok and strict2.reason == "LOW_CONFIDENCE"
    assert rt2.predict(feat, strict=False).ok


def test_non_strict_ignores_thresholds(bank_path, bank_rows):
    rt = RobustKNNRuntime(bank_path, min_confidence=1.5, min_margin=1.5, verbose=False)
    pred = rt.predict(get_features(bank_rows[0]), strict=False)
    assert pred.ok and pred.reason == "OK"


# ---------------------------------------------------------------------------
# verbose
# ---------------------------------------------------------------------------

def test_verbose_flag_controls_bank_print(bank_path, capsys):
    RobustKNNRuntime(bank_path, verbose=False)
    assert capsys.readouterr().out == ""
    RobustKNNRuntime(bank_path)
    assert "Banco cargado" in capsys.readouterr().out
