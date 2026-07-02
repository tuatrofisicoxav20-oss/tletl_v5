from __future__ import annotations

"""Tests de tools/bank_probe.py y tools/duel_lab.py sobre el banco real."""

from tools.bank_probe import probe_bank, format_report
from tools.duel_lab import evaluate_knn, duel


def test_bank_probe_real(bank_path):
    stats = probe_bank(bank_path)
    assert stats["valid_samples"] > 2000
    assert stats["shared_features"] >= 15
    assert stats["missing_labels"] == []           # los 7 gestos presentes
    assert 0.0 < stats["balance_ratio"] <= 1.0
    # no debe haber muestras sin orientación en el banco activo
    assert stats["rows_without_orientation"] == 0


def test_bank_probe_report_is_text(bank_path):
    txt = format_report(probe_bank(bank_path))
    assert "bank probe" in txt
    assert "PINCH" in txt


def test_evaluate_knn_holdout_is_decent(bank_path):
    """En holdout (sin auto-engaño) el KNN debe acertar la mayoría."""
    res = evaluate_knn(bank_path, k=13, holdout_frac=0.2, seed=3, max_test=80)
    assert res["test"] >= 40
    assert res["train"] > 1500
    assert res["accuracy"] >= 0.75, f"accuracy holdout baja: {res['accuracy']}"


def test_duel_orders_by_accuracy(bank_path):
    configs = [{"k": 13, "orientation_weight": 0.45}, {"k": 5, "orientation_weight": 0.45}]
    results = duel(bank_path, configs, holdout_frac=0.2, seed=5, max_test=60)
    assert len(results) == 2
    # ordenado descendente por accuracy
    assert results[0]["accuracy"] >= results[1]["accuracy"]
