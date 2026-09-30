"""Tests de tools/bank_probe.py, tools/duel_lab.py y tools/healthcheck.py sobre el banco real."""

from __future__ import annotations

import tools.bank_probe as bank_probe
import tools.duel_lab as duel_lab
import tools.healthcheck as healthcheck
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


# ---------------------------------------------------------------------------
# Docstrings de módulo (antes iban DESPUÉS de `from __future__` y no eran docstring)
# ---------------------------------------------------------------------------

def test_tools_have_real_module_docstrings():
    for mod in (healthcheck, bank_probe, duel_lab):
        assert mod.__doc__, f"{mod.__name__} sin docstring de módulo"
        assert "Uso" in mod.__doc__


def test_tool_defaults_follow_runtime_paths(bank_path):
    from tletl_core.paths import default_bank_path
    assert bank_probe.DEFAULT_BANK == default_bank_path()
    assert duel_lab.DEFAULT_BANK == default_bank_path()
    assert bank_probe.DEFAULT_BANK == bank_path


# ---------------------------------------------------------------------------
# healthcheck: rutas resueltas + check del banco vía bank_path_from_config
# ---------------------------------------------------------------------------

def test_healthcheck_main_ok_and_prints_resolved_paths(capsys, monkeypatch):
    monkeypatch.delenv("TLETL_GESTURE_BANK", raising=False)
    from tletl_core.config import (adaptive_path_from_config, bank_path_from_config,
                                   bus_path_from_config, load_config)
    cfg = load_config()
    assert healthcheck.main() == 0
    out = capsys.readouterr().out
    assert "healthcheck OK" in out
    assert str(bank_path_from_config(cfg)) in out
    assert str(bus_path_from_config(cfg)) in out
    assert str(adaptive_path_from_config(cfg)) in out
    assert "critic min_conf" in out


def test_healthcheck_fails_when_configured_bank_missing(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("TLETL_GESTURE_BANK", str(tmp_path / "no_existe.jsonl"))
    assert healthcheck.main() == 1
    out = capsys.readouterr().out
    assert "banco no encontrado" in out and "no_existe.jsonl" in out
