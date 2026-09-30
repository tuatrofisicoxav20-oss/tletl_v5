"""Tests de tools/gesture_bank.py: helpers de carga/conteo y la caché de conteos
(antes se releía el JSONL completo en cada frame)."""

from __future__ import annotations

import json

import pytest

from tools import gesture_bank as gb
from tools.gesture_bank import (
    LABELS,
    SampleCounter,
    append_sample,
    build_parser,
    count_by_label,
    format_counts,
    load_samples,
    short_counts,
    tracker_config,
)


def test_load_samples_missing_file_is_empty(tmp_path):
    assert load_samples(tmp_path / "nope.jsonl") == []


def test_load_samples_skips_broken_lines(tmp_path):
    p = tmp_path / "bank.jsonl"
    p.write_text('{"label": "FIST", "features": {}}\n\nnot json\n{"label": "PINCH", "features": {}}\n', encoding="utf-8")
    rows = load_samples(p)
    assert [r["label"] for r in rows] == ["FIST", "PINCH"]


def test_count_and_format_cover_all_labels():
    rows = [{"label": "FIST"}, {"label": "FIST"}, {"label": "PINCH"}, {"no_label": 1}, "basura"]
    counts = count_by_label(rows)
    assert counts == {"FIST": 2, "PINCH": 1, "?": 2}
    text = format_counts(counts)
    assert text.startswith("OPEN_PALM:0") and "FIST:2" in text and "PINCH:1" in text
    assert all(lbl in text for lbl in LABELS)
    assert short_counts(rows) == text


def test_sample_counter_loads_once_and_updates_in_memory(tmp_path, monkeypatch):
    p = tmp_path / "bank.jsonl"
    append_sample(p, "FIST", {"f": 1.0})
    append_sample(p, "FIST", {"f": 1.0}, handedness="Left")
    append_sample(p, "POINT", {"f": 1.0})
    counter = SampleCounter.from_file(p)
    assert counter.get("FIST") == 2 and counter.get("POINT") == 1 and counter.total == 3

    # a partir de aquí NO se vuelve a leer el archivo
    monkeypatch.setattr(gb, "load_samples", lambda path: (_ for _ in ()).throw(AssertionError("releyó el JSONL")))
    counter.add("PINCH", 2)
    counter.add("FIST")
    assert counter.get("PINCH") == 2 and counter.get("FIST") == 3 and counter.total == 6
    assert "PINCH:2" in counter.short() and "FIST:3" in counter.short()
    assert counter.get("NEUTRAL") == 0


def test_sample_counter_matches_file_after_appends(tmp_path):
    p = tmp_path / "bank.jsonl"
    counter = SampleCounter.from_file(p)
    assert counter.total == 0
    for label in ("OPEN_PALM", "OPEN_PALM", "THREE"):
        append_sample(p, label, {"x": 0.5})
        counter.add(label)
    fresh = SampleCounter.from_file(p)
    assert fresh.counts == counter.counts == {"OPEN_PALM": 2, "THREE": 1}
    with open(p, encoding="utf-8") as fh:
        assert all(json.loads(line)["meta"]["handedness"] == "Right" for line in fh)


def test_parser_backend_and_config_defaults():
    ns = build_parser().parse_args([])
    assert ns.backend is None and ns.det_conf is None and ns.track_conf is None and ns.camera is None
    assert ns.model_complexity is None and ns.dataset == "gesture_bank.jsonl"
    ns2 = build_parser().parse_args(["--backend", "tasks", "--det-conf", "0.5", "--dual-hand", "-c", "2"])
    assert ns2.backend == "tasks" and ns2.det_conf == 0.5 and ns2.dual_hand and ns2.camera == 2
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--backend", "bogus"])


def test_tracker_config_cli_over_toml():
    cfg = {"tracker": {"backend": "auto", "det_conf": 0.72, "track_conf": 0.72, "model_complexity": 1}}
    ns = build_parser().parse_args([])
    assert tracker_config(ns, cfg) == cfg["tracker"]
    ns = build_parser().parse_args(["--backend", "legacy", "--track-conf", "0.3", "--model-complexity", "0"])
    trk = tracker_config(ns, cfg)
    assert trk["backend"] == "legacy" and trk["track_conf"] == 0.3 and trk["model_complexity"] == 0
    assert trk["det_conf"] == 0.72 and cfg["tracker"]["backend"] == "auto"   # el original no se toca
