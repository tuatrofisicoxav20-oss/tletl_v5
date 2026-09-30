"""Tests de tools/fetch_models.py con un opener falso: NUNCA se toca la red."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from tools import fetch_models as fm

BIG = b"m" * (2 * 1024 * 1024)   # 2.0 MB: por encima de MIN_BYTES y se imprime en MB


def _opener(payload: bytes, calls: list | None = None):
    def open_url(url):
        if calls is not None:
            calls.append(url)
        return io.BytesIO(payload)
    return open_url


def test_urls_are_official_and_names_match_tracker():
    assert set(fm.MODELS) == {"hand_landmarker.task", "gesture_recognizer.task"}
    for name, url in fm.MODELS.items():
        assert url.startswith("https://storage.googleapis.com/mediapipe-models/")
        assert url.endswith(name)


def test_download_writes_file_atomically(tmp_path):
    dest = tmp_path / "models" / "hand_landmarker.task"
    calls = []
    status, size = fm.download("http://x/hand", dest, opener=_opener(BIG, calls), chunk_size=1000)
    assert status == "downloaded" and size == len(BIG)
    assert dest.read_bytes() == BIG
    assert not dest.with_name(dest.name + ".part").exists()
    assert calls == ["http://x/hand"]


def test_download_skips_existing_unless_force(tmp_path):
    dest = tmp_path / "gesture_recognizer.task"
    dest.write_bytes(BIG)
    calls = []
    assert fm.download("http://x/g", dest, opener=_opener(BIG, calls)) == ("skipped", len(BIG))
    assert calls == []
    status, _ = fm.download("http://x/g", dest, opener=_opener(BIG + b"new", calls), force=True)
    assert status == "downloaded" and calls == ["http://x/g"] and dest.stat().st_size == len(BIG) + 3


def test_download_rejects_small_file(tmp_path):
    dest = tmp_path / "hand_landmarker.task"
    with pytest.raises(fm.FetchError) as exc:
        fm.download("http://x/h", dest, opener=_opener(b"<html>portal cautivo</html>"))
    assert "pequeña" in str(exc.value)
    assert not dest.exists() and not dest.with_name(dest.name + ".part").exists()


def test_download_network_error_cleans_up(tmp_path):
    dest = tmp_path / "hand_landmarker.task"

    def broken(url):
        raise ConnectionError("sin red")

    with pytest.raises(fm.FetchError) as exc:
        fm.download("http://x/h", dest, opener=broken)
    assert "sin red" in str(exc.value)
    assert not dest.exists() and not dest.with_name(dest.name + ".part").exists()


def test_existing_small_file_is_replaced(tmp_path):
    dest = tmp_path / "hand_landmarker.task"
    dest.write_bytes(b"truncado")
    status, size = fm.download("http://x/h", dest, opener=_opener(BIG))
    assert status == "downloaded" and size == len(BIG)


def test_fetch_models_exit_codes_and_log(tmp_path):
    logs = []

    def opener(url):
        if "gesture" in url:
            raise TimeoutError("timeout")
        return io.BytesIO(BIG)

    code = fm.fetch_models(list(fm.MODELS), tmp_path, opener=opener, log=logs.append)
    assert code == 1
    assert (tmp_path / "hand_landmarker.task").is_file()
    assert not (tmp_path / "gesture_recognizer.task").exists()
    assert any(line.startswith("[ok] hand_landmarker.task") and "MB" in line for line in logs)
    assert any(line.startswith("[FAIL] gesture_recognizer.task") for line in logs)

    code = fm.fetch_models(list(fm.MODELS), tmp_path, opener=_opener(BIG), log=logs.append)
    assert code == 0
    assert any("ya estaba" in line for line in logs)
    assert fm.fetch_models(["nope.task"], tmp_path, opener=_opener(BIG), log=logs.append) == 1


def test_check_models_reports_missing(tmp_path):
    logs = []
    assert fm.check_models(list(fm.MODELS), tmp_path, log=logs.append) == 1
    assert sum("[MISSING]" in line for line in logs) == 2
    (tmp_path / "hand_landmarker.task").write_bytes(BIG)
    (tmp_path / "gesture_recognizer.task").write_bytes(BIG)
    assert fm.check_models(list(fm.MODELS), tmp_path, log=lambda s: None) == 0


def test_main_check_and_unknown_model(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(fm, "_default_opener", lambda url: (_ for _ in ()).throw(AssertionError("red!")))
    assert fm.main(["--dir", str(tmp_path), "--check"]) == 1
    assert "MISSING" in capsys.readouterr().out
    assert fm.main(["--dir", str(tmp_path), "bogus.task"]) == 2


def test_main_downloads_with_patched_opener(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fm, "_default_opener", _opener(BIG))
    assert fm.main(["--dir", str(tmp_path), "hand_landmarker.task"]) == 0
    assert (tmp_path / "hand_landmarker.task").stat().st_size == len(BIG)
    assert "descargado" in capsys.readouterr().out


def test_default_dir_follows_tletl_home(monkeypatch, tmp_path):
    monkeypatch.setenv("TLETL_HOME", str(tmp_path))
    from apps.common.hand_tracker import default_models_dir
    assert default_models_dir() == tmp_path / "models"


def test_human_size():
    assert fm.human_size(10) == "10 B"
    assert fm.human_size(2048) == "2.0 KB"
    assert fm.human_size(8_000_000).endswith("MB")
