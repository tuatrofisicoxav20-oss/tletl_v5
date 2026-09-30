"""Tests de tletl_core/config.py: defaults completos, merge profundo, env overrides."""

from __future__ import annotations

from tletl_core.config import (
    CRITIC_MIN_CONF_DEFAULTS,
    DEFAULTS,
    ENV_MAP,
    critic_min_conf,
    load_config,
)


def test_all_default_sections_present_with_real_toml():
    cfg = load_config()
    for section, values in DEFAULTS.items():
        assert section in cfg, section
        for key in values:
            assert key in cfg[section], f"{section}.{key}"


def test_defaults_are_not_shared_between_calls(tmp_path):
    a = load_config(tmp_path / "nope.toml")
    b = load_config(tmp_path / "nope.toml")
    a["guard"]["dangerous"].append("BANANA")
    a["critic"]["min_conf"]["PINCH"] = 0.01
    assert "BANANA" not in b["guard"]["dangerous"]
    assert b["critic"]["min_conf"]["PINCH"] == CRITIC_MIN_CONF_DEFAULTS["PINCH"]
    assert DEFAULTS["critic"]["min_conf"]["PINCH"] == CRITIC_MIN_CONF_DEFAULTS["PINCH"]


def test_nested_table_is_deep_merged(tmp_path):
    toml = tmp_path / "t.toml"
    toml.write_text("[critic.min_conf]\nPINCH = 0.50\n", encoding="utf-8")
    cfg = load_config(toml)
    assert cfg["critic"]["min_conf"]["PINCH"] == 0.50
    # los demás gestos NO se pierden
    assert cfg["critic"]["min_conf"]["FIST"] == CRITIC_MIN_CONF_DEFAULTS["FIST"]
    assert cfg["critic"]["enabled"] is True


def test_critic_min_conf_helper_fills_missing_and_uppercases(tmp_path):
    cfg = load_config(tmp_path / "nope.toml")
    cfg["critic"]["min_conf"] = {"pinch": 0.4, "bogus": "x"}
    table = critic_min_conf(cfg)
    assert table["PINCH"] == 0.4
    assert set(CRITIC_MIN_CONF_DEFAULTS) <= set(table)
    assert table["FIST"] == CRITIC_MIN_CONF_DEFAULTS["FIST"]


def test_env_overrides_cover_new_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("TLETL_CAMERA", "2")
    monkeypatch.setenv("TLETL_HEADLESS", "yes")
    monkeypatch.setenv("TLETL_TRACKER_BACKEND", "tasks")
    monkeypatch.setenv("TLETL_BUS_UDP", "127.0.0.1:5055")
    monkeypatch.setenv("TLETL_DOMINANT_HAND", "Left")
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg["camera"]["index"] == 2
    assert cfg["camera"]["headless"] is True
    assert cfg["tracker"]["backend"] == "tasks"
    assert cfg["bus"]["udp_target"] == "127.0.0.1:5055"
    assert cfg["fedora"]["dominant_hand"] == "Left"


def test_flag_env_false_values(monkeypatch, tmp_path):
    for value in ("0", "false", "False", "no", "off", ""):
        monkeypatch.setenv("TLETL_DRY_RUN", value)
        assert load_config(tmp_path / "nope.toml")["fedora"]["dry_run"] is False


def test_bad_env_value_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("TLETL_AI_V47_K", "trece")
    assert load_config(tmp_path / "nope.toml")["classifier"]["k"] == 13


def test_env_map_targets_exist_in_defaults():
    for env_key, (section, key, _) in ENV_MAP.items():
        assert key in DEFAULTS[section], f"{env_key} -> {section}.{key} no existe en DEFAULTS"


def test_real_toml_pinch_threshold_matches_defaults():
    """El toml del repo y los DEFAULTS deben contar la misma historia."""
    cfg = load_config()
    assert cfg["critic"]["min_conf"]["PINCH"] == CRITIC_MIN_CONF_DEFAULTS["PINCH"]
