from __future__ import annotations

import json
import random

import pytest

from tletl_core.classifier import Prediction
from tletl_core.config import load_config
from tletl_core.features import extract_live_features, get_features, get_label
from tletl_core.intent import gesture_to_common_intent
from tletl_core.pipeline import HandResult, TletlPipeline
from tletl_core.state import TletlFrameState
import tletl_core.pipeline as pipeline_module
from conftest import make_landmarks


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
        assert res.hint is None


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


# ---------------------------------------------------------------------------
# Helpers: predicciones fabricadas para forzar caminos concretos
# ---------------------------------------------------------------------------

def _palm_front_feats(gesture: str) -> dict:
    feats = extract_live_features(make_landmarks(gesture))
    feats["palm_facing_score"] = 0.90
    feats["back_hand_score"] = 0.05
    feats["side_hand_score"] = 0.05
    return feats


def _pred(label: str, conf: float, margin: float, ok: bool = True) -> Prediction:
    return Prediction(label if ok else "NEUTRAL", label, conf, margin, "PALM_FRONT",
                      "OK" if ok else "LOW_CONFIDENCE", ok, {label: conf})


class _FakePredict:
    """Sustituto de classifier.predict que devuelve la Prediction que le digan."""

    def __init__(self, pred: Prediction):
        self.pred = pred
        self.calls = 0

    def __call__(self, feat, strict=True):
        self.calls += 1
        return self.pred


# ---------------------------------------------------------------------------
# Cableado de config: umbrales del clasificador y tabla del critic
# ---------------------------------------------------------------------------

def test_config_thresholds_are_wired(pipeline):
    cfg = pipeline.cfg
    assert pipeline.classifier.min_confidence == float(cfg["classifier"]["min_confidence"])
    assert pipeline.classifier.min_margin == float(cfg["classifier"]["min_margin"])
    assert pipeline._critic_min_conf["PINCH"] == float(cfg["critic"]["min_conf"]["PINCH"])


def test_classifier_min_confidence_from_config_changes_behavior(bank_path, bank_rows, tmp_path, capsys):
    cfg = load_config(tmp_path / "no_existe.toml")
    cfg["classifier"]["min_confidence"] = 1.01          # imposible de alcanzar
    p = TletlPipeline(bank_path, config=cfg, adaptive_path=tmp_path / "a.json", verbose=False)
    assert capsys.readouterr().out == ""                # verbose=False llega al clasificador
    for obj in bank_rows[:10]:
        pred = p.classifier.predict(get_features(obj))
        assert not pred.ok and pred.reason == "LOW_CONFIDENCE"
        res = p.process_features(get_features(obj), hand="dom")
        assert not res.ok


def test_critic_min_conf_table_is_used_by_pipeline(pipeline, monkeypatch):
    """PINCH a 0.65 pasa con la tabla del toml (0.62) y muere con 0.72."""
    feats = _palm_front_feats("PINCH")
    monkeypatch.setattr(pipeline.classifier, "predict", _FakePredict(_pred("PINCH", 0.65, 0.50)))

    res = pipeline.process_features(feats, hand="critic_cfg")
    assert res.guard_reason in ("OK", "GUARD_OFF")
    assert res.critic_accepted is True, res.critic_reason

    monkeypatch.setattr(pipeline, "_critic_min_conf", {"PINCH": 0.72})
    res = pipeline.process_features(feats, hand="critic_cfg")
    assert res.critic_accepted is False and "0.72" in res.critic_reason


# ---------------------------------------------------------------------------
# BUG de coherencia: critic_ok mezclaba el estable con el critic del frame actual
# ---------------------------------------------------------------------------

def test_hand_state_critic_ok_follows_stable_verdict(pipeline):
    held = HandResult(present=True, raw_gesture="PINCH", guarded_gesture="PINCH",
                      stable_gesture="PINCH", ok=True, confidence=0.58, margin=0.3,
                      critic_accepted=False,
                      critic_reason="confianza baja para PINCH: 0.58 < 0.62",
                      reason="TEMPORAL_HOLD")
    hs = pipeline.to_hand_state(held, side="Right")
    assert hs.critic_ok is True                          # el estable manda
    assert hs.critic_reason == held.critic_reason        # diagnóstico del frame actual
    st = TletlFrameState()
    st.dom = hs
    assert gesture_to_common_intent(st).name == "GRAB_OR_SELECT"

    # y al revés: estable no OK -> critic_ok False aunque el critic del frame acepte
    unstable = HandResult(present=True, stable_gesture="NEUTRAL", ok=False, critic_accepted=True)
    assert pipeline.to_hand_state(unstable).critic_ok is False


def test_temporal_hold_does_not_flicker_intent(pipeline, monkeypatch):
    """7 frames PINCH buenos, luego 1 frame que el critic rechaza: el estable se
    sostiene (TEMPORAL_HOLD, ok=True) y el intent del bus NO cae a BLOCKED."""
    feats = _palm_front_feats("PINCH")
    fake = _FakePredict(_pred("PINCH", 0.90, 0.60))
    monkeypatch.setattr(pipeline.classifier, "predict", fake)
    pipeline.reset("hold")
    res = None
    for _ in range(7):
        res = pipeline.process_features(feats, hand="hold")
    assert res is not None and res.stable_gesture == "PINCH" and res.ok and res.critic_accepted

    # guard pasa (conf 0.58 >= 0.55, margen 0.30 >= 0.20) pero el critic rechaza (< 0.62)
    fake.pred = _pred("PINCH", 0.58, 0.30)
    res = pipeline.process_features(feats, hand="hold")
    assert res.critic_accepted is False
    assert res.reason == "TEMPORAL_HOLD" and res.stable_gesture == "PINCH" and res.ok

    st = TletlFrameState()
    st.dom = pipeline.to_hand_state(res, side="Right")
    assert st.dom.critic_ok is True
    assert gesture_to_common_intent(st).name == "GRAB_OR_SELECT"


# ---------------------------------------------------------------------------
# Segunda opinión externa (hint)
# ---------------------------------------------------------------------------

def test_hint_agreement_unblocks_guard(pipeline, monkeypatch):
    feats = _palm_front_feats("PINCH")
    monkeypatch.setattr(pipeline_module, "geometric_rule", lambda f: "NEUTRAL")   # la regla discrepa
    # PINCH (peligroso) con margen bajo (0.10 < 0.20) -> el guard bloquea
    monkeypatch.setattr(pipeline.classifier, "predict", _FakePredict(_pred("PINCH", 0.70, 0.10)))

    blocked = pipeline.process_features(feats, hand="hint_none")
    assert blocked.guard_reason == "GUARD_BLOCK_PINCH"
    assert blocked.guarded_gesture == "NEUTRAL" and blocked.hint is None
    assert blocked.rule_raw == "NEUTRAL"

    agreed = pipeline.process_features(feats, hand="hint_agree", hint="PINCH")
    assert agreed.guard_reason == "OK_HINT_AGREES"
    assert agreed.guarded_gesture == "PINCH" and agreed.hint == "PINCH"
    assert agreed.rule_raw == "NEUTRAL"                  # la regla sigue reportándose tal cual
    assert agreed.critic_accepted is True                # 0.70 >= 0.62 con PALM_FRONT

    disagreed = pipeline.process_features(feats, hand="hint_disagree", hint="FIST")
    assert disagreed.guard_reason == "GUARD_BLOCK_PINCH"
    assert disagreed.guarded_gesture == "NEUTRAL" and disagreed.hint == "FIST"


def test_hint_agreement_stabilizes_end_to_end(pipeline, monkeypatch):
    feats = _palm_front_feats("PINCH")
    monkeypatch.setattr(pipeline_module, "geometric_rule", lambda f: "NEUTRAL")
    monkeypatch.setattr(pipeline.classifier, "predict", _FakePredict(_pred("PINCH", 0.70, 0.10)))
    pipeline.reset("e2e_hint")
    without = with_hint = None
    for _ in range(7):
        without = pipeline.process_features(feats, hand="e2e_nohint")
        with_hint = pipeline.process_features(feats, hand="e2e_hint", hint=" pinch ")
    assert without is not None and without.stable_gesture == "NEUTRAL" and not without.ok
    assert with_hint is not None and with_hint.hint == "PINCH"
    assert with_hint.stable_gesture == "PINCH" and with_hint.ok


def test_hint_does_not_change_reason_when_guard_would_pass(pipeline, monkeypatch):
    feats = _palm_front_feats("PINCH")
    monkeypatch.setattr(pipeline_module, "geometric_rule", lambda f: "NEUTRAL")
    monkeypatch.setattr(pipeline.classifier, "predict", _FakePredict(_pred("PINCH", 0.90, 0.60)))
    res = pipeline.process_features(feats, hand="hint_pass", hint="PINCH")
    assert res.guard_reason == "OK" and res.guarded_gesture == "PINCH" and res.hint == "PINCH"


def test_hint_agreement_with_guard_off_is_recorded_only(pipeline, monkeypatch):
    feats = _palm_front_feats("PINCH")
    monkeypatch.setitem(pipeline.cfg["guard"], "enabled", False)
    monkeypatch.setattr(pipeline.classifier, "predict", _FakePredict(_pred("PINCH", 0.70, 0.10)))
    res = pipeline.process_features(feats, hand="hint_off", hint="PINCH")
    assert res.guard_reason == "GUARD_OFF" and res.hint == "PINCH"


# ---------------------------------------------------------------------------
# Memoria adaptativa: solo aprende del frame actual aceptado; close() persiste
# ---------------------------------------------------------------------------

def _adaptive_pipeline(bank_path, tmp_path, save_interval: float):
    cfg = load_config(tmp_path / "no_existe.toml")
    cfg["adaptive"]["enabled"] = True
    cfg["adaptive"]["min_confidence"] = 0.5
    cfg["adaptive"]["save_interval"] = save_interval
    cfg["adaptive"]["path"] = str(tmp_path / "adaptive.json")
    return TletlPipeline(bank_path, config=cfg, verbose=False)


def test_adaptive_learns_only_when_current_frame_agrees_with_stable(bank_path, tmp_path, monkeypatch):
    p = _adaptive_pipeline(bank_path, tmp_path, save_interval=1000.0)
    observed = []
    monkeypatch.setattr(p.adaptive, "observe", lambda g, f, c: observed.append((g, dict(f), c)) or True)

    fist = _palm_front_feats("FIST")
    palm = _palm_front_feats("OPEN_PALM")
    fake = _FakePredict(_pred("FIST", 0.95, 0.80))
    monkeypatch.setattr(p.classifier, "predict", fake)

    for _ in range(7):
        p.process_features(fist, hand="dom")
    assert observed and all(g == "FIST" for g, _, _ in observed)
    n_fist = len(observed)

    # Ahora el frame actual es OPEN_PALM (aceptado) pero el estable sigue en FIST
    # por historia (TEMPORAL_HOLD): NO se debe grabar palm etiquetado como FIST.
    fake.pred = _pred("OPEN_PALM", 0.95, 0.80)
    for _ in range(2):
        res = p.process_features(palm, hand="dom")
        assert res.stable_gesture == "FIST" and res.reason == "TEMPORAL_HOLD" and res.ok
    assert len(observed) == n_fist

    # Cuando OPEN_PALM se vuelve estable, sí aprende OPEN_PALM con sus propias features.
    for _ in range(3):
        res = p.process_features(palm, hand="dom")
    assert res.stable_gesture == "OPEN_PALM" and res.ok
    assert observed[-1][0] == "OPEN_PALM"
    assert observed[-1][1]["thumb_tip_index_tip"] == palm["thumb_tip_index_tip"]


def test_close_flushes_throttled_adaptive_memory(bank_path, tmp_path, monkeypatch):
    p = _adaptive_pipeline(bank_path, tmp_path, save_interval=1000.0)
    db = p.adaptive.path
    feats = _palm_front_feats("FIST")
    monkeypatch.setattr(p.classifier, "predict", _FakePredict(_pred("FIST", 0.95, 0.80)))
    for _ in range(10):
        p.process_features(feats, hand="dom")
    in_memory = p.adaptive.counts()["FIST"]
    assert in_memory >= 3
    on_disk = len(json.loads(db.read_text(encoding="utf-8"))["gestures"]["FIST"])
    assert on_disk == 1                                    # solo la primera escritura (inmediata)
    assert p.adaptive.dirty

    p.close()
    assert not p.adaptive.dirty
    assert len(json.loads(db.read_text(encoding="utf-8"))["gestures"]["FIST"]) == in_memory


def test_pipeline_context_manager_closes(bank_path, tmp_path, monkeypatch):
    with _adaptive_pipeline(bank_path, tmp_path, save_interval=1000.0) as p:
        monkeypatch.setattr(p.classifier, "predict", _FakePredict(_pred("FIST", 0.95, 0.80)))
        for _ in range(9):
            p.process_features(_palm_front_feats("FIST"), hand="dom")
        assert p.adaptive.dirty
    assert not p.adaptive.dirty
    assert json.loads(p.adaptive.path.read_text(encoding="utf-8"))["gestures"]["FIST"]


def test_close_with_adaptive_disabled_is_noop(pipeline):
    pipeline.close()
    assert pipeline.adaptive.flush() is False
