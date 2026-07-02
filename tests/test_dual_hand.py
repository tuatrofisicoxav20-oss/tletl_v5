from __future__ import annotations

import pytest

from tletl_core.config import load_config
from tletl_core.pipeline import TletlPipeline
from tletl_core.features import get_features, get_label
from tletl_core.state import TletlFrameState


@pytest.fixture(scope="module")
def pipeline(bank_path):
    return TletlPipeline(bank_path, config=load_config(),
                         adaptive_path="/tmp/tletl_adaptive_dual_ignore.json")


def test_temporal_filters_are_independent_per_hand(pipeline):
    """Cada mano mantiene su propio TemporalFilter: no comparten historia."""
    pipeline.reset()
    tf_dom = pipeline._temporal_for("dom")
    tf_mod = pipeline._temporal_for("mod")
    assert tf_dom is not tf_mod


def test_two_hands_classify_independently(pipeline, bank_rows):
    """dom con FIST y mod con OPEN_PALM deben producir gestos independientes."""
    fist = [get_features(o) for o in bank_rows if get_label(o) == "FIST"][:20]
    palm = [get_features(o) for o in bank_rows if get_label(o) == "OPEN_PALM"][:20]
    assert fist and palm

    pipeline.reset()
    dom_res = mod_res = None
    for i in range(20):
        dom_res = pipeline.process_features(fist[i % len(fist)], hand="dom")
        mod_res = pipeline.process_features(palm[i % len(palm)], hand="mod")

    # las dos manos no colapsan al mismo gesto por compartir estado
    assert dom_res is not None and mod_res is not None
    assert dom_res.raw_gesture != mod_res.raw_gesture or dom_res.stable_gesture != mod_res.stable_gesture


def test_frame_state_holds_both_hands(pipeline, bank_rows):
    """El TletlFrameState puede llevar dom y mod presentes a la vez."""
    fist = [get_features(o) for o in bank_rows if get_label(o) == "FIST"][:10]
    palm = [get_features(o) for o in bank_rows if get_label(o) == "OPEN_PALM"][:10]

    pipeline.reset()
    dom_res = mod_res = None
    for i in range(10):
        dom_res = pipeline.process_features(fist[i % len(fist)], hand="dom")
        mod_res = pipeline.process_features(palm[i % len(palm)], hand="mod")

    st = TletlFrameState()
    st.dom = pipeline.to_hand_state(dom_res, side="Right")
    st.mod = pipeline.to_hand_state(mod_res, side="Left")
    assert st.dom.present and st.mod.present
    assert st.dom.side == "Right" and st.mod.side == "Left"
