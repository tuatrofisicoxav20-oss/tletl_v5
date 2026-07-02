from __future__ import annotations

from tletl_core.intent import gesture_to_common_intent
from tletl_core.state import TletlFrameState, TletlHandState


def _state(gesture, critic_ok=True, present=True):
    st = TletlFrameState()
    st.dom = TletlHandState(present=present, gesture=gesture, critic_ok=critic_ok,
                            confidence=0.9, critic_reason="OK")
    return st


def test_no_hand_is_none():
    intent = gesture_to_common_intent(_state("PINCH", present=False))
    assert intent.name == "NONE"


def test_blocked_when_critic_fails():
    intent = gesture_to_common_intent(_state("PINCH", critic_ok=False))
    assert intent.name == "BLOCKED"


def test_pinch_is_grab():
    assert gesture_to_common_intent(_state("PINCH")).name == "GRAB_OR_SELECT"


def test_fist_is_safety_stop():
    assert gesture_to_common_intent(_state("FIST")).name == "SAFETY_STOP"


def test_three_is_mode_next():
    assert gesture_to_common_intent(_state("THREE")).name == "MODE_NEXT"
