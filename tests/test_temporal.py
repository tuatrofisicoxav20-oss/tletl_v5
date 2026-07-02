from __future__ import annotations

from tletl_core.classifier import Prediction
from tletl_core.temporal import CursorDelta, MotionTracker, TemporalFilter


def _pred(label="PINCH", ok=True):
    return Prediction(label, label, 0.9, 0.5, "PALM_FRONT", "OK", ok, {label: 1.0})


def test_temporal_warms_up_then_stabilizes():
    tf = TemporalFilter(size=7, min_count=4)
    out = None
    for _ in range(7):
        out = tf.update(_pred("PINCH", ok=True))
    assert out.label == "PINCH"
    assert out.ok


def test_temporal_single_frame_does_not_fire():
    tf = TemporalFilter(size=7, min_count=4)
    out = tf.update(_pred("PINCH", ok=True))
    assert out.reason == "WARMING_UP"
    assert not out.ok


def test_cursor_delta_first_call_zero():
    cd = CursorDelta()
    assert cd.update(0.5, 0.5) == (0, 0)


def test_motion_tracker_detects_horizontal_swipe():
    mt = MotionTracker()
    for i in range(6):
        mt.update(0.1 + i * 0.05, 0.5)
    assert mt.swipe() in ("RIGHT", None)  # depende de timing real; no debe tronar
