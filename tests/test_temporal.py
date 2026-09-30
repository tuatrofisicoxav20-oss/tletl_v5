from __future__ import annotations

import time

import pytest

from tletl_core.classifier import Prediction
from tletl_core.temporal import CursorDelta, Hold, MotionTracker, TemporalFilter


def _pred(label="PINCH", ok=True):
    return Prediction(label, label, 0.9, 0.5, "PALM_FRONT", "OK", ok, {label: 1.0})


class FakeClock:
    """Reloj inyectable: los tests de MotionTracker/Hold son deterministas."""

    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


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


# ---------------------------------------------------------------------------
# MotionTracker con reloj inyectado (antes dependía de time.time() real)
# ---------------------------------------------------------------------------

def _feed(mt: MotionTracker, clock: FakeClock, points, step: float):
    for x, y in points:
        mt.update(x, y)
        clock.advance(step)


def test_motion_tracker_detects_horizontal_swipe():
    clock = FakeClock()
    mt = MotionTracker(clock=clock)
    _feed(mt, clock, [(0.1 + i * 0.05, 0.5) for i in range(6)], step=0.05)   # dx=0.25 en 0.25 s
    assert mt.swipe() == "RIGHT"
    assert len(mt.points) == 0            # el swipe consume los puntos
    assert mt.swipe() is None


def test_motion_tracker_left_and_vertical_swipes():
    clock = FakeClock()
    mt = MotionTracker(clock=clock)
    _feed(mt, clock, [(0.8 - i * 0.05, 0.5) for i in range(6)], step=0.05)
    assert mt.swipe() == "LEFT"

    clock.advance(1.0)                    # fuera del cooldown
    _feed(mt, clock, [(0.5, 0.2 + i * 0.05) for i in range(6)], step=0.05)
    assert mt.swipe() == "DOWN"

    clock.advance(1.0)
    _feed(mt, clock, [(0.5, 0.8 - i * 0.05) for i in range(6)], step=0.05)
    assert mt.swipe() == "UP"


def test_motion_tracker_slow_motion_is_not_a_swipe():
    clock = FakeClock()
    mt = MotionTracker(clock=clock)
    _feed(mt, clock, [(0.1 + i * 0.05, 0.5) for i in range(6)], step=0.30)   # 1.5 s > 0.90 s
    assert mt.swipe() is None


def test_motion_tracker_cooldown_after_swipe():
    clock = FakeClock()
    mt = MotionTracker(clock=clock)
    _feed(mt, clock, [(0.1 + i * 0.05, 0.5) for i in range(6)], step=0.05)
    assert mt.swipe() == "RIGHT"
    # segundo swipe dentro de los 0.55 s de cooldown -> ignorado
    _feed(mt, clock, [(0.1 + i * 0.05, 0.5) for i in range(6)], step=0.02)
    assert mt.swipe() is None
    clock.advance(0.6)
    assert mt.swipe() == "RIGHT"          # mismos puntos, cooldown vencido


def test_motion_tracker_needs_five_points_and_velocity():
    clock = FakeClock()
    mt = MotionTracker(clock=clock)
    _feed(mt, clock, [(0.1, 0.5), (0.4, 0.5)], step=0.1)
    assert mt.swipe() is None
    assert mt.last_velocity == pytest.approx(3.0)   # 0.3 en 0.1 s


# ---------------------------------------------------------------------------
# Hold con reloj inyectado
# ---------------------------------------------------------------------------

def test_hold_progress_is_deterministic():
    clock = FakeClock(0.0)
    hold = Hold(clock=clock)
    assert hold.progress("OPEN_PALM", 2.0) == 0.0
    clock.advance(1.0)
    assert hold.progress("OPEN_PALM", 2.0) == pytest.approx(0.5)
    clock.advance(1.5)
    assert hold.progress("OPEN_PALM", 2.0) == 1.0          # saturado en 1.0
    assert hold.progress("THREE", 2.0) == 0.0               # otro nombre reinicia
    hold.reset()
    assert hold.name is None and hold.progress("THREE", 2.0) == 0.0


def test_default_clock_is_wall_time():
    """API compatible: sin argumentos siguen usando time.time()."""
    assert MotionTracker()._clock is time.time
    assert Hold()._clock is time.time
    assert MotionTracker(maxlen=5).points.maxlen == 5
