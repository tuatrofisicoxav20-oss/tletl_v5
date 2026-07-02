from __future__ import annotations

import math

from tletl_core.geometry import Point, clamp, dist, palm_center, palm_scale
from conftest import make_landmarks


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(99, 0, 10) == 10


def test_dist_simple():
    a = Point(0, 0, 0)
    b = Point(3, 4, 0)
    assert math.isclose(dist(a, b), 5.0)


def test_palm_scale_positive():
    lm = make_landmarks("OPEN_PALM")
    assert palm_scale(lm) > 0


def test_palm_center_in_range():
    lm = make_landmarks("OPEN_PALM")
    cx, cy = palm_center(lm)
    assert 0.0 <= cx <= 1.0
    assert 0.0 <= cy <= 1.0
