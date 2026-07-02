from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass
class Point:
    x: float
    y: float
    z: float


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def dist(a: Point, b: Point) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def points_from_mediapipe(hand_landmarks: Any) -> List[Point]:
    return [Point(float(p.x), float(p.y), float(p.z)) for p in hand_landmarks.landmark]


def palm_scale(lm: List[Point]) -> float:
    wrist_to_middle = dist(lm[0], lm[9])
    palm_width = dist(lm[5], lm[17])
    return max((wrist_to_middle + palm_width) / 2.0, 1e-6)


def palm_center(lm: List[Point]) -> Tuple[float, float]:
    ids = [0, 5, 9, 13, 17]
    return sum(lm[i].x for i in ids) / len(ids), sum(lm[i].y for i in ids) / len(ids)
