from __future__ import annotations

import math
import time
from collections import Counter, deque
from typing import Deque, Optional, Tuple

from .classifier import Prediction
from .geometry import clamp


class TemporalFilter:
    def __init__(self, size: int = 7, min_count: int = 4):
        self.history: Deque[str] = deque(maxlen=size)
        self.last = "NEUTRAL"
        self.min_count = min_count

    def update(self, pred: Prediction) -> Prediction:
        if pred.ok:
            self.history.append(pred.raw_label)
        else:
            self.history.append("NEUTRAL")
        if len(self.history) < self.history.maxlen:
            return Prediction("NEUTRAL", pred.raw_label, pred.confidence, pred.margin, pred.orientation, "WARMING_UP", False, pred.votes)

        label, count = Counter(self.history).most_common(1)[0]
        if count >= self.min_count:
            self.last = label
            if pred.ok and label == pred.raw_label:
                return Prediction(label, pred.raw_label, pred.confidence, pred.margin, pred.orientation, pred.reason, True, pred.votes)
            return Prediction(label, pred.raw_label, pred.confidence, pred.margin, pred.orientation, "TEMPORAL_HOLD", label != "NEUTRAL", pred.votes)
        return Prediction(self.last, pred.raw_label, pred.confidence, pred.margin, pred.orientation, "UNSTABLE_TEMPORAL", False, pred.votes)

    def reset(self) -> None:
        self.history.clear()
        self.last = "NEUTRAL"


class MotionTracker:
    def __init__(self, maxlen: int = 9):
        self.points: Deque[Tuple[float, float, float]] = deque(maxlen=maxlen)
        self.last_swipe_time = 0.0
        self.last_velocity = 0.0

    def update(self, x: float, y: float) -> None:
        now = time.time()
        if self.points:
            t0, x0, y0 = self.points[-1]
            dt = max(now - t0, 1e-3)
            self.last_velocity = math.sqrt((x - x0) ** 2 + (y - y0) ** 2) / dt
        self.points.append((now, x, y))

    def swipe(self) -> Optional[str]:
        if len(self.points) < 5:
            return None
        now = time.time()
        if now - self.last_swipe_time < 0.55:
            return None

        t0, x0, y0 = self.points[0]
        t1, x1, y1 = self.points[-1]
        dt = max(t1 - t0, 1e-3)
        dx = x1 - x0
        dy = y1 - y0

        if dt <= 0.90 and abs(dx) > 0.12 and abs(dx) > abs(dy) * 1.45:
            self.last_swipe_time = now
            self.points.clear()
            return "RIGHT" if dx > 0 else "LEFT"
        if dt <= 0.90 and abs(dy) > 0.12 and abs(dy) > abs(dx) * 1.45:
            self.last_swipe_time = now
            self.points.clear()
            return "DOWN" if dy > 0 else "UP"
        return None

    def reset(self) -> None:
        self.points.clear()
        self.last_velocity = 0.0


class CursorDelta:
    def __init__(self):
        self.prev: Optional[Tuple[float, float]] = None

    def update(self, x: float, y: float, gain: int = 1550, max_step: int = 70) -> Tuple[int, int]:
        if self.prev is None:
            self.prev = (x, y)
            return 0, 0
        px, py = self.prev
        self.prev = (x, y)
        dx = x - px
        dy = y - py
        if abs(dx) < 0.0045:
            dx = 0.0
        if abs(dy) < 0.0045:
            dy = 0.0
        return int(clamp(dx * gain, -max_step, max_step)), int(clamp(dy * gain, -max_step, max_step))

    def reset(self) -> None:
        self.prev = None


class Hold:
    def __init__(self):
        self.name: Optional[str] = None
        self.t0 = 0.0

    def progress(self, name: str, seconds: float) -> float:
        now = time.time()
        if self.name != name:
            self.name = name
            self.t0 = now
        return clamp((now - self.t0) / max(seconds, 1e-3), 0.0, 1.0)

    def reset(self) -> None:
        self.name = None
        self.t0 = 0.0
