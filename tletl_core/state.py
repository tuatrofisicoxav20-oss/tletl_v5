from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

Point2D = Tuple[float, float]


@dataclass
class TletlHandState:
    present: bool = False
    side: str = "unknown"

    gesture: str = "NO_HAND"
    raw_gesture: str = "NO_HAND"
    stable_gesture: str = "NO_HAND"

    orientation: str = "UNKNOWN"
    confidence: float = 0.0
    margin: float = 0.0

    critic_ok: bool = False
    critic_reason: str = "NO_HAND"

    palm: Optional[Point2D] = None
    index: Optional[Point2D] = None
    middle: Optional[Point2D] = None
    thumb: Optional[Point2D] = None

    velocity: Point2D = (0.0, 0.0)
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class TletlIntent:
    name: str = "NONE"
    active: bool = False
    strength: float = 0.0
    mode: str = "NONE"
    reason: str = ""


@dataclass
class TletlFrameState:
    version: int = 4
    app_version: str = "4.9-core-split"
    timestamp: float = 0.0

    frame_width: int = 0
    frame_height: int = 0
    fps: float = 0.0

    dom: TletlHandState = field(default_factory=TletlHandState)
    mod: TletlHandState = field(default_factory=TletlHandState)
    intent: TletlIntent = field(default_factory=TletlIntent)

    mode: str = "NONE"
    action: str = "NOOP"

    selected: bool = False
    grabbed: bool = False
    snap_enabled: bool = False

    transform: Dict[str, Any] = field(default_factory=lambda: {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "rot_x": 0.0,
        "rot_y": 0.0,
        "rot_z": 0.0,
        "scale": 1.0,
    })

    extra: Dict[str, Any] = field(default_factory=dict)
