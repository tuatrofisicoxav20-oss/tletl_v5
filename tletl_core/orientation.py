from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .geometry import Point, clamp, palm_scale


ORIENTATION_KEYS = {
    "palm_normal_x", "palm_normal_y", "palm_normal_z", "palm_roll", "palm_yaw", "palm_pitch",
    "palm_facing_score", "back_hand_score", "side_hand_score", "palm_flatness_score",
    "wrist_depth_score", "thumb_side_score", "finger_depth_spread",
}


def _vec(a: Point, b: Point) -> np.ndarray:
    return np.array([b.x - a.x, b.y - a.y, b.z - a.z], dtype=float)


def _norm_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3, dtype=float)
    return v / n


def compute_orientation(lm: List[Point]) -> Dict[str, float]:
    """Palm orientation from MediaPipe normalized 3D landmarks.

    MediaPipe z is relative/noisy. These are orientation features, not industrial metrology.
    Still useful, unlike pretending palm/back do not exist.
    """
    scale = palm_scale(lm)
    wrist = lm[0]
    index_mcp = lm[5]
    middle_mcp = lm[9]
    pinky_mcp = lm[17]

    across = _vec(pinky_mcp, index_mcp)
    up = _vec(wrist, middle_mcp)
    normal = _norm_vec(np.cross(across, up))

    nx, ny, nz = [float(x) for x in normal]

    roll = math.degrees(math.atan2(across[1], across[0])) if np.linalg.norm(across[:2]) > 1e-9 else 0.0
    yaw = math.degrees(math.atan2(nx, max(abs(nz), 1e-6)))
    pitch = math.degrees(math.atan2(ny, max(abs(nz), 1e-6)))

    abs_nz = abs(nz)
    side_score = clamp(1.0 - abs_nz, 0.0, 1.0)
    palm_facing = clamp((nz + 1.0) / 2.0, 0.0, 1.0)
    back_hand = clamp((-nz + 1.0) / 2.0, 0.0, 1.0)

    mcp_ids = [5, 9, 13, 17]
    z_vals = [lm[i].z for i in mcp_ids]
    z_spread = max(z_vals) - min(z_vals)
    palm_flatness = clamp(1.0 - abs(z_spread) / max(scale, 1e-6), 0.0, 1.0)
    wrist_depth_score = (lm[0].z - lm[9].z) / scale
    thumb_side_score = (lm[4].x - lm[5].x) / scale
    tip_z = [lm[i].z for i in [4, 8, 12, 16, 20]]
    finger_depth_spread = (max(tip_z) - min(tip_z)) / scale

    return {
        "palm_normal_x": nx,
        "palm_normal_y": ny,
        "palm_normal_z": nz,
        "palm_roll": roll / 180.0,
        "palm_yaw": yaw / 180.0,
        "palm_pitch": pitch / 180.0,
        "palm_facing_score": palm_facing,
        "back_hand_score": back_hand,
        "side_hand_score": side_score,
        "palm_flatness_score": palm_flatness,
        "wrist_depth_score": wrist_depth_score,
        "thumb_side_score": thumb_side_score,
        "finger_depth_spread": finger_depth_spread,
    }


def orientation_bucket(feat: Dict[str, float]) -> str:
    pf = float(feat.get("palm_facing_score", 0.0))
    bh = float(feat.get("back_hand_score", 0.0))
    sh = float(feat.get("side_hand_score", 0.0))
    if sh >= max(pf, bh) and sh > 0.40:
        return "SIDE_HAND"
    if bh >= pf and bh > 0.35:
        return "BACK_HAND"
    if pf > 0.35:
        return "PALM_FRONT"
    return "UNKNOWN"
