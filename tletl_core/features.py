from __future__ import annotations

from typing import Dict, List, Optional

import math

from .geometry import Point, dist, palm_center, palm_scale
from .orientation import compute_orientation

LABELS = ["OPEN_PALM", "FIST", "POINT", "VICTORY", "PINCH", "THREE", "NEUTRAL"]
LABEL_KEYS = ("label", "gesture", "target", "class", "y")
NON_FEATURE_KEYS = set(LABEL_KEYS) | {
    "timestamp", "time", "created_at", "source", "orientation", "bucket", "handedness", "score",
    "landmarks", "raw_landmarks", "hand_landmarks", "features",
}
EXCLUDED_FEATURES = {
    "palm_center_x", "palm_center_y", "screen_x", "screen_y", "cx", "cy",
}
META_KEYS = NON_FEATURE_KEYS | EXCLUDED_FEATURES


def extract_live_features(lm: List[Point]) -> Dict[str, float]:
    scale = palm_scale(lm)
    wrist = lm[0]
    features: Dict[str, float] = {}

    fingers = {
        "index": (8, 6, 5),
        "middle": (12, 10, 9),
        "ring": (16, 14, 13),
        "pinky": (20, 18, 17),
    }
    for name, (tip, pip, mcp) in fingers.items():
        features[f"{name}_tip_wrist"] = dist(lm[tip], wrist) / scale
        features[f"{name}_pip_wrist"] = dist(lm[pip], wrist) / scale
        features[f"{name}_mcp_wrist"] = dist(lm[mcp], wrist) / scale
        features[f"{name}_tip_mcp"] = dist(lm[tip], lm[mcp]) / scale
        features[f"{name}_tip_pip"] = dist(lm[tip], lm[pip]) / scale
        features[f"{name}_vertical"] = (lm[pip].y - lm[tip].y) / scale
        features[f"{name}_curl"] = (dist(lm[tip], wrist) - dist(lm[pip], wrist)) / scale
        features[f"{name}_tip_y_mcp"] = (lm[mcp].y - lm[tip].y) / scale
        features[f"{name}_tip_z_mcp"] = (lm[tip].z - lm[mcp].z) / scale

    features["thumb_tip_wrist"] = dist(lm[4], wrist) / scale
    features["thumb_ip_wrist"] = dist(lm[3], wrist) / scale
    features["thumb_tip_index_mcp"] = dist(lm[4], lm[5]) / scale
    features["thumb_tip_index_tip"] = dist(lm[4], lm[8]) / scale
    features["thumb_tip_middle_tip"] = dist(lm[4], lm[12]) / scale
    features["thumb_horizontal"] = abs(lm[4].x - lm[3].x) / scale
    features["thumb_vertical"] = abs(lm[4].y - lm[3].y) / scale
    features["thumb_tip_palm"] = dist(lm[4], lm[9]) / scale

    features["index_middle_spread"] = dist(lm[8], lm[12]) / scale
    features["middle_ring_spread"] = dist(lm[12], lm[16]) / scale
    features["ring_pinky_spread"] = dist(lm[16], lm[20]) / scale
    features["index_pinky_spread"] = dist(lm[8], lm[20]) / scale
    features["index_ring_spread"] = dist(lm[8], lm[16]) / scale
    features["thumb_pinky_spread"] = dist(lm[4], lm[20]) / scale
    features["palm_width"] = dist(lm[5], lm[17]) / scale
    features["wrist_middle_mcp"] = dist(lm[0], lm[9]) / scale

    cx, cy = palm_center(lm)
    features["palm_center_x"] = cx
    features["palm_center_y"] = cy

    features.update(compute_orientation(lm))
    return features


def get_label(obj: dict) -> Optional[str]:
    for k in LABEL_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().upper()
    return None


def get_features(obj: dict) -> Dict[str, float]:
    src = obj.get("features") if isinstance(obj.get("features"), dict) else obj
    out: Dict[str, float] = {}
    for k, v in src.items():
        if k in NON_FEATURE_KEYS:
            continue
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[k] = float(v)
    return out
