from __future__ import annotations

from .state import TletlFrameState, TletlIntent


def gesture_to_common_intent(state: TletlFrameState) -> TletlIntent:
    """
    Convierte gesto final en intención común.

    Esta capa NO debe saber si está en Fedora o Blender.
    Solo produce intención abstracta.
    """

    hand = state.dom

    if not hand.present:
        return TletlIntent(name="NONE", active=False, strength=0.0, mode=state.mode, reason="NO_HAND")

    if not hand.critic_ok:
        return TletlIntent(
            name="BLOCKED",
            active=False,
            strength=hand.confidence,
            mode=state.mode,
            reason=hand.critic_reason,
        )

    g = hand.gesture
    conf = float(hand.confidence)

    if g == "PINCH":
        return TletlIntent(name="GRAB_OR_SELECT", active=True, strength=conf, mode=state.mode, reason="PINCH")
    if g == "POINT":
        return TletlIntent(name="POINT", active=True, strength=conf, mode=state.mode, reason="POINT")
    if g == "OPEN_PALM":
        return TletlIntent(name="RELEASE_OR_TOGGLE", active=True, strength=conf, mode=state.mode, reason="OPEN_PALM")
    if g == "FIST":
        return TletlIntent(name="SAFETY_STOP", active=True, strength=conf, mode=state.mode, reason="FIST")
    if g == "THREE":
        return TletlIntent(name="MODE_NEXT", active=True, strength=conf, mode=state.mode, reason="THREE")
    if g == "VICTORY":
        return TletlIntent(name="SECONDARY_ACTION", active=True, strength=conf, mode=state.mode, reason="VICTORY")
    if g == "NEUTRAL":
        return TletlIntent(name="IDLE", active=False, strength=conf, mode=state.mode, reason="NEUTRAL")

    return TletlIntent(name="UNKNOWN", active=False, strength=conf, mode=state.mode, reason=g)
