"""tests/test_gesture_modeler.py — máquina de estados gesto -> ops (sin ningún CAD).

Los deltas se calculan entre frames consecutivos: un frame aislado NUNCA mueve.
El reloj se inyecta (`now`) para probar los holds (THREE 0.8 s, FIST 3 s) y la
obsolescencia del bus sin dormir.
"""

from __future__ import annotations

import math

import pytest

from apps.autocad_control.gesture_modeler import (
    MIN_SCALE,
    SOLID_KINDS,
    GestureModeler,
    Scene,
    Solid,
    format_op,
    frame_timestamp,
    hand_gesture,
    hand_palm,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _state(dom="NO_HAND", dpalm=None, mod="NO_HAND", mpalm=None, ts=None) -> dict:
    st = {
        "dom": {"gesture": dom, "palm": list(dpalm) if dpalm else None, "present": dom != "NO_HAND"},
        "mod": {"gesture": mod, "palm": list(mpalm) if mpalm else None, "present": mod != "NO_HAND"},
        "intent": {"name": "NONE", "active": False},
        "mode": "NONE",
    }
    if ts is not None:
        st["timestamp"] = ts
    return st


def _feed(m: GestureModeler, frames, *, dt=0.05, start=0.0):
    """Alimenta frames consecutivos separados `dt` s; devuelve la lista de ops por frame."""
    out = []
    t = start
    for fr in frames:
        out.append(m.update(fr, t))
        t += dt
    return out


def _modeler(**kw) -> GestureModeler:
    kw.setdefault("gain", 10.0)
    kw.setdefault("stale_after", None)
    return GestureModeler(**kw)


def _with_box(**kw) -> GestureModeler:
    m = _modeler(**kw)
    m.scene.add("box", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    return m


def _names(ops):
    return [op[0] for op in ops]


# ── Solid / Scene ────────────────────────────────────────────────────────────

def test_solid_validates_kind_and_casts_vectors():
    s = Solid(id="3", kind="box", center=[1, 2, 3], size=(2, 2, 2))
    assert s.id == 3 and s.center == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError):
        Solid(id=1, kind="torus")
    with pytest.raises(ValueError):
        Solid(id=1, kind="box", center=(1, 2))


def test_solid_dimensions_apply_scale():
    s = Solid(id=1, kind="cylinder", size=(2.0, 2.0, 5.0), scale=0.5)
    assert s.dimensions == (1.0, 1.0, 2.5)
    assert Solid.from_dict(s.to_dict()) == s


def test_scene_add_assigns_incremental_ids_and_selects():
    sc = Scene()
    a = sc.add("box")
    b = sc.add("sphere", (1, 1, 1), (2, 2, 2))
    assert (a.id, b.id) == (1, 2)
    assert sc.selected is b and sc.next_id == 3 and len(sc) == 2
    assert sc.get(99) is None and sc.get(None) is None


def test_scene_apply_move_rotate_scale():
    sc = Scene()
    s = sc.add("cone", (0, 0, 0), (1, 1, 1))
    assert sc.apply(("move", s.id, 1.0, -2.0, 0.5))
    assert sc.apply(("rotate", s.id, math.pi / 4))
    assert sc.apply(("scale", s.id, 3.0))
    assert s.center == (1.0, -2.0, 0.5)
    assert s.rotation_z == pytest.approx(math.pi / 4)
    assert s.scale == pytest.approx(3.0)


def test_scene_apply_ignores_unknown_ids_and_non_scene_ops():
    sc = Scene()
    sc.add("box")
    assert not sc.apply(("move", 42, 1, 1, 1))
    assert not sc.apply(("mode", "CREATE"))
    assert not sc.apply(("end_session",))
    assert not sc.apply(("select", 42))
    assert sc.selected_id is None          # select de id inexistente deselecciona
    assert not sc.apply(("add", Solid(id=1, kind="box")))   # id duplicado no se agrega
    assert len(sc) == 1


def test_scene_add_op_stores_a_copy():
    sc = Scene()
    original = Solid(id=7, kind="wedge", center=(1, 1, 1))
    sc.apply(("add", original))
    original.center = (9, 9, 9)
    assert sc.get(7).center == (1.0, 1.0, 1.0)
    assert sc.next_id == 8


def test_scene_dict_roundtrip_and_selected_fallback():
    sc = Scene(units="m")
    sc.add("box", (1, 2, 3), (4, 5, 6), rotation_z=0.3, scale=1.5)
    sc.add("sphere")
    data = sc.to_dict()
    back = Scene.from_dict(data)
    assert back.to_dict() == data and back.units == "m"
    data["selected_id"] = 999                     # selección inválida -> último sólido
    assert Scene.from_dict(data).selected_id == 2
    with pytest.raises(ValueError):
        Scene.from_dict({"nope": 1})


# ── TRANSFORM: traslación por delta ──────────────────────────────────────────

def test_single_pinch_frame_does_not_move():
    m = _with_box()
    assert m.update(_state("PINCH", (0.8, 0.2)), 0.0) == []


def test_pinch_delta_moves_selected_with_gain_and_inverted_y():
    m = _with_box(gain=10.0)
    m.update(_state("PINCH", (0.5, 0.5)), 0.0)
    ops = m.update(_state("PINCH", (0.6, 0.3)), 0.05)     # +0.1 en x, la mano SUBE (y pantalla baja)
    assert ops == [("move", 1, pytest.approx(1.0), pytest.approx(2.0), 0.0)]


def test_move_ops_are_applied_to_own_scene():
    m = _with_box(gain=10.0)
    _feed(m, [_state("PINCH", (0.5, 0.5)), _state("PINCH", (0.4, 0.5)), _state("PINCH", (0.3, 0.5))])
    assert m.scene.get(1).center == pytest.approx((-2.0, 0.0, 0.0))
    assert m.ops_emitted == 2


def test_no_selected_solid_no_move_ops():
    m = _modeler()                                   # escena vacía
    ops = _feed(m, [_state("PINCH", (0.5, 0.5)), _state("PINCH", (0.9, 0.1))])
    assert ops == [[], []]
    assert "seleccionado" in m.status


def test_open_palm_release_prevents_jump():
    m = _with_box(gain=10.0)
    ops = _feed(m, [_state("PINCH", (0.5, 0.5)), _state("OPEN_PALM", (0.9, 0.9)),
                    _state("PINCH", (0.9, 0.9)), _state("PINCH", (0.95, 0.9))])
    assert ops[1] == [] and ops[2] == []             # soltar no mueve; re-agarrar no salta
    assert ops[3] == [("move", 1, pytest.approx(0.5), pytest.approx(0.0), 0.0)]


def test_fist_resets_continuity_and_emits_nothing():
    m = _with_box(gain=10.0)
    ops = _feed(m, [_state("PINCH", (0.5, 0.5)), _state("FIST", (0.9, 0.9)), _state("PINCH", (0.9, 0.9))])
    assert ops == [[], [], []]
    assert "seguridad" in m.status or m.status        # status legible
    assert m.scene.get(1).center == (0.0, 0.0, 0.0)


def test_neutral_and_no_hand_emit_nothing():
    m = _with_box()
    ops = _feed(m, [_state("NEUTRAL", (0.5, 0.5)), _state("NO_HAND"), {}, _state("POINT", (0.2, 0.2))])
    assert ops == [[], [], [], []]


# ── TRANSFORM: rotación / Z / escala ─────────────────────────────────────────

def test_mod_pinch_rotates_z_by_horizontal_delta():
    m = _with_box(rot_gain=2.0)
    m.update(_state(mod="PINCH", mpalm=(0.4, 0.5)), 0.0)
    ops = m.update(_state(mod="PINCH", mpalm=(0.5, 0.5)), 0.05)
    assert ops == [("rotate", 1, pytest.approx(0.1 * 2.0 * math.pi))]
    assert m.scene.get(1).rotation_z == pytest.approx(0.2 * math.pi)


def test_z_gain_off_by_default_and_on_when_set():
    m = _with_box()
    m.update(_state(mod="PINCH", mpalm=(0.5, 0.6)), 0.0)
    assert m.update(_state(mod="PINCH", mpalm=(0.5, 0.4)), 0.05) == []     # solo vertical: nada
    m2 = _with_box(z_gain=5.0)
    m2.update(_state(mod="PINCH", mpalm=(0.5, 0.6)), 0.0)
    ops = m2.update(_state(mod="PINCH", mpalm=(0.5, 0.4)), 0.05)
    assert ops == [("move", 1, 0.0, 0.0, pytest.approx(1.0))]      # sube 0.2 * 5


def test_two_hand_spread_scales_up_and_close_scales_down():
    m = _with_box(scale_gain=2.0)
    m.update(_state("PINCH", (0.4, 0.5), "OPEN_PALM", (0.6, 0.5)), 0.0)          # dist 0.2
    ops = m.update(_state("PINCH", (0.4, 0.5), "OPEN_PALM", (0.9, 0.5)), 0.05)   # dist 0.5
    scale_ops = [op for op in ops if op[0] == "scale"]
    assert scale_ops == [("scale", 1, pytest.approx(1.6))]                       # 1 + 0.3*2
    assert m.scene.get(1).scale == pytest.approx(1.6)
    ops = m.update(_state("PINCH", (0.4, 0.5), "OPEN_PALM", (0.5, 0.5)), 0.10)   # dist 0.1
    factor = [op for op in ops if op[0] == "scale"][0][2]
    assert factor < 1.0 and m.scene.get(1).scale == pytest.approx(0.8)


def test_scale_never_below_min_scale():
    m = _with_box(scale_gain=50.0)
    m.update(_state("PINCH", (0.05, 0.5), "OPEN_PALM", (0.95, 0.5)), 0.0)
    m.update(_state("PINCH", (0.5, 0.5), "OPEN_PALM", (0.5, 0.5)), 0.05)
    assert m.scene.get(1).scale == pytest.approx(MIN_SCALE)


def test_scale_needs_two_frames_and_correct_gesture_pair():
    m = _with_box()
    assert m.update(_state("PINCH", (0.4, 0.5), "OPEN_PALM", (0.6, 0.5)), 0.0) == []
    # dom OPEN_PALM + mod OPEN_PALM no escala (y rompe la distancia previa)
    assert m.update(_state("OPEN_PALM", (0.4, 0.5), "OPEN_PALM", (0.9, 0.5)), 0.05) == []


def test_deadzone_accumulates_small_deltas():
    m = _with_box(gain=10.0, deadzone=0.05)
    ops = _feed(m, [_state("PINCH", (0.50, 0.5)), _state("PINCH", (0.52, 0.5)),
                    _state("PINCH", (0.54, 0.5)), _state("PINCH", (0.56, 0.5))])
    assert ops[1] == [] and ops[2] == []
    assert ops[3] == [("move", 1, pytest.approx(0.6), pytest.approx(0.0), 0.0)]   # acumulado 0.06


# ── Holds: THREE (modo) ──────────────────────────────────────────────────────

def test_three_hold_toggles_mode_once_per_hold():
    m = _modeler(mode_hold=0.8)
    three = _state("THREE", (0.5, 0.5))
    assert m.update(three, 0.0) == []
    assert m.update(three, 0.5) == []
    assert m.update(three, 0.8) == [("mode", "CREATE")]
    assert m.mode == "CREATE"
    assert m.update(three, 2.0) == []                   # sigue sostenido: no re-dispara
    assert m.update(_state("NEUTRAL", (0.5, 0.5)), 2.1) == []
    assert m.update(three, 2.2) == []
    assert m.update(three, 3.1) == [("mode", "TRANSFORM")]


def test_three_short_hold_does_not_toggle():
    m = _modeler(mode_hold=0.8)
    ops = _feed(m, [_state("THREE", (0.5, 0.5))] * 10, dt=0.05)   # 0.45 s en total
    assert all(o == [] for o in ops) and m.mode == "TRANSFORM"


def test_mode_toggle_breaks_continuity():
    m = _with_box(gain=10.0, mode_hold=0.1)
    m.update(_state("PINCH", (0.5, 0.5)), 0.0)
    m.update(_state("THREE", (0.5, 0.5)), 0.05)
    m.update(_state("THREE", (0.5, 0.5)), 0.2)          # -> CREATE
    m.update(_state("THREE", (0.5, 0.5)), 0.25)
    m.update(_state("THREE", (0.5, 0.5)), 0.4)          # sigue en CREATE (mismo hold)
    m.update(_state("NEUTRAL"), 0.45)
    m.update(_state("THREE", (0.5, 0.5)), 0.5)
    m.update(_state("THREE", (0.5, 0.5)), 0.7)          # -> TRANSFORM
    assert m.mode == "TRANSFORM"
    assert m.update(_state("PINCH", (0.9, 0.9)), 0.75) == []   # sin salto


def test_set_mode_validates():
    m = _modeler()
    m.set_mode("CREATE")
    assert m.mode == "CREATE"
    with pytest.raises(ValueError):
        m.set_mode("FLY")


# ── CREATE ───────────────────────────────────────────────────────────────────

def test_create_pinch_rising_edge_spawns_and_selects():
    m = _modeler(gain=2.0)
    m.set_mode("CREATE")
    ops = m.update(_state("PINCH", (0.75, 0.25)), 0.0)
    assert _names(ops) == ["add", "select"]
    solid = ops[0][1]
    assert isinstance(solid, Solid) and solid.kind == "box" and solid.id == 1
    assert solid.center == pytest.approx((0.5, 0.5, 0.0))
    assert ops[1] == ("select", 1)
    assert m.scene.selected_id == 1 and len(m.scene) == 1


def test_create_holding_pinch_spawns_once_and_release_allows_another():
    m = _modeler()
    m.set_mode("CREATE")
    ops = _feed(m, [_state("PINCH", (0.5, 0.5)), _state("PINCH", (0.6, 0.5)), _state("PINCH", (0.7, 0.5)),
                    _state("OPEN_PALM", (0.7, 0.5)), _state("PINCH", (0.2, 0.2))])
    assert [_names(o) for o in ops] == [["add", "select"], [], [], [], ["add", "select"]]
    assert [s.id for s in m.scene.solids] == [1, 2] and m.scene.selected_id == 2


def test_victory_cycles_kind_on_rising_edge_and_wraps():
    m = _modeler()
    m.set_mode("CREATE")
    seen = []
    for _ in range(len(SOLID_KINDS)):
        ops = _feed(m, [_state("VICTORY", (0.5, 0.5)), _state("VICTORY", (0.5, 0.5)), _state("NEUTRAL")])
        assert ops[0] == [("kind", m.kind)] and ops[1] == [] and ops[2] == []
        seen.append(m.kind)
    assert seen == list(SOLID_KINDS[1:]) + [SOLID_KINDS[0]]


def test_kind_is_used_when_spawning():
    m = _modeler()
    m.set_mode("CREATE")
    _feed(m, [_state("VICTORY", (0.5, 0.5)), _state("NEUTRAL"), _state("VICTORY", (0.5, 0.5)), _state("NEUTRAL")])
    ops = m.update(_state("PINCH", (0.5, 0.5)), 1.0)
    assert ops[0][1].kind == "sphere"


def test_spawn_position_maps_palm_to_xy_plane():
    m = _modeler(gain=100.0)
    m.set_mode("CREATE")
    corner = m.update(_state("PINCH", (0.0, 1.0)), 0.0)[0][1]      # abajo-izquierda de la pantalla
    assert corner.center == pytest.approx((-50.0, -50.0, 0.0))
    assert m.palm_to_world((0.5, 0.5)) == (0.0, 0.0, 0.0)


def test_spawn_size_default_is_tenth_of_gain_and_overridable():
    assert GestureModeler(gain=40.0).spawn_size == pytest.approx(4.0)
    assert GestureModeler(gain=40.0, spawn_size=7.0).spawn_size == 7.0
    m = _modeler(gain=40.0)
    m.set_mode("CREATE")
    assert m.update(_state("PINCH", (0.5, 0.5)), 0.0)[0][1].size == pytest.approx((4.0, 4.0, 4.0))


def test_create_mode_ignores_mod_hand_and_transform_gestures():
    m = _with_box()
    m.set_mode("CREATE")
    ops = _feed(m, [_state("PINCH", (0.5, 0.5), "PINCH", (0.2, 0.5)),
                    _state("PINCH", (0.6, 0.5), "PINCH", (0.4, 0.5))])
    assert _names(ops[0]) == ["add", "select"] and ops[1] == []   # ni mueve ni rota


# ── Bus obsoleto / fin de sesión / reloj ─────────────────────────────────────

def test_stale_frame_resets_and_emits_nothing():
    m = _with_box(gain=10.0, stale_after=1.0)
    m.update(_state("PINCH", (0.5, 0.5), ts=100.0), 100.0)
    assert m.update(_state("PINCH", (0.9, 0.9), ts=100.0), 101.5) == []      # frame viejo
    assert m.stale_frames == 1 and "obsoleto" in m.status
    assert m.update(_state("PINCH", (0.9, 0.9), ts=101.6), 101.6) == []      # fresco: sin salto
    assert m.update(_state("PINCH", (1.0, 0.9), ts=101.7), 101.7)[0][0] == "move"


def test_stale_gap_does_not_rearm_pinch_edge():
    m = _modeler(stale_after=1.0)
    m.set_mode("CREATE")
    assert _names(m.update(_state("PINCH", (0.5, 0.5), ts=10.0), 10.0)) == ["add", "select"]
    assert m.update(_state("PINCH", (0.5, 0.5), ts=10.0), 12.0) == []        # obsoleto
    assert m.update(_state("PINCH", (0.5, 0.5), ts=12.1), 12.1) == []        # vuelve con la pinza cerrada: nada
    assert len(m.scene) == 1


def test_stale_disabled_when_none_or_without_timestamp():
    m = _with_box(gain=10.0, stale_after=None)
    m.update(_state("PINCH", (0.5, 0.5), ts=1.0), 500.0)
    assert m.update(_state("PINCH", (0.6, 0.5), ts=1.0), 500.1) != []
    m2 = _with_box(gain=10.0, stale_after=1.0)
    m2.update(_state("PINCH", (0.5, 0.5)), 500.0)                              # sin timestamp: no se juzga
    assert m2.update(_state("PINCH", (0.6, 0.5)), 500.1) != []


def test_fist_hold_emits_end_session_once():
    m = _with_box(end_hold=3.0)
    fist = _state("FIST", (0.5, 0.5))
    assert m.update(fist, 0.0) == []
    assert m.update(fist, 2.9) == []
    assert m.update(fist, 3.0) == [("end_session",)]
    assert m.update(fist, 10.0) == []
    m.update(_state("NEUTRAL"), 10.1)
    assert m.update(fist, 10.2) == [] and m.update(fist, 13.3) == [("end_session",)]


def test_end_hold_none_disables_end_session():
    m = _with_box(end_hold=None)
    assert _feed(m, [_state("FIST", (0.5, 0.5))] * 5, dt=2.0) == [[]] * 5


def test_clock_is_injectable_when_now_is_omitted():
    t = {"now": 0.0}
    m = _modeler(mode_hold=0.8, clock=lambda: t["now"])
    three = _state("THREE", (0.5, 0.5))
    m.update(three)
    t["now"] = 1.0
    assert m.update(three) == [("mode", "CREATE")]


def test_reset_clears_edges_and_holds_but_keeps_scene_and_mode():
    m = _with_box(mode_hold=0.5)
    m.set_mode("CREATE")
    m.update(_state("THREE", (0.5, 0.5)), 0.0)
    m.reset()
    assert m.update(_state("THREE", (0.5, 0.5)), 0.6) == []      # el hold empezó de nuevo
    assert m.mode == "CREATE" and len(m.scene) == 1


# ── Helpers de lectura y formato ─────────────────────────────────────────────

def test_hand_gesture_fallbacks():
    assert hand_gesture({"dom": {"gesture": "PINCH"}}, "dom") == "PINCH"
    assert hand_gesture({"dom": {"gesture": "", "stable_gesture": "FIST"}}, "dom") == "FIST"
    assert hand_gesture({"hands": {"mod": {"gesture": "THREE"}}}, "mod") == "THREE"
    assert hand_gesture({}, "dom") == "NO_HAND"
    assert hand_gesture("basura", "dom") == "NO_HAND"   # type: ignore[arg-type]


def test_hand_palm_formats():
    assert hand_palm({"dom": {"palm": [0.1, 0.2]}}, "dom") == (0.1, 0.2)
    assert hand_palm({"dom": {"palm": {"x": 0.3, "y": 0.4}}}, "dom") == (0.3, 0.4)
    assert hand_palm({"dom": {"palm": None}}, "dom") is None
    assert hand_palm({"dom": {"palm": ["a", "b"]}}, "dom") is None
    assert frame_timestamp({"timestamp": 12.5}) == 12.5
    assert frame_timestamp({"timestamp": 0}) is None and frame_timestamp({}) is None


def test_format_op_covers_all_ops():
    solid = Solid(id=3, kind="cone", center=(1, 2, 3), size=(4, 4, 4))
    texts = [format_op(op) for op in (("add", solid), ("select", 3), ("move", 3, 1.0, -2.0, 0.0),
                                      ("rotate", 3, math.pi / 2), ("scale", 3, 1.25), ("mode", "CREATE"),
                                      ("kind", "wedge"), ("end_session",), ("otra", 1))]
    assert texts[0].startswith("add #3 cone") and "select #3" == texts[1]
    assert "move #3" in texts[2] and "+90.00°" in texts[3] and "x1.2500" in texts[4]
    assert texts[5] == "mode CREATE" and texts[6] == "kind wedge" and texts[7] == "end_session"
    assert texts[8] == "otra 1"


def test_describe_mentions_mode_kind_and_selection():
    m = _with_box()
    m.update(_state("PINCH", (0.5, 0.5), "OPEN_PALM", (0.7, 0.7)), 0.0)
    txt = m.describe()
    assert "modo=TRANSFORM" in txt and "tipo=box" in txt and "sel=#1" in txt
    assert "dom=PINCH" in txt and "mod=OPEN_PALM" in txt


# ── Obsolescencia por avance del timestamp (relojes desincronizados) ─────────

def test_receiver_clock_ahead_does_not_mark_live_stream_stale():
    """Cliente UDP con el reloj 30 s adelantado respecto a Fedora: los frames
    siguen llegando con timestamp creciente, así que NO son obsoletos (con la
    regla vieja `now - ts > stale_after` todos lo eran y AutoCAD nunca se movía)."""
    m = _with_box(gain=10.0, stale_after=1.0)
    skew = 30.0
    first = m.update(_state("PINCH", (0.5, 0.5), ts=100.0), 100.0 + skew)
    assert first == [] and m.stale_frames == 1            # solo el primero, por prudencia
    assert m.update(_state("PINCH", (0.5, 0.5), ts=100.05), 100.05 + skew) == []   # fresco: prima
    ops = m.update(_state("PINCH", (0.6, 0.5), ts=100.10), 100.10 + skew)
    assert _names(ops) == ["move"] and m.stale_frames == 1


def test_receiver_clock_behind_is_fine_too():
    m = _with_box(gain=10.0, stale_after=1.0)
    m.update(_state("PINCH", (0.5, 0.5), ts=100.0), 70.0)
    assert m.stale_frames == 0
    assert _names(m.update(_state("PINCH", (0.6, 0.5), ts=100.05), 70.05)) == ["move"]


def test_same_frame_redelivered_becomes_stale_after_timeout():
    m = _with_box(gain=10.0, stale_after=1.0)
    same = _state("PINCH", (0.5, 0.5), ts=50.0)
    m.update(same, 50.0)
    assert m.update(same, 50.5) == []          # mismo frame reentregado: sin delta, aún no obsoleto
    assert m.stale_frames == 0
    assert m.update(same, 51.2) == []          # > 1 s sin avance del timestamp: obsoleto
    assert m.stale_frames == 1


def test_timestamp_jumping_backwards_is_a_new_stream():
    """La app se reinició o el reloj del emisor se ajustó: un salto atrás grande
    no deja al modelador esperando a que el ts 'alcance' al anterior."""
    m = _with_box(gain=10.0, stale_after=1.0)
    m.update(_state("PINCH", (0.5, 0.5), ts=1000.0), 1000.0)
    m.update(_state("PINCH", (0.5, 0.5), ts=1000.05), 1000.05)
    assert m.update(_state("PINCH", (0.5, 0.5), ts=10.0), 1000.10) == []          # flujo nuevo: prima
    assert m.stale_frames == 0
    assert _names(m.update(_state("PINCH", (0.6, 0.5), ts=10.05), 1000.15)) == ["move"]
    # un retroceso pequeño (reordenamiento UDP) NO es flujo nuevo: no se juzga obsoleto de inmediato
    assert m.update(_state("PINCH", (0.6, 0.5), ts=10.04), 1000.20) == []
    assert m.stale_frames == 0
