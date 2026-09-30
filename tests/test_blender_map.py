"""
tests/test_blender_map.py
Tests para la lógica de mapeo del addon de Blender y herramientas auxiliares.

Todos los tests corren SIN bpy ni mediapipe instalados. El mapeo es por DELTA
entre frames (BlenderGestureMapper mantiene la palma anterior), así que un único
frame aislado no produce traslación: hace falta una secuencia.
"""
from __future__ import annotations

import importlib
import json
import sys
import unittest.mock as mock

import pytest


def _import_blocking(module_path: str, blocked: dict) -> object:
    """Importa module_path forzando que los módulos en `blocked` fallen."""
    for key in list(sys.modules.keys()):
        if key == module_path or key.endswith("." + module_path.split(".")[-1]):
            del sys.modules[key]
    with mock.patch.dict(sys.modules, blocked):
        return importlib.import_module(module_path)


_addon = _import_blocking("apps.blender_control.tletl_blender_addon", {"bpy": None})
BlenderGestureMapper   = _addon.BlenderGestureMapper
map_state_to_transform = _addon.map_state_to_transform
normalize_transform    = _addon.normalize_transform

_gb = _import_blocking("tools.gesture_bank", {"mediapipe": None, "cv2": None})
append_sample  = _gb.append_sample
build_parser   = _gb.build_parser
KEY_TO_GESTURE = _gb.KEY_TO_GESTURE
LABELS         = _gb.LABELS


# ── Helpers de estado ────────────────────────────────────────────────────────

def _make_state(dom_gesture="NO_HAND", dom_palm=None, mod_gesture="NO_HAND", mod_palm=None) -> dict:
    return {
        "dom": {"gesture": dom_gesture, "palm": list(dom_palm) if dom_palm else None,
                "present": dom_gesture != "NO_HAND"},
        "mod": {"gesture": mod_gesture, "palm": list(mod_palm) if mod_palm else None,
                "present": mod_gesture != "NO_HAND"},
        "intent": {"name": "NONE", "active": False},
        "transform": {k: (1.0 if k == "scale" else 0.0) for k in
                      ("x", "y", "z", "rot_x", "rot_y", "rot_z", "scale")},
        "mode": "NONE",
    }


def _zero():
    return {"x": 0.0, "y": 0.0, "z": 0.0, "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0, "scale": 1.0}


# ── Imports ──────────────────────────────────────────────────────────────────

def test_addon_imports_without_bpy():
    assert _addon.bpy is None
    assert callable(map_state_to_transform)
    assert callable(BlenderGestureMapper)


def test_gesture_bank_imports_without_mediapipe():
    assert callable(build_parser)
    assert callable(append_sample)


# ── Traslación por delta (necesita 2 frames) ─────────────────────────────────

def test_single_frame_does_not_translate():
    """Un solo frame de PINCH no mueve (no hay palma previa para el delta)."""
    m = BlenderGestureMapper(smoothing=1.0)
    out = m.update(_make_state("PINCH", (0.8, 0.2)), _zero())
    assert out["x"] == pytest.approx(0.0)
    assert out["y"] == pytest.approx(0.0)


def test_pinch_translates_on_second_frame_right():
    """Palma moviéndose a la derecha entre frames → x positivo."""
    m = BlenderGestureMapper(gain=4.0, smoothing=1.0)
    m.update(_make_state("PINCH", (0.5, 0.5)), _zero())          # establece prev
    out = m.update(_make_state("PINCH", (0.7, 0.5)), _zero())     # delta +0.2 en x
    assert out["x"] > 0.0


def test_pinch_translates_left_negative():
    m = BlenderGestureMapper(gain=4.0, smoothing=1.0)
    m.update(_make_state("PINCH", (0.5, 0.5)), _zero())
    out = m.update(_make_state("PINCH", (0.3, 0.5)), _zero())
    assert out["x"] < 0.0


def test_pinch_up_moves_y_positive():
    """Mano sube (y de pantalla baja) → y de mundo sube (positivo)."""
    m = BlenderGestureMapper(gain=4.0, smoothing=1.0)
    m.update(_make_state("PINCH", (0.5, 0.5)), _zero())
    out = m.update(_make_state("PINCH", (0.5, 0.3)), _zero())
    assert out["y"] > 0.0


# ── Safety / release ─────────────────────────────────────────────────────────

def test_fist_returns_current_unchanged():
    current = {"x": 3.5, "y": -1.2, "z": 0.8, "rot_x": 0.1, "rot_y": 0.2, "rot_z": 0.3, "scale": 2.0}
    m = BlenderGestureMapper()
    out = m.update(_make_state("FIST", (0.8, 0.2)), current.copy())
    for k, v in current.items():
        assert out[k] == pytest.approx(v)


def test_fist_resets_continuity():
    """Tras FIST, el siguiente PINCH no debe pegar un salto (continuidad rota)."""
    m = BlenderGestureMapper(gain=4.0, smoothing=1.0)
    m.update(_make_state("PINCH", (0.5, 0.5)), _zero())
    m.update(_make_state("FIST", (0.9, 0.9)), _zero())   # resetea prev
    out = m.update(_make_state("PINCH", (0.9, 0.9)), _zero())  # primer frame tras reset: sin delta
    assert out["x"] == pytest.approx(0.0)
    assert out["y"] == pytest.approx(0.0)


def test_open_palm_release_no_move():
    current = {"x": 1.0, "y": 2.0, "z": 3.0, "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.5, "scale": 1.5}
    m = BlenderGestureMapper()
    out = m.update(_make_state("OPEN_PALM", (0.7, 0.3)), current.copy())
    for k, v in current.items():
        assert out[k] == pytest.approx(v)


# ── Rotación por mod PINCH ───────────────────────────────────────────────────

def test_mod_pinch_rotates_z_on_second_frame():
    m = BlenderGestureMapper(smoothing=1.0)
    m.update(_make_state(mod_gesture="PINCH", mod_palm=(0.4, 0.5)), _zero())
    out = m.update(_make_state(mod_gesture="PINCH", mod_palm=(0.7, 0.5)), _zero())
    assert out["rot_z"] != 0.0


# ── Escala por distancia entre manos ─────────────────────────────────────────

def test_two_hand_spread_increases_scale():
    """dom PINCH + mod OPEN_PALM con manos separándose → scale crece."""
    m = BlenderGestureMapper(smoothing=1.0, scale_gain=2.0)
    # frame 1: distancia 0.2
    m.update(_make_state("PINCH", (0.4, 0.5), "OPEN_PALM", (0.6, 0.5)), _zero())
    # frame 2: distancia 0.6 (manos más separadas)
    out = m.update(_make_state("PINCH", (0.2, 0.5), "OPEN_PALM", (0.8, 0.5)), _zero())
    assert out["scale"] > 1.0


def test_two_hand_close_decreases_scale():
    m = BlenderGestureMapper(smoothing=1.0, scale_gain=2.0)
    m.update(_make_state("PINCH", (0.1, 0.5), "OPEN_PALM", (0.9, 0.5)), _zero())  # dist 0.8
    out = m.update(_make_state("PINCH", (0.45, 0.5), "OPEN_PALM", (0.55, 0.5)), _zero())  # dist 0.1
    assert out["scale"] < 1.0


def test_scale_never_negative():
    m = BlenderGestureMapper(smoothing=1.0, scale_gain=50.0)
    m.update(_make_state("PINCH", (0.05, 0.5), "OPEN_PALM", (0.95, 0.5)), _zero())
    out = m.update(_make_state("PINCH", (0.5, 0.5), "OPEN_PALM", (0.5, 0.5)), _zero())
    assert out["scale"] >= 0.0


# ── Smoothing ────────────────────────────────────────────────────────────────

def test_smoothing_zero_no_movement():
    m = BlenderGestureMapper(gain=10.0, smoothing=0.0)
    m.update(_make_state("PINCH", (0.5, 0.5)), _zero())
    out = m.update(_make_state("PINCH", (0.9, 0.1)), _zero())
    assert out["x"] == pytest.approx(0.0)
    assert out["y"] == pytest.approx(0.0)


# ── normalize_transform ──────────────────────────────────────────────────────

def test_normalize_transform_fills_defaults():
    out = normalize_transform({"x": 5.0})
    assert out["x"] == 5.0
    assert out["scale"] == 1.0
    assert set(out.keys()) == {"x", "y", "z", "rot_x", "rot_y", "rot_z", "scale"}


# ── gesture_bank ─────────────────────────────────────────────────────────────

def test_append_sample_writes_valid_jsonl(tmp_path):
    dataset = tmp_path / "test_bank.jsonl"
    append_sample(dataset, "PINCH", {"thumb_tip_wrist": 1.23}, handedness="Right")
    lines = dataset.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["label"] == "PINCH"
    assert record["features"]["thumb_tip_wrist"] == pytest.approx(1.23)
    assert record["meta"]["handedness"] == "Right"
    assert "timestamp" in record


def test_append_sample_multiple_writes(tmp_path):
    dataset = tmp_path / "bank.jsonl"
    for label in ["OPEN_PALM", "FIST", "PINCH"]:
        append_sample(dataset, label, {"f": 0.0})
    lines = dataset.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    labels = [json.loads(ln)["label"] for ln in lines]
    assert labels == ["OPEN_PALM", "FIST", "PINCH"]


def test_key_to_gesture_mapping():
    assert len(KEY_TO_GESTURE) == 7
    for i, label in enumerate(LABELS):
        assert KEY_TO_GESTURE[ord(str(i + 1))] == label


# ══════════════════════════════════════════════════════════════════════════════
#  Suavizado REAL (target acumulado + exponencial): no se pierde movimiento
# ══════════════════════════════════════════════════════════════════════════════

def _run_static(m, gesture, palm, current, n, **kw):
    """Aplica n updates con el mismo gesto y la palma quieta, encadenando current."""
    for _ in range(n):
        current = m.update(_make_state(gesture, palm, **kw), current)
    return current


def test_smoothing_one_is_immediate():
    """smoothing=1.0 == comportamiento validado: el delta completo en el mismo frame."""
    m = BlenderGestureMapper(gain=10.0, smoothing=1.0)
    m.update(_make_state("PINCH", (0.5, 0.5)), _zero())
    out = m.update(_make_state("PINCH", (0.7, 0.4)), _zero())
    assert out["x"] == pytest.approx(2.0)
    assert out["y"] == pytest.approx(1.0)


def test_smoothing_converges_to_full_displacement():
    """Antes: smoothing=0.35 tiraba el 65 % del movimiento. Ahora converge a gain*delta."""
    gain, delta = 10.0, 0.2
    m = BlenderGestureMapper(gain=gain, smoothing=0.35)
    cur = m.update(_make_state("PINCH", (0.5, 0.5)), _zero())
    cur = m.update(_make_state("PINCH", (0.5 + delta, 0.5)), cur)     # único delta
    assert 0.0 < cur["x"] < gain * delta                               # parcial en el primer frame
    assert cur["x"] == pytest.approx(gain * delta * 0.35)
    cur = _run_static(m, "PINCH", (0.5 + delta, 0.5), cur, 20)         # mano quieta, sigue PINCH
    assert cur["x"] == pytest.approx(gain * delta, abs=1e-3)
    assert cur["y"] == pytest.approx(0.0, abs=1e-9)


def test_smoothing_monotone_and_never_overshoots():
    m = BlenderGestureMapper(gain=10.0, smoothing=0.5)
    cur = m.update(_make_state("PINCH", (0.5, 0.5)), _zero())
    cur = m.update(_make_state("PINCH", (0.6, 0.5)), cur)
    prev = cur["x"]
    for _ in range(30):
        cur = m.update(_make_state("PINCH", (0.6, 0.5)), cur)
        assert prev <= cur["x"] <= 1.0 + 1e-12
        prev = cur["x"]


def test_smoothing_zero_target_still_tracked_but_object_static():
    m = BlenderGestureMapper(gain=10.0, smoothing=0.0)
    cur = m.update(_make_state("PINCH", (0.5, 0.5)), _zero())
    cur = m.update(_make_state("PINCH", (0.9, 0.1)), cur)
    assert cur == _zero()
    assert m.target["x"] == pytest.approx(4.0)   # acumulado, pero el objeto no se mueve


def test_target_resyncs_when_channel_released():
    """Sueltas (OPEN_PALM) a mitad del suavizado: el objeto se queda donde está."""
    m = BlenderGestureMapper(gain=10.0, smoothing=0.35)
    cur = m.update(_make_state("PINCH", (0.5, 0.5)), _zero())
    cur = m.update(_make_state("PINCH", (0.7, 0.5)), cur)
    partial = cur["x"]
    assert 0.0 < partial < 2.0
    cur = m.update(_make_state("OPEN_PALM", (0.7, 0.5)), cur)
    assert cur["x"] == pytest.approx(partial)
    assert m.target["x"] == pytest.approx(partial)          # target == current
    cur = _run_static(m, "NEUTRAL", None, cur, 5)
    assert cur["x"] == pytest.approx(partial)               # no sigue deslizándose


def test_target_resyncs_on_fist():
    m = BlenderGestureMapper(gain=10.0, smoothing=0.35)
    cur = m.update(_make_state("PINCH", (0.5, 0.5)), _zero())
    cur = m.update(_make_state("PINCH", (0.7, 0.5)), cur)
    partial = cur["x"]
    cur = m.update(_make_state("FIST", (0.7, 0.5)), cur)
    assert cur["x"] == pytest.approx(partial)
    assert m.target is None
    cur = _run_static(m, "NEUTRAL", None, cur, 5)
    assert cur["x"] == pytest.approx(partial)
    # y el siguiente PINCH sólo ceba
    cur = m.update(_make_state("PINCH", (0.1, 0.1)), cur)
    assert cur["x"] == pytest.approx(partial)


def test_target_resyncs_on_reset():
    m = BlenderGestureMapper(gain=10.0, smoothing=0.35)
    cur = m.update(_make_state("PINCH", (0.5, 0.5)), _zero())
    cur = m.update(_make_state("PINCH", (0.7, 0.5)), cur)
    m.reset()
    assert m.target is None
    out = m.update(_make_state("PINCH", (0.7, 0.5)), cur)
    assert out == pytest.approx(cur)


def test_target_resyncs_when_object_moved_manually():
    """El usuario mueve/deshace el objeto a mano: el resto del delta viejo se descarta."""
    m = BlenderGestureMapper(gain=10.0, smoothing=0.35)
    cur = m.update(_make_state("PINCH", (0.5, 0.5)), _zero())
    cur = m.update(_make_state("PINCH", (0.7, 0.5)), cur)
    manual = dict(cur); manual["x"] = -3.0                # Ctrl+Z / G en Blender
    out = m.update(_make_state("PINCH", (0.7, 0.5)), manual)
    assert out["x"] == pytest.approx(-3.0)
    out = _run_static(m, "PINCH", (0.7, 0.5), out, 10)
    assert out["x"] == pytest.approx(-3.0)
    # y el arrastre sigue funcionando desde la nueva posición
    out = m.update(_make_state("PINCH", (0.8, 0.5)), out)
    assert out["x"] > -3.0


def test_tiny_differences_do_not_resync():
    """Diferencias <= 1e-6 (ruido de float32 de Blender) no rompen el suavizado."""
    import struct
    f32 = lambda v: struct.unpack("f", struct.pack("f", v))[0]  # noqa: E731
    m = BlenderGestureMapper(gain=10.0, smoothing=0.35)
    far = {**_zero(), "x": 100.0, "y": 100.0}
    cur = m.update(_make_state("PINCH", (0.5, 0.5)), far)
    cur = m.update(_make_state("PINCH", (0.7, 0.5)), cur)
    for _ in range(25):
        # Blender devuelve lo escrito redondeado a float32 (error ~1e-5 en x=100)
        cur = {k: f32(v) for k, v in cur.items()}
        cur = m.update(_make_state("PINCH", (0.7, 0.5)), cur)
    assert cur["x"] == pytest.approx(102.0, abs=1e-3)   # todo el delta llegó igual


def test_configure_updates_gains_without_reset():
    m = BlenderGestureMapper(gain=1.0, smoothing=1.0)
    m.update(_make_state("PINCH", (0.5, 0.5)), _zero())
    m.configure(gain=20.0, smoothing=1.0, z_gain=None)
    out = m.update(_make_state("PINCH", (0.6, 0.5)), _zero())
    assert out["x"] == pytest.approx(2.0)      # continuidad intacta, nueva ganancia
    assert m.z_gain == 0.0


# ── Z con la mano modificadora ───────────────────────────────────────────────

def test_mod_pinch_vertical_lifts_z_when_z_gain_on():
    m = BlenderGestureMapper(smoothing=1.0, z_gain=5.0, rot_gain=2.0)
    m.update(_make_state(mod_gesture="PINCH", mod_palm=(0.5, 0.6)), _zero())
    out = m.update(_make_state(mod_gesture="PINCH", mod_palm=(0.5, 0.4)), _zero())
    assert out["z"] == pytest.approx(1.0)     # mano sube 0.2 * 5
    assert out["rot_z"] == pytest.approx(0.0)  # sin delta horizontal: no rota
    out2 = m.update(_make_state(mod_gesture="PINCH", mod_palm=(0.6, 0.4)), out)
    assert out2["rot_z"] == pytest.approx(0.1 * 2.0 * 3.141592653589793)
    assert out2["z"] == pytest.approx(1.0)


def test_z_gain_zero_keeps_validated_behavior():
    m = BlenderGestureMapper(smoothing=1.0)          # z_gain default 0.0
    m.update(_make_state(mod_gesture="PINCH", mod_palm=(0.5, 0.6)), _zero())
    out = m.update(_make_state(mod_gesture="PINCH", mod_palm=(0.5, 0.2)), _zero())
    assert out["z"] == 0.0


def test_defaults_recalibrated():
    assert _addon.DEFAULT_GAIN == 10.0
    assert _addon.DEFAULT_SMOOTHING == 0.5
    assert _addon.DEFAULT_Z_GAIN == 0.0
    assert _addon.DEFAULT_INTERVAL == pytest.approx(0.033)
    assert _addon.DEFAULT_STALE_AFTER == 1.0


# ── Escala no uniforme: helpers puros de la capa bpy ─────────────────────────

def test_scale_reference_uses_largest_axis():
    assert _addon.scale_reference((1.0, 2.0, 1.0)) == 2.0
    assert _addon.scale_reference((-3.0, 1.0, 1.0)) == 3.0     # espejo: valor absoluto
    assert _addon.scale_reference((0.0, 0.0, 0.0)) == 1.0      # degenerado: no divide por 0


def test_apply_scale_factor_preserves_proportions_and_sign():
    assert _addon.apply_scale_factor((1.0, 2.0, 1.0), 1.5) == pytest.approx((1.5, 3.0, 1.5))
    assert _addon.apply_scale_factor((-1.0, 1.0, 0.5), 2.0) == pytest.approx((-2.0, 2.0, 1.0))
    assert _addon.scale_factor(2.0, 3.0) == pytest.approx(1.5)
    assert _addon.scale_factor(0.0, 3.0) == 1.0


def test_non_uniform_object_keeps_shape_through_mapper():
    """Objeto (1, 2, 1): escalar con gestos debe dar (k, 2k, k), nunca uniforme."""
    m = BlenderGestureMapper(smoothing=1.0, scale_gain=2.0)
    obj_scale = (1.0, 2.0, 1.0)
    cur = {**_zero(), "scale": _addon.scale_reference(obj_scale)}
    cur = m.update(_make_state("PINCH", (0.4, 0.5), "OPEN_PALM", (0.6, 0.5)), cur)
    new = m.update(_make_state("PINCH", (0.2, 0.5), "OPEN_PALM", (0.8, 0.5)), cur)
    f = _addon.scale_factor(cur["scale"], new["scale"])
    assert f > 1.0
    sx, sy, sz = _addon.apply_scale_factor(obj_scale, f)
    assert sy == pytest.approx(2.0 * sx) and sz == pytest.approx(sx)
    # la referencia del objeto escalado coincide con lo que devolvió el mapper (sin resync espurio)
    assert _addon.scale_reference((sx, sy, sz)) == pytest.approx(new["scale"])


def test_changed_channels_only_reports_real_changes():
    a = _zero()
    assert _addon.changed_channels(a, dict(a)) == []
    b = {**a, "x": 1e-12}                          # ruido: no se escribe
    assert _addon.changed_channels(a, b) == []
    c = {**a, "y": 0.5, "rot_z": 0.1}
    assert _addon.changed_channels(a, c) == ["y", "rot_z"]
    assert _addon.changed_channels({}, {"scale": 2.0}) == ["scale"]
