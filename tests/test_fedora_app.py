from __future__ import annotations

"""Smoke tests de la app Fedora SIN mediapipe ni cámara.

Verifican que el módulo importa (mediapipe es lazy) y que las piezas puras
(parser, split_hands, FedoraActions en dry_run) funcionan.
"""

import types


def test_import_without_mediapipe():
    # El módulo debe importarse aunque mediapipe no esté instalado.
    from apps.fedora_control import main
    assert hasattr(main, "build_parser")
    assert hasattr(main, "run")
    assert main.MODES == ["NAVEGADOR", "CURSOR", "VENTANAS"]


def test_parser_defaults():
    from apps.fedora_control.main import build_parser
    ns = build_parser().parse_args([])
    assert ns.camera == 0
    assert ns.dry_run is False
    ns2 = build_parser().parse_args(["--dry-run", "--k", "9"])
    assert ns2.dry_run is True and ns2.k == 9


def test_actions_dry_run_does_not_call_system(capsys):
    from apps.fedora_control.actions import FedoraActions
    act = FedoraActions(dry_run=True)
    act.click_left()
    act.scroll_up()
    out = capsys.readouterr().out
    assert "[DRY]" in out
    assert "click_left" in out


def _fake_result(labels):
    """Construye un objeto tipo MediaPipe result con N manos y handedness dados."""
    def hand(label):
        cls = types.SimpleNamespace(label=label)
        return types.SimpleNamespace(classification=[cls])
    res = types.SimpleNamespace()
    res.multi_hand_landmarks = [object() for _ in labels]
    res.multi_handedness = [hand(lbl) for lbl in labels]
    return res


def test_split_hands_assigns_dominant():
    from apps.fedora_control.main import split_hands
    res = _fake_result(["Left", "Right"])
    dom, mod = split_hands(res, "Right")
    # la mano "Right" es la segunda -> debe ir a dom
    assert dom is res.multi_hand_landmarks[1]
    assert mod is res.multi_hand_landmarks[0]


def test_split_hands_single_hand_goes_dom():
    from apps.fedora_control.main import split_hands
    res = _fake_result(["Right"])
    dom, mod = split_hands(res, "Right")
    assert dom is res.multi_hand_landmarks[0]
    assert mod is None


def test_split_hands_no_hands():
    from apps.fedora_control.main import split_hands
    res = types.SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
    dom, mod = split_hands(res, "Right")
    assert dom is None and mod is None


def test_split_hands_two_non_dominant_are_distinct():
    """Regresión: si ninguna mano es la dominante, dom y mod NO deben ser el mismo objeto."""
    from apps.fedora_control.main import split_hands
    res = _fake_result(["Left", "Left"])  # dominante por defecto = Right; ninguna coincide
    dom, mod = split_hands(res, "Right")
    assert dom is not None and mod is not None
    assert dom is not mod
