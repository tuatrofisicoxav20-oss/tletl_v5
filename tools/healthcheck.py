from __future__ import annotations

"""Healthcheck de Tletl v5.

Importa todo el core, carga el banco, reporta nº de muestras y features útiles,
y valida que gesture_to_common_intent corre sobre un state dummy.
Sale con código 0 si todo está bien, 1 si algo falla.

Uso:  python -m tools.healthcheck
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANK = ROOT / "datasets" / "tletl_gesture_bank_v2_features.jsonl"


def main() -> int:
    print("== Tletl v5 healthcheck ==")
    try:
        from tletl_core import __version__
        from tletl_core.classifier import RobustKNNRuntime
        from tletl_core.config import load_config
        from tletl_core.intent import gesture_to_common_intent
        from tletl_core.state import TletlFrameState, TletlHandState
        from tletl_core import geometry, features, orientation, temporal, bus  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] import del core: {exc!r}")
        return 1
    print(f"[ok] core importado (v{__version__})")

    cfg = load_config()
    print(f"[ok] config cargada: k={cfg['classifier']['k']}, "
          f"adaptive={cfg['adaptive']['enabled']}, lowlight={cfg['lowlight']['enabled']}")

    if not BANK.exists():
        print(f"[FAIL] banco no encontrado: {BANK}")
        return 1
    try:
        runtime = RobustKNNRuntime(BANK, k=cfg["classifier"]["k"],
                                   orientation_weight=cfg["classifier"]["orientation_weight"])
    except Exception as exc:
        print(f"[FAIL] no se pudo cargar el banco: {exc!r}")
        return 1
    print(f"[ok] banco: {len(runtime.samples)} muestras (balanceadas), "
          f"{len(runtime.keys)} features útiles")
    print(f"     labels: {dict(runtime.counts)}")

    # state dummy -> intent
    st = TletlFrameState()
    st.dom = TletlHandState(present=True, gesture="PINCH", critic_ok=True, confidence=0.9)
    intent = gesture_to_common_intent(st)
    if intent.name != "GRAB_OR_SELECT":
        print(f"[FAIL] intent inesperada para PINCH: {intent.name}")
        return 1
    print(f"[ok] intent dummy: PINCH -> {intent.name}")

    print("== healthcheck OK ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
