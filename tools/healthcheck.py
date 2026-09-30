"""Healthcheck de Tletl v5.

Importa todo el core, resuelve las rutas de runtime (banco, bus, memoria
adaptativa) con las mismas reglas que las apps, carga el banco, reporta nº de
muestras y features útiles, y valida que gesture_to_common_intent corre sobre
un state dummy. Sale con código 0 si todo está bien, 1 si algo falla.

Uso:  python -m tools.healthcheck
"""

from __future__ import annotations

import sys


def main() -> int:
    print("== Tletl v5 healthcheck ==")
    try:
        from tletl_core import __version__
        from tletl_core.classifier import RobustKNNRuntime
        from tletl_core.config import (
            adaptive_path_from_config,
            bank_path_from_config,
            bus_path_from_config,
            critic_min_conf,
            load_config,
        )
        from tletl_core.intent import gesture_to_common_intent
        from tletl_core.state import TletlFrameState, TletlHandState
        from tletl_core import geometry, features, orientation, temporal, bus  # noqa: F401
    except Exception as exc:  # pragma: no cover
        print(f"[FAIL] import del core: {exc!r}")
        return 1
    print(f"[ok] core importado (v{__version__})")

    cfg = load_config()
    cl = cfg["classifier"]
    print(f"[ok] config cargada: k={cl['k']}, min_conf={cl['min_confidence']}, "
          f"min_margin={cl['min_margin']}, adaptive={cfg['adaptive']['enabled']}, "
          f"lowlight={cfg['lowlight']['enabled']}")
    print(f"     critic min_conf: {critic_min_conf(cfg)}")

    # Rutas resueltas con la MISMA regla que usan las apps (config + env + ~/.tletl).
    bank = bank_path_from_config(cfg)
    bus_path = bus_path_from_config(cfg)
    adaptive_path = adaptive_path_from_config(cfg)
    print(f"     banco:    {bank}")
    print(f"     bus:      {bus_path}")
    print(f"     adaptive: {adaptive_path}")

    # Detector de manos: qué backend se usaría y si el modelo está (sin cargarlo).
    try:
        from apps.common.hand_tracker import describe_backend, format_backend_summary
        info = describe_backend(cfg["tracker"])
        print(f"     tracker:  {format_backend_summary(info)}")
    except Exception as exc:  # noqa: BLE001 - el healthcheck del core no depende de mediapipe
        print(f"     tracker:  no evaluado ({type(exc).__name__}: {exc})")

    if not bank.exists():
        print(f"[FAIL] banco no encontrado: {bank}")
        return 1
    try:
        runtime = RobustKNNRuntime(
            bank,
            k=int(cl["k"]),
            orientation_weight=float(cl["orientation_weight"]),
            min_confidence=float(cl["min_confidence"]),
            min_margin=float(cl["min_margin"]),
            verbose=False,
        )
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
