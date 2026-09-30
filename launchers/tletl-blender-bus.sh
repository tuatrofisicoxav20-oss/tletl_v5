#!/usr/bin/env bash
# Alimenta el bus (~/.tletl/tletl_state.json) para Blender SIN ejecutar acciones de Fedora.
# Reutiliza la app Fedora en dry-run: clasifica, aplica seguridad y escribe el bus,
# pero no envía nada a ydotool. El addon de Blender lee ese bus.
#
# Por default corre SIN ventana (--headless): en una laptop sin segunda pantalla
# la ventana de OpenCV solo estorba sobre Blender. Pasa --window para verla
# (útil para calibrar). Cualquier otro argumento va directo a la app
# (p.ej. --camera 1, --backend tasks, --gesture-hint).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate || true
export TLETL_DRY_RUN=1
HEADLESS="--headless"
for arg in "$@"; do
  [ "$arg" = "--window" ] && HEADLESS=""
done
exec python -m apps.fedora_control.main --dry-run $HEADLESS "$@"
