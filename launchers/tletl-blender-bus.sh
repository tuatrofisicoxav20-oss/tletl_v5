#!/usr/bin/env bash
# Alimenta el bus (tletl_state.json) para Blender SIN ejecutar acciones de Fedora.
# Reutiliza la app Fedora en dry-run: clasifica, aplica seguridad y escribe el bus,
# pero no envía nada a ydotool. El addon de Blender lee ese bus.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate || true
export TLETL_DRY_RUN=1
exec python -m apps.fedora_control.main --dry-run "$@"
