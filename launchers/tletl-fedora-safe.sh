#!/usr/bin/env bash
# Dry-run: muestra gestos y acciones SIN ejecutarlas (no toca Fedora).
# Es la forma de validar gestos, el backend del detector (panel: "backend:...")
# y la segunda opinión (--gesture-hint) antes de dar control real.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate || true
export TLETL_DRY_RUN=1
exec python -m apps.fedora_control.main --dry-run "$@"
