#!/usr/bin/env bash
# Dry-run: muestra gestos y acciones SIN ejecutarlas (no toca Fedora).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate || true
export TLETL_DRY_RUN=1
exec python -m apps.fedora_control.main --dry-run "$@"
