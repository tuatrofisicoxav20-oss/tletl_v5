#!/usr/bin/env bash
# Corre la app de control de Fedora (acciones reales vía ydotool).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate || true
exec python -m apps.fedora_control.main "$@"
