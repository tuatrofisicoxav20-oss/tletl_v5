#!/usr/bin/env bash
# Sesión de modelado CAD por gestos: lee el bus (archivo o UDP), crea/mueve sólidos
# y exporta un DXF al terminar (Ctrl+C o FIST 3 s). Ver apps/autocad_control/README.md.
#   ./launchers/tletl-cad-session.sh --export modelo.dxf [--units mm --gain 100]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate || true
exec python -m apps.autocad_control.session "$@"
