#!/usr/bin/env bash
# Empaqueta el addon de Blender como dist/tletl_blender_addon-<version>.zip
# (instalable como addon legacy y como extensión de Blender 4.2+).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate || true
exec python -m tools.build_blender_addon "$@"
