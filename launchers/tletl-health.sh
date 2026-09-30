#!/usr/bin/env bash
# Healthcheck: importa el core, carga el banco, valida intent. Sale 0 si todo bien.
# Para comprobar además los modelos del detector: ./launchers/tletl-fetch-models.sh --check
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate || true
exec python -m tools.healthcheck "$@"
