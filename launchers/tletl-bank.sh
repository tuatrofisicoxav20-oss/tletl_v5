#!/usr/bin/env bash
# Abre la herramienta de captura de gestos (escribe al banco JSONL).
# Usa el mismo detector que la app (--backend auto|legacy|tasks; default [tracker].backend).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate || true
exec python -m tools.gesture_bank "$@"
