#!/usr/bin/env bash
# Abre la herramienta de captura de gestos (escribe al banco JSONL).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate || true
exec python -m tools.gesture_bank "$@"
