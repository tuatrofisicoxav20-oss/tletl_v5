#!/usr/bin/env bash
# Descarga los modelos .task de MediaPipe (hand_landmarker + gesture_recognizer)
# a ~/.tletl/models (o $TLETL_HOME/models). Necesario para el backend "tasks"
# (mediapipe >= 0.10.31 / 1.x, sin API legacy). Acepta --dir, --force, --check.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate || true
exec python -m tools.fetch_models "$@"
