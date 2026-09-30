#!/usr/bin/env bash
# Corre la app de control de Fedora (acciones reales vía ydotool).
# Requisitos: ydotool instalado, daemon activo (systemctl --user enable --now ydotool)
# y tu usuario en el grupo input. Si falta el modelo del detector:
# ./launchers/tletl-fetch-models.sh. Argumentos extra van a la app (--camera 1, --backend tasks...).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d .venv ] && source .venv/bin/activate || true
exec python -m apps.fedora_control.main "$@"
