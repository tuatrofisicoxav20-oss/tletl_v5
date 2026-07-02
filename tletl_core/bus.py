from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

from .state import TletlFrameState


def _safe_data(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"No puedo serializar objeto tipo {type(obj)!r}")


class TletlStateBus:
    """
    Bus común para exportar estado de Tletl.

    Fedora puede usarlo para debug.
    Blender puede leerlo para manipular objetos.
    """

    def __init__(self, path: str | Path = "tletl_state.json"):
        self.path = Path(path)

    def write(self, state: TletlFrameState | Dict[str, Any]) -> None:
        data = _safe_data(state)
        data.setdefault("timestamp", time.time())

        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def read(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))
