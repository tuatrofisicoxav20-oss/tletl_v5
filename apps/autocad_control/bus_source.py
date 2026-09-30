"""apps/autocad_control/bus_source.py — de dónde salen los frames del bus.

Dos fuentes reales con la misma interfaz (`poll() -> dict | None`, `age()`,
`close()`) y una guionada para tests:

  FileBusSource  lee el JSON que escribe TletlStateBus (misma máquina: Fedora).
                 `poll()` solo devuelve el frame cuando el archivo CAMBIÓ; si no,
                 None (así un bus congelado no se re-alimenta 20 veces por segundo).
  UdpBusSource   recibe los datagramas de UdpStatePublisher (otra máquina:
                 AutoCAD en Windows). Envuelve tletl_core.bus.UdpStateReceiver.
  ScriptedBusSource  entrega una lista fija de frames y luego marca `finished`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tletl_core.bus import UdpStateReceiver, parse_udp_target
from tletl_core.paths import default_bus_path

DEFAULT_UDP_HOST = "0.0.0.0"
DEFAULT_UDP_PORT = 5055


def _timestamp_of(state: Dict[str, Any]) -> Optional[float]:
    ts = state.get("timestamp")
    if isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts > 0:
        return float(ts)
    return None


class FileBusSource:
    """Lee el bus de archivo. `read()` devuelve el dict o {} (nunca lanza);
    `poll()` devuelve el frame solo si cambió desde la última llamada."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser() if path else default_bus_path()
        self._last_text: Optional[str] = None
        self.last_state: Dict[str, Any] = {}
        self.reads = 0
        self.finished = False

    def read(self) -> Dict[str, Any]:
        text = self._read_text()
        return self._parse(text) if text is not None else {}

    def poll(self) -> Optional[Dict[str, Any]]:
        text = self._read_text()
        if text is None or text == self._last_text:
            return None
        self._last_text = text
        state = self._parse(text)
        if not state:
            return None
        self.last_state = state
        self.reads += 1
        return state

    def age(self, now: Optional[float] = None) -> float:
        """Segundos desde el último frame válido leído por poll() (inf si ninguno)."""
        ts = _timestamp_of(self.last_state)
        if ts is None:
            return float("inf")
        return max(0.0, (now if now is not None else time.time()) - ts)

    def close(self) -> None:
        return None

    def describe(self) -> str:
        return f"archivo {self.path}"

    # -- internos --
    def _read_text(self) -> Optional[str]:
        try:
            return self.path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return None

    @staticmethod
    def _parse(text: str) -> Dict[str, Any]:
        if not text.strip():
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


class UdpBusSource:
    """Recibe el bus por UDP (lo que publica `[bus] udp_target` / TLETL_BUS_UDP)."""

    def __init__(self, host: str = DEFAULT_UDP_HOST, port: int = DEFAULT_UDP_PORT):
        self.receiver = UdpStateReceiver(host, int(port))
        self.address: Tuple[str, int] = self.receiver.address
        self.last_state: Dict[str, Any] = {}
        self.finished = False

    @classmethod
    def from_target(cls, target: str | None) -> "UdpBusSource":
        """'host:puerto' (o '' => 0.0.0.0:5055)."""
        parsed = parse_udp_target(target) if target else None
        if parsed is None:
            return cls()
        return cls(parsed[0], parsed[1])

    def poll(self) -> Optional[Dict[str, Any]]:
        state = self.receiver.poll()
        if state is not None:
            self.last_state = state
        return state

    def age(self, now: Optional[float] = None) -> float:
        ts = _timestamp_of(self.last_state)
        if ts is None:
            return float("inf")
        return max(0.0, (now if now is not None else time.time()) - ts)

    @property
    def received(self) -> int:
        return self.receiver.received

    def close(self) -> None:
        self.receiver.close()

    def describe(self) -> str:
        return f"udp {self.address[0]}:{self.address[1]}"


class ScriptedBusSource:
    """Entrega `frames` uno por poll(); `None` en la lista significa 'sin frame
    nuevo'. Al agotarse marca `finished=True` (los loops terminan solos)."""

    def __init__(self, frames: Iterable[Optional[Dict[str, Any]]]):
        self._frames: List[Optional[Dict[str, Any]]] = list(frames)
        self._index = 0
        self.last_state: Dict[str, Any] = {}
        self.finished = not self._frames

    def poll(self) -> Optional[Dict[str, Any]]:
        if self._index >= len(self._frames):
            self.finished = True
            return None
        frame = self._frames[self._index]
        self._index += 1
        if self._index >= len(self._frames):
            self.finished = True
        if frame is not None:
            self.last_state = frame
        return frame

    def age(self, now: Optional[float] = None) -> float:
        ts = _timestamp_of(self.last_state)
        if ts is None:
            return float("inf")
        return max(0.0, (now if now is not None else time.time()) - ts)

    def close(self) -> None:
        return None

    def describe(self) -> str:
        return f"guion ({len(self._frames)} frames)"


def open_source(*, bus: str | Path | None = None, udp: str | None = None):
    """Fábrica para los CLIs: `--udp HOST:PORT` gana sobre `--bus PATH`; sin nada,
    el archivo por default (~/.tletl/tletl_state.json)."""
    if udp is not None:
        return UdpBusSource.from_target(udp)
    return FileBusSource(bus)
