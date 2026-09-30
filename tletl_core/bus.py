"""tletl_core/bus.py — el ÚNICO canal entre la app de cámara y sus clientes.

Escribe `TletlFrameState` como JSON de forma atómica (tmp + rename) para que
Blender/CAD nunca lean un archivo a medias, y opcionalmente lo publica por UDP
para clientes en otra máquina (p. ej. AutoCAD en Windows).

Decisiones:
  - JSON compacto por default: se escribe ~30 veces por segundo.
  - Las `features` de cada mano (≈60 floats) NO viajan salvo `include_features`:
    ningún cliente las usa y triplican el tamaño del frame.
  - `read()` nunca lanza: archivo ausente, vacío o corrupto -> {}.
  - `timestamp` siempre es real (antes, un state con timestamp=0.0 se escribía
    tal cual y los clientes no podían detectar un bus obsoleto).

REGLA DURA: aquí no hay cv2-window, ydotool ni bpy. Sockets y archivos sí.
"""

from __future__ import annotations

import copy
import json
import os
import socket
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .paths import default_bus_path, ensure_parent
from .state import TletlFrameState

MAX_UDP_PAYLOAD = 65_000  # bytes; por encima el datagrama no cabe y se descarta


def _safe_data(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return copy.deepcopy(obj)
    raise TypeError(f"No puedo serializar objeto tipo {type(obj)!r}")


def parse_udp_target(target: str | None) -> Optional[Tuple[str, int]]:
    """'host:puerto' -> (host, puerto). Vacío/None -> None. Puerto inválido -> ValueError."""
    if not target:
        return None
    text = str(target).strip()
    if not text:
        return None
    host, sep, port = text.rpartition(":")
    if not sep or not host:
        raise ValueError(f"udp_target debe ser host:puerto, recibí {target!r}")
    return host, int(port)


class UdpStatePublisher:
    """Publica cada frame del bus como un datagrama UDP (JSON en UTF-8)."""

    def __init__(self, target: str | Tuple[str, int]):
        parsed = parse_udp_target(target) if isinstance(target, str) else target
        if parsed is None:
            raise ValueError("UdpStatePublisher necesita un destino host:puerto")
        self.target: Tuple[str, int] = (parsed[0], int(parsed[1]))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except OSError:  # pragma: no cover - plataformas sin SO_BROADCAST
            pass
        self.sent = 0
        self.dropped = 0
        self.last_error: Optional[str] = None

    def send(self, payload: str | bytes) -> bool:
        raw = payload.encode("utf-8") if isinstance(payload, str) else payload
        if len(raw) > MAX_UDP_PAYLOAD:
            self.dropped += 1
            self.last_error = f"payload de {len(raw)} bytes excede {MAX_UDP_PAYLOAD}"
            return False
        try:
            self.sock.sendto(raw, self.target)
        except OSError as exc:
            self.dropped += 1
            self.last_error = str(exc)
            return False
        self.sent += 1
        return True

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:  # pragma: no cover
            pass


class UdpStateReceiver:
    """Recibe frames publicados por UdpStatePublisher. `poll()` no bloquea."""

    def __init__(self, host: str = "0.0.0.0", port: int = 5055):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, int(port)))
        self.sock.setblocking(False)
        self.address: Tuple[str, int] = self.sock.getsockname()[:2]
        self.last_state: Dict[str, Any] = {}
        self.received = 0
        self.malformed = 0

    def poll(self) -> Optional[Dict[str, Any]]:
        """Devuelve el frame MÁS RECIENTE disponible (drena la cola) o None."""
        latest: Optional[Dict[str, Any]] = None
        while True:
            try:
                raw, _ = self.sock.recvfrom(MAX_UDP_PAYLOAD + 1024)
            except BlockingIOError:
                break
            except OSError:
                break
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.malformed += 1
                continue
            if isinstance(data, dict):
                latest = data
                self.received += 1
        if latest is not None:
            self.last_state = latest
        return latest

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:  # pragma: no cover
            pass


class TletlStateBus:
    """
    Bus común para exportar estado de Tletl.

    Fedora puede usarlo para debug. Blender/CAD lo leen para manipular objetos.
    """

    def __init__(self, path: str | Path | None = None, *, compact: bool = True,
                 include_features: bool = False, udp_target: str | None = None):
        self.path = Path(path).expanduser() if path else default_bus_path()
        ensure_parent(self.path)
        self.compact = bool(compact)
        self.include_features = bool(include_features)
        self.udp: Optional[UdpStatePublisher] = None
        if udp_target:
            self.udp = UdpStatePublisher(udp_target)
        self.writes = 0
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    def serialize(self, state: TletlFrameState | Dict[str, Any]) -> str:
        data = _safe_data(state)
        if not data.get("timestamp"):
            data["timestamp"] = time.time()
        if not self.include_features:
            for side in ("dom", "mod"):
                hand = data.get(side)
                if isinstance(hand, dict) and hand.get("features"):
                    hand["features"] = {}
        if self.compact:
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        return json.dumps(data, indent=2, ensure_ascii=False)

    def write(self, state: TletlFrameState | Dict[str, Any]) -> None:
        payload = self.serialize(state)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self.path)
            self.writes += 1
            self.last_error = None
        except OSError as exc:
            self.last_error = str(exc)
            raise
        if self.udp is not None:
            self.udp.send(payload)

    def read(self) -> Dict[str, Any]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return {}
        if not text.strip():
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def age(self, now: float | None = None) -> float:
        """Segundos desde el último frame escrito (inf si no hay bus legible)."""
        data = self.read()
        ts = data.get("timestamp")
        if not isinstance(ts, (int, float)) or ts <= 0:
            return float("inf")
        return max(0.0, (now if now is not None else time.time()) - float(ts))

    def close(self) -> None:
        if self.udp is not None:
            self.udp.close()
            self.udp = None
