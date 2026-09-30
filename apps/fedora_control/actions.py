"""apps/fedora_control/actions.py — adaptador Fedora/navegador sobre ydotool.

Aquí sí existen ydotool y acciones del sistema. El núcleo común no debe importar
este archivo. Que el cerebro no cargue la mochila.

Requisitos en Fedora (Wayland o X11) para que las acciones lleguen al sistema:
  1. Instalar:                 sudo dnf install ydotool
  2. Daemon corriendo:         systemctl --user enable --now ydotool
                               (o el servicio de sistema: sudo systemctl enable --now ydotool)
  3. Permiso sobre /dev/uinput: sudo usermod -aG input $USER  y volver a iniciar
     sesión (el paquete de Fedora trae la regla udev para el grupo `input`).
  Comprobación rápida: `ydotool mousemove -x 5 -y 0` debe mover el cursor.

Comportamiento ante fallos:
  - Si `ydotool` no está en PATH (y no es dry-run) se avisa UNA vez al arrancar,
    `available` queda en False y todas las acciones son no-op silenciosas: nada
    de un error impreso por cada frame de `mousemove`.
  - Si ydotool existe pero falla (ydotoold apagado, sin permisos) se imprime UN
    aviso con la causa probable y se contabiliza en `failures`.

Códigos de tecla: son los KEY_* de Linux (input-event-codes.h): 29=LeftCtrl,
42=LeftShift, 56=LeftAlt, 15=Tab, 28=Enter, 103/108=Up/Down, 104/109=PgUp/PgDn,
105/106=Left/Right, 125=Super.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import List, Optional

YDOTOOL_SETUP_HINT = ("instala ydotool, arranca el daemon (systemctl --user enable --now ydotool) "
                      "y añade tu usuario al grupo input (sudo usermod -aG input $USER)")


def key_combo_sequence(*codes: int) -> str:
    """Secuencia ydotool para una combinación: pulsa en orden y suelta en orden inverso.

    key_combo_sequence(29, 15) -> "29:1 15:1 15:0 29:0"  (Ctrl+Tab)
    """
    if not codes:
        return ""
    down = " ".join(f"{int(c)}:1" for c in codes)
    up = " ".join(f"{int(c)}:0" for c in reversed(codes))
    return f"{down} {up}"


class FedoraActions:
    """Adaptador Fedora/navegador.

    dry_run=True imprime `[DRY] label: cmd` y no ejecuta nada (lo usan los
    launchers *-safe y *-blender-bus, y los tests).
    """

    def __init__(self, dry_run: bool = False, binary: str = "ydotool", command_timeout: float = 2.0):
        self.dry_run = bool(dry_run)
        self.binary = binary
        self.command_timeout = float(command_timeout)
        self.available = True
        self.failures = 0
        self._warned_failure = False
        if not self.dry_run:
            self.available = shutil.which(binary) is not None
            if not self.available:
                print(f"[TLETL] AVISO: '{binary}' no está en PATH: las acciones de Fedora serán no-op. "
                      f"Para control real, {YDOTOOL_SETUP_HINT}.")

    # ------------------------------------------------------------------
    def run(self, cmd: List[str], label: str) -> None:
        if self.dry_run:
            print(f"[DRY] {label}: {' '.join(cmd)}")
            return
        if not self.available:
            return
        try:
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  check=False, timeout=self.command_timeout)
        except Exception as exc:  # FileNotFoundError, TimeoutExpired, PermissionError...
            self._note_failure(f"{label}: {exc}")
            return
        if proc.returncode != 0:
            self._note_failure(f"{label}: {self.binary} devolvió {proc.returncode} "
                               f"(¿ydotoold corriendo? ¿usuario en grupo input?)")

    def _note_failure(self, message: str) -> None:
        self.failures += 1
        if not self._warned_failure:
            self._warned_failure = True
            print(f"[ydotool error] {message} — {YDOTOOL_SETUP_HINT}. Se silencian los siguientes errores.")

    # ------------------------------------------------------------------
    def key(self, seq: str, label: str) -> None:
        self.run([self.binary, "key", *seq.split()], label)

    def key_combo(self, *codes: int, label: Optional[str] = None) -> None:
        """Pulsa una combinación (p.ej. key_combo(29, 15) = Ctrl+Tab)."""
        self.key(key_combo_sequence(*codes), label or "combo+" + "+".join(str(c) for c in codes))

    def click_left(self) -> None:
        self.run([self.binary, "click", "0xC0"], "click_left")

    def click_right(self) -> None:
        self.run([self.binary, "click", "0xC1"], "click_right")

    def mouse_down(self) -> None:
        self.run([self.binary, "click", "0x40"], "mouse_down")

    def mouse_up(self) -> None:
        self.run([self.binary, "click", "0x80"], "mouse_up")

    def mousemove(self, dx: int, dy: int) -> None:
        if dx or dy:
            self.run([self.binary, "mousemove", "-x", str(dx), "-y", str(dy)], f"mousemove {dx},{dy}")

    def scroll_down(self) -> None:
        self.key("108:1 108:0", "scroll_down")

    def scroll_up(self) -> None:
        self.key("103:1 103:0", "scroll_up")

    def page_down(self) -> None:
        self.key("109:1 109:0", "page_down")

    def page_up(self) -> None:
        self.key("104:1 104:0", "page_up")

    def next_tab(self) -> None:
        self.key_combo(29, 15, label="next_tab")            # Ctrl+Tab

    def prev_tab(self) -> None:
        self.key_combo(29, 42, 15, label="prev_tab")        # Ctrl+Shift+Tab

    def back(self) -> None:
        self.key_combo(56, 105, label="browser_back")       # Alt+Left

    def forward(self) -> None:
        self.key_combo(56, 106, label="browser_forward")    # Alt+Right

    def overview(self) -> None:
        self.key("125:1 125:0", "overview")                  # Super

    def enter(self) -> None:
        self.key("28:1 28:0", "enter")
