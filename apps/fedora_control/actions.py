from __future__ import annotations

import subprocess
from typing import List


class FedoraActions:
    """Adaptador Fedora/navegador.

    Aquí sí existen ydotool y acciones del sistema.
    El núcleo común no debe importar este archivo. Que el cerebro no cargue la mochila.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def run(self, cmd: List[str], label: str) -> None:
        if self.dry_run:
            print(f"[DRY] {label}: {' '.join(cmd)}")
            return
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception as exc:
            print(f"[ydotool error] {exc}")

    def key(self, seq: str, label: str) -> None:
        self.run(["ydotool", "key", *seq.split()], label)

    def click_left(self) -> None:
        self.run(["ydotool", "click", "0xC0"], "click_left")

    def click_right(self) -> None:
        self.run(["ydotool", "click", "0xC1"], "click_right")

    def mouse_down(self) -> None:
        self.run(["ydotool", "click", "0x40"], "mouse_down")

    def mouse_up(self) -> None:
        self.run(["ydotool", "click", "0x80"], "mouse_up")

    def mousemove(self, dx: int, dy: int) -> None:
        if dx or dy:
            self.run(["ydotool", "mousemove", "-x", str(dx), "-y", str(dy)], f"mousemove {dx},{dy}")

    def scroll_down(self) -> None:
        self.key("108:1 108:0", "scroll_down")

    def scroll_up(self) -> None:
        self.key("103:1 103:0", "scroll_up")

    def page_down(self) -> None:
        self.key("109:1 109:0", "page_down")

    def page_up(self) -> None:
        self.key("104:1 104:0", "page_up")

    def next_tab(self) -> None:
        self.key("29:1 15:1 15:0 29:0", "next_tab")

    def prev_tab(self) -> None:
        self.key("29:1 42:1 15:1 15:0 42:0 29:0", "prev_tab")

    def back(self) -> None:
        self.key("56:1 105:1 105:0 56:0", "browser_back")

    def forward(self) -> None:
        self.key("56:1 106:1 106:0 56:0", "browser_forward")

    def overview(self) -> None:
        self.key("125:1 125:0", "overview")

    def enter(self) -> None:
        self.key("28:1 28:0", "enter")
