"""
tletl_core/lowlight.py

LowLightEnhancer: CLAHE + gamma adaptativo por luminancia media.
Solo usa cv2 para procesamiento de imagen puro y numpy — sin ventanas,
sin ydotool, sin bpy.

Umbrales de luminancia (canal Y en YCrCb, rango 0-255):
  < 62  → modo "dark"   (corrección fuerte,  gamma inv=1/0.62 ≈ 1.61)
  < 88  → modo "medium" (corrección media,   gamma inv=1/0.76 ≈ 1.32)
  < 112 → modo "low"    (corrección suave,   gamma inv=1/0.88 ≈ 1.14)
  >= 112 → modo "normal" (sin corrección)
"""

import cv2
import numpy as np


class LowLightEnhancer:
    """Preprocesado suave para baja luz.

    Mejora contraste/brillo antes de pasar el frame al clasificador.
    No requiere MediaPipe ni ningún componente de sistema.
    """

    # Umbrales de luminancia (canal Y YCrCb, 0-255)
    _LUMA_NORMAL: float = 112.0   # >= este valor → sin corrección
    _LUMA_MEDIUM: float = 88.0    # >= este y < NORMAL → corrección suave
    _LUMA_LOW: float = 62.0       # >= este y < MEDIUM → corrección media
    # < LUMA_LOW → corrección fuerte ("dark")

    def __init__(self, clip_limit: float = 2.0, gamma_dark: float = 1.6):
        """
        Args:
            clip_limit:  límite de recorte para CLAHE (default 2.0).
            gamma_dark:  gamma para el modo más oscuro (<62 luma).
                         Se almacena pero los gammas intermedios se derivan
                         de la escala original del código fuente.
        """
        self.clip_limit = float(clip_limit)
        self.gamma_dark = float(gamma_dark)
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(8, 8),
        )
        self.last_luma: float = 0.0
        self.last_mode: str = "normal"

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _build_lut(self, gamma: float) -> np.ndarray:
        """Construye tabla LUT para corrección de gamma."""
        inv_gamma = 1.0 / gamma
        return np.array(
            [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
            dtype=np.uint8,
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Aplica realce de baja luz si la luminancia media lo requiere.

        Args:
            frame: imagen BGR uint8 (H, W, 3).

        Returns:
            Imagen BGR uint8 con el mismo shape que la entrada.
            Actualiza self.last_luma y self.last_mode.
        """
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        mean_luma = float(np.mean(y))
        self.last_luma = mean_luma

        # Sin corrección: luz suficiente
        if mean_luma >= self._LUMA_NORMAL:
            self.last_mode = "normal"
            return frame

        # Aplicar CLAHE al canal de luminancia
        y2 = self.clahe.apply(y)

        # Seleccionar gamma según nivel de oscuridad
        if mean_luma < self._LUMA_LOW:
            # Corrección fuerte
            gamma = 0.62
            self.last_mode = "dark"
        elif mean_luma < self._LUMA_MEDIUM:
            # Corrección media
            gamma = 0.76
            self.last_mode = "medium"
        else:
            # Corrección suave (>= 88 y < 112)
            gamma = 0.88
            self.last_mode = "low"

        lut = self._build_lut(gamma)
        y3 = cv2.LUT(y2, lut)
        merged = cv2.merge([y3, cr, cb])
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
