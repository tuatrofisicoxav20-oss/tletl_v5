"""
tletl_core/lowlight.py

LowLightEnhancer: CLAHE + gamma adaptativo por luminancia media.
Solo usa cv2 para procesamiento de imagen puro y numpy — sin ventanas,
sin ydotool, sin bpy.

Umbrales de luminancia (canal Y en YCrCb, rango 0-255) y exponente gamma que
se aplica al canal Y tras CLAHE, `y_out = 255·(y/255)^exponente` (exponente
< 1 ACLARA, > 1 oscurece). Los tres exponentes se derivan de `gamma_dark`
(default 1.6):

  < 62   → "dark"   : exponente = 1 / gamma_dark                    (1.6 → 0.625)
  < 88   → "medium" : exponente = 1 / (1 + (gamma_dark - 1)·0.53)   (1.6 → 0.759)
  < 112  → "low"    : exponente = 1 / (1 + (gamma_dark - 1)·0.227)  (1.6 → 0.880)
  >= 112 → "normal" : sin corrección

Con gamma_dark=1.6 se recuperan (±0.005) los exponentes históricos 0.62 /
0.76 / 0.88 que el código anterior llevaba como constantes. A mayor
gamma_dark, más aclara cada modo (las fracciones 0.53 / 0.227 son la parte de
la "fuerza" (gamma_dark - 1) que conservan los modos medium y low).

BUG corregido (v5.2): las constantes 0.62 / 0.76 / 0.88 se pasaban a una LUT
estilo OpenCV `(i/255)^(1/gamma)`, es decir se invertían DOS veces: el canal Y
se elevaba a 1/0.62 ≈ 1.61 y el "realce de baja luz" OSCURECÍA los frames
oscuros (un gris 128 salía en 84). Además `gamma_dark` se guardaba sin usarse
y la LUT se reconstruía en cada frame; ahora las tres LUT se calculan una sola
vez en el constructor.
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
    _LUMA_MEDIUM: float = 88.0    # >= este y < NORMAL → corrección suave ("low")
    _LUMA_LOW: float = 62.0       # >= este y < MEDIUM → corrección media ("medium")
    # < LUMA_LOW → corrección fuerte ("dark")

    # Fracción de (gamma_dark - 1) que conservan los modos intermedios.
    _MEDIUM_FRACTION: float = 0.53
    _LOW_FRACTION: float = 0.227

    def __init__(self, clip_limit: float = 2.0, gamma_dark: float = 1.6):
        """
        Args:
            clip_limit:  límite de recorte para CLAHE (default 2.0).
            gamma_dark:  gamma del modo más oscuro (<62 luma), > 0. Los modos
                         medium y low se derivan de él (ver docstring del módulo).
        """
        self.clip_limit = float(clip_limit)
        self.gamma_dark = float(gamma_dark)
        if not self.gamma_dark > 0.0:
            raise ValueError(f"gamma_dark debe ser > 0, recibí {gamma_dark!r}")
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(8, 8),
        )
        # Exponentes por modo (y_out = 255·(y/255)^exponente) y sus LUT, una vez.
        self.exponents = {
            "dark": 1.0 / self.gamma_dark,
            "medium": 1.0 / (1.0 + (self.gamma_dark - 1.0) * self._MEDIUM_FRACTION),
            "low": 1.0 / (1.0 + (self.gamma_dark - 1.0) * self._LOW_FRACTION),
        }
        self._luts = {mode: self._build_lut(exp) for mode, exp in self.exponents.items()}
        self.last_luma: float = 0.0
        self.last_mode: str = "normal"

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    @staticmethod
    def _build_lut(exponent: float) -> np.ndarray:
        """Tabla LUT de 256 entradas: 255·(i/255)^exponent, truncado a uint8."""
        return np.array(
            [(i / 255.0) ** exponent * 255 for i in range(256)],
            dtype=np.uint8,
        )

    def mode_for(self, mean_luma: float) -> str:
        """Modo de corrección para una luminancia media (sin tocar el frame)."""
        if mean_luma >= self._LUMA_NORMAL:
            return "normal"
        if mean_luma < self._LUMA_LOW:
            return "dark"
        if mean_luma < self._LUMA_MEDIUM:
            return "medium"
        return "low"

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
        mode = self.mode_for(mean_luma)
        self.last_mode = mode

        # Sin corrección: luz suficiente
        if mode == "normal":
            return frame

        # CLAHE al canal de luminancia + gamma del modo (LUT precalculada)
        y2 = self.clahe.apply(y)
        y3 = cv2.LUT(y2, self._luts[mode])
        merged = cv2.merge([y3, cr, cb])
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
