"""
tletl_core/adaptive.py

AdaptiveGestureMemory: aprende muestras de gestos en tiempo de ejecución
y las persiste de forma atómica en un JSON separado del banco principal.

Diseño:
- APAGADO POR DEFAULT (enabled=False).
- Nunca guarda features de posición absoluta.
- Cap de muestras por gesto para evitar archivos ilimitados.
- Escritura atómica (escribe a .tmp y luego renombra) para evitar
  archivos corruptos si el proceso es interrumpido.
- Sin dependencias de ventanas, ydotool ni bpy.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict


# Features de posición absoluta que NUNCA se guardan
_EXCLUDED_FEATURES: frozenset[str] = frozenset(
    {
        "palm_center_x",
        "palm_center_y",
        "screen_x",
        "screen_y",
        "cx",
        "cy",
    }
)


class AdaptiveGestureMemory:
    """Memoria adaptativa de gestos.

    Aprende muestras de features de la mano y las persiste en JSON.
    Por defecto está desactivada (enabled=False) para no interferir con
    el flujo principal hasta que el usuario lo habilite explícitamente.
    """

    def __init__(
        self,
        path,
        *,
        enabled: bool = False,
        min_confidence: float = 0.78,
        max_samples_per_gesture: int = 120,
    ):
        """
        Args:
            path: ruta al archivo JSON de persistencia (str o Path).
                  NO debe ser el banco principal de gestos.
            enabled: si False (default), observe() siempre devuelve False
                     y nunca escribe al disco.
            min_confidence: confianza mínima para aceptar una muestra.
            max_samples_per_gesture: cap de muestras por gesto.
        """
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.min_confidence = float(min_confidence)
        self.max_samples_per_gesture = max(1, int(max_samples_per_gesture))

        self._data: Dict[str, Any] = {"gestures": {}}
        if self.enabled:
            self._load()

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Carga el JSON si existe. Silencia errores."""
        if not self.path.exists():
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
            loaded = json.loads(raw)
            if isinstance(loaded, dict) and isinstance(loaded.get("gestures"), dict):
                self._data = loaded
        except Exception:
            # Archivo corrupto u otro error: empezar vacío
            pass

    def _save_atomic(self) -> None:
        """Escribe el JSON de forma atómica (tmp + rename)."""
        tmp_path = self.path.with_suffix(".tmp")
        payload = json.dumps(self._data, indent=2, ensure_ascii=False)
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, self.path)

    # ------------------------------------------------------------------
    # Limpieza de features
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_features(features: dict) -> Dict[str, float]:
        """Excluye keys de posición absoluta y valores no finitos."""
        clean: Dict[str, float] = {}
        for key, value in features.items():
            if key in _EXCLUDED_FEATURES:
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                clean[key] = round(float(value), 6)
        return clean

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def observe(
        self,
        gesture: str,
        features: dict,
        confidence: float,
    ) -> bool:
        """Intenta registrar una muestra de gesto.

        Args:
            gesture:    identificador del gesto (p.ej. "FIST").
            features:   diccionario de features de la mano.
            confidence: confianza de la predicción (0.0–1.0).

        Returns:
            True si la muestra fue guardada, False en caso contrario.
        """
        if not self.enabled:
            return False

        if float(confidence) < self.min_confidence:
            return False

        clean = self._clean_features(features)
        if not clean:
            return False

        gestures = self._data.setdefault("gestures", {})
        bucket: list = gestures.setdefault(gesture, [])

        # Aplicar cap
        if len(bucket) >= self.max_samples_per_gesture:
            # Eliminar las muestras más antiguas para hacer espacio
            excess = len(bucket) - self.max_samples_per_gesture + 1
            del bucket[:excess]

        bucket.append(clean)
        self._data["updated_at"] = time.time()

        self._save_atomic()
        return True

    def counts(self) -> Dict[str, int]:
        """Devuelve el número de muestras por gesto."""
        return {
            g: len(v)
            for g, v in self._data.get("gestures", {}).items()
            if isinstance(v, list)
        }
