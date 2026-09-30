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
- Escritura THROTTLED: antes cada observe() reescribía el JSON completo
  (una vez por frame con la memoria activa). Ahora la primera escritura de la
  sesión es inmediata y las siguientes se agrupan cada `save_interval`
  segundos; `flush()` persiste lo pendiente (el pipeline lo llama en close()).
- Las muestras aprendidas SÍ se usan: `samples()` las devuelve en el formato
  que RobustKNNRuntime acepta como `extra_samples` (antes se guardaban y nada
  las leía, así que el "aprendizaje en vivo" no tenía efecto).
- Sin dependencias de ventanas, ydotool ni bpy.
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .paths import ensure_parent


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
        save_interval: float = 2.0,
        clock: Callable[[], float] = time.time,
    ):
        """
        Args:
            path: ruta al archivo JSON de persistencia (str o Path).
                  NO debe ser el banco principal de gestos.
            enabled: si False (default), observe() siempre devuelve False
                     y nunca escribe al disco.
            min_confidence: confianza mínima para aceptar una muestra.
            max_samples_per_gesture: cap de muestras por gesto.
            save_interval: segundos mínimos entre escrituras a disco. La
                     primera escritura tras un cambio es inmediata si aún no
                     se ha guardado nada en esta sesión; las siguientes esperan
                     `save_interval`. 0 = escribir en cada observe().
            clock: fuente de tiempo inyectable (tests deterministas).
        """
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.min_confidence = float(min_confidence)
        self.max_samples_per_gesture = max(1, int(max_samples_per_gesture))
        self.save_interval = max(0.0, float(save_interval))
        self._clock = clock
        self._dirty = False
        self._last_save: Optional[float] = None   # None = nada guardado en esta sesión

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
        """Escribe el JSON de forma atómica (tmp + rename).

        Crea el directorio padre si falta: la ruta por default vive en
        ~/.tletl/, que en una máquina nueva no existe, y antes el primer
        observe() con confianza alta tumbaba la app con FileNotFoundError.
        """
        ensure_parent(self.path)
        tmp_path = self.path.with_suffix(".tmp")
        payload = json.dumps(self._data, indent=2, ensure_ascii=False)
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self.path)
        self._dirty = False
        self._last_save = self._clock()

    def _save_if_due(self) -> bool:
        """Guarda si nunca se guardó en esta sesión o si ya pasó save_interval."""
        if not self._dirty:
            return False
        now = self._clock()
        if self._last_save is not None and (now - self._last_save) < self.save_interval:
            return False
        self._save_atomic()
        return True

    def flush(self) -> bool:
        """Persiste los cambios pendientes (si los hay). Devuelve True si escribió."""
        if not self._dirty:
            return False
        self._save_atomic()
        return True

    @property
    def dirty(self) -> bool:
        """True si hay muestras en memoria que aún no se escribieron a disco."""
        return self._dirty

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
            True si la muestra fue registrada (en memoria; el disco se
            actualiza según save_interval), False en caso contrario.
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
        self._dirty = True

        self._save_if_due()
        return True

    def counts(self) -> Dict[str, int]:
        """Devuelve el número de muestras por gesto."""
        return {
            g: len(v)
            for g, v in self._data.get("gestures", {}).items()
            if isinstance(v, list)
        }

    def samples(self) -> List[Tuple[str, Dict[str, float]]]:
        """Muestras aprendidas como lista de (gesto, features), listas para
        `RobustKNNRuntime(extra_samples=...)`. Ignora entradas malformadas."""
        out: List[Tuple[str, Dict[str, float]]] = []
        gestures = self._data.get("gestures", {})
        if not isinstance(gestures, dict):
            return out
        for gesture, bucket in gestures.items():
            if not isinstance(bucket, list):
                continue
            for feat in bucket:
                if not isinstance(feat, dict):
                    continue
                clean = {
                    str(k): float(v)
                    for k, v in feat.items()
                    if isinstance(v, (int, float)) and math.isfinite(float(v))
                }
                if clean:
                    out.append((str(gesture).upper(), clean))
        return out
