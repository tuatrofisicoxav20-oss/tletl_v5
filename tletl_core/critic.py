"""
Tletl v5 - crítico estricto (portado desde ai_critic_v47.py).

Mapeo de claves de orientación viejo (v4.7) -> nuevo (v5)
----------------------------------------------------------
Las CLAVES son idénticas en nombre (ambas versiones usan las mismas 13 features):
  palm_normal_x/y/z, palm_roll, palm_yaw, palm_pitch,
  palm_facing_score, back_hand_score, side_hand_score,
  palm_flatness_score, wrist_depth_score, thumb_side_score, finger_depth_spread

Diferencias de CÓMPUTO (mismo nombre, distinto valor):
  - v4.7: palm_facing_score = sigmoid(-5*nz)  → rango [0,1] con transición suave
  - v5:   palm_facing_score = clamp((nz+1)/2)  → lineal normalizado
  - v4.7: side_hand_score   = clamp(abs(nx)*1.35)
  - v5:   side_hand_score   = clamp(1.0 - abs(nz))  → complemento de frontalidad z
  - v4.7: palm_roll, palm_yaw, palm_pitch en grados (float libre)
  - v5:   palm_roll, palm_yaw, palm_pitch normalizados /180 (rango ~[-1,1])

Diferencias de API de orientación:
  - v4.7: has_orientation_features(features, strict=True) + orientation_summary(features) -> str
  - v5:   orientation_bucket(features) -> {'PALM_FRONT','BACK_HAND','SIDE_HAND','UNKNOWN'}

Lógica portada:
  - MIN_CONF por gesto (idénticos umbrales salvo PINCH: 0.72 -> 0.62 en v5.2;
    la tabla vive en tletl_core.config y [critic.min_conf] la sobreescribe)
  - ACTION_GESTURES más estrictos
  - Validación de orientación para gestos de acción (SIDE_HAND/UNKNOWN se bloquean)
  - Reglas blandas de palm_flatness_score y finger_depth_spread

REGLA DURA: este módulo no importa cv2, ydotool, bpy ni mediapipe. Lógica pura.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import CRITIC_MIN_CONF_DEFAULTS
from .orientation import ORIENTATION_KEYS, orientation_bucket


VALID_GESTURES = {"OPEN_PALM", "FIST", "POINT", "VICTORY", "PINCH", "THREE", "NEUTRAL"}

# Umbrales mínimos de confianza por gesto. FUENTE ÚNICA: tletl_core.config
# (CRITIC_MIN_CONF_DEFAULTS; [critic.min_conf] del toml los sobreescribe vía
# pipeline). Vienen de v4.7 salvo PINCH, que bajó de 0.72 a 0.62 en v5.2 por la
# evidencia de la validación diurna. Antes este módulo tenía su propia copia
# con PINCH 0.72 y strict_critic() sin tabla contradecía a la config.
MIN_CONF: Dict[str, float] = dict(CRITIC_MIN_CONF_DEFAULTS)

# Gestos que disparan acciones: más estrictos en orientación y confianza.
ACTION_GESTURES = {"PINCH", "POINT", "VICTORY", "THREE", "FIST"}

# Buckets de orientación que se consideran inválidos para gestos de acción.
_INVALID_ACTION_BUCKETS = {"SIDE_HAND", "UNKNOWN"}


@dataclass
class CriticResult:
    accepted: bool
    gesture: str       # gesto final (el mismo si accepted, o fallback si rechazado)
    reason: str
    fallback: str = "NEUTRAL"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _has_orientation_features(features: Dict[str, Any]) -> bool:
    """Verifica que el dict tenga todas las claves de orientación del core v5."""
    return all(
        k in features and isinstance(features.get(k), (int, float))
        for k in ORIENTATION_KEYS
    )


def resolve_min_conf(min_conf: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Tabla efectiva de umbrales: MIN_CONF del módulo + overrides (claves en
    mayúsculas). Los gestos que no vengan en `min_conf` conservan el default."""
    if not min_conf:
        return MIN_CONF
    table = dict(MIN_CONF)
    for gesture, value in min_conf.items():
        try:
            table[str(gesture).upper()] = float(value)
        except (TypeError, ValueError):
            continue
    return table


def strict_critic(
    gesture: str,
    confidence: float,
    features: Optional[Dict[str, Any]] = None,
    votes: Optional[Any] = None,
    *,
    require_orientation: bool = True,
    min_conf: Optional[Dict[str, float]] = None,
) -> CriticResult:
    """
    Crítico estricto para el pipeline Tletl v5.

    Parámetros
    ----------
    gesture : str
        Gesto predicho por el clasificador.
    confidence : float
        Confianza de la predicción (0..1).
    features : dict, optional
        Features extraídas por extract_live_features (incluye orientación).
    votes : list o dict, optional
        Votos de múltiples modelos/reglas. No usado activamente en v5 (reservado).
    require_orientation : bool
        Si True y el gesto es de acción, valida que la orientación sea coherente.
    min_conf : dict, optional
        Umbral de confianza mínima por gesto (p.ej. `[critic.min_conf]` del toml
        vía `tletl_core.config.critic_min_conf`). Los gestos ausentes usan
        MIN_CONF del módulo. Antes la tabla estaba hardcodeada y el toml se
        ignoraba: PINCH moría de día con 0.62–0.70 < 0.72 (validación 2026-07-02).

    Retorna
    -------
    CriticResult con:
      - accepted: True si el gesto pasa todos los filtros.
      - gesture: gesto final (el predicho si accepted, fallback si rechazado).
      - reason: explicación textual de la decisión.
      - fallback: siempre "NEUTRAL" (campo fijo para compatibilidad de pipeline).
    """
    features = features or {}
    gesture = str(gesture or "UNKNOWN")
    conf = _safe_float(confidence)
    thresholds = resolve_min_conf(min_conf)

    # --- Gesto inválido ---
    if gesture not in VALID_GESTURES:
        return CriticResult(
            accepted=False,
            gesture="NEUTRAL",
            reason=f"gesto inválido o desconocido: {gesture!r}",
        )

    # --- NEUTRAL es el estado de reposo y el fallback seguro de todo rechazo:
    #     se acepta SIEMPRE para que el reposo sea IDLE (no BLOCKED) aunque la
    #     confianza del KNN sea baja. No tiene umbral de acción. ---
    if gesture == "NEUTRAL":
        return CriticResult(accepted=True, gesture="NEUTRAL", reason="reposo (NEUTRAL)")

    # --- Umbral de confianza mínima ---
    min_conf_gesture = thresholds.get(gesture, 0.72)
    if conf < min_conf_gesture:
        return CriticResult(
            accepted=False,
            gesture="NEUTRAL",
            reason=f"confianza baja para {gesture}: {conf:.2f} < {min_conf_gesture:.2f}",
        )

    # --- Validación de orientación para gestos de acción ---
    if require_orientation and gesture in ACTION_GESTURES:
        orientation_ok = _has_orientation_features(features)

        if not orientation_ok:
            # Sin features de orientación: aceptar solo si la confianza es muy alta
            high_conf_threshold = max(0.82, min_conf_gesture + 0.08)
            if conf >= high_conf_threshold:
                return CriticResult(
                    accepted=True,
                    gesture=gesture,
                    reason="sin features de orientación; aceptado por confianza alta",
                )
            return CriticResult(
                accepted=False,
                gesture="NEUTRAL",
                reason="faltan features de orientación para gesto de acción",
            )

        bucket = orientation_bucket(features)
        if bucket in _INVALID_ACTION_BUCKETS:
            return CriticResult(
                accepted=False,
                gesture="NEUTRAL",
                reason=f"orientación incoherente para {gesture}: {bucket}",
            )

    # --- Reglas blandas de orientación (portadas de v4.7) ---
    palm_flat = _safe_float(features.get("palm_flatness_score"))
    side = _safe_float(features.get("side_hand_score"))
    depth_spread = _safe_float(features.get("finger_depth_spread"))

    if gesture == "OPEN_PALM":
        if palm_flat < 0.34 and conf < 0.86:
            return CriticResult(
                accepted=False,
                gesture="NEUTRAL",
                reason=f"OPEN_PALM poco plana: flat={palm_flat:.2f}",
            )
        if side > 0.82 and conf < 0.88:
            return CriticResult(
                accepted=False,
                gesture="NEUTRAL",
                reason=f"OPEN_PALM muy lateral: side={side:.2f}",
            )

    if gesture in {"VICTORY", "THREE", "POINT"}:
        if depth_spread > 0.78 and conf < 0.84:
            return CriticResult(
                accepted=False,
                gesture="NEUTRAL",
                reason=f"{gesture} con demasiada diferencia de profundidad: {depth_spread:.2f}",
            )

    if gesture == "PINCH":
        if palm_flat < 0.12 and conf < 0.84:
            return CriticResult(
                accepted=False,
                gesture="NEUTRAL",
                reason=f"PINCH inestable: palma poco plana flat={palm_flat:.2f}",
            )

    # --- Aceptado ---
    return CriticResult(
        accepted=True,
        gesture=gesture,
        reason=f"aceptado: {gesture} conf={conf:.2f}",
    )
