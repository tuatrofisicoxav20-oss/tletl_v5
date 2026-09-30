"""tletl_core/pipeline.py — el orquestador único de Tletl v5.

Encadena: features -> classifier(KNN) -> guard(IA vs regla [+ pista externa])
-> critic(estricto) -> temporal(estabiliza, uno por mano) -> [adaptive(opcional)].

Cualquier app (Fedora, Blender, mañana otra) consume este pipeline y hereda
exactamente la misma seguridad y el mismo clasificador. Un solo cerebro,
muchos clientes.

Cableado de config (v5.2): antes `[classifier] min_confidence/min_margin` y
`[critic.min_conf]` del toml se ignoraban (umbrales hardcodeados) y la memoria
adaptativa grababa muestras que nadie leía. Ahora los tres llegan a donde deben.

Ciclo de vida: llama `close()` (o usa `with TletlPipeline(...) as p:`) al
terminar para persistir la memoria adaptativa pendiente (escritura throttled).

REGLA DURA: este módulo no importa cv2-window, ydotool ni bpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .adaptive import AdaptiveGestureMemory
from .classifier import Prediction, RobustKNNRuntime
from .config import adaptive_path_from_config, critic_min_conf, load_config
from .critic import CriticResult, strict_critic
from .guard import geometric_rule, rule_guard
from .orientation import orientation_bucket
from .state import TletlHandState
from .temporal import TemporalFilter

# Etiquetas de reposo/ausencia: nunca se aprenden como gesto.
_NON_GESTURES = ("NEUTRAL", "NO_HAND")


@dataclass
class HandResult:
    """Resultado del pipeline para UNA mano, ya pasado por toda la seguridad."""
    present: bool = False
    raw_gesture: str = "NEUTRAL"        # lo que dijo el KNN crudo
    guarded_gesture: str = "NEUTRAL"    # tras rule_guard
    stable_gesture: str = "NEUTRAL"     # tras TemporalFilter (el que la app debe usar)
    ok: bool = False                    # estable y aceptado
    confidence: float = 0.0
    margin: float = 0.0
    orientation: str = "UNKNOWN"
    rule_raw: str = "NEUTRAL"           # veredicto de la regla geométrica (sin pista)
    guard_reason: str = "OK"
    critic_accepted: bool = False       # critic del frame ACTUAL (no del estable)
    critic_reason: str = ""
    reason: str = ""
    votes: Dict[str, float] = field(default_factory=dict)
    hint: Optional[str] = None          # segunda opinión externa (normalizada) o None


class TletlPipeline:
    def __init__(self, bank_path: str | Path, config: Optional[Dict[str, Any]] = None,
                 adaptive_path: Optional[str | Path] = None, *, verbose: bool = True):
        self.cfg = config or load_config()
        c = self.cfg

        # Memoria adaptativa PRIMERO (apagada por default vía config). Vive en
        # ~/.tletl/, NUNCA junto al banco (antes se escribía en datasets/). Si
        # está activa, sus muestras entran al KNN como `extra_samples`: antes
        # se grababan y ningún componente las leía (aprendizaje sin efecto).
        if adaptive_path is None:
            adaptive_path = adaptive_path_from_config(self.cfg)
        a = c["adaptive"]
        self.adaptive = AdaptiveGestureMemory(
            adaptive_path,
            enabled=bool(a["enabled"]),
            min_confidence=float(a["min_confidence"]),
            max_samples_per_gesture=int(a["max_samples_per_gesture"]),
            save_interval=float(a["save_interval"]),
        )
        extra_samples = self.adaptive.samples() if self.adaptive.enabled else None

        cl = c["classifier"]
        self.classifier = RobustKNNRuntime(
            bank_path,
            k=int(cl["k"]),
            orientation_weight=float(cl["orientation_weight"]),
            min_confidence=float(cl["min_confidence"]),
            min_margin=float(cl["min_margin"]),
            extra_samples=extra_samples,
            verbose=verbose,
        )
        self._strict = bool(cl.get("strict", True))

        # Tabla de umbrales del critic desde [critic.min_conf] (antes ignorada).
        self._critic_min_conf: Dict[str, float] = critic_min_conf(self.cfg)

        # Un TemporalFilter independiente por mano (dom, mod, ...).
        self._temporal: Dict[str, TemporalFilter] = {}

    # ------------------------------------------------------------------
    def _temporal_for(self, hand: str) -> TemporalFilter:
        tf = self._temporal.get(hand)
        if tf is None:
            t = self.cfg["temporal"]
            tf = TemporalFilter(size=int(t["size"]), min_count=int(t["min_count"]))
            self._temporal[hand] = tf
        return tf

    def reset(self, hand: Optional[str] = None) -> None:
        if hand is None:
            for tf in self._temporal.values():
                tf.reset()
        elif hand in self._temporal:
            self._temporal[hand].reset()

    def close(self) -> None:
        """Persiste la memoria adaptativa pendiente. Llamar al cerrar la app."""
        self.adaptive.flush()

    def __enter__(self) -> "TletlPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _normalize_hint(hint: Optional[str]) -> Optional[str]:
        if hint is None:
            return None
        text = str(hint).strip().upper()
        return text or None

    # ------------------------------------------------------------------
    def process_features(self, features: Dict[str, float], hand: str = "dom",
                         hint: Optional[str] = None) -> HandResult:
        """Pasa las features de UNA mano por todo el pipeline de seguridad.

        `hint` es una segunda opinión INDEPENDIENTE del KNN (p.ej. el Gesture
        Recognizer de MediaPipe, que se conectará después): una etiqueta de
        LABELS ("FIST", "OPEN_PALM", "POINT", "VICTORY", "PINCH", "THREE",
        "NEUTRAL") o None. Si coincide con el gesto crudo del KNN, el guard la
        toma como acuerdo: su argumento `rule_raw` pasa a ser el propio gesto
        crudo, así que un gesto peligroso con confianza/margen bajos NO se
        bloquea por discrepar de la regla geométrica (dos detectores
        independientes coinciden); en ese caso `guard_reason` es
        "OK_HINT_AGREES". Si la pista discrepa o no viene, nada cambia: guard y
        critic actúan como siempre. La pista se guarda en `HandResult.hint`.
        """
        g = self.cfg["guard"]
        cr = self.cfg["critic"]

        # 1) Clasificador KNN
        pred: Prediction = self.classifier.predict(features, strict=self._strict)

        # 2) Guard: segunda opinión geométrica + árbitro (+ pista externa)
        rule_raw = geometric_rule(features)
        hint_label = self._normalize_hint(hint)
        hint_agrees = hint_label is not None and hint_label == pred.raw_label
        if g.get("enabled", True):
            guard_kwargs = dict(
                dangerous=list(g["dangerous"]),
                conf_threshold=float(g["conf_threshold"]),
                margin_threshold=float(g["margin_threshold"]),
            )
            guarded, guard_reason = rule_guard(pred, rule_raw, **guard_kwargs)
            if hint_agrees and guard_reason.startswith("GUARD_BLOCK"):
                # KNN y detector externo coinciden: la regla geométrica deja de
                # ser voto de bloqueo (rule_raw := gesto crudo).
                guarded, guard_reason = rule_guard(pred, pred.raw_label, **guard_kwargs)
                if not guard_reason.startswith("GUARD_BLOCK"):
                    guard_reason = "OK_HINT_AGREES"
        else:
            guarded, guard_reason = pred.raw_label, "GUARD_OFF"

        guard_blocked = guarded == "NEUTRAL" and guard_reason.startswith("GUARD_BLOCK")

        # 3) Critic estricto sobre el gesto ya guardado
        if cr.get("enabled", True):
            critic: CriticResult = strict_critic(
                guarded, pred.confidence, features, pred.votes,
                require_orientation=bool(cr.get("require_orientation", True)),
                min_conf=self._critic_min_conf,
            )
        else:
            critic = CriticResult(accepted=True, gesture=guarded, reason="CRITIC_OFF")

        # 4) Construir una Prediction "post-seguridad" para el filtro temporal
        safe_ok = bool(pred.ok and not guard_blocked and critic.accepted)
        safe_label = critic.gesture if critic.accepted else "NEUTRAL"
        post = Prediction(
            label=safe_label,
            raw_label=safe_label if safe_ok else "NEUTRAL",
            confidence=pred.confidence,
            margin=pred.margin,
            orientation=pred.orientation,
            reason=critic.reason,
            ok=safe_ok,
            votes=pred.votes,
        )
        stable = self._temporal_for(hand).update(post)

        # 5) Aprendizaje adaptativo (no-op si está apagado). Solo aprende cuando
        #    el frame ACTUAL pasó toda la seguridad con el MISMO gesto que el
        #    estable: durante un TEMPORAL_HOLD el estable se sostiene por la
        #    historia mientras el frame actual puede ser una transición o
        #    NEUTRAL, y antes esas features se grababan etiquetadas con el gesto
        #    estable (envenenaba la memoria que ahora sí alimenta al KNN).
        if (stable.ok and safe_ok and safe_label == stable.label
                and stable.label not in _NON_GESTURES):
            self.adaptive.observe(stable.label, features, pred.confidence)

        return HandResult(
            present=True,
            raw_gesture=pred.raw_label,
            guarded_gesture=guarded,
            stable_gesture=stable.label,
            ok=bool(stable.ok),
            confidence=float(pred.confidence),
            margin=float(pred.margin),
            orientation=pred.orientation or orientation_bucket(features),
            rule_raw=rule_raw,
            guard_reason=guard_reason,
            critic_accepted=critic.accepted,
            critic_reason=critic.reason,
            reason=stable.reason,
            votes=dict(pred.votes),
            hint=hint_label,
        )

    # ------------------------------------------------------------------
    def to_hand_state(self, result: HandResult, side: str = "unknown",
                      features: Optional[Dict[str, float]] = None) -> TletlHandState:
        """Convierte un HandResult en TletlHandState (para que la app sea delgada).

        `critic_ok` es el veredicto ESTABLE (`result.ok`): el filtro temporal ya
        incorpora guard + critic vía safe_ok. Antes se hacía
        `result.ok and result.critic_accepted`, mezclando el estable con el
        critic del frame ACTUAL: durante un TEMPORAL_HOLD el gesto estable se
        mantiene (ok=True) pero un solo frame rechazado volteaba critic_ok a
        False y el intent del bus parpadeaba a BLOCKED mientras las apps seguían
        actuando sobre el gesto estable. `critic_reason` sí es el del frame
        actual (diagnóstico).
        """
        return TletlHandState(
            present=result.present,
            side=side,
            gesture=result.stable_gesture,
            raw_gesture=result.raw_gesture,
            stable_gesture=result.stable_gesture,
            orientation=result.orientation,
            confidence=result.confidence,
            margin=result.margin,
            critic_ok=bool(result.ok),
            critic_reason=result.critic_reason,
            features=features or {},
        )
