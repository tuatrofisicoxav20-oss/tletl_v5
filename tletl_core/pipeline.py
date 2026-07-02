"""tletl_core/pipeline.py — el orquestador único de Tletl v5.

Encadena: features -> classifier(KNN) -> guard(IA vs regla) -> critic(estricto)
-> temporal(estabiliza, uno por mano) -> [adaptive(opcional)].

Cualquier app (Fedora, Blender, mañana otra) consume este pipeline y hereda
exactamente la misma seguridad y el mismo clasificador. Un solo cerebro,
muchos clientes.

REGLA DURA: este módulo no importa cv2-window, ydotool ni bpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .adaptive import AdaptiveGestureMemory
from .classifier import Prediction, RobustKNNRuntime
from .config import load_config
from .critic import CriticResult, strict_critic
from .guard import geometric_rule, rule_guard
from .orientation import orientation_bucket
from .state import TletlHandState
from .temporal import TemporalFilter


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
    rule_raw: str = "NEUTRAL"
    guard_reason: str = "OK"
    critic_accepted: bool = False
    critic_reason: str = ""
    reason: str = ""
    votes: Dict[str, float] = field(default_factory=dict)


class TletlPipeline:
    def __init__(self, bank_path: str | Path, config: Optional[Dict[str, Any]] = None,
                 adaptive_path: Optional[str | Path] = None):
        self.cfg = config or load_config()
        c = self.cfg
        self.classifier = RobustKNNRuntime(
            bank_path,
            k=int(c["classifier"]["k"]),
            orientation_weight=float(c["classifier"]["orientation_weight"]),
        )
        self._strict = bool(c["classifier"].get("strict", True))

        # Un TemporalFilter independiente por mano (dom, mod, ...).
        self._temporal: Dict[str, TemporalFilter] = {}

        # Memoria adaptativa (apagada por default vía config).
        if adaptive_path is None:
            adaptive_path = Path(bank_path).expanduser().resolve().parent / "tletl_adaptive_runtime.json"
        a = c["adaptive"]
        self.adaptive = AdaptiveGestureMemory(
            adaptive_path,
            enabled=bool(a["enabled"]),
            min_confidence=float(a["min_confidence"]),
            max_samples_per_gesture=int(a["max_samples_per_gesture"]),
        )

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

    # ------------------------------------------------------------------
    def process_features(self, features: Dict[str, float], hand: str = "dom") -> HandResult:
        """Pasa las features de UNA mano por todo el pipeline de seguridad."""
        g = self.cfg["guard"]
        cr = self.cfg["critic"]

        # 1) Clasificador KNN
        pred: Prediction = self.classifier.predict(features, strict=self._strict)

        # 2) Guard: segunda opinión geométrica + árbitro
        rule_raw = geometric_rule(features)
        if g.get("enabled", True):
            guarded, guard_reason = rule_guard(
                pred, rule_raw,
                dangerous=list(g["dangerous"]),
                conf_threshold=float(g["conf_threshold"]),
                margin_threshold=float(g["margin_threshold"]),
            )
        else:
            guarded, guard_reason = pred.raw_label, "GUARD_OFF"

        guard_blocked = guarded == "NEUTRAL" and guard_reason.startswith("GUARD_BLOCK")

        # 3) Critic estricto sobre el gesto ya guardado
        if cr.get("enabled", True):
            critic: CriticResult = strict_critic(
                guarded, pred.confidence, features, pred.votes,
                require_orientation=bool(cr.get("require_orientation", True)),
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

        # 5) Aprendizaje adaptativo (no-op si está apagado)
        if stable.ok and stable.label not in ("NEUTRAL", "NO_HAND"):
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
        )

    # ------------------------------------------------------------------
    def to_hand_state(self, result: HandResult, side: str = "unknown",
                      features: Optional[Dict[str, float]] = None) -> TletlHandState:
        """Convierte un HandResult en TletlHandState (para que la app sea delgada)."""
        return TletlHandState(
            present=result.present,
            side=side,
            gesture=result.stable_gesture,
            raw_gesture=result.raw_gesture,
            stable_gesture=result.stable_gesture,
            orientation=result.orientation,
            confidence=result.confidence,
            margin=result.margin,
            critic_ok=result.ok and result.critic_accepted,
            critic_reason=result.critic_reason,
            features=features or {},
        )
