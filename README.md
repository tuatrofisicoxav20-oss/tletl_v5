# Tletl v5

Control gestual por cámara para **Fedora** (navegador, cursor, ventanas) y **Blender**
(mover objetos en el viewport). Detecta la mano con MediaPipe, extrae features
geométricas, clasifica el gesto con un KNN robusto sobre un banco de muestras, lo
pasa por una capa de seguridad (guard + critic + temporal) y traduce el gesto a una
**intención abstracta** que cada app ejecuta a su manera.

Un solo cerebro (`tletl_core`), muchos clientes (Fedora, Blender, …).

## Arquitectura (flujo de datos)

```
cámara → frame → [lowlight.enhance] → MediaPipe Hands (max_num_hands=2)
   → por cada mano: landmarks → features.extract_live_features
   → pipeline.process_features():
        classifier.predict        (KNN robusto sobre el banco)
        → guard.rule_guard        (cruza IA vs regla geométrica; bloquea peligrosos en conflicto)
        → critic.strict_critic    (umbral por gesto + orientación coherente)
        → temporal.TemporalFilter (estabiliza: no actuar por 1 frame suelto; uno POR MANO)
        → [adaptive.observe]      (opcional, OFF por default: aprende si conf alta)
   → state.TletlFrameState{dom, mod, intent} ← intent.gesture_to_common_intent
   → bus.write(state) → tletl_state.json
        ├─→ apps/fedora_control : intent/gesto → ydotool
        └─→ apps/blender_control: addon lee el bus → mueve el objeto con bpy
```

## La regla del core (no se rompe)

`tletl_core/` **jamás** importa cv2-window, ydotool ni bpy. El core decide la
intención; la app la ejecuta. Si el core necesitara "hacer algo de sistema", está
mal diseñado. (Excepción razonada: `lowlight.py` usa `cv2` solo para procesamiento
de imagen puro —CLAHE/gamma—, nunca ventanas ni entrada/salida del SO.)

## Instalación

```bash
cd tletl_v5
python3.12 -m venv .venv && source .venv/bin/activate   # mediapipe requiere Python 3.12
pip install -r requirements.txt
```

> El **core y sus tests no necesitan mediapipe** (trabajan sobre landmarks ya
> extraídos). mediapipe solo hace falta para la app en vivo y la captura de gestos.

## Uso

```bash
./launchers/tletl-health.sh          # healthcheck: core + banco + intent
./launchers/tletl-fedora-safe.sh     # DRY-RUN: muestra gestos y acciones sin ejecutarlas
./launchers/tletl-fedora.sh          # control real de Fedora (ydotool)
./launchers/tletl-bank.sh --dual-hand# capturar gestos propios al banco
./launchers/tletl-blender-bus.sh     # alimentar el bus para Blender (sin tocar Fedora)
```

Diagnóstico del clasificador y el banco (no necesitan cámara):

```bash
python -m tools.bank_probe           # estadísticas del banco (distribución, balance, features)
python -m tools.duel_lab             # compara configs del KNN con holdout honesto (accuracy real)
```

Para Blender, ver `apps/blender_control/README.md` (instalar el addon, activar,
correr el bus y mover un objeto con PINCH).

## Gestos → acciones (app Fedora)

Control se activa/desactiva manteniendo **OPEN_PALM**. **FIST** = pausa de seguridad.
**THREE** = cambiar de modo. **Q/ESC** salir, `t` toggle, `m` modo.

| Modo | POINT | PINCH | VICTORY + swipe | OPEN_PALM + swipe |
|------|-------|-------|-----------------|-------------------|
| **NAVEGADOR** | scroll/page según zona vertical | click | cambiar pestaña (←/→) | atrás / adelante |
| **CURSOR** | mover cursor | drag / click | — | mover cursor |
| **VENTANAS** | — | enter | — | overview (swipe ↑) |

La **mano dominante** controla Fedora; la **otra mano** se escribe al bus para que
Blender la use (p. ej. una agarra y la otra rota/escala).

## Configuración

Todo en `config/tletl.toml` (k del KNN, umbrales de guard/critic, temporal, lowlight,
adaptive, cámara). Cada clave admite override por variable de entorno
(`TLETL_AI_V47_K`, `TLETL_DRY_RUN`, `TLETL_ADAPTIVE`, …). Ver `tletl_core/config.py`.

## Estado

| Capacidad | Estado |
|---|---|
| 3 modos Fedora (NAVEGADOR/CURSOR/VENTANAS) | ✅ |
| Dual-hand (dom + mod, filtros temporales independientes) | ✅ |
| Guard + Critic (seguridad anti-clicks-fantasma) | ✅ |
| Low-light (CLAHE + gamma) | ✅ |
| Gesture bank (captura, `--dual-hand`) | ✅ |
| Blender addon (lógica de mapeo testeada sin bpy) | ✅ código / ⏳ prueba en Blender |
| Adaptive (aprendizaje en vivo) | ⚠️ experimental, **OFF por default** |

> **Adaptive**: si se activa mal puede degradar la clasificación. Aprende a un JSON
> aparte (nunca al banco principal) y excluye features de posición absoluta.

## Tests

```bash
python -m pytest tests/ -q     # toda la suite del core + apps
```
