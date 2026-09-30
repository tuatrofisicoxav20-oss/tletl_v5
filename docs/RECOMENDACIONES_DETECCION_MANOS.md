# Recomendaciones: detección de manos y precisión de gestos en Tletl

Fecha: 2026-09-30. Documento de decisión. Lo escribo con lo que se pudo **medir**
en este entorno (CPU, sin cámara) y con la evidencia de la validación física
del 2026-07-02. Donde algo no se pudo probar en vivo lo digo tal cual.

---

## 0. Resumen en una página

| # | Qué | Veredicto | Estado en el repo |
|---|-----|-----------|-------------------|
| 1 | **MediaPipe Hand Landmarker (Tasks API)** — `google-ai-edge/mediapipe` | **Recomendación al cien.** Mismos 21 landmarks que hoy, así que el banco de 2 652 muestras sigue valiendo. CPU: 13.8 ms/frame medidos aquí. Y es obligatorio: la API vieja (`mp.solutions`) **ya no existe** en mediapipe ≥ 0.10.31 / 1.0 (verificado con 1.0.1). | **APROBADO por ti el 2026-09-30 y aplicado como backend principal.** `[tracker] backend = "auto"` usa Tasks siempre que exista y descarga el modelo solo si falta (`auto_download = true`); la API vieja queda únicamente como respaldo. Pendiente: prueba con cámara real (tú). |
| 2 | **MediaPipe Gesture Recognizer (Tasks)** como *segunda opinión* | **Recomendado, opcional.** Reconoce puño, palma, índice y victoria de forma independiente al KNN. Cuando ambos coinciden, el guard deja pasar el gesto aunque la confianza del KNN sea justa; cuando no, todo sigue igual. Ataca directo a los dos fallos medidos (FIST fantasma, PINCH que moría por umbral). No sabe PINCH ni THREE: para eso está el KNN. | **Adaptado, apagado por default** (`[tracker] gesture_hint = true` para activarlo). Necesita tu visto bueno para dejarlo encendido. |
| 3 | **Enriquecer el banco** (no es un repo, es tu cámara) | **Es lo que más precisión da por hora invertida.** El banco actual se capturó de noche; FIST en pose de escritorio y PINCH de día fallan por falta de muestras, no por el detector. | Herramienta lista: `./launchers/tletl-bank.sh --dual-hand`. Guía en §4. |
| 4 | RTMPose-hand vía `rtmlib` (Tau-J/rtmlib) | Viable en CPU (ONNX Runtime/OpenVINO), 21 puntos en el mismo orden, **pero solo 2D**: se pierden las 13 features de orientación y las de profundidad. Plan B si MediaPipe fallara con oclusiones. | No adaptado. Costo estimado: 1 día (adapter + desactivar orientación en el critic). |
| 5 | YOLO11-pose entrenado en `hand-keypoints` (Ultralytics) | Un solo modelo detecta y da 21 puntos (etiquetados con MediaPipe, mismo orden). CPU ≈ 20–40 ms con `yolo11n-pose`. Solo 2D. Licencia **AGPL-3.0** (ojo si algún día distribuyes Tletl). | No adaptado. Plan C. |
| 6 | WiLoR (rolpotamias/WiLoR, ECCV 2024) | Estado del arte en malla 3D de la mano. **Rechazado**: exige CUDA (PyTorch 2.0 + CUDA 11.7, transformer ViT). Rompe la regla "todo en CPU, cero CUDA" del máster. | No. |
| 7 | HaMeR (geopavlakos/hamer, CVPR 2024) | Igual que WiLoR: ViT-H, GPU, registro en MANO. **Rechazado.** | No. |

**Estado de las decisiones:** #1 ya tiene tu visto bueno y está aplicado
(`[tracker] backend = "auto"` prefiere Tasks). Quedan por decidir: activar
`gesture_hint = true` después de probarlo 10 minutos en dry-run, y dedicar
una sesión de 20 minutos a capturar muestras diurnas de FIST y PINCH. Con
eso se atacan las tres causas raíz que salieron en la validación.

---

## 1. Por qué MediaPipe Hand Landmarker (Tasks) y no otra cosa

**Hecho verificado en este entorno:** `pip install mediapipe` instala hoy la
1.0.1 y `hasattr(mediapipe, "solutions")` devuelve `False`. La app tal como
estaba (`mp.solutions.hands`) truena al arrancar con cualquier mediapipe
reciente. Esto ya lo habías anotado en la validación (instalaste 0.10.21 a
mano). No es un "nice to have": sin esto Tletl solo corre con una versión
congelada que dejará de instalar en Python nuevos.

**Compatibilidad total con el banco.** El HandLandmarker devuelve exactamente
los mismos 21 landmarks normalizados (x, y, z) con los mismos índices (0
muñeca, 1–4 pulgar, 5–8 índice, …). `extract_live_features` no cambia una
línea, el KNN no cambia, las 2 652 muestras siguen siendo válidas. Cualquier
otro detector con otra topología (o sin z) invalida parte del banco.

**Rendimiento medido aquí (CPU, XNNPACK, frame 640×360 sin mano):**
13.8 ms/frame en modo VIDEO con `num_hands=2`. Con mano presente sube algo
(corre el modelo de landmarks por mano), pero sigue muy por debajo de los
33 ms de un frame a 30 fps. Los ~15 fps que mediste no venían del detector,
venían del KNN en Python puro (ver §3).

**Qué se hizo para adaptarlo** (todo en `apps/`, el core no toca MediaPipe):

1. `apps/common/hand_tracker.py`: clase `HandTracker` con dos backends,
   `tasks` (HandLandmarker, principal) y `legacy` (mp.solutions, solo
   respaldo). Ambos devuelven la misma lista de `HandDetection(landmarks: 21
   Point, handedness, score)`. `backend = "auto"` elige tasks siempre que la
   API Tasks exista; cae a legacy solo si no.
2. `tools/fetch_models.py` + `launchers/tletl-fetch-models.sh`: descarga
   `hand_landmarker.task` y `gesture_recognizer.task` a `~/.tletl/models/`.
   Con `[tracker] auto_download = true` la app lo baja sola la primera vez
   si falta (unos 8 MB, una sola vez).
3. `apps/fedora_control/main.py` y `tools/gesture_bank.py` ya no importan
   mediapipe directamente: usan el tracker. Así el día que quieras probar
   rtmlib o YOLO, es un backend más en un solo archivo.
4. `[camera] proc_width = 640`: el frame se reduce solo para el detector
   (los landmarks son normalizados, nada más cambia). Ojo, medido aquí: el
   HandLandmarker tarda lo mismo a 1280×720 que a 640×360 (10.8 ms; él ya
   redimensiona por dentro), así que esto solo ahorra la conversión de color
   y la copia del frame. La mejora grande de fps viene del KNN (§3).

**Cómo lo pruebas tú (5 minutos):**

```bash
source .venv/bin/activate
pip install -U mediapipe                 # 1.x, sin API legacy
./launchers/tletl-fetch-models.sh        # descarga los .task a ~/.tletl/models
./launchers/tletl-health.sh              # debe decir backend=tasks
./launchers/tletl-fedora-safe.sh         # dry-run: en el panel verás "backend:tasks"
```

Si la mano se detecta y los gestos salen como antes, listo. Si notas que
la mano "parpadea", sube `[tracker] track_conf` a 0.8 o baja `det_conf` a 0.6.

---

## 2. Gesture Recognizer como segunda opinión (opcional, tu visto bueno)

El modelo `gesture_recognizer.task` de MediaPipe clasifica por sí solo
`Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Victory`, `Thumb_Up`,
`Thumb_Down`, `ILoveYou`. Los cuatro primeros mapean 1:1 a `FIST`,
`OPEN_PALM`, `POINT`, `VICTORY`. Está entrenado con miles de manos distintas
de la tuya, así que es una opinión **independiente** del banco.

Cómo se conectó (sin tocar la regla del core):

- `apps/common/gesture_hint.py` corre el recognizer sobre el mismo frame y
  traduce el nombre al vocabulario de Tletl.
- `tletl_core/pipeline.py::process_features(..., hint=...)`: si el hint
  coincide con el gesto crudo del KNN, el guard lo trata como "la regla
  geométrica está de acuerdo" y NO bloquea por confianza baja. Si no
  coincide, no cambia nada (guard, critic y filtro temporal siguen igual).
  Motivo del diseño conservador: no quiero que un modelo que no conoce
  PINCH ni THREE pueda vetar esos gestos.

Costo: un modelo más por frame (~8 MB, CPU). Medido aquí: 10.6 ms/frame
sin mano (el landmarker solo: 10.8 ms). Con manos presentes ambos suben algo.
Si tu laptop va justa, déjalo apagado: es una mejora de precisión, no de fps.

Qué esperar: menos `GUARD_BLOCK_FIST`/`GUARD_BLOCK_POINT` cuando el gesto es
claro, y una fuente extra en el panel (`hint:FIST`) para diagnosticar.

Para activarlo: `[tracker] gesture_hint = true` en `config/tletl.toml`
(o `TLETL_GESTURE_HINT=1`). Pruébalo primero con `tletl-fedora-safe.sh`.

---

## 3. Rendimiento: dónde estaba el cuello de botella

| Componente | Antes | Ahora |
|---|---|---|
| KNN `predict()` | bucle Python sobre ~2 363 muestras × ~60 features por mano y por frame | matriz numpy; misma respuesta bit a bit (hay un test que lo comprueba contra la implementación de referencia) |
| Frame al detector | 1280×720 completo | reducido a `proc_width` (640) solo para detectar |
| Bus | JSON con indentación y ~60 features por mano, 30 veces/s | JSON compacto sin features (`[bus] include_features = false`) |
| Banco de gestos (tool) | releía el JSONL completo **en cada frame** | conteos en memoria |
| Ventana OpenCV | siempre | `--headless` para alimentar Blender/CAD sin ventana |

Medido en este entorno (200 predicciones, k=13, banco balanceado 2 363×67):
`predict()` pasó de **58.1 ms** a **0.95 ms** por mano (≈61×). A 58 ms por
predicción y dos manos, el solo clasificador ya limitaba el bucle a ~15 fps,
que es exactamente lo que mediste en la validación. Ahora el detector de
MediaPipe (10–14 ms) es el costo dominante y el bucle puede ir a la
velocidad de la cámara.

Bonus encontrado en la misma revisión: el realce de baja luz (CLAHE + gamma)
**oscurecía** los frames oscuros por una doble inversión de la gamma (el
exponente quedaba en 1.61 en vez de 0.625). Está corregido y con test.

Lo que **no** cambia: el clasificador sigue siendo el KNN sobre tu banco,
CPU, sin CUDA, sin modelos grandes. Solo corre más rápido.

---

## 4. Precisión de gestos: qué falló de verdad y qué se hizo

De `docs/VALIDACION_FISICA_v5.md`:

| Síntoma medido | Causa raíz | Qué se hizo | Qué falta (tú) |
|---|---|---|---|
| PINCH 0/10 de día; raw correcto 68/80 con conf 0.62–0.70 | umbral fijo 0.72 del critic | `[critic.min_conf] PINCH = 0.62` (configurable por gesto, ya no está en el código) | capturar 60–100 PINCH de día |
| FIST 0/10 en pose de escritorio (sale POINT); 80/80 a distancia | el banco no tiene puños en esa pose | — (es dato, no código) | capturar 60–100 FIST en pose de escritorio |
| 3 SAFETY_STOP fantasma en 2 min al agarrar cosas | un solo frame de FIST pausaba | `[fedora] safety_hold = 0.35` s de FIST estable antes de pausar | ajustar si sigue molestando |
| 1 scroll fantasma (POINT) | gesto real breve | hint del Gesture Recognizer + filtro temporal | — |
| Click imposible en modo CURSOR a 15 fps | umbrales tap/drag fijos (0.24/0.32 s) menores que la latencia del filtro temporal | `[fedora] tap_max / drag_min` + KNN rápido (más fps) | probar clicks |
| `intent` en el bus parpadeaba a BLOCKED | `critic_ok` usaba el critic del frame actual, no el veredicto estable | corregido en `pipeline.to_hand_state` | — |

### Cómo capturar las muestras que faltan (20 minutos)

```bash
./launchers/tletl-bank.sh --dataset datasets/tletl_gesture_bank_v2_features.jsonl --dual-hand
```

1. De día, sentado como trabajas. Tecla `2` (FIST), `A` (autosave), mantén
   el puño 40 s moviéndolo un poco. Repite a la distancia de uso de Blender.
2. Tecla `5` (PINCH), `A`, 40 s, variando ángulo de la muñeca.
3. Tecla `7` (NEUTRAL), `A`, 60 s haciendo cosas normales (teclear, taza).
   Esto es lo que baja los falsos positivos.
4. `python -m tools.bank_probe` para ver el balance y
   `python -m tools.duel_lab` para confirmar que la accuracy holdout no bajó.

El banco es sagrado: haz `cp datasets/tletl_gesture_bank_v2_features.jsonl
datasets/backup_$(date +%F).jsonl` antes, y commitea el resultado.

### Gestos propios (más adelante)

Si un día quieres gestos nuevos (p. ej. "OK", "pulgar arriba"), hay dos
caminos honestos: (a) añadir la etiqueta a `LABELS` y capturarla al banco —
el KNN no necesita reentrenar; (b) entrenar un Gesture Recognizer propio en
**MediaPipe Studio** (interfaz web, sin código) y cargarlo como hint. El
paquete `mediapipe-model-maker` dejó de actualizarse y hoy rompe con
TensorFlow reciente; no lo recomiendo.

---

## 5. Los candidatos que descarté, con el motivo exacto

- **WiLoR** (`rolpotamias/WiLoR`): localización + reconstrucción 3D con
  transformer; probado por sus autores con PyTorch 2.0 y CUDA 11.7. En CPU la
  inferencia del ViT no baja de cientos de ms por mano. Contradice la regla 4
  del máster ("todo en CPU, cero CUDA, cero modelos grandes").
- **HaMeR** (`geopavlakos/hamer`): mismo perfil (ViT-H, MANO, GPU). Además
  exige registrarse para descargar el modelo MANO.
- **rtmlib / RTMPose-hand** (`Tau-J/rtmlib`): sí corre en CPU con ONNX
  Runtime u OpenVINO y da 21 puntos en el orden COCO-WholeBody (compatible
  con el índice de MediaPipe). Lo que pierde es la **z**: la mitad de las
  features de orientación (`palm_normal_*`, `*_depth_*`, `*_tip_z_mcp`) se
  vuelven cero y el critic de orientación habría que desactivarlo. Necesita
  además un detector de manos separado (RTMDet). Es un buen plan B si algún
  día MediaPipe falla con oclusiones fuertes, no un plan A.
- **YOLO11-pose + hand-keypoints** (Ultralytics): un solo modelo, 21 puntos
  etiquetados con MediaPipe (misma topología), CPU razonable con `n`. Misma
  limitación 2D, y la licencia AGPL-3.0 obliga a liberar tu código si lo
  distribuyes. Plan C.
- **Hardware** (Ultraleap/Leap Motion, cámaras de profundidad): mejor señal,
  pero cambia el proyecto entero (SDK propio, otro banco). Fuera de alcance.
- **Redes de baja luz** (Zero-DCE, SCI): sustituirían CLAHE+gamma. Son
  pequeñas y correrían en CPU, pero no hay evidencia de que la baja luz sea
  hoy tu problema (la validación fue de día y de noche y el detector vio la
  mano). No lo haría todavía.

---

## 6. Lo que necesito de ti

1. ~~Visto bueno a Hand Landmarker~~ **dado y aplicado.** Falta tu prueba con
   cámara: `./launchers/tletl-fedora-safe.sh` y confirmar `backend:tasks` en
   el panel.
2. **Visto bueno a `gesture_hint = true`** tras probarlo en dry-run.
3. **Sesión de captura** de FIST/PINCH/NEUTRAL de día (§4).
4. Si quieres el plan B (rtmlib) o C (YOLO), dime y lo adapto como backend;
   es un día de trabajo cada uno y ninguno mejora la z.

Fuentes consultadas: repositorio y paper de WiLoR (`github.com/rolpotamias/WiLoR`,
arXiv 2409.12259); HaMeR (`github.com/geopavlakos/hamer`); rtmlib
(`github.com/Tau-J/rtmlib`); dataset hand-keypoints de Ultralytics
(`docs.ultralytics.com/datasets/pose/hand-keypoints`); guía de Hand
Landmarker y de personalización del Gesture Recognizer de Google AI Edge
(`ai.google.dev/edge/mediapipe`); notas de versiones de mediapipe en GitHub.
