# Tletl v5

Control gestual por cámara, todo en CPU, para **Fedora** (navegador, cursor,
ventanas), **Blender** (mover, rotar, escalar y crear objetos) y **CAD**
(AutoCAD por COM desde Windows, DXF offline, FreeCAD). Detecta la mano con
MediaPipe, extrae 67 features geométricas, clasifica el gesto con un KNN
sobre un banco de muestras propio, lo pasa por tres capas de seguridad (guard +
critic + filtro temporal) y publica una **intención abstracta** en un bus JSON
que cada app ejecuta a su manera.

Versión del core: **5.2.0**. Suite: **487 tests verdes** sin cámara.
Documentación completa (todo el repo, módulo por módulo):
[`docs/DOCUMENTACION_TLETL_V5.md`](docs/DOCUMENTACION_TLETL_V5.md).
Cambios de esta versión: [`docs/CHANGELOG_v5.2.md`](docs/CHANGELOG_v5.2.md).

## Arquitectura (flujo de datos)

```
cámara → flip → [lowlight.enhance] → HandTracker (MediaPipe Hand Landmarker, Tasks API; legacy mp.solutions de respaldo)
   → por cada mano: 21 landmarks → features.extract_live_features (67 features)
   → pipeline.process_features(features, hand, hint?):
        classifier.predict        KNN k=13 sobre el banco, vectorizado (≈1 ms)
        → guard.rule_guard        cruza IA vs regla geométrica (+ hint opcional del Gesture Recognizer)
        → critic.strict_critic    umbral por gesto ([critic.min_conf]) + orientación coherente
        → temporal.TemporalFilter ventana 7 / mínimo 4, UNO por mano
        → [adaptive.observe]      OFF por default
   → TletlFrameState{dom, mod, intent} ← intent.gesture_to_common_intent
   → bus.write(state) → ~/.tletl/tletl_state.json  (+ UDP opcional)
        ├─→ apps/fedora_control : gesto de la mano dominante → ydotool
        ├─→ apps/blender_control: addon (timer) → bpy / bmesh
        └─→ apps/autocad_control: GestureModeler → AutoCAD COM | DXF (ezdxf) | FreeCAD
```

**La regla del core (no se rompe):** `tletl_core/` jamás importa ventanas de
OpenCV, ydotool, bpy ni mediapipe. El core decide la intención; la app la
ejecuta. (`lowlight.py` usa `cv2` solo para procesamiento de imagen puro.)

## Instalación

```bash
cd tletl_v5
python3.12 -m venv .venv && source .venv/bin/activate   # mediapipe requiere Python 3.12
pip install -r requirements.txt                           # numpy, opencv-python, mediapipe (sin tope)
./launchers/tletl-fetch-models.sh                         # hand_landmarker.task + gesture_recognizer.task -> ~/.tletl/models/
./launchers/tletl-health.sh                               # debe decir backend=tasks y "healthcheck OK"
```

- mediapipe ≥ 0.10.31 y 1.x ya no traen `mp.solutions`; el detector principal
  es el **Hand Landmarker de la Tasks API** (backend `tasks`). Si el modelo
  falta, la app lo descarga sola (`[tracker] auto_download = true`, ~8 MB).
- El core y sus tests **no necesitan mediapipe**; solo la app en vivo y la
  captura del banco.
- Control real de Fedora: `sudo dnf install ydotool`, `systemctl --user enable
  --now ydotool`, `sudo usermod -aG input $USER` (y reiniciar sesión).
- Opcionales: `pip install ezdxf` (DXF), `pip install pywin32` (solo en el
  Windows con AutoCAD), `sudo dnf install freecad`.

## Uso

```bash
./launchers/tletl-health.sh                 # core + config + rutas + detector + banco + intent
./launchers/tletl-fedora-safe.sh            # DRY-RUN con ventana: valida gestos y el backend sin tocar nada
./launchers/tletl-fedora.sh                 # control real de Fedora (ydotool)
./launchers/tletl-blender-bus.sh            # alimenta el bus para Blender/CAD: dry-run y SIN ventana (--window para verla)
./launchers/tletl-bank.sh --dual-hand       # capturar gestos propios al banco
./launchers/tletl-fetch-models.sh --check   # ¿están los modelos de MediaPipe?
./launchers/tletl-build-blender-addon.sh    # -> dist/tletl_blender_addon-5.2.0.zip
./launchers/tletl-cad-session.sh --export modelo.dxf --units mm   # modelar sólidos por gestos y exportar DXF
```

Cualquier argumento extra va a la app: `--camera 1`, `--backend tasks`,
`--gesture-hint`, `--headless`/`--window`, `--proc-width 0`, `--fourcc ""`…

Diagnóstico sin cámara:

```bash
python -m tools.bank_probe                       # distribución y balance del banco
python -m tools.duel_lab                         # accuracy holdout del KNN por configuración
python -m tools.bench_tracker --synthetic        # ms/frame del detector
python -m pytest tests/                          # suite completa (487)
```

## Gestos

Siete etiquetas: `OPEN_PALM`, `FIST`, `POINT`, `VICTORY`, `PINCH`, `THREE`,
`NEUTRAL` (mano relajada que no actúa). La **mano dominante**
(`[fedora].dominant_hand`, `Right`) controla; la otra (`mod`) va al bus para
Blender/CAD.

### App Fedora

El control arranca **OFF**. `OPEN_PALM` sostenido 1.25 s lo enciende (2 s lo
apaga). `FIST` sostenido 0.35 s = pausa de seguridad. `THREE` 0.85 s = siguiente
modo. Teclas: `q`/`ESC` salir, `t` toggle, `m` modo.

| Modo | POINT | PINCH | VICTORY + swipe | OPEN_PALM + swipe |
|---|---|---|---|---|
| **NAVEGADOR** | scroll ↑/↓ o PageUp/PageDown según la altura de la mano | click | pestaña ← / → | atrás / adelante |
| **CURSOR** | mover cursor | tap ≤ 0.35 s = click; ≥ 0.40 s = arrastre | — | mover cursor |
| **VENTANAS** | — | Enter | — | overview (swipe ↑) |

### Blender (modos del addon; detalle en `apps/blender_control/README.md`)

`THREE` sostenido 0.8 s alterna **TRANSFORM ⇄ CREATE**. `FIST` es seguridad en
ambos.

| Modo | Gesto | Efecto |
|---|---|---|
| TRANSFORM | dom `PINCH` | arrastra el objeto activo en XY (Gain 10, suavizado real 0.5) |
| | dom `OPEN_PALM` | suelta sin salto |
| | mod `PINCH` horizontal / vertical | rota en Z / eleva en Z (Z gain 0 = apagado) |
| | dom `PINCH` + mod `OPEN_PALM` | escala uniforme por distancia entre manos (respeta escalas no uniformes) |
| CREATE | dom `PINCH` | crea la primitiva actual en la palma (cube → sphere → cylinder → cone → plane) |
| | dom `VICTORY` | cicla la primitiva |

### CAD (detalle en `apps/autocad_control/README.md`)

Mismos gestos que Blender con sólidos box → cylinder → sphere → cone → wedge;
`FIST` 3 s termina la sesión y exporta el DXF. Tres rutas: AutoCAD en Windows
por UDP + COM, DXF offline en Fedora (ezdxf), macro de FreeCAD.

## Configuración

Todo en `config/tletl.toml`; cada clave con default, significado y variable de
entorno en la [documentación §4](docs/DOCUMENTACION_TLETL_V5.md#4-configuración-completa).
Precedencia: flags de la CLI > env vars > toml > defaults de `tletl_core/config.py`.

| Sección | Qué controla |
|---|---|
| `[classifier]` | `k`, `orientation_weight`, `strict`, `min_confidence`, `min_margin` del KNN |
| `[guard]` | umbrales y lista de gestos peligrosos del cruce IA vs regla |
| `[critic]`, `[critic.min_conf]` | orientación requerida y confianza mínima **por gesto** (PINCH 0.62) |
| `[temporal]` | tamaño de ventana (7) y mínimo (4) del filtro |
| `[lowlight]` | CLAHE + gamma (`gamma_dark` 1.6) |
| `[adaptive]` | aprendizaje en vivo (OFF), cap, ruta, `save_interval` |
| `[fedora]` | mano dominante, `dry_run`, `safety_hold` 0.35, `tap_max` 0.35, `drag_min` 0.40, `action_conf` 0.52 |
| `[bus]` | ruta (`""` = `~/.tletl/tletl_state.json`), `include_features`, `udp_target` |
| `[camera]` | índice, 1280×720@30, `max_num_hands` 2, `proc_width` 640, `headless`, `fourcc` MJPG |
| `[tracker]` | `backend` auto/tasks/legacy, `model_path`, `auto_download`, `det_conf`/`track_conf` 0.72, `gesture_hint` (OFF), `gesture_model_path`, `gesture_hint_min_score` |
| `[paths]` | ruta del banco |

Env vars principales: `TLETL_DRY_RUN`, `TLETL_HEADLESS`, `TLETL_CAMERA`,
`TLETL_STATE_PATH`, `TLETL_HOME`, `TLETL_BUS_UDP`, `TLETL_TRACKER_BACKEND`,
`TLETL_TRACKER_MODEL`, `TLETL_GESTURE_HINT`, `TLETL_ADAPTIVE`, `TLETL_LOWLIGHT`,
`TLETL_AI_V47_K`, `TLETL_GESTURE_BANK`.

## Rutas de runtime

Nada de runtime se escribe en el repo. Una sola regla (`tletl_core/paths.py`),
replicada en el lector de Blender y en la macro de FreeCAD:

```
$TLETL_HOME (default ~/.tletl)/
   tletl_state.json              el bus   ($TLETL_STATE_PATH lo redirige)
   tletl_adaptive_runtime.json   memoria adaptativa ($TLETL_ADAPTIVE_PATH)
   models/hand_landmarker.task   modelos de MediaPipe ($TLETL_TRACKER_MODEL)
   models/gesture_recognizer.task
```

El banco vive en `datasets/tletl_gesture_bank_v2_features.jsonl` (2 652
muestras; es sagrado: respaldo antes de tocarlo).

## Estado

| Capacidad | Código y tests | Validado en vivo |
|---|---|---|
| 3 modos Fedora (NAVEGADOR/CURSOR/VENTANAS) | ✅ 487 tests, smoke del bucle con cámara falsa | ✅ informal en v5.1 (2026-07-02); **v5.2 pendiente** (click/drag, pausa FIST, fps) |
| Detector MediaPipe Tasks (HandLandmarker) | ✅ modelo real cargado en tests, 11.5 ms/frame CPU | ⏳ falta cámara real |
| Dual-hand (dom + mod, filtros independientes) | ✅ | ✅ v5.1 |
| Guard + critic + temporal | ✅ | ⚠️ v5.1: 5/7 gestos perfectos; FIST y PINCH condición-sensibles; 4 FP en 2 min |
| KNN vectorizado (58 ms → 1 ms) | ✅ idéntico bit a bit | ⏳ fps reales por medir |
| Low-light (CLAHE + gamma) | ✅ bug de gamma corregido | ⏳ |
| Gesture Recognizer como hint | ✅ OFF por default | ⏳ decisión del usuario |
| Banco (captura `--dual-hand`, Tasks o legacy) | ✅ | ⏳ faltan muestras diurnas de FIST/PINCH |
| Blender addon (zip, TRANSFORM/CREATE, suavizado real, escala no uniforme, bus obsoleto) | ✅ con bpy/bmesh falsos + bus real | ✅ mover/rotar/escalar/FIST en v5.1; **v5.2 pendiente** |
| CAD ruta 2: DXF offline (ezdxf) | ✅ export + relectura + audit | ⏳ abrir el DXF en un CAD |
| CAD ruta 1: AutoCAD por UDP + COM (Windows) | ✅ contra un ModelSpace falso | ❌ nunca ejecutado contra AutoCAD |
| CAD ruta 3: macro FreeCAD | ✅ helpers puros | ❌ nunca ejecutado dentro de FreeCAD |
| Adaptive (aprendizaje en vivo) | ✅ ahora sí alimenta al KNN | ⚠️ experimental, **OFF por default** |

Qué falta probar, checklist en vivo y troubleshooting: [documentación §12](docs/DOCUMENTACION_TLETL_V5.md#12-lo-que-no-está-probado-checklist-en-vivo-y-troubleshooting).
Decisiones pendientes (hint, banco diurno, planes B/C): [§13](docs/DOCUMENTACION_TLETL_V5.md#13-decisiones-pendientes-del-usuario)
y [`docs/RECOMENDACIONES_DETECCION_MANOS.md`](docs/RECOMENDACIONES_DETECCION_MANOS.md).
Validación física de v5.1: [`docs/VALIDACION_FISICA_v5.md`](docs/VALIDACION_FISICA_v5.md).

## Tests

```bash
python -m pytest tests/         # 487 tests; ninguno necesita cámara, Blender ni CAD
python -m pytest tests/ -rs     # muestra los saltados (3 usan el modelo real de MediaPipe; 56 necesitan ezdxf)
```
