# Tletl v5 — documentación completa del repositorio

Versión del core: **5.2.0** (`tletl_core/__init__.py`). Fecha: 2026-09-30.
Estado de la suite: **487 tests verdes** (`python -m pytest tests/`), sin cámara,
sin Blender, sin AutoCAD ni FreeCAD en el entorno donde se escribió esto.

Este documento describe el repositorio tal como está en el commit `04b3b09`,
módulo por módulo, leído del código y verificado ejecutando las herramientas.
Donde algo no se pudo ejecutar aquí (cámara, Blender real, COM de AutoCAD,
FreeCAD) se dice explícitamente. Los otros documentos del repo no se
duplican; se enlazan:

- `docs/VALIDACION_FISICA_v5.md` — la hoja de validación con cámara del 2026-07-02 (v5.1).
- `docs/RECOMENDACIONES_DETECCION_MANOS.md` — la decisión sobre MediaPipe Tasks, el hint y el banco.
- `docs/MASTER_PROMPT_TLETL_V5.md` — el prompt de auditoría original (reglas duras del proyecto).
- `docs/CHANGELOG_v5.2.md` — qué cambió en v5.2, por commit.
- `apps/blender_control/README.md` y `apps/autocad_control/README.md` — guías de cada app.

## Índice

1. [Qué es Tletl y cómo funciona](#1-qué-es-tletl-y-cómo-funciona)
2. [Mapa del repositorio](#2-mapa-del-repositorio)
3. [Instalación y puesta en marcha](#3-instalación-y-puesta-en-marcha)
4. [Configuración completa](#4-configuración-completa)
5. [El core, módulo por módulo](#5-el-core-módulo-por-módulo)
6. [El bus](#6-el-bus)
7. [Las apps](#7-las-apps)
8. [Herramientas y launchers](#8-herramientas-y-launchers)
9. [Tests](#9-tests)
10. [Rendimiento](#10-rendimiento)
11. [Bugs corregidos en v5.2](#11-bugs-corregidos-en-v52)
12. [Lo que NO está probado, checklist en vivo y troubleshooting](#12-lo-que-no-está-probado-checklist-en-vivo-y-troubleshooting)
13. [Decisiones pendientes del usuario](#13-decisiones-pendientes-del-usuario)

---

## 1. Qué es Tletl y cómo funciona

Tletl es control gestual por cámara web, todo en CPU. Una app de cámara detecta
una o dos manos con MediaPipe, convierte los 21 landmarks de cada mano en un
vector de features geométricas, clasifica el gesto con un KNN sobre un banco de
muestras propio, pasa el resultado por tres capas de seguridad (guard, critic,
filtro temporal) y publica el resultado en un archivo JSON ("el bus"). Los
clientes leen ese bus y hacen algo con él: la propia app manda teclas y ratón a
Fedora con `ydotool`; un addon de Blender mueve/rota/escala/crea objetos; y un
paquete CAD modela sólidos (AutoCAD por COM desde Windows, DXF offline, o
FreeCAD).

Hay una sola regla que gobierna la estructura: **`tletl_core/` decide la
intención y nunca ejecuta nada del sistema**. No importa ventanas de OpenCV,
ydotool, bpy ni mediapipe (la única excepción es `lowlight.py`, que usa `cv2`
solo para procesamiento de imagen). Todo lo que habla con hardware o con una
aplicación vive en `apps/`.

### Flujo de datos (por frame)

```
cámara (cv2.VideoCapture, MJPG 1280x720@30)
  → cv2.flip(frame, 1)                      espejo: la lateralidad de MediaPipe asume selfie
  → resize a [camera].proc_width (640)      solo la copia que ve el detector
  → tletl_core.lowlight.LowLightEnhancer    CLAHE + gamma si la luma media < 112
  → apps.common.hand_tracker.HandTracker    backend "tasks" (HandLandmarker) o "legacy" (mp.solutions)
        → lista de HandDetection(21 Point, handedness, score)
  → apps.fedora_control.main.split_hands    dom = la mano con [fedora].dominant_hand; mod = la otra
  → por cada mano:
      tletl_core.features.extract_live_features(landmarks)  → dict de 67 features
      [apps.common.gesture_hint.GestureHint]                → hint FIST/OPEN_PALM/POINT/VICTORY (opcional, OFF)
      tletl_core.pipeline.TletlPipeline.process_features(features, hand, hint):
          classifier.RobustKNNRuntime.predict   KNN k=13 sobre el banco (vectorizado)
          guard.geometric_rule + rule_guard     regla geométrica vs IA; bloquea gestos peligrosos en conflicto
          critic.strict_critic                  umbral por gesto ([critic.min_conf]) + orientación coherente
          temporal.TemporalFilter               ventana de 7, mínimo 4 iguales; UNO por mano
          [adaptive.AdaptiveGestureMemory]      aprende solo si está encendida (OFF por default)
        → HandResult → pipeline.to_hand_state → TletlHandState
  → tletl_core.intent.gesture_to_common_intent(state)  → TletlIntent (GRAB_OR_SELECT, SAFETY_STOP, …)
  → tletl_core.bus.TletlStateBus.write(state)          → ~/.tletl/tletl_state.json (+ UDP opcional)
        ├─ la propia app: gesto dom + modo → apps.fedora_control.actions.FedoraActions (ydotool)
        ├─ apps/blender_control/tletl_blender_addon.py: timer lee el bus → bpy
        └─ apps/autocad_control/*: GestureModeler → AutoCAD COM / DXF / FreeCAD
```

Vocabulario fijo de gestos (`tletl_core/features.py::LABELS`), en el orden en
que la herramienta del banco los asigna a las teclas 1–7:

| # | Gesto | Qué es | Intent (`intent.py`) |
|---|---|---|---|
| 1 | `OPEN_PALM` | palma abierta, dedos separados | `RELEASE_OR_TOGGLE` |
| 2 | `FIST` | puño cerrado | `SAFETY_STOP` |
| 3 | `POINT` | solo índice extendido | `POINT` |
| 4 | `VICTORY` | índice y medio | `SECONDARY_ACTION` |
| 5 | `PINCH` | pulgar e índice juntos, resto sin cerrar del todo | `GRAB_OR_SELECT` |
| 6 | `THREE` | tres dedos (índice, medio, anular) | `MODE_NEXT` |
| 7 | `NEUTRAL` | mano relajada que NO debe actuar | `IDLE` |

Además existen las etiquetas de estado `NO_HAND` (no hay mano) y, en el KNN,
`UNKNOWN` como `raw_label` cuando no hubo vecinos válidos.

---

## 2. Mapa del repositorio

Todo lo que hay en el repo, con una o dos líneas por archivo. Los archivos
generados (`dist/`, `.venv/`, `__pycache__/`) están en `.gitignore`.

### Raíz

| Archivo | Qué es |
|---|---|
| `README.md` | Portada: instalación, launchers, tablas de gestos, estado. |
| `pyproject.toml` | Metadatos del paquete (`tletl`, `requires-python >= 3.12`), dependencias (`numpy`, `opencv-python`, `mediapipe>=0.10.21` sin tope), extras `dev` (pytest) y `cad` (ezdxf), `pytest.ini_options` (`testpaths = tests`, `addopts = -q`). `version = "5.2.0"`, igual que el core. |
| `requirements.txt` | Lo mismo para `pip install -r`. Explica por qué mediapipe va sin tope: el tracker soporta las dos APIs. |
| `.gitignore` | Basura regenerable (`.venv/`, `dist/`, `__pycache__/`…) y artefactos de runtime por si alguien los apunta al repo (`tletl_state.json`, `tletl_adaptive_runtime.json`, `*.task`, `*.dxf`). Nunca ignora `datasets/` ni `_archive/`. |
| `.claude/settings.local.json` | Permisos locales de la herramienta de sesión. No está en git; no forma parte de Tletl. |

### `config/`

| Archivo | Qué es |
|---|---|
| `config/tletl.toml` | El único archivo de parámetros. Cada clave documentada en §4. Las rutas vacías (`""`) significan "usa el default de `tletl_core/paths.py`". |

### `datasets/`

| Archivo | Qué es |
|---|---|
| `datasets/tletl_gesture_bank_v2_features.jsonl` | El banco de gestos: 2 652 líneas JSON (7.35 MB), una muestra por línea. Es "sagrado": no se regenera ni se limpia sin respaldo. Formato en §5.2. |

### `docs/`

| Archivo | Qué es |
|---|---|
| `docs/DOCUMENTACION_TLETL_V5.md` | Este documento. |
| `docs/CHANGELOG_v5.2.md` | Cambios de v5.2 por área, con hashes. |
| `docs/RECOMENDACIONES_DETECCION_MANOS.md` | Decisión técnica sobre detectores (MediaPipe Tasks aprobado; rtmlib/YOLO como planes B/C; WiLoR/HaMeR descartados), el hint y el plan de captura del banco. |
| `docs/VALIDACION_FISICA_v5.md` | Hoja de validación con cámara del 2026-07-02 sobre v5.1: 5/7 gestos perfectos, FIST y PINCH condición-sensibles, 4 falsos positivos en 2 min, Blender OK. Su §5 es la lista de bugs que motivó v5.2. |
| `docs/MASTER_PROMPT_TLETL_V5.md` | Prompt máster de auditoría (reglas duras, fases F1–F7, criterios de "hecha"). |

### `tletl_core/` — el cerebro (sin ventanas, sin ydotool, sin bpy, sin mediapipe)

| Archivo | Qué es |
|---|---|
| `tletl_core/__init__.py` | `__version__ = "5.2.0"`. |
| `tletl_core/paths.py` | UNA regla para las rutas de runtime: bus, memoria adaptativa, banco, `TLETL_HOME`. |
| `tletl_core/config.py` | `load_config()`: DEFAULTS completos + `config/tletl.toml` (merge profundo un nivel) + env vars (`ENV_MAP`). Helpers de rutas y `critic_min_conf()`. |
| `tletl_core/geometry.py` | `Point`, `dist`, `clamp`, `palm_scale`, `palm_center`, `points_from_mediapipe` (API legacy). |
| `tletl_core/features.py` | `LABELS`, claves meta, `extract_live_features(lm)` (67 features), `get_label`/`get_features` para leer el banco. |
| `tletl_core/orientation.py` | 13 features de orientación de la palma y `orientation_bucket()` (PALM_FRONT / BACK_HAND / SIDE_HAND / UNKNOWN). |
| `tletl_core/classifier.py` | `RobustKNNRuntime`: carga y balancea el banco, z-normaliza, pesa, predice vectorizado con numpy. `Prediction`. |
| `tletl_core/guard.py` | `geometric_rule()` (clasificador por reglas, sin banco) y `rule_guard()` (árbitro IA vs regla). |
| `tletl_core/critic.py` | `strict_critic()`: umbral de confianza por gesto y coherencia de orientación. `CriticResult`. |
| `tletl_core/temporal.py` | `TemporalFilter` (estabiliza), `MotionTracker` (swipes), `CursorDelta`, `Hold`. Relojes inyectables. |
| `tletl_core/adaptive.py` | `AdaptiveGestureMemory`: aprendizaje en vivo, OFF por default, JSON aparte del banco, escritura throttled. |
| `tletl_core/lowlight.py` | `LowLightEnhancer`: CLAHE + gamma por luminancia media (la única parte del core que importa `cv2`). |
| `tletl_core/pipeline.py` | `TletlPipeline`: orquesta classifier → guard → critic → temporal → adaptive. `HandResult`. |
| `tletl_core/intent.py` | `gesture_to_common_intent()`: gesto estable → intención abstracta. |
| `tletl_core/state.py` | Dataclasses del bus: `TletlHandState`, `TletlIntent`, `TletlFrameState`. |
| `tletl_core/bus.py` | `TletlStateBus` (escritura atómica JSON + UDP opcional), `UdpStatePublisher`, `UdpStateReceiver`, `parse_udp_target`. |

### `apps/`

| Archivo | Qué es |
|---|---|
| `apps/__init__.py` | Vacío (paquete). |
| `apps/common/__init__.py` | Docstring: aquí sí se puede importar mediapipe/cv2 (lazy). |
| `apps/common/hand_tracker.py` | `HandTracker` con backends `tasks` (HandLandmarker, principal) y `legacy` (mp.solutions, respaldo). Descarga de modelos, `describe_backend()`, dibujo de landmarks. |
| `apps/common/gesture_hint.py` | `GestureHint`: Gesture Recognizer de MediaPipe como segunda opinión (OFF por default). |
| `apps/fedora_control/__init__.py` | Vacío. |
| `apps/fedora_control/main.py` | La app de cámara: bucle principal, modos NAVEGADOR/CURSOR/VENTANAS, panel, bus, CLI. |
| `apps/fedora_control/actions.py` | `FedoraActions`: adaptador sobre `ydotool` (click, drag, scroll, pestañas, overview, enter). |
| `apps/blender_control/__init__.py` | Vacío. |
| `apps/blender_control/README.md` | Guía del addon (instalación zip, gestos por modo, prefs, troubleshooting). |
| `apps/blender_control/state_reader.py` | Lector del bus sin bpy ni `tletl_core` (Blender no puede importar el core). Replica la regla de rutas. |
| `apps/blender_control/tletl_blender_addon.py` | El addon: lógica pura (`BlenderGestureMapper`, `GestureSession`) + capa bpy (timer, prefs, operadores, panel N). Trae una copia inline del lector por si se instala suelto. |
| `apps/autocad_control/__init__.py` | Docstring de las tres rutas; no importa submódulos a propósito. |
| `apps/autocad_control/README.md` | Guía de las tres rutas CAD, gestos, unidades, limitaciones, troubleshooting. |
| `apps/autocad_control/gesture_modeler.py` | `GestureModeler`, `Scene`, `Solid`: bus → ops de modelado. Puro. |
| `apps/autocad_control/bus_source.py` | `FileBusSource`, `UdpBusSource`, `ScriptedBusSource`, `open_source()`. |
| `apps/autocad_control/dxf_export.py` | `export_scene_dxf()` con ezdxf (MESH por sólido, capa por tipo, R2010) + JSON de escena + CLI de re-export. |
| `apps/autocad_control/session.py` | Ruta 2: bucle poll → modeler → consola → DXF al terminar. `run_loop()` compartido. |
| `apps/autocad_control/autocad_client.py` | Ruta 1: cliente Windows, UDP → COM (pywin32). `AutoCADBackend`, `DryRunBackend`. |
| `apps/autocad_control/freecad_macro.py` | Ruta 3: macro de FreeCAD (QTimer + Part::Box/Cylinder/Sphere/Cone + cuña). `MinimalModeler` de respaldo. |

### `tools/`

| Archivo | Qué es |
|---|---|
| `tools/__init__.py` | Vacío. |
| `tools/healthcheck.py` | Importa el core, resuelve rutas, describe el backend del detector, carga el banco, valida un intent. Exit 0/1. |
| `tools/bank_probe.py` | Estadísticas del banco (distribución, balance, features útiles, problemas). |
| `tools/duel_lab.py` | Compara configuraciones del KNN con holdout honesto (train/test sin solape). |
| `tools/gesture_bank.py` | Captura de muestras al banco con la cámara (teclas 1–7, SPACE, A, Q; `--dual-hand`). |
| `tools/fetch_models.py` | Descarga `hand_landmarker.task` y `gesture_recognizer.task` a `~/.tletl/models/`. |
| `tools/bench_tracker.py` | Mide ms/frame de `HandTracker.process` por backend (sintético o cámara). |
| `tools/build_blender_addon.py` | Empaqueta el addon como `dist/tletl_blender_addon-<versión>.zip` con `blender_manifest.toml`. |

### `launchers/` (todos: `cd` a la raíz, activan `.venv` si existe, `exec python -m …`)

| Script | Qué lanza |
|---|---|
| `launchers/tletl-health.sh` | `tools.healthcheck`. |
| `launchers/tletl-fedora.sh` | `apps.fedora_control.main` (control real vía ydotool). |
| `launchers/tletl-fedora-safe.sh` | Lo mismo con `TLETL_DRY_RUN=1 --dry-run` (no ejecuta nada). |
| `launchers/tletl-blender-bus.sh` | Dry-run **y** `--headless` por default (pasa `--window` para ver la cámara): alimenta el bus para Blender/CAD. |
| `launchers/tletl-bank.sh` | `tools.gesture_bank`. |
| `launchers/tletl-fetch-models.sh` | `tools.fetch_models` (acepta `--dir`, `--force`, `--check`). |
| `launchers/tletl-build-blender-addon.sh` | `tools.build_blender_addon`. |
| `launchers/tletl-cad-session.sh` | `apps.autocad_control.session` (ruta 2, DXF). |

### `tests/` (29 archivos + `conftest.py`, 487 tests; detalle en §9)

`conftest.py` genera landmarks sintéticos por gesto (`make_landmarks`) y expone
el banco real (`bank_path`, `bank_rows`). Ningún test necesita cámara; los
cuatro que usan modelos reales de MediaPipe se saltan solos si no están.

### `_archive/`

| Archivo | Qué es |
|---|---|
| `_archive/MANIFEST.md` | Qué hay en cada tar.gz, de dónde viene cada módulo de v5 (trazabilidad) y cómo restaurar. |
| `_archive/tletl_control_v3.tar.gz` | Fósil: 7 `.py` monolíticos (~8.9k líneas). |
| `_archive/tletl_control_v4_modular.tar.gz` | Primer split limpio. |
| `_archive/tletl_control_v4_1_ai_integrado.tar.gz` | La cantera de v5 (core + apps + backups internos). |
| `_archive/tletl_backups.tar.gz` | Zip del avance "core split" v4.9 con su `.sha256`. |

### Generado (no en git)

| Ruta | Qué es |
|---|---|
| `dist/tletl_blender_addon-5.2.0.zip` | Salida de `tools/build_blender_addon.py`. |
| `~/.tletl/tletl_state.json` | El bus (lo escribe la app). |
| `~/.tletl/tletl_state.json.tmp` | Archivo temporal de la escritura atómica. |
| `~/.tletl/tletl_adaptive_runtime.json` | Memoria adaptativa (solo si `[adaptive] enabled = true`). |
| `~/.tletl/models/*.task` | Modelos de MediaPipe (7.5 MB + 8.0 MB). |
| `.venv/models/*.task` | Los tests buscan los modelos aquí primero (ver §9). |

---

## 3. Instalación y puesta en marcha

### 3.1 Python y dependencias

mediapipe exige Python 3.12 (no hay ruedas para 3.13/3.14 al escribir esto).

```bash
cd tletl_v5
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, opencv-python, mediapipe
pip install pytest                        # o: pip install -e '.[dev]'
pip install ezdxf                         # opcional, solo para exportar DXF (ruta 2 de CAD); o pip install -e '.[cad]'
```

El core y sus tests **no necesitan mediapipe**: trabajan sobre landmarks ya
extraídos. mediapipe solo hace falta para la app en vivo, el banco y
`bench_tracker`. Si solo vas a alimentar el bus sin ventana (`--headless`),
`opencv-python-headless` sirve igual; no instales los dos a la vez.

Versiones con las que se ejecutó todo lo de este documento: Python 3.12.3,
mediapipe 1.0.1, numpy 2.5.3, ezdxf 1.4.4, pytest 9.1.1.

### 3.2 Modelos de MediaPipe (backend `tasks`)

mediapipe ≥ 0.10.31 y 1.x ya no traen `mp.solutions`; el detector principal es
el Hand Landmarker de la Tasks API y necesita un archivo `.task`:

```bash
./launchers/tletl-fetch-models.sh            # descarga hand_landmarker.task y gesture_recognizer.task a ~/.tletl/models/
./launchers/tletl-fetch-models.sh --check    # solo informa qué falta (exit 1 si falta algo)
./launchers/tletl-fetch-models.sh --force    # re-descarga (si un .task quedó corrupto)
```

No es obligatorio correrlo a mano: con `[tracker] auto_download = true`
(default) la app descarga `hand_landmarker.task` la primera vez que falta. La
descarga va a un `.part` y se renombra al final; se rechaza cualquier archivo
menor de 1 MB (un proxy o portal cautivo devuelve HTML con código 200). Las URLs
son las de `storage.googleapis.com/mediapipe-models/...` (float16, `latest`);
`urllib` respeta `HTTPS_PROXY`.

### 3.3 ydotool (solo para control real de Fedora)

La app manda teclas y ratón con `ydotool`, que funciona en Wayland y X11 a
través de `/dev/uinput`:

```bash
sudo dnf install ydotool
systemctl --user enable --now ydotool      # o el servicio de sistema: sudo systemctl enable --now ydotool
sudo usermod -aG input $USER               # el paquete de Fedora trae la regla udev para el grupo input
# cerrar sesión y volver a entrar; comprobar:
ydotool mousemove -x 5 -y 0                # debe mover el cursor
```

Si `ydotool` no está en `PATH`, la app avisa **una vez** al arrancar y todas las
acciones son no-op (no un error por frame). Si está pero falla (daemon apagado,
sin permiso sobre uinput), imprime un aviso con la causa probable y cuenta los
fallos (`FedoraActions.failures`). Con `--dry-run` no se necesita ydotool.

### 3.4 Comprobación

```bash
./launchers/tletl-health.sh
```

Salida esperada (real, de este entorno):

```
== Tletl v5 healthcheck ==
[ok] core importado (v5.2.0)
[ok] config cargada: k=13, min_conf=0.47, min_margin=0.18, adaptive=False, lowlight=True
     critic min_conf: {'OPEN_PALM': 0.7, 'FIST': 0.68, 'POINT': 0.69, 'VICTORY': 0.72, 'PINCH': 0.62, 'THREE': 0.72, 'NEUTRAL': 0.55}
     banco:    /home/user/tletl_v5/datasets/tletl_gesture_bank_v2_features.jsonl
     bus:      /root/.tletl/tletl_state.json
     adaptive: /root/.tletl/tletl_adaptive_runtime.json
     tracker:  detector backend=tasks (Tasks API disponible) | mediapipe 1.0.1 | tasks:sí legacy:no | modelo /root/.tletl/models/hand_landmarker.task (FALTA) | auto_download:on | gesture_hint:off
[ok] banco: 2363 muestras (balanceadas), 67 features útiles
     labels: {'OPEN_PALM': 328, 'FIST': 355, 'POINT': 355, 'VICTORY': 284, 'PINCH': 355, 'THREE': 331, 'NEUTRAL': 355}
[ok] intent dummy: PINCH -> GRAB_OR_SELECT
== healthcheck OK ==
```

"modelo … (FALTA)" no es fallo del healthcheck: dice que el `.task` aún no
está y que `auto_download` lo bajará.

### 3.5 Arrancar

```bash
./launchers/tletl-fedora-safe.sh              # dry-run con ventana: valida gestos y el backend ("backend:tasks" en el panel)
./launchers/tletl-fedora.sh                   # control real de Fedora
./launchers/tletl-blender-bus.sh              # alimenta el bus para Blender/CAD sin ventana y sin tocar Fedora
./launchers/tletl-blender-bus.sh --window     # igual, con ventana (para calibrar)
./launchers/tletl-bank.sh --dual-hand         # capturar muestras al banco
```

Cualquier argumento extra va a la app (`--camera 1`, `--backend tasks`,
`--gesture-hint`, `--proc-width 0`…); ver §7.1.

### 3.6 Blender

```bash
./launchers/tletl-build-blender-addon.sh      # -> dist/tletl_blender_addon-5.2.0.zip
```

Instalar el zip en Blender: *Edit > Preferences > Add-ons > (˅) Install from
Disk…* (4.0–4.1 o como addon legacy) o *Get Extensions > (˅) Install from
Disk…* (4.2+ como extensión; el manifiesto pide `blender_version_min = 4.2.0`).
Activar "Tletl Gesture Control", en la vista 3D `N` > pestaña **Tletl** >
*Iniciar Tletl* con un objeto seleccionado. **No instales el `.py` suelto**:
desde v5.2 funciona (copia inline del lector), pero el zip es la vía soportada.
Detalle en `apps/blender_control/README.md`.

### 3.7 CAD (opcional)

- **DXF en Fedora** (probado aquí): `pip install ezdxf`, luego
  `./launchers/tletl-cad-session.sh --export modelo.dxf --units mm` con el bus
  corriendo en otra terminal.
- **AutoCAD en Windows** (no probado aquí): en Windows, Python 3.12 +
  `pip install pywin32` + el repo clonado; en Fedora `[bus] udp_target =
  "IP_WINDOWS:5055"`; en Windows `python -m apps.autocad_control.autocad_client
  --udp 0.0.0.0:5055 --dry-run` primero.
- **FreeCAD en Fedora** (no ejecutado aquí): `sudo dnf install freecad`,
  `export TLETL_REPO=/ruta/tletl_v5`, ejecutar `freecad_macro.py` desde
  *Macro > Macros…*.

Detalle en `apps/autocad_control/README.md`.

---

## 4. Configuración completa

Fuente única: `config/tletl.toml`. Precedencia, de menor a mayor:
`tletl_core/config.py::DEFAULTS` → `config/tletl.toml` → variables de entorno
(`ENV_MAP`) → flags de la CLI de la app (`apps/fedora_control/main.py::apply_cli_overrides`).
`load_config()` **siempre** devuelve todas las secciones y claves de `DEFAULTS`
(las apps indexan sin `.get()` defensivo); las tablas anidadas (`[critic.min_conf]`)
se fusionan clave a clave, así un toml que solo cambie `PINCH` conserva el resto.
Las env vars booleanas aceptan cualquier valor salvo `""`, `0`, `false`, `no`,
`off` (`_flag`). Una env var que no convierte (p. ej. `TLETL_AI_V47_K=abc`) se
ignora en silencio.

Verificado clave a clave contra `DEFAULTS` y `ENV_MAP` de `tletl_core/config.py`
y contra el toml del repo (coinciden).

### `[classifier]`

| Clave | Default | Significado | Env |
|---|---|---|---|
| `k` | 13 | vecinos del KNN | `TLETL_AI_V47_K` |
| `orientation_weight` | 0.45 | peso de las features de orientación/profundidad en la distancia | `TLETL_AI_V47_ORIENTATION_WEIGHT` |
| `strict` | true | aplicar los umbrales de confianza/margen del KNN (`predict(strict=...)`) | — |
| `min_confidence` | 0.47 | por debajo, el KNN devuelve `NEUTRAL` con razón `LOW_CONFIDENCE` | — |
| `min_margin` | 0.18 | margen mínimo entre top-1 y top-2; por debajo `LOW_MARGIN` | — |

### `[guard]`

| Clave | Default | Significado | Env |
|---|---|---|---|
| `enabled` | true | activa `rule_guard` (si no, `guard_reason = "GUARD_OFF"`) | — |
| `conf_threshold` | 0.55 | la IA debe superar esto (o el margen) para imponerse a la regla geométrica | — |
| `margin_threshold` | 0.20 | ídem para el margen | — |
| `dangerous` | `["PINCH","POINT","VICTORY","THREE","FIST"]` | gestos que se bloquean a `NEUTRAL` si IA y regla discrepan con confianza baja | — |

### `[critic]` y `[critic.min_conf]`

| Clave | Default | Significado | Env |
|---|---|---|---|
| `enabled` | true | activa `strict_critic` (si no, `critic_reason = "CRITIC_OFF"`) | — |
| `require_orientation` | true | exige orientación coherente para gestos de acción | — |
| `min_conf.OPEN_PALM` | 0.70 | confianza mínima del KNN para aceptar el gesto | — |
| `min_conf.FIST` | 0.68 | | — |
| `min_conf.POINT` | 0.69 | | — |
| `min_conf.VICTORY` | 0.72 | | — |
| `min_conf.PINCH` | **0.62** | bajó de 0.72 por la validación diurna (§11) | — |
| `min_conf.THREE` | 0.72 | | — |
| `min_conf.NEUTRAL` | 0.55 | está en la tabla pero no se usa: el critic acepta `NEUTRAL` siempre | — |

### `[temporal]`

| Clave | Default | Significado | Env |
|---|---|---|---|
| `size` | 7 | frames de historia por mano | `TLETL_TEMPORAL_SIZE` |
| `min_count` | 4 | cuántos iguales dentro de la ventana para fijar un gesto | `TLETL_TEMPORAL_MIN` |

### `[lowlight]`

| Clave | Default | Significado | Env |
|---|---|---|---|
| `enabled` | true | CLAHE + gamma antes del detector | `TLETL_LOWLIGHT` |
| `clip_limit` | 2.0 | `clipLimit` de CLAHE (tiles 8×8) | — |
| `gamma_dark` | 1.6 | gamma del modo más oscuro; los intermedios se derivan (§5.11) | — |

### `[adaptive]`

| Clave | Default | Significado | Env |
|---|---|---|---|
| `enabled` | **false** | aprendizaje en vivo (experimental) | `TLETL_ADAPTIVE` |
| `min_confidence` | 0.78 | solo aprende frames con confianza ≥ esto | — |
| `max_samples_per_gesture` | 120 | cap FIFO por gesto | — |
| `path` | `""` | `""` ⇒ `~/.tletl/tletl_adaptive_runtime.json` | `TLETL_ADAPTIVE_PATH` |
| `save_interval` | 2.0 | segundos mínimos entre escrituras a disco (0 = cada `observe`) | — |

### `[fedora]`

| Clave | Default | Significado | Env |
|---|---|---|---|
| `dominant_hand` | `"Right"` | mano que controla Fedora; la otra va al bus como `mod` | `TLETL_DOMINANT_HAND` |
| `dry_run` | false | imprime `[DRY] …` en vez de ejecutar ydotool | `TLETL_DRY_RUN` |
| `safety_hold` | 0.35 | segundos de FIST estable antes de pausar el control | — |
| `tap_max` | 0.35 | CURSOR: PINCH más corto que esto = click | — |
| `drag_min` | 0.40 | CURSOR: PINCH que alcanza esto = arrastre | — |
| `action_conf` | 0.52 | confianza mínima del KNN para ejecutar una acción | — |

### `[bus]`

| Clave | Default | Significado | Env |
|---|---|---|---|
| `path` | `""` | `""` ⇒ `~/.tletl/tletl_state.json`. Relativa ⇒ relativa a la raíz del repo | `TLETL_STATE_PATH` |
| `include_features` | false | true ⇒ cada mano lleva su dict de features en el bus (triplica el JSON) | — |
| `udp_target` | `""` | `"host:puerto"` para publicar cada frame también por UDP | `TLETL_BUS_UDP` |

### `[blender]`

| Clave | Default | Significado | Env |
|---|---|---|---|
| `bus_path` | `""` | alias LEGACY de `[bus].path` (se lee si `[bus].path` está vacío). Ganancia/suavizado ya no viven aquí sino en las prefs del addon | — |

### `[camera]`

| Clave | Default | Significado | Env |
|---|---|---|---|
| `index` | 0 | índice de `cv2.VideoCapture` | `TLETL_CAMERA` |
| `width` | 1280 | resolución pedida | — |
| `height` | 720 | | — |
| `fps` | 30 | fps pedidos; en headless también limita el bucle | — |
| `max_num_hands` | 2 | manos que el detector busca | — |
| `proc_width` | 640 | ancho al que se reduce la copia que ve el detector (0 = sin reducir) | — |
| `headless` | false | sin ventana OpenCV (solo bus; sin teclas q/t/m) | `TLETL_HEADLESS` |
| `fourcc` | `"MJPG"` | formato V4L2 pedido antes del tamaño; `""` = no tocar | `TLETL_CAMERA_FOURCC` |

### `[tracker]`

| Clave | Default | Significado | Env |
|---|---|---|---|
| `backend` | `"auto"` | `auto` = tasks si la Tasks API importa, si no legacy; `tasks`/`legacy` fuerzan | `TLETL_TRACKER_BACKEND` |
| `model_path` | `""` | ruta a `hand_landmarker.task`; `""` ⇒ `~/.tletl/models/hand_landmarker.task`; si es directorio se le añade el nombre | `TLETL_TRACKER_MODEL` |
| `auto_download` | true | descarga el `.task` si falta | `TLETL_TRACKER_AUTO_DOWNLOAD` |
| `det_conf` | 0.72 | `min_hand_detection_confidence` (y `min_hand_presence_confidence` en tasks) | — |
| `track_conf` | 0.72 | `min_tracking_confidence` | — |
| `model_complexity` | 1 | solo backend legacy (0, 1, 2) | — |
| `gesture_hint` | **false** | segunda opinión con `gesture_recognizer.task` | `TLETL_GESTURE_HINT` |
| `gesture_model_path` | `""` | ruta a `gesture_recognizer.task`; `""` ⇒ junto al modelo de manos | `TLETL_GESTURE_MODEL` |
| `gesture_hint_min_score` | 0.5 | score mínimo del recognizer para tomar su opinión en cuenta | — |

### `[paths]`

| Clave | Default | Significado | Env |
|---|---|---|---|
| `bank` | `""` | `""` ⇒ `datasets/tletl_gesture_bank_v2_features.jsonl` del repo | `TLETL_GESTURE_BANK` |

### Variables de entorno fuera de `ENV_MAP`

| Env | Dónde se lee | Efecto |
|---|---|---|
| `TLETL_HOME` | `tletl_core/paths.py`, `apps/common/hand_tracker.py`, `state_reader.py`, `freecad_macro.py` | directorio base de runtime (default `~/.tletl`): bus, memoria adaptativa y `models/` |
| `TLETL_TRACKER_MODEL`, `TLETL_GESTURE_MODEL` | además de `ENV_MAP`, `hand_tracker.resolve_model_path` / `gesture_hint.resolve_gesture_model_path` las vuelven a leer directamente | la env var gana sobre el toml por dos vías; no hay flag CLI para estas rutas, así que no hay conflicto |
| `TLETL_REPO` | `apps/autocad_control/freecad_macro.py` | ruta del repo para que la macro use el `GestureModeler` completo |
| `GLOG_minloglevel` | `hand_tracker._import_mediapipe` la fija a `2` si no existe | baja el ruido de glog de mediapipe (en 1.x los `W0000 inference_feedback_manager` salen igual y son inofensivos) |

Regla de rutas (`tletl_core/paths.py`), idéntica en `state_reader.py` y
`freecad_macro.py` (hay un test que compara las implementaciones):

```
bus       = $TLETL_STATE_PATH    > $TLETL_HOME/tletl_state.json            > ~/.tletl/tletl_state.json
adaptive  = $TLETL_ADAPTIVE_PATH > $TLETL_HOME/tletl_adaptive_runtime.json > ~/.tletl/tletl_adaptive_runtime.json
banco     = $TLETL_GESTURE_BANK  > <repo>/datasets/tletl_gesture_bank_v2_features.jsonl
modelos   = $TLETL_TRACKER_MODEL > [tracker].model_path > $TLETL_HOME/models/hand_landmarker.task
```

Se eligió `~/.tletl` y no `XDG_RUNTIME_DIR` (tmpfs) porque Blender instalado
como Flatpak no ve `/run/user/<uid>`; ese bug ya costó una validación.

---

## 5. El core, módulo por módulo

### 5.1 `geometry.py`

- `Point(x, y, z)` — dataclass; los landmarks del detector llegan así (x, y
  normalizados 0..1 respecto al frame ya espejado, z relativo y ruidoso).
- `dist(a, b)`, `clamp(v, lo, hi)`.
- `palm_scale(lm)` = `max((dist(0,9) + dist(5,17)) / 2, 1e-6)`: la escala con la
  que se normalizan todas las distancias (independiente del tamaño de la mano en
  pantalla).
- `palm_center(lm)` = media de x e y de los landmarks 0, 5, 9, 13, 17. Es lo que
  el bus publica como `palm`.
- `points_from_mediapipe(hand_landmarks)` — convierte el objeto de la API
  legacy (`.landmark`); el backend tasks ya no lo necesita.

### 5.2 `features.py` y el banco

**`extract_live_features(lm) -> dict`** produce **67 claves**, todas
adimensionales (divididas por `palm_scale`):

| Grupo | Claves | Cómo se calculan |
|---|---|---|
| Por dedo (`index`, `middle`, `ring`, `pinky`; tip/pip/mcp = 8/6/5, 12/10/9, 16/14/13, 20/18/17) — 9 × 4 = 36 | `<dedo>_tip_wrist`, `_pip_wrist`, `_mcp_wrist` | distancia de tip/pip/mcp a la muñeca (0) |
| | `<dedo>_tip_mcp`, `_tip_pip` | distancia tip–mcp y tip–pip |
| | `<dedo>_vertical` | `(pip.y − tip.y)`: positivo si la punta está por encima de la falange |
| | `<dedo>_curl` | `dist(tip, muñeca) − dist(pip, muñeca)`: positivo si el dedo está estirado |
| | `<dedo>_tip_y_mcp`, `_tip_z_mcp` | `(mcp.y − tip.y)`, `(tip.z − mcp.z)` |
| Pulgar — 8 | `thumb_tip_wrist`, `thumb_ip_wrist`, `thumb_tip_index_mcp`, `thumb_tip_index_tip`, `thumb_tip_middle_tip`, `thumb_horizontal`, `thumb_vertical`, `thumb_tip_palm` | distancias 4–0, 3–0, 4–5, 4–8, 4–12; \|4.x−3.x\|, \|4.y−3.y\|; 4–9 |
| Aperturas — 8 | `index_middle_spread`, `middle_ring_spread`, `ring_pinky_spread`, `index_pinky_spread`, `index_ring_spread`, `thumb_pinky_spread`, `palm_width`, `wrist_middle_mcp` | distancias entre puntas (8–12, 12–16, 16–20, 8–20, 8–16, 4–20), 5–17, 0–9 |
| Posición — 2 | `palm_center_x`, `palm_center_y` | centro de la palma. **Excluidas del KNN y de la memoria adaptativa** (`EXCLUDED_FEATURES`) |
| Orientación — 13 | ver §5.3 | `compute_orientation(lm)` |

`get_label(obj)` acepta las claves `label`, `gesture`, `target`, `class`, `y`
(en mayúsculas). `get_features(obj)` lee `obj["features"]` si existe (si no, el
propio objeto), descarta las claves de `NON_FEATURE_KEYS` y cualquier valor no
numérico o no finito.

**Formato del banco** (`datasets/tletl_gesture_bank_v2_features.jsonl`): una
línea JSON por muestra:

```json
{"label": "PINCH", "features": {"index_tip_wrist": 1.53, "...": 0.0}, "meta": {"source": "training-lab-v4.6-manual", "...": "..."}, "timestamp": 1777824596.72}
```

Lo que hay hoy en el archivo (medido con `tools/bank_probe.py` y un recuento directo):

| Dato | Valor |
|---|---|
| Líneas / muestras válidas | 2 652 / 2 652 (0 malformadas) |
| Por gesto | OPEN_PALM 328, FIST 380, POINT 355, VICTORY 284, PINCH 571, THREE 331, NEUTRAL 403 |
| Balance min/max | 0.497 |
| Claves por muestra | 69 en todas (67 útiles para el KNN + `palm_center_x/y`) |
| `meta.source` | `training-lab-v4.6-manual` 1 552, `training-lab-v4.6-correction` 817, `ai-duel-lab` 260, `ai-duel-lab-space` 23 |
| `meta.handedness` | ausente en las 2 652 (se añade solo en capturas nuevas con `tools/gesture_bank.py`) |
| Otros campos de `meta` (según origen) | `focus`, `pred`, `conf`, `hard`, `teacher`, `critic_reason`, `critic_suggestions`, `votes`, `specialist` — diagnósticos del laboratorio v4.6/v4.7, no los usa nadie en v5 |

**Un detalle que ningún documento decía y conviene saber**: el banco lo produjo
el extractor de v4.6 y sus claves no son las mismas que las de
`extract_live_features` actual.

- 13 claves están en el banco pero **nunca se producen en vivo**:
  `closed_fingers_mean`, `fist_compact_score`, `index_middle_length_gap`,
  `ring_pinky_length_gap`, `index_tip_palm`, `middle_tip_palm`, `ring_tip_palm`,
  `pinky_tip_palm`, `pinch_cleanliness`, `pinch_index_to_mcp`,
  `pinch_thumb_index`, `pinch_thumb_middle`, `pinch_thumb_ring`.
- 11 claves se producen en vivo pero **no existen en el banco** (y por tanto
  no entran al KNN): `index_ring_spread`, `thumb_pinky_spread`,
  `thumb_tip_palm` y los ocho `<dedo>_tip_y_mcp` / `<dedo>_tip_z_mcp`.
- El KNN usa las **67** claves del banco; una consulta en vivo comparte **54**
  con cada muestra. La distancia se calcula solo sobre claves compartidas y
  exige al menos `max(10, int(67·0.35)) = 23`, así que funciona, y las
  muestras nuevas capturadas con la herramienta (67 claves de vivo) conviven
  con las viejas sin problema. Pero "el banco y el extractor usan las mismas
  67 features" es impreciso: son 54 en común. Las muestras nuevas de
  `tools/gesture_bank.py` sí llevan `meta.handedness`.

Las 21 posiciones de landmarks siguen el orden de MediaPipe (0 muñeca, 1–4
pulgar, 5–8 índice, 9–12 medio, 13–16 anular, 17–20 meñique); ambos backends
del tracker devuelven exactamente eso, por lo que el banco sigue siendo válido
con la Tasks API.

### 5.3 `orientation.py`

`compute_orientation(lm)` devuelve 13 claves (`ORIENTATION_KEYS`):

| Clave | Cálculo |
|---|---|
| `palm_normal_x/y/z` | normal unitaria = `cross(across, up)` con `across = 17→5`, `up = 0→9` |
| `palm_roll` | `atan2(across.y, across.x)` en grados / 180 |
| `palm_yaw` | `atan2(nx, max(\|nz\|, 1e-6))` / 180 |
| `palm_pitch` | `atan2(ny, max(\|nz\|, 1e-6))` / 180 |
| `palm_facing_score` | `clamp((nz + 1) / 2)` |
| `back_hand_score` | `clamp((−nz + 1) / 2)` |
| `side_hand_score` | `clamp(1 − \|nz\|)` |
| `palm_flatness_score` | `clamp(1 − (max z − min z de los MCP 5,9,13,17) / scale)` |
| `wrist_depth_score` | `(z0 − z9) / scale` |
| `thumb_side_score` | `(x4 − x5) / scale` |
| `finger_depth_spread` | `(max − min de z de las puntas 4,8,12,16,20) / scale` |

`orientation_bucket(feat)`: `SIDE_HAND` si `side ≥ max(facing, back)` y
`side > 0.40`; si no `BACK_HAND` si `back ≥ facing` y `back > 0.35`; si no
`PALM_FRONT` si `facing > 0.35`; si no `UNKNOWN`. La z de MediaPipe es
relativa y ruidosa: esto es orientación aproximada, no metrología (lo dice el
propio módulo).

### 5.4 `classifier.py` — el KNN

```python
RobustKNNRuntime(bank_path, k=13, orientation_weight=0.45,
                 min_confidence=0.47, min_margin=0.18,
                 extra_samples=None, verbose=True)
runtime.predict(features: dict, strict=True) -> Prediction
runtime.distance(a: dict, b: dict) -> float        # referencia por muestra (inf si comparten pocas claves)
runtime._predict_reference(features, strict)       # el bucle viejo, solo para tests de equivalencia
```

`Prediction(label, raw_label, confidence, margin, orientation, reason, ok, votes)`:
`raw_label` es lo que ganó el voto; `label` es `raw_label` si `ok`, si no
`NEUTRAL`; `orientation` es `orientation_bucket(features)`.

**Carga (`load()`)**:

1. Lee el JSONL; se queda con las filas cuya etiqueta esté en `LABELS` y con
   ≥ 12 features numéricas (`MIN_FEATURES_PER_SAMPLE`). Añade `extra_samples`
   (memoria adaptativa) validadas con el mismo criterio. Menos de 20 filas ⇒
   `RuntimeError`.
2. Claves útiles = las presentes en al menos `max(10, 25 % de las filas)`
   muestras (hoy 67). Menos de 15 ⇒ `RuntimeError`.
3. **Balanceo**: `cap = max(220, mediana de muestras por clase)` (hoy 355); las
   clases con más muestras se submuestrean con paso uniforme. Resultado hoy:
   2 363 muestras (PINCH 571→355, NEUTRAL 403→355, FIST 380→355).
4. Media y desviación por clave sobre el banco balanceado; `std = max(std, 0.055)`.
5. **Pesos por clave**: 1.0 base; `orientation_weight` (0.45) si la clave es de
   orientación o empieza por `palm_` o contiene `depth`/`roll`/`yaw`/`pitch`;
   ×1.25 si contiene `thumb_tip_index_tip` o `pinch`; ×1.10 si contiene
   `curl`, `vertical`, `tip_mcp` o `tip_wrist`. Distribución hoy: 14 claves a
   0.45, 29 a 1.0, 18 a 1.1, 6 a 1.25.
6. Construye la matriz `F×N` (features × muestras, NaN donde falta la clave).

**Distancia**: `sqrt(Σ w·z² / used)` con `z = (q − s) / std`, sumando solo las
claves finitas en ambos lados; si `used < max(10, int(F·0.35))` (= 23) la
distancia es `inf` y esa muestra no puede ser vecina. Antes era el centinela
finito `1e9`: con features vacías, las primeras k muestras del banco "ganaban"
y el clasificador devolvía `OPEN_PALM` con confianza 1.0.

**Voto**: los k vecinos con distancia finita, ordenados por (distancia, índice
del banco); cada uno vota `1 / (d + 1e-6)` por su etiqueta. `confidence =
top / total`, `margin = (top − second) / total`. En modo `strict`:
`conf < min_confidence` ⇒ `LOW_CONFIDENCE`; `margin < min_margin` ⇒
`LOW_MARGIN`; entre los 5 más cercanos hay menos de 3 con la etiqueta ganadora
⇒ `SPLIT_NEIGHBORS`. Cualquiera de los tres deja `ok=False` y `label=NEUTRAL`.
Sin vecinos válidos (dict vacío, todo NaN) ⇒ `Prediction("NEUTRAL", "UNKNOWN",
0, 0, …, "NO_VOTES", ok=False)`.

**Vectorización (v5.2)**: la distancia a las N muestras se calcula de golpe,
pero la suma sobre features se acumula fila a fila en el mismo orden que el
bucle viejo (no la suma pairwise de numpy) y los k vecinos se eligen con
`lexsort` por (distancia, índice). Resultado: **bit a bit idéntico** al bucle
de referencia (test `test_vectorized_predict_matches_reference` sobre el banco
real; reproducido aquí: 200/200 predicciones idénticas). 58.1 ms → 0.95 ms por
predicción (§10).

### 5.5 `guard.py`

`geometric_rule(features) -> str` — clasificador por reglas, sin banco:

- Sin ninguna clave `<dedo>_vertical/_curl/_tip_mcp` ⇒ `NEUTRAL` (antes un dict
  vacío contaba los cuatro dedos como cerrados y devolvía `FIST`).
- Dedo extendido si `(vertical > 0.08 y curl > 0.03)` o `tip_mcp > 0.55`.
- Ninguno extendido ⇒ `FIST`; los cuatro ⇒ `OPEN_PALM`; índice+medio+anular ⇒
  `THREE`; índice+medio ⇒ `VICTORY`; solo índice ⇒ `POINT`;
  `thumb_tip_index_tip < 0.42` con ≤ 1 dedo largo extendido ⇒ `PINCH`; si no
  `NEUTRAL`.

`rule_guard(prediction, rule_raw, *, dangerous, conf_threshold, margin_threshold) -> (gesto, razón)`:
si `raw_label ∈ dangerous`, la regla dice otra cosa y `(conf < 0.55 o margin <
0.20)` ⇒ `("NEUTRAL", "GUARD_BLOCK_<raw>")`; en cualquier otro caso `(raw, "OK")`.
Es decir: el guard solo interviene cuando la IA no está segura **y** la
geometría la contradice.

### 5.6 `critic.py`

`strict_critic(gesture, confidence, features=None, votes=None, *, require_orientation=True, min_conf=None) -> CriticResult(accepted, gesture, reason, fallback="NEUTRAL")`, en este orden:

1. Gesto fuera de `VALID_GESTURES` ⇒ rechazado.
2. `NEUTRAL` ⇒ **siempre aceptado** (es el reposo y el fallback; así el reposo
   es `IDLE` y no `BLOCKED`).
3. `conf < min_conf[gesto]` ⇒ rechazado, razón `confianza baja para PINCH: 0.61 < 0.62`.
4. Si `require_orientation` y el gesto es de acción (`PINCH, POINT, VICTORY,
   THREE, FIST`): sin las 13 claves de orientación se acepta solo con
   `conf ≥ max(0.82, min_conf + 0.08)`; con ellas, bucket `SIDE_HAND` o
   `UNKNOWN` ⇒ rechazado ("orientación incoherente").
5. Reglas blandas: `OPEN_PALM` con `palm_flatness < 0.34` y `conf < 0.86`, o
   `side > 0.82` y `conf < 0.88` ⇒ rechazado; `VICTORY/THREE/POINT` con
   `finger_depth_spread > 0.78` y `conf < 0.84` ⇒ rechazado; `PINCH` con
   `palm_flatness < 0.12` y `conf < 0.84` ⇒ rechazado.
6. Aceptado, razón `aceptado: PINCH conf=0.87`.

La tabla `min_conf` llega desde `[critic.min_conf]` vía
`config.critic_min_conf(cfg)`. `critic.MIN_CONF` (la tabla del módulo, usada si
llamas `strict_critic` sin `min_conf`) es una copia de
`config.CRITIC_MIN_CONF_DEFAULTS`, así que módulo, config y toml dicen lo mismo
(PINCH 0.62); hay un test que lo garantiza.

### 5.7 `temporal.py`

- `TemporalFilter(size=7, min_count=4).update(pred) -> Prediction`: mete en la
  ventana `raw_label` si `pred.ok`, si no `NEUTRAL`. Mientras la ventana no
  esté llena devuelve `NEUTRAL` con razón `WARMING_UP` (ok=False). Si la
  etiqueta más frecuente aparece ≥ `min_count` veces es el gesto estable: si el
  frame actual es `ok` y coincide, devuelve `ok=True` con la razón del frame;
  si no, `TEMPORAL_HOLD` con `ok = (estable ≠ NEUTRAL)`. Si nada alcanza
  `min_count`, devuelve el último estable con `UNSTABLE_TEMPORAL` (ok=False).
  `reset()` vacía la ventana. Con 7/4 a 30 fps un gesto tarda ≥ 4 frames
  (~0.13 s) en fijarse y un frame suelto no dispara nada.
- `MotionTracker(maxlen=9, clock)`: `update(x, y)` guarda `(t, x, y)`;
  `swipe()` necesita ≥ 5 puntos, respeta un enfriamiento de 0.55 s y, si la
  ventana dura ≤ 0.90 s, devuelve `LEFT/RIGHT` cuando `|dx| > 0.12` y `|dx| >
  1.45·|dy|`, o `UP/DOWN` cuando `|dy| > 0.12` y `|dy| > 1.45·|dx|`; al
  detectar un swipe vacía los puntos.
- `CursorDelta.update(x, y, gain=1550, max_step=70) -> (dx, dy)` en píxeles:
  primer frame 0, zona muerta 0.0045 en normalizado, clamp ±70 px.
- `Hold(clock).progress(name, seconds) -> 0..1`: cuánto lleva sostenido `name`;
  cambiar de nombre reinicia.

Los relojes son inyectables (`clock=`) para que los tests sean deterministas.

### 5.8 `adaptive.py`

`AdaptiveGestureMemory(path, *, enabled=False, min_confidence=0.78, max_samples_per_gesture=120, save_interval=2.0, clock=time.time)`.

- `observe(gesture, features, confidence) -> bool`: no-op si está apagada o
  `confidence < min_confidence`; limpia claves de posición (`palm_center_x/y`,
  `screen_x/y`, `cx`, `cy`) y valores no finitos; cap FIFO por gesto; marca
  `dirty` y guarda si toca.
- Escritura **throttled**: la primera de la sesión es inmediata, las
  siguientes esperan `save_interval`; `flush()` persiste lo pendiente (el
  pipeline lo llama en `close()`); `save_interval=0` escribe en cada `observe`.
  Atómica (`.tmp` + `replace`), crea el directorio padre.
- `samples() -> [(label, features)]`: lo que `RobustKNNRuntime(extra_samples=…)`
  acepta. Antes se guardaba y nadie lo leía: el "aprendizaje en vivo" no tenía
  efecto. `counts()`, `dirty`.
- Formato en disco: `{"gestures": {"FIST": [ {feature: valor, …}, … ]}, "updated_at": <unix>}`.

Sigue siendo experimental y **OFF por default**: si se activa con muestras
malas degrada el KNN (entran al balanceo y a la normalización como si fueran
del banco).

### 5.9 `intent.py`

`gesture_to_common_intent(state: TletlFrameState) -> TletlIntent(name, active, strength, mode, reason)`
mira solo `state.dom`: sin mano ⇒ `NONE`; `critic_ok=False` ⇒ `BLOCKED` con
`reason = critic_reason`; luego la tabla de §1. `strength` es la confianza;
`active` es False para `IDLE`, `BLOCKED`, `NONE`, `UNKNOWN`.

### 5.10 `pipeline.py`

```python
TletlPipeline(bank_path, config=None, adaptive_path=None, *, verbose=True)
pipeline.process_features(features, hand="dom", hint=None) -> HandResult
pipeline.to_hand_state(result, side="unknown", features=None) -> TletlHandState
pipeline.reset(hand=None)      # vacía el filtro temporal de una mano (o de todas)
pipeline.close()               # flush de la memoria adaptativa; también `with TletlPipeline(...) as p:`
```

`HandResult`: `present`, `raw_gesture` (KNN crudo), `guarded_gesture`,
`stable_gesture` (**el que la app debe usar**), `ok` (estable y aceptado),
`confidence`, `margin`, `orientation`, `rule_raw` (veredicto geométrico),
`guard_reason` (`OK`, `GUARD_BLOCK_X`, `OK_HINT_AGREES`, `GUARD_OFF`),
`critic_accepted` y `critic_reason` (del frame actual, diagnóstico), `reason`
(del filtro temporal), `votes`, `hint`.

Orden por mano: (1) `predict(strict=[classifier].strict)`; (2)
`geometric_rule` + `rule_guard`; si viene `hint` y coincide con `raw_label` y
el guard bloqueó, se re-evalúa con `rule_raw := raw_label` (dos detectores
independientes coinciden) y la razón pasa a `OK_HINT_AGREES`; si el hint
discrepa o no viene, nada cambia; (3) `strict_critic` sobre el gesto guardado
con la tabla `[critic.min_conf]`; (4) se construye una `Prediction`
"post-seguridad" (`safe_ok = pred.ok ∧ ¬guard_block ∧ critic.accepted`,
etiqueta `NEUTRAL` si no) y pasa al `TemporalFilter` de esa mano (uno por
`hand`); (5) la memoria adaptativa observa **solo** si el frame actual pasó
toda la seguridad con el mismo gesto que el estable y no es `NEUTRAL/NO_HAND`
(antes, durante un `TEMPORAL_HOLD`, se grababan transiciones etiquetadas con el
gesto estable).

`to_hand_state`: `critic_ok = result.ok` (el veredicto **estable**). Antes era
`result.ok and result.critic_accepted`, que mezclaba el estable con el critic
del frame actual y hacía parpadear `intent` a `BLOCKED` durante un hold
mientras las apps seguían actuando.

### 5.11 `lowlight.py`

`LowLightEnhancer(clip_limit=2.0, gamma_dark=1.6).enhance(frame_bgr) -> frame_bgr`.
Convierte a YCrCb, mide la luma media del canal Y y elige modo:

| Luma media | Modo | Exponente (`y_out = 255·(y/255)^exp`) con `gamma_dark = 1.6` |
|---|---|---|
| ≥ 112 | `normal` | sin tocar (devuelve el mismo frame) |
| 88–112 | `low` | `1 / (1 + 0.6·0.227)` = 0.880 |
| 62–88 | `medium` | `1 / (1 + 0.6·0.53)` = 0.759 |
| < 62 | `dark` | `1 / 1.6` = 0.625 |

Aplica CLAHE (tiles 8×8) al canal Y y luego la LUT del modo (las tres LUT se
calculan una vez en el constructor). `mode_for(luma)`, `last_luma`, `last_mode`
(la app lo muestra como `LowLight:dark`). Exponente < 1 aclara. El bug de v5.1
(§11): las constantes 0.62/0.76/0.88 se pasaban como *gamma* a una LUT
`(i/255)^(1/gamma)`, así que el exponente efectivo era 1.61 y el "realce"
oscurecía. Con el frame texturizado de los tests (luma 29.6): v5.2 → 112.9;
con el bug → 34.4. El commit `ce5ab89` reporta, con otro frame, 29.5 → 16.5
(bug) y → 84.8 (fix); la dirección es la misma.

### 5.12 `state.py`, `paths.py`, `config.py`, `bus.py`

- `state.py`: `TletlHandState`, `TletlIntent`, `TletlFrameState` (campos en §6).
- `paths.py`: `repo_root()`, `tletl_home()`, `default_bus_path()`,
  `default_adaptive_path()`, `default_bank_path()`, `resolve_path(configured,
  default, relative_to=None)`, `ensure_parent(path)`. Constantes `BUS_FILENAME`,
  `ADAPTIVE_FILENAME`, `BANK_RELATIVE`.
- `config.py`: `load_config(path=None)`, `DEFAULTS`, `ENV_MAP`,
  `CRITIC_MIN_CONF_DEFAULTS`, `bus_path_from_config(cfg)` (`TLETL_STATE_PATH` >
  `[bus].path` > `[blender].bus_path` > default), `adaptive_path_from_config`,
  `bank_path_from_config`, `critic_min_conf(cfg)` (garantiza los 7 gestos).
- `bus.py`: §6.

---

## 6. El bus

`tletl_core/bus.py::TletlStateBus(path=None, *, compact=True, include_features=False, udp_target=None)`.

- `write(state)`: serializa, escribe a `<path>.tmp` y hace `os.replace` (los
  lectores nunca ven un archivo a medias). Si falla, guarda `last_error` y
  **relanza** `OSError` (la app lo captura, cuenta `bus_err` y sigue). Después
  publica el mismo payload por UDP si hay `udp_target`.
- `serialize(state)`: acepta `TletlFrameState` o dict (lo copia, no lo muta);
  si `timestamp` es 0/ausente pone `time.time()`; salvo `include_features`,
  vacía `dom.features` y `mod.features`; JSON compacto (`separators=(",", ":")`).
- `read() -> dict`: **nunca lanza**: archivo ausente, vacío, corrupto o JSON no
  objeto ⇒ `{}`.
- `age(now=None) -> float`: segundos desde el `timestamp` del último frame; `inf`
  si no hay bus legible.
- `close()`: cierra el socket UDP.

Dónde vive: `~/.tletl/tletl_state.json` (regla de §4). La app imprime la ruta
al arrancar (`[TLETL] bus -> …`) y la muestra en el panel (`Bus:~/.tletl/…`).

### 6.1 Esquema JSON (lo que se escribe, campo a campo)

Ejemplo real generado con `TletlStateBus` (compacto en disco, ~1.1 KB con dos
manos y sin features; aquí formateado):

```json
{
  "version": 5,
  "app_version": "5.2.0-core-fedora",
  "timestamp": 1790000000.123456,
  "frame_width": 1280, "frame_height": 720, "fps": 29.4,
  "dom": {
    "present": true, "side": "Right",
    "gesture": "PINCH", "raw_gesture": "PINCH", "stable_gesture": "PINCH",
    "orientation": "PALM_FRONT", "confidence": 0.87, "margin": 0.74,
    "critic_ok": true, "critic_reason": "aceptado: PINCH conf=0.87",
    "palm": [0.52, 0.48], "index": null, "middle": null, "thumb": null,
    "velocity": [0.0, 0.0], "features": {}
  },
  "mod": { "...": "misma estructura; present=false y gesture=NO_HAND si no hay segunda mano" },
  "intent": {"name": "GRAB_OR_SELECT", "active": true, "strength": 0.87, "mode": "NAVEGADOR", "reason": "PINCH"},
  "mode": "NAVEGADOR", "action": "Click",
  "selected": true, "grabbed": true, "snap_enabled": false,
  "transform": {"x": 0.0, "y": 0.0, "z": 0.0, "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0, "scale": 1.0},
  "extra": {"swipe": "-", "control": true, "lowlight": "normal", "backend": "tasks", "hint": "", "headless": false}
}
```

| Campo | Tipo | Quién lo pone | Significado |
|---|---|---|---|
| `version` | int | `TletlFrameState` | versión del esquema: **5** |
| `app_version` | str | la app | `"<core>-core-fedora"` (`"5.2.0-core-fedora"`); default del dataclass `"5.2-core"` |
| `timestamp` | float | la app (`time.time()`); el bus lo rellena si viene 0 | unix del frame. Base de la detección de bus obsoleto |
| `frame_width`, `frame_height` | int | la app | tamaño del frame de cámara (no del reducido) |
| `fps` | float | la app | fps suavizados (EMA α=0.15) del bucle |
| `dom`, `mod` | objeto | pipeline + app | mano dominante (la de `[fedora].dominant_hand`) y modificadora |
| `dom.present` | bool | | hay mano |
| `dom.side` | str | | `"Right"`/`"Left"` según MediaPipe (frame ya espejado; no se intercambia); `"unknown"` sin mano |
| `dom.gesture` | str | | **el gesto estable** (igual a `stable_gesture`); `"NO_HAND"` sin mano |
| `dom.raw_gesture` | str | | lo que dijo el KNN crudo (`UNKNOWN` si no hubo vecinos) |
| `dom.stable_gesture` | str | | salida del filtro temporal |
| `dom.orientation` | str | | `PALM_FRONT` / `BACK_HAND` / `SIDE_HAND` / `UNKNOWN` |
| `dom.confidence`, `dom.margin` | float | | del KNN del frame actual |
| `dom.critic_ok` | bool | | veredicto **estable** (`HandResult.ok`); es lo que decide `intent` |
| `dom.critic_reason` | str | | razón del critic del frame actual (diagnóstico) |
| `dom.palm` | [x, y] o null | la app | centro de la palma normalizado 0..1, origen arriba-izquierda, frame espejado. **Es lo que usan Blender y CAD** |
| `dom.index`, `dom.middle`, `dom.thumb` | null | — | reservados; la app v5.2 no los rellena |
| `dom.velocity` | [0.0, 0.0] | — | reservado; no se rellena |
| `dom.features` | objeto | pipeline | `{}` salvo `[bus] include_features = true` (entonces el dict de 67 features) |
| `intent.name` | str | `intent.py` | `NONE`, `BLOCKED`, `IDLE`, `GRAB_OR_SELECT`, `POINT`, `RELEASE_OR_TOGGLE`, `SAFETY_STOP`, `MODE_NEXT`, `SECONDARY_ACTION`, `UNKNOWN` |
| `intent.active` | bool | | True solo para los intents de acción |
| `intent.strength` | float | | confianza de la mano dominante |
| `intent.mode` | str | | copia de `mode` |
| `intent.reason` | str | | gesto, o `critic_reason` si `BLOCKED`, o `NO_HAND` |
| `mode` | str | la app | `NAVEGADOR` / `CURSOR` / `VENTANAS` |
| `action` | str | la app | etiqueta de la última acción (`"Click"`, `"Scroll abajo"`, `"Modo CURSOR"`, `"Pausa seguridad"`, `"-"`) |
| `selected` | bool | la app | control ON/OFF (mismo valor que `extra.control`) |
| `grabbed` | bool | la app | `dom.gesture == PINCH` y estable/aceptado |
| `snap_enabled` | bool | — | reservado, siempre false |
| `transform` | objeto | — | reservado (siempre ceros y scale 1.0); el addon de Blender **no** lo lee, calcula el suyo por deltas de `palm` |
| `extra.swipe` | str | la app | `LEFT/RIGHT/UP/DOWN` o `"-"` |
| `extra.control` | bool | la app | control ON/OFF (el panel de Blender lo muestra; es informativo, el addon actúa igual) |
| `extra.lowlight` | str | la app | `normal/low/medium/dark/off` |
| `extra.backend` | str | la app | `tasks`/`legacy` |
| `extra.hint` | str | la app | hint del Gesture Recognizer para la mano dominante o `""` |
| `extra.headless` | bool | la app | si corre sin ventana |

Los lectores (`state_reader.py`, `gesture_modeler.py`, `freecad_macro.py`)
aceptan `palm` como lista o como `{"x":…, "y":…}` y también el formato antiguo
`{"hands": {"dom": …}}`; `gesture_modeler` toma `stable_gesture` si falta `gesture`.

### 6.2 Obsolescencia

- La app escribe ~30 veces/s. Si se cierra o se congela, el archivo queda con
  el último `timestamp`.
- **Blender**: `bus_status()` compara `timestamp` con el reloj local; si la
  edad supera *Stale after* (1.0 s por default) declara `BUS OBSOLETO` y el
  addon no actúa (resetea la continuidad) hasta que vuelvan frames frescos.
  Sin `timestamp` válido, la edad es `inf`.
- **CAD** (`GestureModeler._is_stale`): solo el **primer** frame se juzga
  contra el reloj local; después solo importa si el `timestamp` **avanza**
  (o retrocede más de 5 s = flujo reiniciado). Así un cliente Windows con el
  reloj desfasado respecto a Fedora no ve todo como obsoleto. Además
  `FileBusSource.poll()` solo entrega el frame si el texto del archivo cambió.
- `bus.age()` sirve para cualquier otro cliente.

### 6.3 UDP

Con `[bus] udp_target = "192.168.1.50:5055"` (o `TLETL_BUS_UDP`) cada `write`
manda además el mismo JSON como datagrama (`UdpStatePublisher`; `SO_BROADCAST`
activado; payload > 65 000 bytes se descarta y se cuenta en `dropped`). El
receptor es `UdpStateReceiver(host="0.0.0.0", port=5055)`; `poll()` no bloquea
y devuelve el **más reciente** (drena la cola; los malformados se cuentan en
`malformed`). Sin cifrado ni autenticación: LAN de confianza. Lo usa
`apps/autocad_control/bus_source.py::UdpBusSource`.

---

## 7. Las apps

### 7.1 Fedora (`apps/fedora_control/main.py`)

Cliente delgado: abre la cámara, corre detector y pipeline, traduce el gesto
**estable** de la mano dominante a `FedoraActions`, dibuja el panel y escribe
el bus. La otra mano solo va al bus (para Blender/CAD).

**CLI** (`python -m apps.fedora_control.main …`; todo default viene del toml,
la CLI sobreescribe):

| Flag | Clave de config que pisa |
|---|---|
| `--bank PATH` | `[paths].bank` / `TLETL_GESTURE_BANK` (relativo al cwd) |
| `--config PATH` | ruta al toml (default `config/tletl.toml`) |
| `--camera N`, `--width`, `--height`, `--fps`, `--proc-width`, `--fourcc` | `[camera].index/width/height/fps/proc_width/fourcc` |
| `--backend {auto,legacy,tasks}`, `--det-conf`, `--track-conf` | `[tracker].backend/det_conf/track_conf` |
| `--action-conf` | `[fedora].action_conf` |
| `--k N` | `[classifier].k` |
| `--dry-run` | `[fedora].dry_run = true` |
| `--headless` / `--window` | `[camera].headless` (`--window` gana) |
| `--gesture-hint` | `[tracker].gesture_hint = true` |

**Arranque**: imprime la línea de `describe_backend` (backend elegido y por
qué, versión de mediapipe, si el modelo está), crea `HandTracker` y, si está
activo, `GestureHint`; abre la cámara (`fourcc` antes del tamaño: en V4L2
decide qué fps ofrece; YUYV suele topar en 10–15 fps a 720p, MJPG da 30) y
reporta lo que la cámara **aceptó** (`obtenido 1280x720@30 MJPG`); crea el bus.

**Estado de control**: arranca **OFF**. Con la mano dominante:

| Gesto sostenido | Efecto | Tiempo |
|---|---|---|
| `OPEN_PALM` | alterna control ON/OFF | 1.25 s para encender, 2.0 s para apagar; ≥ 1 s entre toggles |
| `FIST` (con control ON y conf > 0.55) | **pausa de seguridad** (control OFF, suelta el arrastre) | `[fedora].safety_hold` = 0.35 s de FIST estable, o la mitad si conf ≥ 0.90 |
| `THREE` (control ON) | siguiente modo (NAVEGADOR → CURSOR → VENTANAS → …) | 0.85 s; ≥ 1 s entre cambios |

Una acción solo se ejecuta si `control` está ON, hay mano, el gesto estable es
`ok` y `confidence ≥ [fedora].action_conf` (0.52).

**Gesto → acción por modo** (mano dominante; `cy` = y de la palma, 0 arriba):

| Modo | Gesto | Acción (`FedoraActions` → `ydotool`) | Cadencia |
|---|---|---|---|
| NAVEGADOR | `POINT` con `cy < 0.22` | PageUp (`key 104`) | 0.23 s |
| | `POINT` con `cy > 0.78` | PageDown (`key 109`) | 0.23 s |
| | `POINT` con `cy < 0.39` | scroll arriba (`key 103`, flecha ↑) | 0.11 s |
| | `POINT` con `cy > 0.61` | scroll abajo (`key 108`, flecha ↓) | 0.11 s |
| | `PINCH` | click izquierdo (`click 0xC0`) | 0.34 s |
| | `VICTORY` + swipe → / ← | pestaña siguiente (Ctrl+Tab) / anterior (Ctrl+Shift+Tab) | 0.65 s |
| | `OPEN_PALM` + swipe ← / → | atrás (Alt+←) / adelante (Alt+→); solo si el hold de toggle va < 55 % | 0.75 s |
| CURSOR | `POINT`, `OPEN_PALM` o `PINCH` moviendo la mano | mueve el cursor (`mousemove dx dy`, ganancia 1550, máx. 70 px/frame) | cada frame |
| | `PINCH` ≤ `tap_max` (0.35 s) y soltar | click (`click 0xC0`) | ≥ 0.28 s entre clicks |
| | `PINCH` ≥ `drag_min` (0.40 s) | arrastre: `mouse_down` (`click 0x40`) … `mouse_up` (`click 0x80`) al soltar | — |
| | `PINCH` entre 0.35 y 0.40 s | nada (zona muerta anti-click accidental) | — |
| VENTANAS | `OPEN_PALM` + swipe ↑ | overview de GNOME (`key 125`, Super) | 0.9 s |
| | `PINCH` | Enter (`key 28`) | 0.38 s |

La máquina tap/drag es `PinchClickDrag` (pura, reloj inyectable). Un arrastre
en curso sobrevive a frames inestables mientras el gesto estable siga siendo
`PINCH`; se suelta al perder la mano, apagar el control, cambiar de modo, con
`t`/`m` y en el `finally` del bucle (el botón nunca se queda pulsado). Los
códigos de tecla son los `KEY_*` de Linux (`input-event-codes.h`).

**Teclado** (solo con ventana): `q`/`ESC` salir, `t` alternar control, `m`
siguiente modo. En `--headless` no hay teclas; `Ctrl+C`.

**Panel** (ventana OpenCV, `draw_panel`), líneas:

1. `TLETL v5 ON|OFF | MODO:x | FPS:n | DRY:bool | LowLight:modo | backend:tasks`
2. `DOM[Right]:PINCH raw:PINCH conf:0.87 ok:True hint:-`
3. `MOD[Left]:OPEN_PALM hint:- (al bus para Blender)` o `MOD: (sin segunda mano)`
4. `Guard:OK | Critic:aceptado: … | Intent:GRAB_OR_SELECT | Swipe:-`
5. `Action:Click | Bus:~/.tletl/tletl_state.json age:33ms err:0`
6. `OPEN_PALM hold=ON/OFF | FIST hold=seguridad | THREE=modo | Q/ESC=salir | t=toggle | m=modo`
7. (solo mientras hay un hold en curso) `Toggle: 40%`, `Mode: 70%`, `Seguridad (FIST): 50%`

Más: esqueleto de cada mano (verde dom, naranja mod), círculo amarillo en la
palma dominante con `GESTO conf`, franjas amarillas al 39 %/61 % en NAVEGADOR,
y un círculo verde/rojo arriba a la derecha (control ON/OFF). En headless
imprime cada 2 s: `[TLETL] ON  CURSOR fps:29.8 dom:PINCH:0.87* mod:NO_HAND hint:- intent:GRAB_OR_SELECT bus_err:0`
(`*` = estable/ok) y limita el bucle a `[camera].fps`.

**Errores**: fallo al escribir el bus ⇒ se imprime la primera vez y se cuenta
en el panel (`err:n`), el bucle sigue. `try/finally` libera cámara, ventana,
hint, tracker, bus y pipeline. Sin `ydotool` ⇒ aviso único y no-op.

**Manos**: `split_hands(detections, dominant_label)`: dom = la mano cuya
lateralidad coincide con `[fedora].dominant_hand` (si hay dos iguales, la de
mayor score; si ninguna coincide, la primera); mod = la primera distinta. Una
sola mano siempre es dom. El frame se espeja (`cv2.flip(frame, 1)`) **antes**
de detectar, así la etiqueta Left/Right de MediaPipe sale correcta sin
intercambiarla.

### 7.2 Capa común (`apps/common/`)

**`hand_tracker.py`**:

- `HandTracker.create(cfg_tracker, num_hands=2, *, downloader=None, log=print)`:
  `tasks` ⇒ asegura el modelo (`ensure_model_file`, descarga si
  `auto_download`) y crea `HandLandmarker` en modo VIDEO
  (`num_hands`, `min_hand_detection_confidence = min_hand_presence_confidence =
  det_conf`, `min_tracking_confidence = track_conf`); `legacy` ⇒
  `mp.solutions.hands.Hands(static_image_mode=False, max_num_hands,
  model_complexity, …)`; `auto` ⇒ tasks si la Tasks API importa, y solo si
  crear el landmarker falla **y** existe `mp.solutions` cae a legacy avisando;
  si nada está disponible, `TrackerUnavailable` con la instrucción de
  instalación.
- `tracker.process(rgb, timestamp_ms=None) -> [HandDetection]`: frame RGB
  HxWx3; genera timestamps en ms estrictamente crecientes
  (`MonotonicTimestamps`, obligatorio en modo VIDEO); descarta manos con menos
  de 21 puntos. `HandDetection(landmarks: [Point]×21, handedness "Left"|"Right",
  score)`.
- `describe_backend(cfg) -> dict` y `format_backend_summary(info) -> str`:
  resumen puro sin cargar modelos (lo usan healthcheck, la app y el banco).
- Errores con mensaje: `TrackerModelMissing` (falta el `.task` y no se pudo
  descargar → "Ejecuta ./launchers/tletl-fetch-models.sh"), `TrackerUnavailable`
  (mediapipe no carga; librería nativa ausente → `sudo dnf install mesa-libEGL
  mesa-libGLES`; `.task` corrupto → `--force`).
- `draw_landmarks(frame_bgr, detection, color=…)` reemplaza a
  `mp.solutions.drawing_utils` (21 puntos + `HAND_CONNECTIONS`).

**`gesture_hint.py`**: `GestureHint.create(cfg_tracker, num_hands=2, …) -> GestureHint | None`
(nunca lanza: None si está apagado, falta el modelo o mediapipe no lo carga).
`process(rgb, timestamp_ms) -> [(handedness, gesto_tletl | None)]` mapeando
`Closed_Fist→FIST`, `Open_Palm→OPEN_PALM`, `Pointing_Up→POINT`,
`Victory→VICTORY` (el resto y `None` no opinan) con `score ≥
gesture_hint_min_score`. `hint_for(hints, handedness)` elige el de esa mano.
La app pasa el hint al pipeline con `make_process_features`, que inspecciona
una vez si `process_features` acepta `hint=`.

### 7.3 Blender (`apps/blender_control/`)

Resumen; la guía completa con tablas de gestos, prefs y troubleshooting está en
`apps/blender_control/README.md`.

- El addon (`bl_info` 5.2.0, Blender ≥ 4.0; como extensión ≥ 4.2) registra un
  timer (`bpy.app.timers`, intervalo *Interval* = 0.033 s) que lee el bus con
  `state_reader`, clasifica su estado (`Conectado (N fps app, M ms)`, `BUS
  OBSOLETO`, `Sin bus en …`, `Bus ilegible en …`) y, solo con bus OK, pasa el
  frame por `GestureSession.tick()` y ejecuta los comandos que salen
  (`transform`, `spawn`, `mode`, `primitive`). El timer **nunca muere**: una
  excepción queda en `last_error` y se ve en el panel.
- **Modo TRANSFORM** (objeto activo de tipo controlable: MESH, EMPTY, CURVE,
  LIGHT, CAMERA…): dom `PINCH` arrastra en XY por delta de la palma (×*Gain*
  10.0); dom `OPEN_PALM` suelta sin salto; dom `FIST` congela y descarta el
  suavizado pendiente; mod `PINCH` horizontal rota en Z (*Rot gain* 2.0 ⇒
  `Δx·2·π` rad); mod `PINCH` vertical eleva en Z solo si *Z gain* > 0 (default
  0); dom `PINCH` + mod `OPEN_PALM` escala por la distancia entre manos (*Scale
  gain* 1.5) **como factor sobre los tres ejes** (un objeto (1, 2, 1) se vuelve
  (k, 2k, k); antes se leía `scale.x` y se escribía `(s, s, s)`).
- **Suavizado real**: cada delta se acumula en un *target* y cada tick acerca
  el objeto la fracción *Smoothing* (0.5); con la mano quieta converge al
  desplazamiento completo, no se pierde recorrido. Resincroniza si el usuario
  mueve el objeto a mano (Ctrl+Z, G), con FIST, al soltar y al cambiar de
  objeto. Solo se escriben los canales que cambiaron (no ensucia undo/depsgraph).
  Exige `rotation_mode` Euler; si no, error visible.
- **Modo CREATE**: dom `THREE` ≥ 0.8 s alterna TRANSFORM ⇄ CREATE (una vez por
  sostén); dom `PINCH` en flanco de subida crea la primitiva actual (cube →
  sphere → cylinder → cone → plane; `VICTORY` cicla) con `bmesh` (nunca
  `bpy.ops`: los timers no tienen contexto de operador) en el plano XY
  alrededor del cursor 3D, queda activa y seleccionada.
- Panel N "Tletl": Iniciar/Detener, modo del addon y primitiva, *Crear en el
  cursor*, parámetros (*Gain*, *Smoothing*, *Rot gain*, *Scale gain*, *Z gain*,
  *Stale after*, *Interval*, *Bus path*), estado del bus, control ON/OFF de la
  app, gesto y confianza de cada mano, intent, modo de la app, último error.
- El lector (`state_reader.py`) no importa `tletl_core` (el Python de Blender
  no puede); replica la regla de rutas y hay un test de paridad. El addon
  intenta importarlo como paquete, luego como módulo vecino y, si no, usa una
  copia inline (`STATE_READER_SOURCE` dice cuál).

### 7.4 CAD (`apps/autocad_control/`)

Resumen; detalle, unidades y troubleshooting en `apps/autocad_control/README.md`.

AutoCAD no corre en Linux, así que hay tres rutas con el mismo cerebro puro
(`gesture_modeler.py::GestureModeler`, sin ninguna dependencia de CAD):

| Ruta | Dónde | Cómo | Estado |
|---|---|---|---|
| 1. AutoCAD en vivo | Windows en la LAN | la app publica el bus por UDP (`[bus] udp_target`); `autocad_client.py` lo recibe y aplica ops por COM (pywin32: `AddBox/AddCylinder/AddSphere/AddCone/AddWedge`, `Move`, `Rotate3D`, `ScaleEntity`, `Regen`) | código listo; **no ejecutado contra AutoCAD** (probado con un ModelSpace falso) |
| 2. DXF offline | Fedora | `session.py` lee el bus, modela y al terminar guarda `escena.json` + `.dxf` con ezdxf (MESH por sólido, capas `TLETL_BOX/CYLINDER/SPHERE/CONE/WEDGE`, R2010, `$INSUNITS`) | **probado de punta a punta** (export + relectura + `audit` limpio) |
| 3. FreeCAD en vivo | Fedora | `freecad_macro.py`: QTimer de 50 ms, primitivas `Part::Box/Cylinder/Sphere/Cone` + cuña extruida; usa el `GestureModeler` del repo si `TLETL_REPO` está, si no un `MinimalModeler` inline (sin rotar/escalar) | código listo; **no ejecutado dentro de FreeCAD** (helpers puros probados) |

Gestos (mismos que Blender): `THREE` 0.8 s alterna TRANSFORM/CREATE; `FIST`
seguridad, sostenido 3 s termina la sesión (`--end-hold 0` lo desactiva);
TRANSFORM: dom `PINCH` mueve por delta, mod `PINCH` rota en Z (`Δx·rot_gain·π`),
dom `PINCH` + mod `OPEN_PALM` escala (mín. 0.05×), dom `OPEN_PALM` suelta;
CREATE: dom `PINCH` (flanco) crea box → cylinder → sphere → cone → wedge (`VICTORY`
cicla) en la palma proyectada a z = 0. Un frame aislado nunca mueve nada. Ops:
`("mode", m)`, `("kind", k)`, `("add", Solid)`, `("select", id)`, `("move", id,
dx, dy, dz)`, `("rotate", id, rad)`, `("scale", id, factor)`, `("end_session",)`.
La escena se serializa como `{"format": "tletl-cad-scene", "version": 1, "units",
"next_id", "selected_id", "solids": [{id, kind, center, size, rotation_z, scale}]}`.

CLI común (`session.py` y `autocad_client.py`): `--bus PATH` | `--udp
[HOST:PORT]` (default `0.0.0.0:5055`), `--units {cm,in,m,mm}`, `--gain`
(default por unidades: mm=100, cm=10, m=1, in=4), `--spawn-size` (default
gain/10), `--rot-gain 2.0`, `--scale-gain 1.5`, `--z-gain 0`, `--mode-hold
0.8`, `--end-hold 3.0`, `--stale 1.0`, `--deadzone 0`, `--hz 20`, `--load
escena.json`. `session.py` añade `--export modelo.dxf` (default
`tletl_modelo.dxf`); `autocad_client.py` añade `--dry-run` y `--regen-every 25`.
`dxf_export.py` re-exporta: `python -m apps.autocad_control.dxf_export
escena.json [salida.dxf] [--units …] [--segments 32]`.

---

## 8. Herramientas y launchers

| Herramienta | Qué hace | Flags |
|---|---|---|
| `python -m tools.healthcheck` | importa el core, imprime config efectiva y rutas resueltas (banco, bus, adaptive), describe el backend del detector sin cargarlo, carga el banco (muestras balanceadas, features útiles, conteo por label), valida `PINCH → GRAB_OR_SELECT`. Exit 0/1. | ninguno |
| `python -m tools.bank_probe` | estadísticas del banco: muestras válidas/malformadas, distribución por gesto, balance min/max, features compartidas útiles, labels faltantes, muestras con < 12 features o sin orientación, labels desconocidos. | `--bank PATH` |
| `python -m tools.duel_lab` | holdout honesto: baraja el banco (`--seed 7`), separa `--holdout 0.2` para test, entrena `RobustKNNRuntime` solo con train y mide accuracy (`predict(strict=False)`, `raw_label == label`) para `k=9/0.45`, `k=13/0.45`, `k=13/0.25`, `k=21/0.45`; ordena por accuracy. | `--bank`, `--holdout`, `--seed` |
| `python -m tools.gesture_bank` | captura al banco con la cámara. Teclas `1`–`7` gesto, `SPACE` guarda, `A` autosave (cada `--autosave-interval` 0.8 s), `Q` sale. Escribe `{"label", "features", "meta": {"handedness", …}, "timestamp"}`. Con `--dual-hand` detecta 2 manos y guarda una muestra por mano etiquetada Left/Right. Conteos en memoria (`SampleCounter`), ya no relee el JSONL por frame. Captura fija a 640×480@30. | `--dataset/-d` (default `gesture_bank.jsonl` en el cwd), `--config`, `--camera/-c`, `--width`, `--height`, `--fps`, `--backend {auto,legacy,tasks}`, `--model-complexity`, `--det-conf`, `--track-conf`, `--autosave-interval`, `--dual-hand` |
| `python -m tools.fetch_models` | descarga `hand_landmarker.task` y `gesture_recognizer.task` a `$TLETL_HOME/models`; `.part` + rename; rechaza < 1 MB; nombre desconocido ⇒ exit 2. | `[nombres…]`, `--dir`, `--force`, `--check` |
| `python -m tools.bench_tracker` | ms/frame de `HandTracker.process` (media, p50, p95, máx, manos vistas, "fps solo detector"); frames sintéticos con un blob móvil salvo `--camera N`. | `--synthetic`, `--camera N`, `--frames 100`, `--width 640`, `--height 360`, `--backend {auto,legacy,tasks,all}`, `--num-hands 2` |
| `python -m tools.build_blender_addon` | zip `dist/tletl_blender_addon-<versión>.zip` con `tletl_blender_addon/__init__.py` (copia del addon), `state_reader.py` y `blender_manifest.toml` (id `tletl_blender_addon`, `blender_version_min 4.2.0`, licencia `SPDX:MIT`, tags Object / 3D View). La versión sale de `bl_info` por `ast`, sin importar bpy. | `--out DIR` (default `dist/`), `--print-manifest` |

Los ocho launchers de `launchers/` están en §2; los cuatro que corren la app
aceptan y reenvían cualquier flag de §7.1.

---

## 9. Tests

```bash
source .venv/bin/activate
python -m pytest tests/            # 487 passed en ~22 s (pyproject ya añade -q)
python -m pytest tests/ -rs        # además lista los tests saltados y por qué
```

Estado real de este entorno: **487 passed, 0 skipped**. Baseline al inicio de
v5.2: 112. Ningún test abre una cámara ni necesita Blender/AutoCAD/FreeCAD.
La columna "Tests" es lo que pytest recolecta (`--collect-only`), con los
parametrizados desplegados; suma 487.

| Archivo | Tests | Cubre |
|---|---|---|
| `tests/conftest.py` | — | landmarks sintéticos por gesto (`make_landmarks`), fixtures `bank_path`, `bank_rows`, `open_hand`. |
| `tests/test_geometry.py` | 4 | `clamp`, `dist`, `palm_scale` > 0, `palm_center` en rango. |
| `tests/test_features.py` | 6 | features finitas, orientación presente, PINCH acerca pulgar–índice, `get_label` variantes, `get_features` filtra meta, 7 labels. |
| `tests/test_classifier.py` | 15 | carga del banco real, recupera sus propias muestras, **vectorizado == referencia** (k por defecto y otros k), distancia `inf` con pocas claves, dict vacío ⇒ `NO_VOTES`, umbrales por constructor, `strict=False`, `verbose`. |
| `tests/test_guard.py` | 10 | `geometric_rule` por gesto (incluido dict vacío ⇒ NEUTRAL) y `rule_guard` (bloquea solo peligroso + discrepancia + confianza baja). |
| `tests/test_critic.py` | 19 | umbrales por gesto, NEUTRAL siempre aceptado, orientación requerida/incoherente, reglas blandas, `min_conf` externo y `resolve_min_conf`. |
| `tests/test_temporal.py` | 10 | warm-up y estabilización, un frame no dispara, `CursorDelta`, swipes en 4 direcciones, lento no es swipe, enfriamiento, `Hold` determinista con reloj inyectado. |
| `tests/test_adaptive.py` | 21 | OFF ⇒ nada; guarda con confianza alta; excluye posición; cap; recarga; **primera escritura inmediata y luego throttled**; `flush`; `save_interval=0`; `samples()`; crea el directorio padre. |
| `tests/test_core_adaptive_learning.py` | 6 | `extra_samples` cambian predicción y normalización, se validan, no mutan la entrada; roundtrip memoria → KNN; el pipeline las carga solo si `enabled`. |
| `tests/test_lowlight.py` | 14 | modos por luma, forma/dtype, `gamma_dark=1.6` reproduce 0.62/0.76/0.88 y su salida (±2 niveles), **aclara y más con gamma mayor**, LUT cacheadas, `gamma_dark` inválida. |
| `tests/test_pipeline.py` | 16 | pipeline sobre muestras reales, estabiliza, guard bloquea conflicto, **umbrales del toml cableados**, `critic_ok` sigue el estable (intent no parpadea), hint desbloquea el guard (y no cambia nada si el guard pasaba o está OFF), adaptive aprende solo si el frame actual coincide, `close()`/context manager hacen flush. |
| `tests/test_intent.py` | 5 | NONE sin mano, BLOCKED si critic falla, PINCH/FIST/THREE. |
| `tests/test_dual_hand.py` | 3 | filtros temporales independientes por mano, dos manos clasifican aparte, `TletlFrameState` lleva ambas. |
| `tests/test_edge_cases.py` | 12 | features vacías/NaN en classifier y pipeline, `orientation_bucket({})`, critic con NEUTRAL/desconocido, env overrides, config sin toml, bus con dos manos. |
| `tests/test_config.py` | 9 | DEFAULTS completos, merge profundo de `[critic.min_conf]`, env overrides, helpers de rutas. |
| `tests/test_paths.py` | 9 | regla de rutas (`TLETL_STATE_PATH`, `TLETL_HOME`, defaults), `resolve_path`, memoria adaptativa fuera de `datasets/`. |
| `tests/test_bus.py` | 15 | roundtrip, ausente ⇒ `{}`, atómico sin `.tmp`, crea el directorio, **timestamp real aunque el state traiga 0**, compacto y sin features por default, `include_features`, no muta el dict, corrupto ⇒ `{}`, `parse_udp_target`, **UDP publisher → receiver por loopback**, basura ignorada, payload grande descartado. |
| `tests/test_tools.py` | 8 | `bank_probe` sobre el banco real, `duel_lab` holdout "decente" y ordenado, docstrings reales, defaults siguen las rutas de runtime, `healthcheck` OK y falla con banco inexistente. |
| `tests/test_fetch_models.py` | 12 | `download` con opener falso (nunca red): salta si existe, `--force`, `.part` borrado al fallar, rechazo < 1 MB, `check`, CLI y nombres desconocidos. |
| `tests/test_hand_tracker.py` | 42 | rutas de modelos, `ensure_model_file` (descarga inyectada), selección de backend (`auto`/forzado/errores), conversión de resultados legacy y tasks, lateralidad, `MonotonicTimestamps`, `HandTracker.process` con backend falso, dibujo (`importorskip("cv2")`), `GestureHint` (mapeo, `min_score`, `create` nunca lanza) y **4 tests con el modelo real** (abajo). |
| `tests/test_fedora_app.py` | 27 | importa sin mediapipe, **CLI > toml**, `--window` gana a headless, `FedoraActions` dry-run / sin ydotool / con fallo, `split_hands`, `PinchClickDrag` (tap, drag, zona muerta, anti doble click, `release`, simulación a 30 fps), `safety_pause_due`, `detector_size`, `FpsMeter`, wrapper de `hint`, panel, y **tres smoke tests de `run()`** con cámara, tracker y ventana falsos (headless escribe el bus; ventana procesa teclas; click y drag pasan por el filtro temporal). |
| `tests/test_gesture_bank.py` | 7 | `load_samples`, conteos, `SampleCounter` carga una vez y se mantiene igual al archivo, parser y CLI > toml. |
| `tests/test_state_reader.py` | 20 | lector del bus (formatos de `palm`, formato antiguo `hands`, timestamp/edad/stale) y **paridad de `default_bus_path` con `tletl_core.paths`**. |
| `tests/test_blender_map.py` | 35 | `BlenderGestureMapper`: un frame no traslada, direcciones, FIST, release, rotación, escala (nunca negativa), **suavizado real** (converge, monótono, sin overshoot, resync por canal/FIST/reset/movimiento manual, no resync por ruido float32), `z_gain`, defaults recalibrados, escala no uniforme, `changed_channels`; además `append_sample` del banco y que addon y `gesture_bank` importan sin bpy/mediapipe. |
| `tests/test_blender_session.py` | 25 | `GestureSession`: sostén de THREE (una vez por sostén), flancos de PINCH/VICTORY, primitivas, `bus_status` (missing/invalid/stale/ok), `status_lines`, `AddonSettings.from_prefs`, `palm_to_world_xy`. |
| `tests/test_blender_bpy_layer.py` | 13 | la capa bpy con un `bpy`/`bmesh` falsos: escribe solo canales cambiados, escala como factor, rotación no Euler ⇒ error visible, `spawn` con bmesh enlaza/activa/selecciona, el timer sobrevive a excepciones y obedece `interval`, prefs ausentes ⇒ defaults. |
| `tests/test_blender_addon_build.py` | 15 | `bl_info` por `ast`, manifiesto (tagline ≤ 64 sin punto), zip con los 3 miembros, importable como paquete, **la copia inline del lector coincide con `state_reader.py`**. |
| `tests/test_integration_bus.py` | 7 | bus REAL escrito por `TletlStateBus` → `state_reader` → mapper/sesión: traslación, FIST, `palm` sobrevive al JSON, **misma ruta por default app/addon**, fresco conecta y obsoleto bloquea, control/confianza llegan al panel, CREATE crea a través del bus. |
| `tests/test_gesture_modeler.py` | 46 | `Scene`/`Solid` (reductor de ops, serialización, `from_dict` tolerante), `GestureModeler`: holds, flancos, TRANSFORM (move/rotate/scale/release/FIST, deadzone), CREATE, `end_session`, **obsolescencia por avance del timestamp** (reloj desfasado, reinicio del flujo), `format_op`. |
| `tests/test_autocad_control.py` | 56 | `importorskip("ezdxf")`: `FileBusSource` (solo entrega cambios), `UdpBusSource` por loopback, `ScriptedBusSource`; `run_loop`/`CadSession` con reloj y sleep inyectados; **export DXF real, relectura y `audit()` sin errores**, XDATA, capas, `$INSUNITS`, cuña cerrada; `AutoCADBackend` contra un ModelSpace falso (entidad registrada antes de rotar, select inválido cuenta como fallo, errores COM no matan el bucle, `Regen`), `_point` con `pythoncom` falso, CLI (`--dry-run`, exit 2 sin AutoCAD); `freecad_macro` importa sin FreeCAD, rutas, `placement_base`, `MinimalModeler` contra el modeler real. |

**Tests con MediaPipe real** (`tests/test_hand_tracker.py`, al final):
`test_tasks_backend_real_model_black_frame`,
`test_gesture_hint_real_model_black_frame` y
`test_tasks_backend_real_model_noise_frames_and_timestamps` cargan de verdad
`hand_landmarker.task` / `gesture_recognizer.task` y procesan frames negros y
de ruido (sin mano ⇒ listas vacías, timestamps crecientes). Buscan el modelo
primero en `<repo>/.venv/models/<nombre>` y luego en `~/.tletl/models/`; si no
está en ninguno, **se saltan** con `modelo … no descargado
(./launchers/tletl-fetch-models.sh)`; también se saltan si mediapipe no importa
o su librería nativa no carga. `test_tasks_backend_corrupt_model_is_tracker_unavailable_with_hint`
solo necesita que la Tasks API importe (escribe un `.task` de basura). Este
entorno tiene los modelos en `.venv/models/`, por eso no hay skips.
Sin `ezdxf`, los 56 tests de `test_autocad_control.py` se saltan enteros.

---

## 10. Rendimiento

Todo medido en CPU en este entorno (sin GPU, sin cámara), con frames
sintéticos donde hace falta.

| Componente | Antes (v5.1) | Ahora (v5.2) | Cómo se midió |
|---|---|---|---|
| `RobustKNNRuntime.predict` | **58.1 ms** por predicción (bucle Python sobre 2 363 × 67) | **0.95 ms** (numpy, idéntico bit a bit) | commit `ce5ab89`, 200 predicciones, k=13. Reproducido aquí: 56.4 ms → 0.92 ms; 200/200 predicciones idénticas a la referencia |
| MediaPipe HandLandmarker (`tasks`, VIDEO, `num_hands=2`, sin mano) | — | **11.5 ms/frame** media (p50 11.1, p95 14.5, máx 17.5) a 640×360; **11.3 ms** a 1280×720 | `python -m tools.bench_tracker --synthetic --backend tasks --frames 200`. El lead midió 13.8 ms y 10.8 ms en dos corridas; misma banda |
| MediaPipe GestureRecognizer (hint, opcional) | — | **11.0 ms/frame** media (p50 10.7, p95 14.4), sin mano | script equivalente sobre `GestureHint.process`. El lead: 10.6 ms |
| Bus | JSON con indentación + ~67 features por mano | ~1.1 KB compacto sin features | `TletlStateBus.serialize` |
| Banco de gestos (herramienta) | releía el JSONL completo por frame | conteo en memoria | `SampleCounter` |

Con manos presentes el landmarker corre además el modelo de puntos por mano y
sube algo; el propio HandLandmarker redimensiona por dentro, por eso
`proc_width` no cambia su tiempo (11.5 vs 11.3 ms) y solo ahorra la conversión
de color, el realce y la copia (1–3 ms a 720p según la medición del lead).

**Qué limita los fps ahora**: a 58 ms por predicción y dos manos, solo el KNN
ya topaba el bucle en ~15 fps, que es exactamente lo que se midió en la
validación física. Con el KNN a ~1 ms, el costo dominante por frame es el
detector (~11–14 ms) más el hint si está activo (~11 ms), y por encima de eso
manda la cámara: con MJPG a 30 fps el bucle debería ir a la velocidad de la
cámara (33 ms por frame). Esto **no se ha verificado con cámara real**; es lo
que sale de sumar los tiempos medidos.

**Accuracy del KNN** (`python -m tools.duel_lab`, holdout 20 %, seed 7, 2 122
train / 530 test, 3.4 s en total):

| k | orientation_weight | accuracy holdout |
|---|---|---|
| 9 | 0.45 | **0.9717** |
| 13 (default) | 0.45 | 0.9679 |
| 13 | 0.25 | 0.9679 |
| 21 | 0.45 | 0.9585 |

Es accuracy contra el propio banco (capturado de noche); no mide la precisión
en vivo de día ni el efecto del guard/critic. k=9 sale ligeramente mejor que
el default 13 en este holdout; nadie ha decidido cambiarlo.

---

## 11. Bugs corregidos en v5.2

Derivados del `git log` (`aa86780` … `04b3b09`), de los diffs contra el
baseline `f3e025a` y de `docs/VALIDACION_FISICA_v5.md` §5.

### 11.1 Los anotados en la validación física (§5) y su estado

| Bug anotado | Estado en v5.2 | Dónde |
|---|---|---|
| Addon instalado como `.py` suelto truena con `ModuleNotFoundError: state_reader` | **Resuelto**: zip con ambos archivos + manifiesto; el `.py` suelto además funciona por la copia inline del lector | `tools/build_blender_addon.py`, `tletl_blender_addon.py` |
| Blender en Fedora 43 arranca con error de OpenColorIO | No es de Tletl (desfase de paquetes de Fedora) | — |
| Rutas del bus inconsistentes: app en `<cwd>/tletl_state.json`, addon en `~/tletl_state.json` | **Resuelto**: una sola regla en `paths.py`, replicada en `state_reader.py` y `freecad_macro.py`, con test de paridad; el symlink deja de hacer falta | `tletl_core/paths.py` |
| `mediapipe` sin tope ⇒ 0.10.35 sin `mp.solutions` ⇒ la app no arranca | **Resuelto** por la otra vía: backend `tasks` (HandLandmarker) como principal; el tope no se puso a propósito | `apps/common/hand_tracker.py` |
| PINCH muere en el critic de día (0.72 fijo vs 0.62–0.70 real) | **Resuelto en código** (umbral por gesto en `[critic.min_conf]`, PINCH 0.62); **pendiente de dato**: capturar PINCH de día | `config.py`, `critic.py`, `pipeline.py` |
| FIST sensible a pose/distancia (POINT en posición de escritorio) | **Pendiente**: es un problema del banco, no del código; el hint del Gesture Recognizer puede ayudar (§13) | — |
| FIST fantasma al agarrar objetos ⇒ SAFETY_STOP espontáneo | **Mitigado**: la pausa exige FIST estable `safety_hold` = 0.35 s (o conf ≥ 0.90 la mitad); falta verificarlo en vivo | `main.py::safety_pause_due` |

### 11.2 Core

| Bug | Archivo | Síntoma | Causa | Fix |
|---|---|---|---|---|
| KNN a 58 ms por predicción | `tletl_core/classifier.py` | app en vivo a ~15 fps | bucle Python sobre 2 363 muestras × 67 features por mano y por frame | matriz F×N y distancia vectorizada, bit a bit idéntica (0.95 ms) |
| Distancia inválida finita | `classifier.py` | con features vacías predecía `OPEN_PALM` con confianza 1.0 | centinela `1e9` en vez de `inf`: las primeras k muestras del banco colaban como vecinos | `inf`; sin vecinos ⇒ `NEUTRAL/UNKNOWN/NO_VOTES` |
| `[classifier] min_confidence/min_margin` ignorados | `classifier.py`, `pipeline.py` | cambiar el toml no cambiaba nada | 0.47/0.18 hardcodeados en `predict` | parámetros del constructor, cableados desde la config |
| `[critic.min_conf]` inexistente/ignorado | `critic.py`, `config.py`, `pipeline.py` | PINCH bloqueado de día | tabla `MIN_CONF` fija en el módulo | `strict_critic(min_conf=…)` + `critic_min_conf(cfg)` con merge profundo |
| Memoria adaptativa sin efecto | `adaptive.py`, `classifier.py`, `pipeline.py` | "aprendizaje en vivo" grababa y nadie leía | el KNN solo cargaba el banco | `samples()` → `RobustKNNRuntime(extra_samples=…)` antes del balanceo y la normalización |
| Memoria adaptativa junto al banco | `pipeline.py`, `paths.py` | `datasets/tletl_adaptive_runtime.json` al lado del banco sagrado | ruta derivada del banco | `~/.tletl/tletl_adaptive_runtime.json` |
| Escritura por frame de la memoria | `adaptive.py` | un JSON completo reescrito 30 veces/s con la memoria activa | `observe()` guardaba siempre | throttle `save_interval` + `flush()` en `pipeline.close()`; crea el directorio padre (antes `FileNotFoundError` en máquina nueva) |
| `intent` parpadeaba a `BLOCKED` | `pipeline.py::to_hand_state` | Blender/CAD veían `BLOCKED` mientras la app seguía actuando | `critic_ok = ok and critic_accepted` mezclaba estable con frame actual | `critic_ok = result.ok` (veredicto estable) |
| Memoria adaptativa envenenada en holds | `pipeline.py` | transiciones y `NEUTRAL` grabados con el gesto estable | observaba con `stable.ok` sin mirar el frame actual | aprende solo si el frame actual pasó la seguridad con el mismo gesto |
| `geometric_rule({})` = `FIST` | `guard.py` | un dict sin claves de dedos daba SAFETY_STOP de la nada | cuatro dedos "cerrados" por ausencia de evidencia | sin evidencia ⇒ `NEUTRAL` |
| El realce de baja luz oscurecía | `lowlight.py` | frames oscuros salían más oscuros (128 → 83 en Y) | doble inversión: 0.62 como gamma en una LUT `^(1/gamma)` ⇒ exponente 1.61; `gamma_dark` guardado sin usarse; LUT reconstruida por frame | exponentes derivados de `gamma_dark` (0.625/0.759/0.880), LUT cacheadas |
| Tests con tiempo real | `temporal.py` | el test del swipe aceptaba `None` "según el timing" | `time.time()` directo | `clock` inyectable en `MotionTracker`, `Hold`, `AdaptiveGestureMemory` |
| Bus con `timestamp` 0 | `bus.py` | un state con `timestamp=0.0` se escribía tal cual: obsolescencia indetectable | `setdefault` no cubre el 0 | timestamp real si viene 0/ausente; `age()` |
| `bus.read()` lanzaba | `bus.py` | JSON a medias/corrupto tumbaba al lector | `json.loads` sin captura | nunca lanza (⇒ `{}`) |
| Bus pesado | `bus.py` | JSON indentado con ~67 features por mano 30 veces/s | `indent=2`, features siempre | compacto, `include_features=false` |
| Config sin merge profundo | `config.py` | un toml con `[critic.min_conf]` parcial perdía los demás umbrales | `dict.update` plano | `_deep_update` un nivel |

### 11.3 App Fedora y capa común

| Bug | Archivo | Síntoma | Causa | Fix |
|---|---|---|---|---|
| La app no arranca con mediapipe ≥ 0.10.31 / 1.x | `apps/fedora_control/main.py`, `tools/gesture_bank.py` | `AttributeError: module 'mediapipe' has no attribute 'solutions'` | importaban `mp.solutions.hands` directo | `apps/common/hand_tracker.py` con backend `tasks` principal y `legacy` de respaldo |
| `[camera]`/`[tracker]` del toml ignorados | `main.py` | corría a 960×540 aunque el toml dijera 1280×720 | argparse tenía sus propios defaults | defaults `None`; CLI > env > toml > DEFAULTS (`apply_cli_overrides`) |
| Pausa por un solo frame de FIST | `main.py` | agarrar la taza pausaba el control (3× en 2 min) | `if g == "FIST"` sin sostén | `Hold` + `safety_pause_due` con `[fedora].safety_hold` |
| Click imposible en CURSOR | `main.py` | a 15 fps ningún PINCH duraba < 0.24 s tras el filtro temporal | umbrales tap 0.24 / drag 0.32 fijos, menores que la latencia del filtro | `PinchClickDrag` con `tap_max` 0.35 / `drag_min` 0.40 configurables (+ KNN rápido) |
| Botón del ratón pegado | `main.py` | `t`/`m`/excepción dejaban un `mouse_down` sin `mouse_up` | sin `release` ni `finally` | `release_pinch` en toggle/modo/mano perdida y `try/finally` que suelta, libera cámara, ventana, modelos, bus y pipeline |
| Error del bus mataba el bucle | `main.py` | un `OSError` al escribir el bus cerraba la app | sin captura | se imprime una vez, se cuenta (`err:n` en el panel) y sigue |
| Un error por acción sin ydotool | `apps/fedora_control/actions.py` | `[ydotool error]` en cada `mousemove` | `print` en cada excepción | aviso único al arrancar (`shutil.which`), acciones no-op, fallos contados con causa probable |
| Banco releía el JSONL por frame | `tools/gesture_bank.py` | varios ms por frame y creciendo con el banco | `load_samples()` en el bucle solo para mostrar conteos | `SampleCounter` en memoria |
| Lateralidad "corregida" de más | `hand_tracker.py` | (prevención) | MediaPipe asume selfie y las apps ya espejan | documentado: no se intercambia Left/Right en ningún backend |
| `.task` corrupto salía como `RuntimeError` crudo | `hand_tracker.py` | "Unable to open zip archive" sin pista y sin caer a legacy | excepción no traducida | `TrackerUnavailable` con ruta y `--force`; `auto` cae a legacy si existe |
| Test real del Gesture Recognizer se saltaba | `apps/common/gesture_hint.py`, `tests/test_hand_tracker.py` | `hint_enabled` releía la env var y un `--gesture-hint` explícito podía anularse con `TLETL_GESTURE_HINT=0` | precedencia resuelta dos veces | `hint_enabled` lee solo el dict fusionado; el test pasa `gesture_hint` en el dict (`04b3b09`) |

### 11.4 Blender

| Bug | Archivo | Síntoma | Causa | Fix |
|---|---|---|---|---|
| Escala no uniforme destruida | `tletl_blender_addon.py` | un objeto (1, 2, 1) se volvía (s, s, s) al primer tick | se leía `scale.x` y se escribía uniforme | referencia = mayor \|eje\|; se aplica un **factor** a los tres ejes (conserva proporción y espejos) |
| "Smoothing" perdía movimiento | `tletl_blender_addon.py` | con 0.35 se perdía el 65 % del recorrido; la validación acabó en Gain 14 / Smoothing 1.0 | `lerp(out, out + delta, smoothing)` tiraba el resto de cada delta | suavizado exponencial hacia un target acumulado; defaults Gain 10 / Smoothing 0.5 |
| Bus obsoleto no detectado | `tletl_blender_addon.py`, `state_reader.py` | la app cerrada y el addon seguía "conectado" | no se miraba `timestamp` | `bus_status()` con *Stale after*; sin actuar hasta frames frescos |
| Timer moría con una excepción | `tletl_blender_addon.py` | había que reiniciar el addon | excepción no capturada en el callback | `try/except` que deja el texto en `last_error` (panel) y sigue |
| Prefs no encontradas como extensión | `tletl_blender_addon.py` | `KeyError` en `preferences.addons[__name__]` | id fijo | `_ADDON_ID = __package__ or __name__`; `from_prefs()` cae a defaults |
| Rutas `//` de Blender | `tletl_blender_addon.py` | *Bus path* relativo al `.blend` no resolvía | sin `bpy.path.abspath` | resuelto en `AddonSettings.resolved_bus_path` |
| Escritura de todos los canales por tick | `tletl_blender_addon.py` | depsgraph y undo sucios aunque nada se moviera | siempre `location/rotation/scale =` | `changed_channels()` escribe solo lo que cambió |

### 11.5 CAD (código nuevo; corregido durante la revisión antes del commit `92587f3`)

| Bug | Archivo | Detalle |
|---|---|---|
| `Regen(1)` regeneraba todos los viewports | `autocad_client.py` | constantes `acActiveViewport = 0` / `acAllViewports = 1` estaban invertidas |
| Entidad huérfana si `Rotate3D` fallaba | `autocad_client.py` | ahora se registra `entities[id]` **antes** de rotar; si rota falla, se anota y sigue |
| `select` de un id inexistente contaba como aplicado | `autocad_client.py` | ahora `KeyError` ⇒ fallo |
| Cuña abierta (18 vértices) | `dxf_export.py` | `CONVTOSOLID` la rechazaba; ahora `MeshVertexMerger` (6 vértices, 5 caras) |
| Cliente Windows con reloj desfasado veía todo obsoleto | `gesture_modeler.py` | obsolescencia por **avance** del timestamp, no por reloj local (salvo el primer frame) |
| Puntos COM como tupla ⇒ "Invalid argument" | `autocad_client.py` | `_point()` devuelve `VARIANT(VT_ARRAY \| VT_R8, …)` cuando hay `pythoncom` |

### 11.6 Inconsistencias encontradas entre código y documentos al documentar

| Dónde | Qué | Estado |
|---|---|---|
| `pyproject.toml` | decía `version = "5.0.0"` | **corregido**: 5.2.0, igual que `tletl_core.__version__` y `bl_info` |
| `tletl_core/critic.py::MIN_CONF` | tenía su propia copia con `PINCH: 0.72` mientras la config decía 0.62 | **corregido**: `MIN_CONF = dict(CRITIC_MIN_CONF_DEFAULTS)`, fuente única, con test |
| `docs/RECOMENDACIONES…` §1, README viejo, comentario de `[bus] include_features` | "mismas 67 features", "~60 features" | **aclarado**: el banco tiene 67 claves útiles, el extractor en vivo 65, y comparten **54** (§5.2); nada falla; la recomendación ahora lo dice tal cual |
| `tools/bench_tracker.py` (docstring) | citaba un `tools/bench_classifier.py` inexistente | **corregido** |
| `docs/VALIDACION_FISICA_v5.md` §4 | "el addon refresca a ~20 Hz (timer 0.05 s)" | histórico (v5.1); ahora *Interval* = 0.033 s (~30 Hz). Se deja tal cual: es un acta |
| `docs/RECOMENDACIONES…` §0 | "Pendiente: prueba con cámara real (tú)" | sigue pendiente (no hay cámara aquí) |

---

## 12. Lo que NO está probado, checklist en vivo y troubleshooting

### 12.1 Sin probar en este entorno (dicho sin rodeos)

- **Nada con cámara**: ni el backend `tasks` viendo una mano real, ni la
  precisión de gestos con los umbrales nuevos, ni los fps reales, ni el click/
  drag, ni la pausa por FIST con `safety_hold`, ni el hint. Lo que está probado
  es que el bucle completo corre con cámara, tracker y ventana **falsos**
  (`test_run_*_smoke`) y que el modelo real de MediaPipe carga y procesa frames
  sin mano.
- **Blender real**: el addon se ejercitó con un `bpy`/`bmesh` falsos y con un
  bus real en disco. Que Blender 4.x lo registre como extensión desde el zip,
  el panel y el timer real, no.
- **AutoCAD**: sin Windows ni COM aquí; `AutoCADBackend` se probó contra un
  ModelSpace que registra llamadas y un `pythoncom` falso. Los nombres/firmas
  COM son los de la referencia ActiveX de AutoCAD.
- **FreeCAD**: la macro importa y sus helpers puros están probados; no se
  ejecutó dentro de FreeCAD (Part/Placement/QTimer).
- **El banco de día**: sigue siendo el de la noche (`training-lab-v4.6-*`);
  FIST en pose de escritorio y PINCH de día no se arreglan con código.
- **Push a GitHub**: bloqueado por acceso al repo durante la sesión;
  irrelevante para esta documentación.

### 12.2 Checklist de prueba en vivo (para el usuario)

**A. Detector y gestos (10 min)**

```bash
source .venv/bin/activate
./launchers/tletl-fetch-models.sh --check          # los dos [ok]
./launchers/tletl-health.sh                        # backend=tasks, modelo (presente)
./launchers/tletl-fedora-safe.sh                   # dry-run con ventana
```

1. Al arrancar debe imprimir `backend tasks (MediaPipe HandLandmarker)` y
   `obtenido 1280x720@30 MJPG` (si dice YUYV o 15 fps, prueba `--fourcc ""` o
   otra cámara con `--camera 1`).
2. Panel: `FPS` cerca de 30 con una mano; `backend:tasks`.
3. Los 7 gestos, 10 veces cada uno, en pose de escritorio **y** a distancia de
   uso de Blender: cuenta con la línea `DOM[…]:GESTO … ok:True`. Apunta
   confusiones. Referencia previa: FIST 0/10 en escritorio, PINCH 0/10 de día
   (`docs/VALIDACION_FISICA_v5.md` §2).
4. Dos minutos haciendo cosas normales (teclear, taza): anota cada `Intent:`
   de acción y cada `Pausa seguridad` sin gesto deliberado. Meta: 0. Referencia
   previa: 4 en 2 min.
5. Repite 3–4 con `--gesture-hint`: el panel muestra `hint:FIST` etc. Compara
   cuántos `GUARD_BLOCK_*` desaparecen y si baja el fps.

**B. Control de Fedora (5 min)** — `./launchers/tletl-fedora.sh`

6. `ydotool mousemove -x 5 -y 0` mueve el cursor antes de empezar.
7. OPEN_PALM 1.25 s ⇒ `Control ON`; THREE 0.85 s ⇒ cambia de modo; FIST
   sostenido 0.35 s ⇒ `Pausa seguridad`; un puño de medio frame no debe pausar.
8. CURSOR: PINCH corto ⇒ click; PINCH largo ⇒ `Drag ON` … `Drag OFF`; soltar
   la mano en mitad de un arrastre debe soltar el botón.
9. NAVEGADOR: POINT arriba/abajo scroll, VICTORY + swipe cambia pestaña.

**C. Blender (10 min)**

```bash
./launchers/tletl-build-blender-addon.sh
./launchers/tletl-blender-bus.sh --window          # terminal 1
```

10. Instalar el zip (extensión o addon), activar, N > Tletl > *Iniciar Tletl*
    con el cubo seleccionado. El panel debe decir `Conectado (30 fps app, … ms)`.
11. Cerrar la app: en ≤ 1 s el panel debe pasar a `BUS OBSOLETO`; al
    reabrirla, vuelve a `Conectado` sin que el cubo salte.
12. PINCH mueve; OPEN_PALM suelta sin salto; FIST frena; mod PINCH rota; dom
    PINCH + mod OPEN_PALM escala. Prueba con un objeto escalado (1, 2, 1): debe
    conservar la proporción.
13. Mover el cubo a mano (G) o Ctrl+Z con el addon activo: no debe "volver solo".
14. THREE 0.8 s ⇒ `Modo CREATE`; PINCH crea un cubo en la palma; VICTORY cambia
    a esfera; THREE ⇒ TRANSFORM y PINCH mueve lo recién creado.
15. Defaults Gain 10 / Smoothing 0.5: si el tacto no coincide con el Gain 14 /
    Smoothing 1.0 de la validación, anota qué valores te quedan.

**D. CAD (opcional)**

16. Ruta 2: `./launchers/tletl-cad-session.sh --export prueba.dxf`; THREE ⇒
    CREATE, PINCH crea, Ctrl+C ⇒ `prueba.json` + `prueba.dxf`; abrir el DXF
    (`ZOOM E`).
17. Ruta 1 (Windows): `--dry-run` primero (deben llegar frames), luego con un
    dibujo vacío; anota cualquier `Invalid argument` o `com_error`.
18. Ruta 3 (FreeCAD): con `TLETL_REPO` exportado, la consola debe decir
    `modeler=GestureModeler (repo)`.

### 12.3 Troubleshooting

| Síntoma | Causa probable → qué hacer |
|---|---|
| `No encuentro el modelo …/hand_landmarker.task` | `auto_download` apagado o sin red → `./launchers/tletl-fetch-models.sh`; o apunta `[tracker].model_path` / `TLETL_TRACKER_MODEL` al `.task` |
| `descarga demasiado pequeña (… < 1.0 MB); ¿proxy, portal cautivo o URL cambiada?` | la red devolvió HTML → revisa proxy/portal; `--force` tras arreglarlo |
| `MediaPipe Tasks no pudo cargar su librería nativa` | faltan EGL/GLES → `sudo dnf install mesa-libEGL mesa-libGLES` |
| `MediaPipe Tasks no pudo cargar el modelo … (Unable to open zip archive)` | `.task` corrupto/truncado → `./launchers/tletl-fetch-models.sh --force` |
| `backend 'legacy' pedido pero mediapipe 1.0.1 ya no trae mp.solutions` | mediapipe nuevo → `[tracker] backend = "auto"` o `"tasks"` |
| `mediapipe no está instalado o no carga` | `pip install 'mediapipe>=0.10.21'` en Python 3.12 (no hay ruedas para 3.13/3.14) |
| `No pude abrir cámara 0` | índice → `--camera 1`; otra app usa la cámara; permisos de `/dev/video*` |
| FPS ~15 o `obtenido …@15 YUYV` | la cámara no aceptó MJPG a esa resolución → `--fourcc ""`, o baja `--width 960 --height 540` |
| `AVISO: 'ydotool' no está en PATH` | `sudo dnf install ydotool`; con `--dry-run` no hace falta |
| `[ydotool error] … devolvió N (¿ydotoold corriendo? ¿usuario en grupo input?)` | `systemctl --user enable --now ydotool`; `sudo usermod -aG input $USER` y reiniciar sesión |
| Nada actúa aunque el gesto sale bien | control OFF (círculo rojo) → OPEN_PALM 1.25 s o tecla `t`; o `conf < action_conf` (0.52); o `Guard:GUARD_BLOCK_*` / `Critic:confianza baja…` |
| `Critic:confianza baja para PINCH: 0.60 < 0.62` | sube muestras de PINCH al banco (§13) o baja `[critic.min_conf] PINCH` con cuidado (más pinzas fantasma) |
| `Guard:GUARD_BLOCK_FIST` frecuente | regla geométrica discrepa con confianza justa → prueba `--gesture-hint` (si el recognizer coincide, `OK_HINT_AGREES`) |
| `Pausa seguridad` al agarrar cosas | sube `[fedora].safety_hold` (0.35 → 0.5) |
| Click no sale en CURSOR | PINCH dura > `tap_max` (0.35) → suéltalo antes, o sube `tap_max`; recuerda la zona muerta hasta `drag_min` (0.40) |
| `ERROR escribiendo el bus …` / `err:n` en el panel | permisos o disco en `~/.tletl` → `TLETL_STATE_PATH` a otra ruta |
| Blender: `Sin bus en ~/.tletl/tletl_state.json` | la app no corre o escribe en otra ruta → compara la línea `Bus:` del panel de la app con *Bus path*; misma `TLETL_STATE_PATH`/`TLETL_HOME` en ambas terminales; Flatpak solo ve el HOME |
| Blender: `BUS OBSOLETO` con la app abierta | la app está congelada, o Blender y la app no comparten reloj (otra máquina) → sube *Stale after* |
| Blender: el objeto no se mueve | ¿`Conectado`? ¿Dom = PINCH con confianza? ¿modo TRANSFORM? ¿objeto activo controlable? ¿`Error:` en el panel? |
| Blender: `rotation_mode 'QUATERNION' no soportado` | cambia el objeto a Euler XYZ |
| Blender: `Prefs no encontradas: usando defaults` | instalación rara → reinstala desde el zip |
| CAD: `bus=sin frames` en Windows | UDP no llega → `udp_target`/`TLETL_BUS_UDP` en Fedora, IP, firewall UDP 5055 entrante, misma LAN; `--dry-run` |
| CAD: `ezdxf no está instalado` | `pip install ezdxf`; el JSON ya quedó, re-exporta con `dxf_export` |
| CAD: `AutoCAD no tiene un dibujo activo` | abre un dibujo (Ctrl+N) y vuelve a correr |
| CAD: `[autocad] error aplicando …` repetido | AutoCAD tiene un comando abierto o un diálogo modal → Esc |
| FreeCAD: `esta macro debe ejecutarse dentro de FreeCAD` | la corriste con Python normal → Macro > Macros… |
| FreeCAD no rota ni escala | falta `TLETL_REPO` → expórtalo antes de abrir FreeCAD |
| Tests: `3 skipped … modelo … no descargado` | `./launchers/tletl-fetch-models.sh` (o copia los `.task` a `.venv/models/`) |
| Tests: `test_autocad_control.py` saltado entero | `pip install ezdxf` |

---

## 13. Decisiones pendientes del usuario

Las tres cosas que este repo no puede decidir solo; el razonamiento completo
está en `docs/RECOMENDACIONES_DETECCION_MANOS.md`.

1. **`gesture_hint = true` o no** (`[tracker]`, `TLETL_GESTURE_HINT=1`,
   `--gesture-hint`). Está integrado y apagado. Cuesta ~11 ms/frame más de
   CPU; a cambio, cuando el Gesture Recognizer coincide con el KNN, el guard
   deja pasar FIST/OPEN_PALM/POINT/VICTORY con confianza justa
   (`OK_HINT_AGREES`). No sabe PINCH ni THREE y nunca veta nada. Decidir tras
   10 minutos en `tletl-fedora-safe.sh --gesture-hint` (checklist A.5).
2. **Sesión de captura diurna del banco** (20 min, `./launchers/tletl-bank.sh
   --dataset datasets/tletl_gesture_bank_v2_features.jsonl --dual-hand`): 60–100
   FIST en pose de escritorio y a distancia de uso, 60–100 PINCH de día, 60 s
   de NEUTRAL haciendo cosas normales. Es lo único que ataca FIST-como-POINT y
   la confianza baja de PINCH; el umbral 0.62 es un parche razonable, no la
   solución. Antes: `cp datasets/tletl_gesture_bank_v2_features.jsonl
   datasets/backup_$(date +%F).jsonl`; después: `bank_probe` y `duel_lab` para
   confirmar que la accuracy holdout no bajó, y commit.
3. **Planes B/C del detector** (rtmlib / RTMPose-hand; YOLO11-pose): solo si
   MediaPipe falla con oclusiones en la prueba real. Ambos son 2D (se pierden
   las 13 features de orientación y habría que apagar el critic de
   orientación), ~1 día de trabajo cada uno como backend nuevo en
   `hand_tracker.py`; YOLO es AGPL-3.0. WiLoR y HaMeR quedan descartados (CUDA).

Menores, también pendientes de tu criterio: dejar `k = 13` o pasar a 9 (§10);
si el tacto de Blender con Gain 10 / Smoothing 0.5 sustituye al Gain 14 /
Smoothing 1.0 de la validación (checklist C.15); y subir `pyproject.toml` a
5.2.0 (§11.6).
