# Changelog — Tletl v5.2

## 2026-09-30 — v5.2.0

Baseline: `f3e025a` (v5 auditado, 112 tests) + la validación física del
2026-07-02 (`docs/VALIDACION_FISICA_v5.md`, tag histórico `v5.1-validated`).
Cierre: `04b3b09`, **487 tests verdes**, `tletl_core.__version__ = "5.2.0"`.
Detalle de cada bug (síntoma, causa, fix) en
`docs/DOCUMENTACION_TLETL_V5.md` §11.

### Rutas de runtime, config y bus — `aa86780`

- `tletl_core/paths.py` (nuevo): una sola regla para el bus
  (`$TLETL_STATE_PATH` > `$TLETL_HOME/tletl_state.json` > `~/.tletl/tletl_state.json`),
  la memoria adaptativa (`~/.tletl/tletl_adaptive_runtime.json`, ya no en
  `datasets/`) y el banco. Corrige el bug de rutas app ↔ addon de la validación.
- `tletl_core/config.py`: `DEFAULTS` completos para todas las secciones
  (`[bus]`, `[tracker]`, `[paths]`, `[critic.min_conf]`, `[fedora]
  safety_hold/tap_max/drag_min/action_conf`, `[adaptive] path/save_interval`,
  `[camera] proc_width/headless`), merge profundo de tablas anidadas,
  `ENV_MAP` ampliado (`TLETL_STATE_PATH`, `TLETL_CAMERA`, `TLETL_BUS_UDP`,
  `TLETL_HEADLESS`, `TLETL_TRACKER_*`, `TLETL_GESTURE_BANK`, …), helpers
  `bus_path_from_config`, `adaptive_path_from_config`, `bank_path_from_config`,
  `critic_min_conf`. PINCH 0.72 → 0.62 en `[critic.min_conf]`.
- `tletl_core/bus.py`: timestamp siempre real, JSON compacto, features fuera
  del bus por default (`include_features`), `read()` nunca lanza, `age()`,
  publicación UDP opcional (`UdpStatePublisher`, `UdpStateReceiver`,
  `parse_udp_target`).
- `state.py` versión 5 / `app_version` 5.2; `.gitignore` con artefactos de runtime.
- +30 tests (`test_paths.py`, `test_config.py`, `test_bus.py`).

### Decisión de detector y claves de config — `dbc6f30`, `a9feae4`, `ffbbef2`

- `docs/RECOMENDACIONES_DETECCION_MANOS.md`: MediaPipe Hand Landmarker (Tasks
  API) recomendado y aprobado por el usuario como backend principal; Gesture
  Recognizer como segunda opinión opcional; WiLoR/HaMeR descartados (CUDA);
  rtmlib y YOLO11-pose como planes B/C; plan de captura diurna del banco.
- `[tracker] backend = "auto"` prefiere Tasks; claves nuevas `auto_download`,
  `gesture_hint`, `gesture_model_path`, `gesture_hint_min_score`; `[camera]
  fourcc`; env vars correspondientes.

### Core — `ce5ab89`

- `classifier.py`: `predict()` vectorizado con numpy, **bit a bit idéntico** al
  bucle original (verificado sobre las 2 652 muestras): 58.1 ms → 0.95 ms por
  predicción (el KNN limitaba la app a ~15 fps). Distancia inválida = `inf`
  (antes `1e9`: features vacías ⇒ `OPEN_PALM` conf 1.0). `min_confidence`,
  `min_margin`, `verbose` por constructor; `extra_samples` para que la memoria
  adaptativa entre al KNN.
- `pipeline.py`: umbrales del critic por gesto desde `[critic.min_conf]`;
  `critic_ok` refleja el veredicto estable (el intent ya no parpadea a
  `BLOCKED` en `TEMPORAL_HOLD`); `hint` opcional en `process_features`
  (`OK_HINT_AGREES`); la memoria adaptativa solo aprende cuando el frame
  actual coincide con el gesto estable; `close()` / context manager.
- `guard.py`: `geometric_rule({})` ⇒ `NEUTRAL` (antes `FIST`).
- `critic.py`: `strict_critic(min_conf=…)`, `resolve_min_conf`.
- `adaptive.py`: escrituras throttled (`save_interval`), `flush()`, `dirty`,
  `samples()`, crea el directorio padre.
- `lowlight.py`: **bug**: el realce oscurecía los frames oscuros (doble
  inversión de gamma, exponente 1.61 en vez de 0.625); ahora los exponentes se
  derivan de `gamma_dark` y las LUT se cachean.
- `temporal.py`: reloj inyectable en `MotionTracker` y `Hold`.
- `tools/`: docstrings reales, rutas resueltas por config. +68 tests.

### App Fedora y capa común — `0491853`

- `apps/common/hand_tracker.py` (nuevo): `HandTracker` con backends `tasks`
  (HandLandmarker, principal) y `legacy` (mp.solutions, respaldo); mediapipe
  1.0.x no trae `mp.solutions` y la app vieja no arrancaba. Descarga
  automática del modelo a `~/.tletl/models/`, errores con instrucción,
  modelo corrupto detectado, `describe_backend`, dibujo de landmarks.
- `apps/common/gesture_hint.py` (nuevo): Gesture Recognizer como segunda
  opinión (FIST/OPEN_PALM/POINT/VICTORY), OFF por default.
- `apps/fedora_control/main.py`: `[camera]`/`[tracker]`/`[fedora]` del toml se
  usan de verdad (CLI solo sobreescribe); bus por `bus_path_from_config` con
  UDP e `include_features`; `--headless`/`--window`; `proc_width`; pausa por
  FIST con sostén (`safety_hold`) en vez de un frame; `PinchClickDrag` con
  `tap_max`/`drag_min` (a 15 fps el click era imposible); `t`/`m` sueltan el
  arrastre; `try/finally` libera ratón, cámara, ventana, modelos, bus y
  pipeline; errores del bus no matan el bucle; panel con backend, hint y edad
  del bus.
- `apps/fedora_control/actions.py`: aviso único si falta ydotool y acciones
  no-op; fallos contados con causa probable.
- `tools/gesture_bank.py`: usa el tracker; ya no relee el JSONL por frame.
- `tools/fetch_models.py` + `launchers/tletl-fetch-models.sh`;
  `tools/bench_tracker.py`.
- `launchers/tletl-blender-bus.sh` headless por default (`--window` para ver).
- `requirements.txt`/`pyproject.toml`: mediapipe sin tope a propósito;
  `apps.common` y `apps.autocad_control` en `packages`. +122 tests.

### Blender — `bacf952`

- Misma regla de ruta del bus que el core (`state_reader.default_bus_path()`,
  con test de paridad).
- Instalable como zip con `blender_manifest.toml`
  (`tools/build_blender_addon.py`, `launchers/tletl-build-blender-addon.sh`;
  sirve como addon legacy y como extensión 4.2+) y también como `.py` suelto
  (copia inline del lector). `_ADDON_ID = __package__ or __name__`.
- **Bug**: la escala se leía de `scale.x` y se escribía `(s, s, s)`; ahora
  factor sobre los tres ejes y solo se escriben los canales que cambian.
- **Bug**: el "smoothing" perdía movimiento (por eso la validación acabó en
  Gain 14 / Smoothing 1.0); ahora suavizado exponencial hacia un target
  acumulado, con resync al soltar, con FIST, en reset y si el usuario mueve el
  objeto a mano. Defaults Gain 10 / Smoothing 0.5 / Interval 0.033.
- Detección de bus obsoleto/ausente/ilegible con mensaje en el panel; panel
  con estado, intent, confianza, modo, primitiva y último error; timer que
  nunca muere; rutas `//`.
- Modo CREATE: THREE 0.8 s alterna TRANSFORM/CREATE; PINCH crea
  cubo/esfera/cilindro/cono/plano con bmesh en la palma; VICTORY cicla.
  Prefs `z_gain`, `rot_gain`, `scale_gain`, `stale_after`, `interval`.
- README del addon reescrito. +94 tests (incluida la capa bpy con un
  bpy/bmesh falsos).

### CAD — `92587f3`

- `apps/autocad_control/` (nuevo): tres rutas con el mismo modelador puro
  `gesture_modeler.py` (TRANSFORM/CREATE, mismos gestos que Blender,
  obsolescencia juzgada por el avance del timestamp para tolerar relojes
  desincronizados).
  1. AutoCAD en Windows: bus por UDP → `autocad_client.py` (pywin32/COM,
     puntos como `VARIANT`). No ejecutable aquí; probado contra un ModelSpace
     simulado.
  2. DXF en Fedora: `session.py` + `dxf_export.py` (ezdxf, MESH por sólido,
     capa por tipo, R2010, auditado sin errores). Probado de punta a punta.
  3. FreeCAD: `freecad_macro.py` (QTimer + Part::Box/Cylinder/Sphere/Cone).
     No ejecutado dentro de FreeCAD.
- `launchers/tletl-cad-session.sh`. +100 tests.

### Healthcheck y test del hint — `04b3b09`

- `tools/healthcheck.py` imprime qué backend del detector se usaría, la
  versión de mediapipe y si el modelo está, sin cargarlo.
- `gesture_hint.hint_enabled` lee solo el dict de config ya fusionado (la
  precedencia CLI > env > toml la resuelve `load_config`); el test de
  integración con el modelo real del Gesture Recognizer ya no se salta.

### Documentación (esta entrega, sin commit todavía)

- `docs/DOCUMENTACION_TLETL_V5.md` (nuevo): todo el repositorio módulo por
  módulo, config completa, esquema del bus, apps, tools, tests, rendimiento,
  bugs, checklist en vivo, decisiones pendientes.
- `README.md` reescrito al estado actual.
- `docs/CHANGELOG_v5.2.md` (este archivo).

### Pendiente (no es código)

- Prueba con cámara real de todo lo anterior (checklist en la documentación §12).
- Decidir `gesture_hint = true`; capturar FIST/PINCH/NEUTRAL de día al banco.
- (Corregido en el commit de cierre: `pyproject.toml` a 5.2.0 y
  `critic.MIN_CONF` como copia de `config.CRITIC_MIN_CONF_DEFAULTS`.)
