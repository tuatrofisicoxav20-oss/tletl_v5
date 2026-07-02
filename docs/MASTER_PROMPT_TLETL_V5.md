# TLETL V5 — AUDITORÍA DE FASES + EJECUCIÓN DE PENDIENTES

> Documento máster entregado por Emiliano el 2026-07-01 y guardado en el repo
> al cierre del PASO 2 (como exige el propio documento). Resultado de la
> auditoría: F1–F6 DONE, F7 PARTIAL→cerrada en esta misma sesión.
> Ver historial git: tag `v5.0-audit` (baseline) y commits de cierre.

## Contexto
El plan original (MASTER_PROMPT_TLETL_V5_UNIFICACION, 13-jun) se
ejecutó PARCIALMENTE y se varó cuando faltó el banco. El banco ya
está verificado (2652/2652). Este prompt: (1) audita qué fases ya
están hechas, (2) HARD STOP con reporte, (3) solo con aprobación
ejecuta las pendientes en orden.

## Reglas duras (NO romper, aplican siempre)
1. `tletl_core/` JAMÁS importa cv2-window, ydotool, bpy ni nada de
   sistema. El core decide la INTENCIÓN; la app la ejecuta.
2. Nada destructivo hasta Fase 7, y ahí solo mover a `_archive/`
   comprimido. Las carpetas viejas (v3, v4_modular, v4_1) no se tocan.
3. Tests verdes antes de avanzar de fase. Rojos = stop y reporte.
4. Todo en CPU. KNN sobre el banco JSONL. Cero CUDA, cero modelos
   grandes.
5. El banco de gestos es SAGRADO: no se regenera ni se limpia sin
   respaldo previo.

## Arquitectura de referencia
Flujo: cámara → [lowlight.enhance] → MediaPipe Hands (max 2 manos)
→ features.extract_live_features → pipeline.process_hand():
classifier.predict (KNN) → guard.rule_guard → critic.strict_critic
→ temporal.TemporalFilter → [adaptive.observe opcional]
→ TletlFrameState{dom, mod, intent} → bus.write → tletl_state.json
→ apps (Fedora vía ydotool / Blender vía addon).
Clave de la unificación: guard y critic venían de
ai_runtime_v47_live.py / ai_critic_v47.py y TODO debe pasar por
pipeline.py — un solo cerebro, muchos clientes delgados.

## Las 7 fases (criterios de "hecha")
- F1 Esqueleto + banco: repo tletl_v5/ con estructura (core, apps,
  tools, datasets, launchers, config, tests), banco copiado en
  datasets/, healthcheck y bank_probe funcionando.
- F2 Core unificado: geometry, features, classifier, temporal,
  orientation, state, intent, bus completos con sus tests.
- F3 Seguridad en pipeline: guard.py + critic.py portados y TODO el
  flujo pasa por pipeline.py; test_guard, test_critic y
  test_pipeline (end-to-end con banco real) verdes.
- F4 Capacidades rescatadas: lowlight, dual-hand (dom/mod),
  memoria adaptativa de gestos (default OFF), tools/gesture_bank.py
  operativo, y app Fedora como cliente delgado (intent→ydotool en
  apps/fedora_control/actions.py).
- F5 Addon REAL de Blender (bpy): timer/modal cada ~50ms lee
  tletl_state.json vía state_reader → bpy.context.active_object.
  Mapeo: PINCH dominante = grab/mover (obj.location += delta del
  cursor), OPEN_PALM dominante = release, PINCH de mano secundaria
  = obj.rotation_euler, OPEN_PALM secundaria + spread =
  obj.scale por distancia entre manos, FIST = safety stop.
  Suavizado lerp. N-panel "Tletl": estado del bus, gesto por mano,
  enable/disable, sliders de sensibilidad. El addon NO clasifica ni
  abre cámara — solo lee bus y transforma. Sin bpy en este entorno:
  py_compile + test aislado del mapeo state→transform; la prueba
  en Blender real la hace Emiliano (mover un cubo, reportar fps).
- F6 Config + launchers + README: config/tletl.toml consolidando
  TODAS las env vars ([classifier] [guard] [critic] [temporal]
  [lowlight] [adaptive] [fedora] [blender] [camera], con override
  por env var); launchers bash: tletl-fedora.sh, tletl-fedora-safe.sh
  (dry-run), tletl-bank.sh, tletl-blender-bus.sh, tletl-health.sh.
- F7 Archivo y cierre: mover v3, v4_modular, v4_1 y backups a
  _archive/ comprimido, verificación final end-to-end, README de
  cierre.

## PASO 1 — Auditoría (solo lectura)
Para cada fase F1–F7, contrasta el repo actual contra su criterio y
emite veredicto DONE / PARTIAL / PENDING con evidencia concreta
(archivos, tests, wiring). Señala también: de dónde carga hoy la
config k=13 (¿toml o env vars?), si apps/fedora_control ejecuta
acciones reales o es esqueleto, y si el addon de Blender existe o
solo está state_reader.

## HARD STOP
Reporta la tabla de veredictos y una propuesta de orden de
ejecución de las fases pendientes. NO ejecutes nada hasta
aprobación explícita.

## PASO 2 — Ejecución (tras aprobación)
Fases pendientes en orden, una por una: implementar → tests verdes
→ commit checkpoint con tag por fase → siguiente. Al final, guarda
ESTE documento como docs/MASTER_PROMPT_TLETL_V5.md en el repo.

---

## Anexo — Resultado de la auditoría (2026-07-01)

| Fase | Veredicto | Evidencia |
|---|---|---|
| F1 | ✅ DONE | Estructura completa; banco 2652/2652; healthcheck + bank_probe OK |
| F2 | ✅ DONE | 8 módulos del core con tests |
| F3 | ✅ DONE | guard+critic en core; todo pasa por pipeline.py; test_pipeline con banco real |
| F4 | ✅ DONE | lowlight, dual-hand dom/mod, adaptive OFF, gesture_bank, ydotool real |
| F5 | ✅ código / ⏳ prueba viva | Addon 385 L, mapper puro, N-panel, timer 50 ms, tests sin bpy |
| F6 | ✅ DONE | tletl.toml 9 secciones + env overrides; 5 launchers |
| F7 | ✅ cerrada 2026-07-01 | v3/v4_modular/v4_1 (13-jun) + tletl_backups (01-jul, hash verificado) |

- Config k=13 viene de `config/tletl.toml` (override opcional por `TLETL_AI_V47_K`).
- Suite: **112 tests verdes**. Repo git desde 2026-07-01, banco TRACKEADO
  (su ausencia en un zip fue lo que bloqueó v5 — git es la protección).
- Pendiente único: **prueba en Blender real** (instalar addon, mover un cubo
  con PINCH, reportar fps) — a cargo de Emiliano.
