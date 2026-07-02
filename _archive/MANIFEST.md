# _archive — código viejo de Tletl (Fase 7, entierro no destructivo)

Archivado el 2026-06-13 al consolidar **Tletl v5**. Cada carpeta vieja se comprimió
a `.tar.gz` **excluyendo** `.venv/`, `.git/`, `__pycache__/` y `node_modules/`
(basura regenerable, pesaba ~6.7 GB). El código fuente, datasets y scripts se
preservan íntegros dentro de los `.tar.gz`.

> Red de seguridad: NO borrar estos `.tar.gz`. Pueden eliminarse manualmente tras
> ~1 mes de uso estable de v5.

## Contenido archivado

| Archivo | Origen | Archivos dentro | Veredicto |
|---|---|---|---|
| `tletl_control_v3.tar.gz` | `tletl_control_v3/` | 12 | Fósil. 7 .py monolíticos (~8.9k líneas). Solo referencia de ideas. |
| `tletl_control_v4_modular.tar.gz` | `tletl_control_v4_modular/` | 20 | Primer split limpio (~1.6k líneas). Superado por el core. |
| `tletl_control_v4_1_ai_integrado.tar.gz` | `tletl_control_v4_1_ai_integrado/` | 259 | Cantera de v5: core + apps + tletl_control viejo + backups internos. |
| `tletl_backups.tar.gz` | `~/Documentos/tletl_backups/` | 2 | Zip del avance "core split" v4.9 (2026-05-09) + su `.sha256`. Enterrado en el cierre de F7 (2026-07-01), hash verificado antes de retirar el original. |

## Trazabilidad: qué de v5 viene de dónde

| Destino en `tletl_v5/` | Ancestro (dentro de `tletl_control_v4_1_ai_integrado.tar.gz`) |
|---|---|
| `tletl_core/{geometry,features,orientation,classifier,temporal,intent,state,bus}.py` | `tletl_core/*` (copiado tal cual) |
| `tletl_core/guard.py` | `tletl_control/ai_runtime_v47_live.py` (_apply_rule_guard, dicts) + `tletl_control/gestures.py` (reglas geométricas) |
| `tletl_core/critic.py` | `tletl_control/ai_critic_v47.py` (claves de orientación adaptadas al core nuevo) |
| `tletl_core/lowlight.py` | `tletl_control/low_light.py` |
| `tletl_core/adaptive.py` | `tletl_control/profile.py` (AdaptiveGestureMemory) |
| `tletl_core/pipeline.py` | NUEVO (orquesta classifier→guard→critic→temporal→adaptive) |
| `tletl_core/config.py` | NUEVO (consolida env vars dispersas en tletl.toml) |
| `apps/fedora_control/main.py` | `apps/fedora_control/main_core.py` (reescrito como cliente delgado + dual-hand + lowlight) |
| `apps/fedora_control/actions.py` | `apps/fedora_control/action_adapter.py` |
| `apps/blender_control/state_reader.py` | `apps/blender_control/blender_state_reader.py` (extendido) |
| `apps/blender_control/tletl_blender_addon.py` | NUEVO (addon bpy real + mapeo puro) |
| `tools/gesture_bank.py` | `tletl_control/gesture_bank_app.py` (al core, + `--dual-hand`) |
| `datasets/tletl_gesture_bank_v2_features.jsonl` | `datasets/tletl_gesture_bank_v2_features.jsonl` (banco sagrado, copiado sin tocar) |

## Restaurar algo

```bash
tar xzf _archive/tletl_control_v4_1_ai_integrado.tar.gz   # extrae la carpeta completa
```
