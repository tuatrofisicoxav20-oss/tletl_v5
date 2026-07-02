# TLETL V5 — VALIDACIÓN FÍSICA
Fecha: ___ | Luz: buena/media

> Hoja de prueba manual de Emiliano. Se llena EN VIVO frente a la cámara.
> Al terminar: si el veredicto es SÍ, se commitea esta hoja llena y se
> taggea `v5.1-validated`.

## 1. Health
> Cómo: `./launchers/tletl-health.sh`

tletl-health.sh: OK / FALLA

*(Pre-check 2026-07-01 sin cámara: **OK**, exit=0 — core v4.9.1, banco
2363 balanceadas, intent PINCH→GRAB_OR_SELECT. Re-correr el día de la prueba.)*

## 2. Gestos (dry-run, 10 intentos c/u)
> Cómo: `./launchers/tletl-fedora-safe.sh` (exporta `TLETL_DRY_RUN=1`, no
> ejecuta nada). El panel muestra por mano el **gesto estable + confianza**
> (p. ej. `PINCH 0.87`). Cuenta un acierto cuando el gesto estable coincide
> con el que hiciste; si sale otro, anótalo en "confusiones".

OPEN_PALM: __/10 | confusiones: ___
VICTORY:   __/10 | confusiones: ___
PINCH:     __/10 | confusiones: ___
THREE:     __/10 | confusiones: ___
NEUTRAL:   __/10 | confusiones: ___
POINT:     __/10 | confusiones: ___   *(añadido: es el gesto más usado en NAVEGADOR/CURSOR)*
FIST:      __/10 | confusiones: ___   *(añadido: es el gesto de seguridad; conviene medirlo también en dry-run)*

## 3. Examen guard/critic (2 min manos normales)
> Cómo: mismo dry-run. Dos minutos haciendo cosas normales frente a la
> cámara (teclear, gesticular hablando, agarrar la taza). Cuenta cuántas
> veces la línea `Guard:... | Critic:... | Intent:...` dispara un intent
> de acción sin que hicieras el gesto a propósito.

Intents fantasma disparados: __ (meta: 0)

## 4. Blender
> Cómo: terminal 1 → `./launchers/tletl-blender-bus.sh` (alimenta el bus
> sin tocar Fedora). En Blender: Edit > Preferences > Add-ons > Install →
> `apps/blender_control/tletl_blender_addon.py`, activar. Vista 3D →
> N-panel pestaña "Tletl" → **Iniciar Tletl**, con el cubo seleccionado.
> FPS: overlay de estadísticas de Blender o los del panel de la app de
> cámara; el addon refresca a ~20 Hz por diseño (timer 0.05 s).

FPS: __ | Latencia percibida: baja/media/alta
PINCH mueve cubo:        SÍ/NO
Rotación mano 2:         SÍ/NO   *(mod PINCH → rotación Z)*
Scale con spread:        SÍ/NO   *(dom PINCH + mod OPEN_PALM, distancia entre manos)*
FIST frena en seco:      SÍ/NO

## 5. Bugs anotados (NO parchados)
> Regla del máster: en validación solo se ANOTA; se parcha después con
> tests, nunca en caliente.

- ___

## Veredicto: v5.1-validated  SÍ / NO
