# TLETL V5 — VALIDACIÓN FÍSICA
Fecha: 2026-07-02 (madrugada) | Luz: ___

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
> sin tocar Fedora). El addon YA está instalado y activado (2026-07-01,
> headless; el bridge viejo de v3 quedó desactivado para que no mueva el
> objeto en paralelo). Si Blender estaba abierto de antes, ciérralo y
> ábrelo de nuevo. Vista 3D → N-panel pestaña "Tletl" → **Iniciar Tletl**,
> con el cubo seleccionado. FPS: overlay de estadísticas de Blender o los
> del panel de la app de cámara; el addon refresca a ~20 Hz (timer 0.05 s).

FPS: ~15 (bus) | Latencia percibida: **baja** ("completamente fácil de usar")
PINCH mueve cubo:        **SÍ**
Rotación mano 2:         **SÍ**   *(mod PINCH → rotación Z)*
Scale con spread:        **SÍ**   *(dom PINCH + mod OPEN_PALM, distancia entre manos)*
FIST frena en seco:      **SÍ**

Calibración final del usuario: **Gain 14 / Smoothing 1.0** (suavizado del addon
apagado a propósito: el filtro temporal + guard/critic del core ya estabilizan
la señal lo suficiente — hallazgo de la validación, no un default recomendado).

## 5. Bugs anotados (NO parchados)
> Regla del máster: en validación solo se ANOTA; se parcha después con
> tests, nunca en caliente.

- [setup 2026-07-01] El README/addon dice "instalar `tletl_blender_addon.py`"
  como archivo suelto, pero su fallback de import necesita `state_reader.py`
  AL LADO → instalado solo, truena con `ModuleNotFoundError: state_reader`.
  Fix futuro: distribuir como zip/carpeta con ambos archivos y corregir README.
- [sistema, no Tletl] Blender en Fedora 43 arranca con ERROR de OpenColorIO
  (config.ocio v2.5 vs librería 2.4.2, otro desfase de paquetes) — no bloquea
  la validación (solo gestión de color); se arregla cuando Fedora rebuildee.
- [setup 2026-07-02] Rutas del bus INCONSISTENTES por default: el launcher
  escribe `<repo>/tletl_state.json` (cwd-relativo) pero el addon lee
  `~/tletl_state.json` (DEFAULT_BUS_PATH). Con defaults jamás se ven.
  Workaround activo: symlink `~/tletl_state.json` → `<repo>/tletl_state.json`.
  Fix futuro: unificar default (ruta absoluta compartida, p.ej. XDG_RUNTIME_DIR).
- [setup 2026-07-02] `requirements.txt` pide `mediapipe>=0.10.21` SIN tope:
  pip instala 0.10.35, que ya no trae la API legacy `mp.solutions` →
  la app truena al arrancar. Instalado 0.10.21 (con numpy 1.26 + cv2 4.11).
  Fix futuro: pinear `mediapipe>=0.10.21,<0.10.30` o migrar a la Tasks API.
- ___

## Veredicto: v5.1-validated  SÍ / NO
