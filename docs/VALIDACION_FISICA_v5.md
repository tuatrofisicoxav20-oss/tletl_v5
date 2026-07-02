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

**Resultado FORMAL (batería instrumentada 2026-07-02 12:52, luz diurna,
posición de escritorio;** ventanas de 2 s como intento, mayoría de muestras del
bus en vivo**):**

| Gesto | Aciertos | Confusiones |
|---|---|---|
| OPEN_PALM | 9/9 | — |
| FIST | **0/10** | POINT ×10 |
| POINT | 10/10 | — |
| VICTORY | 10/10 | — |
| PINCH | **0/10** | NEUTRAL ×10 |
| THREE | 10/10 | — |
| NEUTRAL | 8/8 | — |

**Diagnóstico (re-test instrumentado 13:14, a la distancia de uso real de
Blender, leyendo raw/stable/critic del bus):**
- **FIST: 80/80 raw y stable, conf 0.92–1.00 (media 0.989)** → el 0/10 de la
  batería es artefacto de pose/distancia: en posición de escritorio el puño
  del usuario clasifica como POINT; a distancia de uso es impecable.
- **PINCH: raw correcto 68/80 (85 %) pero conf 0.62–0.70 < umbral 0.72 del
  critic** → degradado a NEUTRAL (61/80 stable). Solo pasa con conf=1.0.
  Sensible a luz/distancia: la noche anterior pasaba (el cubo se movió).
  Evidencia textual del critic: `confianza baja para PINCH: 0.62 < 0.72` ×16.

*(La validación informal de la madrugada — "casi no se confunde" — se hizo a
distancia de uso real, consistente con el re-test.)*

## 3. Examen guard/critic (2 min manos normales)
> Cómo: mismo dry-run. Dos minutos haciendo cosas normales frente a la
> cámara (teclear, gesticular hablando, agarrar la taza). Cuenta cuántas
> veces la línea `Guard:... | Critic:... | Intent:...` dispara un intent
> de acción sin que hicieras el gesto a propósito.

**Resultado FORMAL (2 min instrumentados, 2026-07-02):** **4 episodios** de
intent activo sin gesto deliberado:
- 1 **accionable**: POINT/scroll @3.8 s (fantasma real).
- 3 × **SAFETY_STOP** (@53.8, 105.2, 106.2 s): FIST detectado al agarrar
  objetos — un puño geométrico real; la intención disparada es la PAUSA de
  seguridad, que no ejecuta acciones pero interrumpiría el control.

**Guard/critic con 4 falsos positivos en 2 min — NO validado formalmente;
requiere revisión de umbral/banco (meta: 0).** La validación informal previa
(~1 h sin fantasmas percibidos) sugiere que en uso real el impacto es bajo,
pero el examen formal no se pasa.

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
- [formal 2026-07-02] **PINCH muere en el critic de día**: umbral fijo 0.72
  vs confianza real 0.62–0.70 a luz diurna. Candidatos v5.2: bajar umbral de
  PINCH, o enriquecer el banco con muestras diurnas (banco actual: fuente
  `training-lab-v4.6-correction`, capturado de noche).
- [formal 2026-07-02] **FIST es sensible a pose/distancia**: en posición de
  escritorio clasifica POINT (0/10); a distancia de uso, 80/80 con conf 0.99.
  Candidato v5.2: muestras de puño en pose de escritorio para el banco.
- [formal 2026-07-02] **FIST fantasma al agarrar objetos** → SAFETY_STOP
  espontáneo (3× en 2 min). Seguro pero molesto; revisar si SAFETY_STOP
  debe exigir hold temporal más largo.
- ___

## Veredicto: v5.1-validated → **SÍ** ✅

Validación física completada el 2026-07-02 (madrugada). §1 health OK,
§4 Blender completa (mover/rotar/escalar/FIST todo SÍ, latencia baja),
§2/§3 aprobadas de forma informal por el usuario. Bugs de setup anotados
en §5 quedan como backlog para v5.2.

### Adenda formal (2026-07-02, mediodía)

La ronda FORMAL instrumentada (batería + re-test diagnóstico contra el bus
en vivo) arroja: **5/7 gestos perfectos**; FIST y PINCH condición-sensibles
con causa raíz diagnosticada (pose/distancia el primero, umbral 0.72 del
critic el segundo); **§3 formal NO pasa** (4 FP en 2 min, desglose arriba).
El tag `v5.1-validated` (sesión informal a distancia de uso real) se mantiene
como histórico; **NO se emite tag nuevo** por la regla FP>0 del máster de
cierre. La recalibración (umbral PINCH / banco diurno / hold de SAFETY_STOP)
es el plan de **v5.2**.
