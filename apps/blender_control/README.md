# Tletl Blender Control

Control y **creación** de objetos 3D en Blender con gestos de mano capturados por
Tletl v5. La app de cámara escribe un bus JSON (`tletl_state.json`) ~30 veces por
segundo; el addon lo lee con un timer y mueve, rota, escala o crea el objeto activo.

## Archivos

| Archivo | Descripción |
|---|---|
| `tletl_blender_addon.py` | Addon de Blender. Toda la lógica (mapeo con suavizado real, máquina de estados TRANSFORM/CREATE, estado del bus) es pura y testeable fuera de Blender; bpy sólo ejecuta comandos. |
| `state_reader.py` | Lector del bus. Sin bpy ni `tletl_core` (el Python de Blender no puede importar el core). Replica la regla de rutas de `tletl_core/paths.py`. |
| `../../tools/build_blender_addon.py` | Empaqueta ambos archivos + `blender_manifest.toml` en `dist/tletl_blender_addon-<versión>.zip`. |
| `../../launchers/tletl-build-blender-addon.sh` | Atajo del empaquetado. |

---

## Instalación del addon

**No instales `tletl_blender_addon.py` suelto**: la validación física lo hizo y
tronó con `ModuleNotFoundError: state_reader`. (Desde v5.2 el archivo suelto ya
funciona gracias a una copia inline del lector, pero el zip es la vía soportada.)

1. Genera el zip:

   ```bash
   ./launchers/tletl-build-blender-addon.sh        # -> dist/tletl_blender_addon-5.2.0.zip
   ```

2. Instálalo en Blender **por cualquiera de las dos vías** (el mismo zip sirve para ambas):

   | Blender | Ruta |
   |---|---|
   | 4.2+ como **extensión** | **Edit > Preferences > Get Extensions** > menú **˅** (arriba a la derecha) > **Install from Disk…** > elige el zip |
   | 4.0–4.1 o como **addon legacy** | **Edit > Preferences > Add-ons** > **Install…** (4.2+: menú **˅** > **Install from Disk…**) > elige el zip |

3. Activa la casilla **"Tletl Gesture Control"**.
4. En la vista 3D pulsa **N** > pestaña **Tletl**.

---

## Uso

### 1. Alimentar el bus

En una terminal, en la raíz del proyecto:

```bash
./launchers/tletl-blender-bus.sh           # clasifica y escribe el bus SIN tocar Fedora (dry-run), sin ventana
./launchers/tletl-blender-bus.sh --window  # igual, pero mostrando la ventana de cámara con el panel
```

### 2. Ruta del bus (app y addon deben coincidir)

Ambos lados usan **la misma regla** (`tletl_core/paths.py` ≡ `state_reader.default_bus_path()`):

```
$TLETL_STATE_PATH  >  $TLETL_HOME/tletl_state.json  >  ~/.tletl/tletl_state.json
```

Con los defaults la app escribe y el addon lee `~/.tletl/tletl_state.json`. Si
exportas `TLETL_STATE_PATH` o `TLETL_HOME` **antes de abrir Blender**, el addon
toma ese default; también puedes escribir la ruta a mano en **Bus path** (prefs
o panel). Vacío = regla común.

### 3. Seleccionar un objeto y arrancar

Selecciona el objeto a controlar (mesh, empty, cámara, luz…), pulsa **Iniciar
Tletl** en el panel. El timer lee el bus cada `interval` segundos (0.033 ≈ 30 Hz).

### 4. Estado del bus en el panel

| Mensaje | Significa |
|---|---|
| `Conectado (N fps app, M ms)` | Frames frescos; `M` es la edad del último frame. |
| `BUS OBSOLETO (x.x s) — ¿corre la app de cámara?` | Hay archivo pero su `timestamp` es más viejo que **Stale after**: la app se cerró o se congeló. El addon **no actúa** (resetea continuidad) hasta que vuelvan frames frescos. |
| `Sin bus en <ruta>` | No existe el archivo: lanza `tletl-blender-bus.sh` o corrige **Bus path**. |
| `Bus ilegible en <ruta>` | El archivo no es JSON válido (escritura a medias, permisos). |

El panel también muestra: control ON/OFF de la app (`extra.control`, informativo),
gesto y confianza de cada mano, intent, modo de la app (NAVEGADOR/CURSOR/VENTANAS),
modo del addon, primitiva actual y el último error del timer. El panel **nunca lee
el bus**: muestra lo que cacheó el último tick.

---

## Gestos

Dos modos del addon, conmutables desde el panel o con la mano dominante:

- **THREE sostenido ≥ 0.8 s** alterna TRANSFORM ⇄ CREATE **una vez por sostén**
  (hay que soltar THREE para volver a alternar).
- **FIST** es seguridad en ambos modos.

### Modo TRANSFORM (objeto activo)

| Gesto | Efecto |
|---|---|
| **PINCH** (dominante) | Arrastra el objeto en XY siguiendo el delta de la palma (derecha → +X, arriba → +Y). |
| **OPEN_PALM** (dominante) | Suelta: rompe la continuidad, no mueve (al volver a PINCH no salta). |
| **FIST** (dominante) | Parada de seguridad: congela todo y descarta el suavizado pendiente. |
| **PINCH** (modificadora), delta horizontal | Rota en Z (`Rot gain`). |
| **PINCH** (modificadora), delta vertical | Eleva/baja en Z si `Z gain` > 0 (default 0 = apagado, como se validó). |
| **PINCH** dom + **OPEN_PALM** mod | Escala uniforme al separar/juntar las manos (`Scale gain`). Respeta escalas no uniformes: un objeto (1, 2, 1) se vuelve (k, 2k, k). |
| **THREE** (dominante) ≥ 0.8 s | Cambia a CREATE. |
| NEUTRAL / POINT / VICTORY / sin mano | Nada. |

Si mueves o deshaces el objeto a mano (G, Ctrl+Z) el addon lo detecta y descarta
el movimiento pendiente: nunca "vuelve solo" a donde iba.

### Modo CREATE (generar modelos)

| Gesto | Efecto |
|---|---|
| **PINCH** (dominante), flanco de subida | Crea la primitiva actual donde está la palma: plano XY alrededor del cursor 3D (`(0.5, 0.5)` = cursor; misma ganancia que la traslación), z = z del cursor. El objeto nuevo queda **activo y seleccionado**. Mantener PINCH no crea más. |
| **VICTORY** (dominante), flanco de subida | Cicla la primitiva: cube → sphere → cylinder → cone → plane → cube. |
| **OPEN_PALM** (dominante) | Nada. |
| **FIST** (dominante) | Seguridad: no crea. |
| **THREE** (dominante) ≥ 0.8 s | Vuelve a TRANSFORM (y ya puedes mover lo recién creado con PINCH). |

Las primitivas se generan con `bmesh` (nunca `bpy.ops`: el timer no tiene contexto
de operador) y se enlazan a la colección activa. El botón **Crear en el cursor**
usa la misma rutina sin gestos.

---

## Preferencias (también en el panel)

| Parámetro | Default | Qué hace |
|---|---|---|
| **Bus path** | regla común | Ruta al `tletl_state.json`. Vacío = default. |
| **Gain** | 10.0 | Unidades de Blender por ancho de cámara recorrido con PINCH (el cubo por defecto mide 2). |
| **Smoothing** | 0.5 | Fracción del camino al target por tick. 1.0 = instantáneo. Es suavizado **real**: con la mano quieta el objeto converge al desplazamiento completo; nada se pierde. |
| **Rot gain** | 2.0 | Medias vueltas en Z por ancho de cámara con mod PINCH. |
| **Scale gain** | 1.5 | Sensibilidad de la escala a la distancia entre manos. |
| **Z gain** | 0.0 | Elevación en Z con el delta vertical de mod PINCH (0 = apagado). |
| **Stale after (s)** | 1.0 | Edad máxima del frame antes de declarar el bus obsoleto. |
| **Interval (s)** | 0.033 | Periodo del timer. |

> Por qué cambiaron los defaults: en la validación (`docs/VALIDACION_FISICA_v5.md`
> §4) el usuario acabó en *Gain 14 / Smoothing 1.0* porque el suavizado viejo
> tiraba el `(1 − smoothing)` de cada delta (con 0.35 se perdía el 65 % del
> movimiento). Con el suavizado real, `Gain 10 / Smoothing 0.5` da el mismo
> recorrido y además alisa el temblor.

---

## Solución de problemas

- **`Sin bus en ~/.tletl/tletl_state.json`** → la app no corre o escribe en otra
  ruta. Comprueba la línea `Bus:` del panel de la app de cámara y el valor de
  `TLETL_STATE_PATH`/`TLETL_HOME` en ambas terminales.
- **`BUS OBSOLETO`** con la app abierta → la app se congeló o Blender y la app no
  comparten reloj (otra máquina): sube **Stale after**.
- **Blender Flatpak** (`flatpak run org.blender.Blender`) **no ve `/run/user/<uid>`**
  (tmpfs de `XDG_RUNTIME_DIR`) ni otras rutas fuera de su sandbox, y tampoco hereda
  las variables que exportes en la terminal si lo lanzas desde el escritorio.
  Mantén el bus en `~/.tletl` (el default, dentro del HOME que el Flatpak sí ve)
  o concede acceso explícito, p. ej. para `TLETL_STATE_PATH=$XDG_RUNTIME_DIR/tletl/tletl_state.json`:
  `flatpak override --user --filesystem=xdg-run/tletl org.blender.Blender` y lanza
  `flatpak run --env=TLETL_STATE_PATH=... org.blender.Blender`.
- **El objeto no se mueve** → mira el panel: ¿`Conectado`? ¿Dom = PINCH con
  confianza alta? Si PINCH sale NEUTRAL, es el critic de la app (umbral en
  `config/tletl.toml`, `[critic.min_conf]`). ¿Modo del addon = TRANSFORM? ¿Hay
  objeto activo de tipo controlable?
- **Se mueve a saltos** → sube **Smoothing** hacia 1.0 sólo si quieres seguimiento
  crudo; para alisar bájalo (0.3–0.5). Nunca se pierde recorrido.
- **`Error: …` en el panel** → el timer capturó una excepción y sigue vivo; el
  texto es la causa (p. ej. `rotation_mode 'QUATERNION' no soportado`: el addon
  escribe `rotation_euler`, cambia el objeto a Euler XYZ).
- **Prefs no encontradas: usando defaults** → Blender registró el addon con otro
  id (instalación manual rara). Reinstala desde el zip.

---

## Banco de gestos

Para capturar muestras de entrenamiento usa la herramienta incluida:

```bash
./launchers/tletl-bank.sh --dataset mis_gestos.jsonl
./launchers/tletl-bank.sh --dataset mis_gestos.jsonl --dual-hand   # con dos manos
```

Teclas: `1`–`7` gesto (OPEN_PALM, FIST, POINT, VICTORY, PINCH, THREE, NEUTRAL),
`SPACE` guardar, `A` autosave, `Q` salir.

---

## Tests

Todo corre sin Blender ni cámara (bpy se simula ausente):

```bash
source .venv/bin/activate
python -m pytest tests/test_blender_map.py tests/test_blender_session.py \
                 tests/test_state_reader.py tests/test_blender_addon_build.py \
                 tests/test_integration_bus.py -q
python -m pytest tests/ -q          # suite completa
```

| Test | Cubre |
|---|---|
| `test_blender_map.py` | Mapper: deltas, safety, suavizado real y resincronizaciones, Z, escala no uniforme. |
| `test_blender_session.py` | Máquina de estados TRANSFORM/CREATE, sostén de THREE, flancos, `bus_status`, settings. |
| `test_state_reader.py` | Lector del bus y **paridad de rutas con `tletl_core.paths`**. |
| `test_blender_addon_build.py` | Zip/manifiesto, import como paquete, copia inline del lector (archivo suelto). |
| `test_integration_bus.py` | Bus REAL de la app → lector → mapper/sesión (obsoleto, control, CREATE). |
