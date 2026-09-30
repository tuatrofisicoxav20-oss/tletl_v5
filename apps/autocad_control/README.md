# Tletl → AutoCAD / DXF / FreeCAD: modelar sólidos con gestos

Modela sólidos primitivos (caja, cilindro, esfera, cono, cuña) con las manos:
crearlos donde apunta la palma, arrastrarlos, rotarlos y escalarlos con los
mismos gestos que ya se validaron en Blender (`docs/VALIDACION_FISICA_v5.md` §4).

## Primero, la verdad incómoda

**AutoCAD no existe para Linux** y la app de cámara de Tletl corre en Fedora.
No hay forma de "abrir AutoCAD en Fedora y moverlo con gestos". Lo que SÍ hay
son exactamente tres rutas que funcionan, y las tres están implementadas:

| Ruta | Dónde corre el CAD | Cómo llega el bus | Estado |
|---|---|---|---|
| **1. AutoCAD en vivo** | Windows en la misma red (LAN) | UDP desde Fedora → `autocad_client.py` (COM/pywin32) | Código listo, **NO probado contra AutoCAD real** (este entorno es Linux, sin COM); probado con un ModelSpace falso |
| **2. DXF offline** | Fedora (sin CAD abierto) | archivo `~/.tletl/tletl_state.json` → `session.py` → `.dxf` (ezdxf) | **Probado de punta a punta** (export + relectura + audit) |
| **3. FreeCAD en vivo** | Fedora (libre, `dnf install freecad`) | archivo del bus → `freecad_macro.py` (QTimer) | Código listo, **NO ejecutado dentro de FreeCAD** aquí; helpers puros probados |

Las tres comparten el mismo cerebro: `gesture_modeler.py` (puro, sin ninguna
dependencia de CAD, con 40+ tests). Los backends solo aplican operaciones.

## Archivos

| Archivo | Qué es |
|---|---|
| `gesture_modeler.py` | `GestureModeler`: bus → ops (`add`/`move`/`rotate`/`scale`/`select`/`mode`/`kind`/`end_session`). `Scene`/`Solid`: modelo de la escena y reductor de ops. Sin CAD. |
| `bus_source.py` | `FileBusSource` (archivo, misma máquina), `UdpBusSource` (otra máquina), `ScriptedBusSource` (tests). |
| `dxf_export.py` | `export_scene_dxf()` con ezdxf (MESH por sólido, capa por tipo, R2010, `$INSUNITS`). JSON de escena para guardar/re-exportar. CLI de re-export. |
| `session.py` | Ruta 2: bucle poll → modeler → consola; al terminar guarda `escena.json` + `.dxf`. También `run_loop()` que usa el cliente de AutoCAD. |
| `autocad_client.py` | Ruta 1: cliente Windows (pywin32/COM). `AutoCADBackend` aplica ops con `AddBox`/`AddCylinder`/`Move`/`Rotate3D`/`ScaleEntity`. |
| `freecad_macro.py` | Ruta 3: macro de FreeCAD (Part::Box/Cylinder/Sphere/Cone + cuña extruida, `Placement`). |
| `../../launchers/tletl-cad-session.sh` | Lanza la ruta 2 con el venv del repo. |

Tests: `tests/test_gesture_modeler.py` y `tests/test_autocad_control.py`.

---

## Ruta 1 — AutoCAD en vivo (Windows en la LAN)

```
Fedora: cámara → tletl_core → bus JSON ──UDP 5055──▶ Windows: autocad_client.py ──COM──▶ AutoCAD
```

### En Fedora (emisor)

1. Averigua la IP del Windows (`ipconfig` allá, p. ej. `192.168.1.50`).
2. Activa la publicación UDP del bus, por config o por variable de entorno:
   ```toml
   # config/tletl.toml
   [bus]
   udp_target = "192.168.1.50:5055"
   ```
   ```bash
   export TLETL_BUS_UDP=192.168.1.50:5055
   ```
3. Corre la app de cámara sin tocar Fedora: `./launchers/tletl-blender-bus.sh`
   (dry-run: clasifica y escribe/publica el bus, no ejecuta acciones).

### En Windows (receptor + AutoCAD)

1. Instala Python 3.12 y `pip install pywin32` (solo eso: el cliente no usa
   numpy/opencv/mediapipe; importa `tletl_core.bus`, `tletl_core.paths` y este paquete).
2. Clona el repo (o copia las carpetas `tletl_core/` y `apps/` con sus `__init__.py`).
3. Firewall de Windows: permitir **UDP 5055 entrante** para `python.exe`
   (Windows lo pregunta la primera vez; si no, *Firewall → Reglas de entrada → Nueva regla → Puerto UDP 5055*).
4. Abre **AutoCAD con un dibujo activo** (Ctrl+N si hace falta). Sin dibujo activo
   `ActiveDocument` falla y el cliente lo dice y sale con código 2.
5. Verifica que llegan frames SIN tocar AutoCAD:
   ```bat
   python -m apps.autocad_control.autocad_client --udp 0.0.0.0:5055 --dry-run
   ```
   Debes ver líneas `[cad] modo=... dom=PINCH ...` con `bus=0.0xs`. Si dice
   `bus=sin frames`, es red/firewall (ver Troubleshooting).
6. Modela:
   ```bat
   python -m apps.autocad_control.autocad_client --udp 0.0.0.0:5055 --units mm --gain 100
   ```
   Cada op se imprime (`[op] add #1 box ...`) y se aplica al ModelSpace. Ctrl+C para salir.
   `--load escena.json` dibuja primero una escena guardada por la ruta 2.

Detalle COM importante (ya resuelto en `_point()`): AutoCAD exige que los puntos
viajen como `VARIANT(VT_ARRAY | VT_R8, [x, y, z])`; una tupla de Python produce
"Invalid argument". Si AutoCAD está ocupado con un comando abierto, la op falla,
se registra (`[autocad] error aplicando ...`) y el bucle sigue: cierra el
comando (Esc) y continúa.

---

## Ruta 2 — DXF offline en Fedora (la que se probó de punta a punta)

```bash
pip install ezdxf                      # o: pip install -e '.[cad]'
./launchers/tletl-blender-bus.sh       # terminal 1: la cámara alimenta el bus
./launchers/tletl-cad-session.sh --export modelo.dxf --units mm   # terminal 2
```

- Empiezas en modo **TRANSFORM** sin sólidos: **THREE 0.8 s** → **CREATE**, **PINCH** crea
  un sólido donde está la palma, **VICTORY** cambia el tipo. THREE 0.8 s de vuelta
  a TRANSFORM para arrastrar/rotar/escalar el seleccionado (el último creado).
- Termina con **Ctrl+C** o **FIST sostenido 3 s**. Se escriben `modelo.json`
  (la escena, para continuar con `--load modelo.json`) y `modelo.dxf`.
- Abre el DXF en AutoCAD (`OPEN`, o arrástralo a la ventana), FreeCAD, LibreCAD,
  QCAD, Fusion, etc. Los sólidos son entidades **MESH** (DXF R2010) en capas
  `TLETL_BOX`, `TLETL_CYLINDER`, `TLETL_SPHERE`, `TLETL_CONE`, `TLETL_WEDGE`, con
  `$INSUNITS` fijado a las unidades elegidas. En AutoCAD, `CONVTOSOLID` convierte
  una malla cerrada en sólido 3D si lo necesitas.
- Sin ezdxf la sesión igual guarda el JSON y te dice cómo re-exportar:
  `python -m apps.autocad_control.dxf_export modelo.json [salida.dxf --units m]`.

El bus también puede llegar por UDP (`--udp 0.0.0.0:5055`) si la cámara está en otra máquina.

---

## Ruta 3 — FreeCAD en vivo en Fedora

```bash
sudo dnf install freecad
export TLETL_REPO=/ruta/a/tletl_v5      # recomendado: usa el GestureModeler completo
./launchers/tletl-blender-bus.sh &      # la cámara alimenta el bus
freecad
```

En FreeCAD: **Macro → Macros…**, botón de carpeta *Ubicación de macros de usuario*
(normalmente `~/.local/share/FreeCAD/Macro/`), copia ahí `freecad_macro.py` (o
elige el archivo dentro del repo) y **Ejecutar**. La macro crea un documento
"Tletl" (o usa el activo) y un `QTimer` de 50 ms que lee
`~/.tletl/tletl_state.json` (misma regla de ruta que el core: `TLETL_STATE_PATH`
> `TLETL_HOME` > `~/.tletl`) y aplica las ops a primitivas `Part`.

- Con `TLETL_REPO` (o la macro ejecutada desde dentro del repo) se usa el mismo
  `GestureModeler` de las otras rutas (mover, rotar, escalar, crear). El FIST de
  3 s NO termina nada en FreeCAD: la macro ignora `end_session` y sigue con su
  timer; se detiene con `stop()` en la consola de Python.
- Sin él, `MinimalModeler` (inline, sin dependencias): crear, cambiar tipo,
  mover, cambiar de modo. Sin rotación ni escala.
- En la consola de Python de FreeCAD: `stop()` detiene el timer, `start(gain=200)` reinicia.
- FreeCAD trabaja en **mm**; el gain por default es 100 (el encuadre = 10 cm).

---

## Gestos (mano dominante = `dom`, la otra = `mod`)

Ambos modos:

| Gesto | Efecto |
|---|---|
| **THREE** sostenido 0.8 s | Alterna **TRANSFORM ↔ CREATE** (un cambio por hold; suelta y vuelve a hacerlo para el siguiente) |
| **FIST** | Parada de seguridad: no toca nada y rompe la continuidad |
| **FIST** sostenido 3 s | Termina la sesión (ruta 2 exporta el DXF). `--end-hold 0` lo desactiva |
| NEUTRAL / sin mano | Nada |
| bus obsoleto (el `timestamp` deja de avanzar > 1 s) | Nada; se rompe la continuidad (sin saltos al volver). Se juzga por el avance del timestamp del flujo, no por el reloj del receptor: un cliente Windows con el reloj desfasado respecto a Fedora funciona igual (solo se descarta el primer frame si el bus en disco era viejo) |

Modo **TRANSFORM** (sobre el sólido seleccionado = el último creado):

| Gesto | Efecto |
|---|---|
| dom **PINCH** y mover | Arrastra en el plano XY por **delta** de la palma (mano arriba = +Y) |
| dom **OPEN_PALM** | Suelta: al volver a pinzar no hay salto |
| mod **PINCH** y mover horizontal | Rota en Z: `Δx · rot_gain · π` radianes, es decir `rot_gain/2` vueltas por ancho de encuadre (con el default 2.0, cruzar todo el encuadre = una vuelta) |
| mod **PINCH** y mover vertical | Sube/baja en Z solo si `--z-gain` > 0 (default off) |
| dom **PINCH** + mod **OPEN_PALM** | Escala: separar las manos agranda, juntarlas achica (mín. 0.05×). La dominante sigue arrastrando (igual que en Blender): mantenla quieta mientras escalas |

Modo **CREATE**:

| Gesto | Efecto |
|---|---|
| dom **PINCH** (al cerrar) | Crea un sólido del tipo actual en la palma proyectada al plano XY (z = 0) y lo selecciona. Mantener la pinza no crea más; suelta y pinza de nuevo para otro |
| dom **VICTORY** (al hacerlo) | Cicla el tipo: box → cylinder → sphere → cone → wedge → box |

Un frame aislado nunca mueve nada (hace falta el frame anterior para el delta):
es "agarrar y arrastrar", no joystick. No hay suavizado extra: el filtro temporal
+ guard/critic del core ya estabilizan la señal (hallazgo de la validación en Blender).

## Unidades y ganancia

- `--units {mm,cm,m,in}` fija `$INSUNITS` del DXF (y las unidades que asume el cliente).
- `--gain` = unidades de dibujo que recorre un sólido cuando la mano cruza **todo**
  el encuadre; también define dónde nace un sólido en CREATE (el centro de la
  cámara es el origen). Defaults por unidades: mm=100, cm=10, m=1, in=4 (≈10 cm).
- `--spawn-size` = tamaño del sólido nuevo (default `gain/10`).
- `--rot-gain 2.0`, `--scale-gain 1.5` (valores del addon de Blender); `--deadzone 0.01`
  ignora temblor de la palma menor a 1 % del encuadre; `--stale 1.0` (0 = no juzgar).
- En AutoCAD las magnitudes son las del dibujo: con un `acadiso.dwt` (mm) y
  `--gain 100`, un cubo nuevo mide 10 mm.

## Limitaciones (honestas)

- **COM no probado en este entorno**: no hay Windows ni AutoCAD aquí. El backend
  se probó con un ModelSpace falso que registra cada llamada y con un `pythoncom`
  falso para el `VARIANT`. Los nombres/firmas son los de la referencia ActiveX de
  AutoCAD. La primera vez que lo corras, hazlo con `--dry-run` y luego con una
  escena vacía.
- **FreeCAD no ejecutado aquí**: se comprueban compilación, importación sin
  FreeCAD, la regla de ruta del bus y el `MinimalModeler` contra el modeler real.
- Solo primitivas (box/cylinder/sphere/cone/wedge), una selección (el último
  creado), sin deshacer por gestos (usa `U`/Ctrl+Z del CAD; la escena JSON no lo sabrá).
- La rotación es solo alrededor de Z; la creación es siempre en z = 0.
- UDP: sin cifrado ni autenticación (LAN de confianza); "el frame más reciente
  gana" si el cliente va más lento que la cámara.
- La ruta 1 exige que la app de Fedora **publique** el bus por UDP (`[bus] udp_target`).

## Troubleshooting

| Síntoma | Causa / solución |
|---|---|
| `bus=sin frames` en el cliente Windows | No llega UDP: revisa `udp_target`/`TLETL_BUS_UDP` en Fedora, la IP, el firewall (UDP 5055 entrante) y que ambas máquinas estén en la misma LAN. Prueba `--dry-run`. |
| `pywin32 no está instalado` | `pip install pywin32` en el Python de Windows que ejecutas. |
| `AutoCAD no tiene un dibujo activo` | Abre un dibujo (Ctrl+N) y vuelve a correr. |
| `Invalid argument` en AddBox/Move | Puntos no VARIANT: no envuelvas `_point()`; reporta la versión de pywin32. |
| `[autocad] error aplicando ...` repetido | AutoCAD tiene un comando abierto o un cuadro modal: Esc / cierra el diálogo. |
| Nada se crea en CREATE | PINCH no llega como estable (critic). Mira `dom=` en la línea de estado; ajusta `[critic.min_conf] PINCH` como en la validación. |
| El sólido "salta" al volver a pinzar | No debería: OPEN_PALM/FIST rompen la continuidad. Si pasa, sube `--stale` o revisa que el bus tenga `timestamp` real. |
| `bus obsoleto` permanente en Windows aunque lleguen frames | Ya no debería ocurrir (la obsolescencia se juzga por el avance del timestamp, no por el reloj local). Si aun así pasa, `--stale 0` desactiva el juicio. |
| `no pude conectar con AutoCAD: com_error` | AutoCAD no está instalado/registrado en ese Windows, o está abierto como otro usuario. Abre AutoCAD primero y corre el cliente en la misma sesión. |
| `ezdxf no está instalado` | `pip install ezdxf`; el JSON ya quedó guardado, re-exporta con `dxf_export`. |
| DXF abre pero no se ve nada | `ZOOM E` (extents). Los sólidos están en capas `TLETL_*`; revisa que no estén apagadas. |
| FreeCAD: `esta macro debe ejecutarse dentro de FreeCAD` | Corriste el archivo con python normal. Úsalo desde Macro → Macros…. |
| FreeCAD no rota ni escala | Falta `TLETL_REPO`: exporta la variable antes de abrir FreeCAD. |

## Tests

```bash
source .venv/bin/activate
pip install ezdxf
python -m pytest tests/test_gesture_modeler.py tests/test_autocad_control.py -q
```
