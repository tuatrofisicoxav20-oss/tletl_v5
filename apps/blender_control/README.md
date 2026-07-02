# Tletl Blender Control

Control de objetos 3D en Blender mediante gestos de mano capturados con Tletl v5.

## Archivos

| Archivo | Descripción |
|---|---|
| `tletl_blender_addon.py` | Addon de Blender. Toda la lógica de mapeo es pura y testeable fuera de Blender. |
| `state_reader.py` | Lector del bus de estado (`tletl_state.json`). Sin dependencias de bpy. |

---

## Instalación del addon

1. Abre Blender 4.x.
2. Ve a **Edit > Preferences > Add-ons**.
3. Haz clic en **Install…** y selecciona `tletl_blender_addon.py`.
4. Activa la casilla del addon **"Tletl Gesture Control"**.
5. En las preferencias del addon configura la ruta al archivo de bus (`bus_path`), por defecto `~/tletl_state.json`.

---

## Uso

### 1. Lanzar el bus de Tletl

Desde la terminal, en la raíz del proyecto:

```bash
python3 -m tletl_core.main --bus ~/tletl_state.json
```

O con el launcher completo si está configurado.

### 2. Seleccionar un objeto en Blender

En cualquier vista 3D selecciona el objeto que quieres controlar (debe ser de tipo MESH).

### 3. Abrir el panel Tletl

Pulsa **N** en la vista 3D para abrir el panel lateral. Ve a la pestaña **Tletl**.

### 4. Iniciar el control

Haz clic en **Iniciar Tletl**. El addon empezará a leer el bus cada ~50 ms.

### 5. Gestos disponibles

| Gesto | Efecto |
|---|---|
| **PINCH** (mano dominante) | Traslada el objeto en XY siguiendo la palma. |
| **OPEN_PALM** (mano dominante) | Suelta el objeto (no mueve). |
| **FIST** (mano dominante) | Parada de seguridad: congela todo movimiento. |
| **PINCH** (mano modificadora) | Rota el objeto en Z según la posición horizontal de la palma. |

### 6. Ajustar parámetros

- **Gain**: amplificación del desplazamiento. Valores altos = movimiento más brusco.
- **Smoothing**: suavizado. 0 = sin suavizado, 1 = instantáneo.

---

## Banco de gestos

Para capturar muestras de entrenamiento usa la herramienta incluida:

```bash
python3 -m tools.gesture_bank --dataset mis_gestos.jsonl
# Con dos manos:
python3 -m tools.gesture_bank --dataset mis_gestos.jsonl --dual-hand
```

Teclas:
- `1`–`7`: seleccionar gesto (OPEN_PALM, FIST, POINT, VICTORY, PINCH, THREE, NEUTRAL)
- `SPACE`: guardar muestra
- `A`: activar/desactivar autosave
- `Q`: salir

---

## Tests

```bash
cd /home/exitili/Documentos/tletl_v5
python3 -m pytest tests/test_blender_map.py -v
```
