"""apps/autocad_control — modelar sólidos en CAD con gestos Tletl.

AutoCAD no existe para Linux y la app de cámara corre en Fedora, así que este
paquete ofrece TRES rutas reales (ver README.md de esta carpeta):

  1. AutoCAD EN VIVO en un Windows de la misma red: la app publica el bus por
     UDP y `autocad_client.py` (Windows, pywin32/COM) dibuja los sólidos.
  2. DXF OFFLINE en Fedora: `session.py` modela con gestos y exporta un .dxf
     (`dxf_export.py`, ezdxf) que se abre en AutoCAD o en cualquier CAD.
  3. FreeCAD EN VIVO en Fedora (libre, nativo): `freecad_macro.py`.

Toda la decisión (gesto -> operación) vive en `gesture_modeler.py`, puro y
testeado; los backends (COM, ezdxf, FreeCAD) solo aplican operaciones.

Este __init__ no importa submódulos a propósito: importar el paquete nunca
debe exigir ezdxf, pywin32 ni FreeCAD.
"""

__all__ = [
    "autocad_client",
    "bus_source",
    "dxf_export",
    "freecad_macro",
    "gesture_modeler",
    "session",
]
