"""
Carga de la muestra de trabajo: 10 imágenes del dataset de Zenodo 14253170 (embriones humanos, contraste Hoffman),
guardadas en la carpeta muestra/ del repo. Devuelve imágenes en escala de grises, float en [0, 1].
Responsable: Olivia Mayo
"""

from pathlib import Path

import numpy as np
from skimage import io

EXTENSIONES_IMAGEN = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Ground truth para el conteo: cantidad de células que corresponde a cada estadio.
# Mórula y blastocisto no están porque ahí los blastómeros ya no se pueden contar uno por uno.
CELULAS_POR_ESTADIO = {"2cell": 2, "4cell": 4, "8cell": 8}

# Carpeta con la muestra de trabajo (unas 10 imágenes): 3 de 2 células, 3 de 4 y 4 de 8.
# Estructura: muestra/<estadio>/<archivo>.png. El origen de cada imagen en el dataset está en muestra/origen.txt.
CARPETA_MUESTRA = "muestra"


def a_gris(imagen):
    """Convierte una imagen RGB (o RGBA) a escala de grises. Si ya es gris, la devuelve igual."""
    if imagen.ndim == 2:
        return imagen
    # Descartamos el canal alfa (transparencia) si lo hay: no tiene información de la muestra
    rgb = imagen[..., :3].astype(np.float64)
    # Promedio ponderado por la sensibilidad del ojo a cada color (luminancia, Rec. 709)
    return 0.2125 * rgb[..., 0] + 0.7154 * rgb[..., 1] + 0.0721 * rgb[..., 2]


def normalizar(imagen, profundidad_bits):
    """Lleva una imagen entera de n bits a float en [0, 1] dividiendo por el máximo nivel posible (2^n - 1)."""
    maximo = 2 ** profundidad_bits - 1
    return imagen.astype(np.float64) / maximo


def cargar_imagen(ruta):
    """Lee una imagen del disco y la devuelve en gris, float en [0, 1]."""
    imagen = io.imread(ruta)

    # La profundidad de píxel sale del tipo de dato: uint8 -> 8 bits (L = 256), uint16 -> 16 bits (L = 65536)
    if imagen.dtype == np.uint8:
        bits = 8
    elif imagen.dtype == np.uint16:
        bits = 16
    else:
        raise ValueError(f"Tipo de dato no soportado en {ruta}: {imagen.dtype}")

    # Primero normalizamos y después pasamos a gris, así la conversión trabaja ya en [0, 1]
    imagen = normalizar(imagen, bits)
    imagen = a_gris(imagen)
    return np.clip(imagen, 0.0, 1.0)


def cargar_muestra(carpeta=CARPETA_MUESTRA):
    """
    Recorre la carpeta de la muestra y devuelve una lista de (ruta, estadio real).
    El estadio es la subcarpeta donde está cada imagen (por ejemplo "8cell").
    No carga las imágenes: se leen después, de a una, con cargar_imagen.
    """
    carpeta = Path(carpeta)
    lista = []
    for ruta in sorted(carpeta.rglob("*")):
        if ruta.suffix.lower() in EXTENSIONES_IMAGEN:
            estadio = ruta.relative_to(carpeta).parts[0]
            lista.append((ruta, estadio))
    if not lista:
        raise FileNotFoundError(f"No se encontraron imágenes en {carpeta}")
    return lista


if __name__ == "__main__":
    # Prueba rápida: python -m src.carga (desde la carpeta del repo)
    muestra = cargar_muestra()
    print(f"Muestra de trabajo: {len(muestra)} imágenes")
    for ruta, estadio in muestra:
        imagen = cargar_imagen(ruta)
        print(f"  {estadio:6s} ({CELULAS_POR_ESTADIO[estadio]} células)  {ruta.name}"
              f"  forma {imagen.shape}  rango [{imagen.min():.2f}, {imagen.max():.2f}]")
