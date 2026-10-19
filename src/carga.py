"""
Carga de imágenes y ground truth (Zenodo 14253170, BBBC003 y Cell Image Library).
Devuelve imágenes en escala de grises, float en [0, 1].
Responsable: Olivia Mayo
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import io

# Tamaño de píxel de BBBC003 informado por el dataset (µm por píxel).
# Permite pasar áreas de píxeles a µm²: area_um2 = area_px * TAM_PIXEL_BBBC003 ** 2
TAM_PIXEL_BBBC003 = 0.42

EXTENSIONES_IMAGEN = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


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


def cargar_cuadros(ruta):
    """
    Lee un archivo que puede tener varios cuadros (secuencia time-lapse en un TIFF de varias páginas)
    y devuelve una lista de imágenes 2D en gris, float en [0, 1]. Si tiene un solo cuadro, la lista tiene uno.
    """
    pila = io.imread(ruta)

    if pila.dtype == np.uint8:
        bits = 8
    elif pila.dtype == np.uint16:
        bits = 16
    else:
        raise ValueError(f"Tipo de dato no soportado en {ruta}: {pila.dtype}")
    pila = normalizar(pila, bits)

    # Formas posibles: (alto, ancho) gris, (alto, ancho, 3 o 4) color,
    # (cuadros, alto, ancho) secuencia en gris, (cuadros, alto, ancho, 3 o 4) secuencia en color
    es_color = pila.ndim >= 3 and pila.shape[-1] in (3, 4)
    if pila.ndim == 2 or (pila.ndim == 3 and es_color):
        cuadros = [pila]
    else:
        cuadros = list(pila)  # recorrer el primer eje = recorrer los cuadros en el tiempo

    return [np.clip(a_gris(cuadro), 0.0, 1.0) for cuadro in cuadros]


def cargar_mascara(ruta):
    """Lee una máscara de ground truth y la devuelve como bool (True = pertenece al objeto)."""
    # Las máscaras de BBBC003 son TIFF de 1 bit comprimidos con CCITT (un RLE pensado para fax).
    # scikit-image no decodifica ese formato sin paquetes extra; Pillow sí.
    mascara = np.array(Image.open(ruta))
    if mascara.ndim == 3:
        mascara = mascara[..., 0]
    return mascara > 0


def cargar_zenodo(carpeta):
    """
    Recorre el dataset de Zenodo y devuelve una lista de (ruta, estadio real).
    Estructura del zip: HumanEmbryo2.0/<estadio>/<train o test>/00001.png
    El estadio es la primera carpeta debajo de `carpeta` (por ejemplo "8cell").
    La división train/test no se usa: no entrenamos ningún modelo, así que se toman todas las imágenes.
    No carga las imágenes (son ~5500): se leen después, de a una, con cargar_imagen.
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


def cargar_cil(carpeta):
    """
    Carga las imágenes de la Cell Image Library y devuelve una lista de (nombre, imagen).
    Las secuencias time-lapse se separan en cuadros: el nombre lleva el número de cuadro (por ejemplo "CIL_123_t005").
    CIL no tiene ground truth, así que estas imágenes se usan para ilustrar y para probar el pipeline
    sobre casos nuevos (morfología en el tiempo, fragmentación, simetría), no para medir error.
    """
    carpeta = Path(carpeta)
    datos = []
    for ruta in sorted(carpeta.rglob("*")):
        if ruta.suffix.lower() not in EXTENSIONES_IMAGEN:
            continue
        cuadros = cargar_cuadros(ruta)
        if len(cuadros) == 1:
            datos.append((ruta.stem, cuadros[0]))
        else:
            for t, cuadro in enumerate(cuadros):
                datos.append((f"{ruta.stem}_t{t:03d}", cuadro))
    if not datos:
        raise FileNotFoundError(f"No se encontraron imágenes en {carpeta}")
    return datos


def leer_conteos_bbbc003(ruta_txt):
    """Lee el archivo de conteos (separado por tabulaciones) y devuelve {nombre_sin_extension: n_celulas}."""
    conteos = {}
    with open(ruta_txt, newline="") as archivo:
        lector = csv.reader(archivo, delimiter="\t")
        next(lector)  # la primera fila es el encabezado
        for fila in lector:
            if len(fila) < 2:
                continue
            nombre = Path(fila[0].strip()).stem
            conteos[nombre] = int(fila[1])
    return conteos


def buscar_mascara(carpeta_mascaras, nombre):
    """Busca la máscara con el mismo nombre que la imagen (en BBBC003 ambas se llaman, por ejemplo, 7_19_M2E5.tif)."""
    for ruta in sorted(Path(carpeta_mascaras).rglob(nombre + "*")):
        # El "_" o "." después del nombre evita que 7_19_M1E1 tome la máscara de 7_19_M1E18
        resto = ruta.stem[len(nombre):]
        if ruta.suffix.lower() in EXTENSIONES_IMAGEN and (resto == "" or resto.startswith("_")):
            return ruta
    return None


def cargar_bbbc003(carpeta):
    """
    Carga BBBC003 y devuelve una lista de (imagen, mascara GT, n° de células GT).
    Estructura esperada dentro de `carpeta`:
        imagenes/                  <- contenido de BBBC003_v1_images.zip
        mascaras/                  <- contenido de BBBC003_v1_foreground.zip
        BBBC003_v1_counts.txt
    """
    carpeta = Path(carpeta)
    conteos = leer_conteos_bbbc003(carpeta / "BBBC003_v1_counts.txt")

    datos = []
    for nombre, n_celulas in conteos.items():
        rutas = [r for r in (carpeta / "imagenes").rglob(nombre + ".*")
                 if r.suffix.lower() in EXTENSIONES_IMAGEN]
        if not rutas:
            print(f"Aviso: no encontré la imagen {nombre}, la salteo")
            continue
        ruta_mascara = buscar_mascara(carpeta / "mascaras", nombre)
        if ruta_mascara is None:
            print(f"Aviso: no encontré la máscara de {nombre}, la salteo")
            continue

        imagen = cargar_imagen(rutas[0])
        mascara = cargar_mascara(ruta_mascara)
        if mascara.shape != imagen.shape:
            raise ValueError(f"{nombre}: la máscara {mascara.shape} no tiene el tamaño de la imagen {imagen.shape}")
        datos.append((imagen, mascara, n_celulas))
    return datos


if __name__ == "__main__":
    # Prueba rápida: python -m src.carga (desde la carpeta del repo)
    bbbc = cargar_bbbc003("data/BBBC003")
    print(f"BBBC003: {len(bbbc)} imágenes")
    imagen, mascara, n = bbbc[0]
    print(f"  forma {imagen.shape}, tipo {imagen.dtype}, rango [{imagen.min():.3f}, {imagen.max():.3f}]")
    print(f"  máscara {mascara.dtype}, {mascara.mean() * 100:.1f}% del área es embrión, {n} células")

    zenodo = cargar_zenodo("data/HumanEmbryo2.0")
    estadios = sorted({estadio for _, estadio in zenodo})
    print(f"Zenodo: {len(zenodo)} imágenes, estadios: {estadios}")

    # CIL es opcional en la prueba: si todavía no bajaron imágenes, se saltea
    if Path("data/CIL").exists():
        cil = cargar_cil("data/CIL")
        print(f"CIL: {len(cil)} imágenes (contando cada cuadro de las secuencias)")
    else:
        print("CIL: todavía no hay carpeta data/CIL, se saltea")
