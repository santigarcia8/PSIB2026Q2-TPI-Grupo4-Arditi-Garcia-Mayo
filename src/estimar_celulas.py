"""
Estimación automática de la cantidad de células (2, 4 u 8) por análisis de textura.
Sin machine learning: se mide la textura del embrión y se decide con umbrales fijos.

Se calculan cuatro medidas de la matriz de co-ocurrencia (GLCM): contraste, homogeneidad, energía y entropía,
y además la DENSIDAD DE BORDES (Canny a escala gruesa), que es la que se usa para decidir:

    En un embrión de 2 células hay pocas membranas grandes; al dividirse aparecen más membranas internas.
    Con sigma grande (2 y 3) Canny ignora el granulado fino del citoplasma y se queda con las membranas,
    así que la longitud de borde por unidad de área sube con la cantidad de células. Se multiplica por el
    radio del embrión para que no dependa del tamaño.

    indice_bordes < UMBRAL_2_4              -> 2 células
    UMBRAL_2_4 <= indice_bordes < UMBRAL_4_8 -> 4 células
    indice_bordes >= UMBRAL_4_8             -> 8 células

Las medidas de la GLCM se calculan y se muestran, pero NO se usan para decidir: en la muestra de 10 imágenes sus
rangos se solapaban entre etapas (la textura global depende más del enfoque y del contraste de la foto).

IMPORTANTE: los umbrales de abajo salen de solo 3 fotos. Corré `python -m src.segmentacion calibrar` con las
10 y ajustalos.
"""
import numpy as np
from scipy import ndimage
from skimage.feature import canny, graycomatrix, graycoprops

ESTADIOS = (2, 4, 8)

NIVELES = 16                  # niveles de gris para la GLCM
DISTANCIAS = (2, 4, 8)        # píxeles
ANGULOS = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

SIGMAS_BORDES = (2.0, 3.0)    # escalas de Canny
UMBRAL_2_4 = 5.9              # AJUSTAR con `calibrar`
UMBRAL_4_8 = 7.5              # AJUSTAR con `calibrar`


def _recorte_inscrito(imagen, mascara):
    """Recorte cuadrado inscrito dentro del embrión (todo es embrión, sin fondo)."""
    dist = ndimage.distance_transform_edt(mascara)
    radio = dist.max()
    fila, col = np.unravel_index(dist.argmax(), dist.shape)
    medio = max(int(radio * np.sqrt(2) / 2), 8)
    f0, c0 = max(fila - medio, 0), max(col - medio, 0)
    return imagen[f0:fila + medio, c0:col + medio]


def medidas_textura(imagen_pre, embrion):
    """Devuelve contraste, homogeneidad, energía y entropía (todas en 0-1) y el índice."""
    embrion = embrion.astype(bool)
    if embrion.sum() < 500:
        raise ValueError("Máscara de embrión vacía o demasiado chica.")

    recorte = _recorte_inscrito(imagen_pre, embrion)
    radio = np.sqrt(embrion.sum() / np.pi)
    interior = ndimage.binary_erosion(embrion, iterations=3)
    dens = [canny(imagen_pre, sigma=sg, mask=interior)[interior].mean() * radio for sg in SIGMAS_BORDES]
    indice_bordes = float(np.mean(dens))

    # Se cuantiza con percentiles para que el contraste no dependa del brillo de la foto
    lo, hi = np.percentile(recorte, [1, 99])
    q = np.clip((recorte - lo) / (hi - lo + 1e-9), 0, 1)
    q = (q * (NIVELES - 1)).astype(np.uint8)

    glcm = graycomatrix(q, distances=list(DISTANCIAS), angles=list(ANGULOS),
                        levels=NIVELES, symmetric=True, normed=True)

    contraste = graycoprops(glcm, "contrast").mean() / (NIVELES - 1) ** 2
    homogeneidad = graycoprops(glcm, "homogeneity").mean()
    energia = graycoprops(glcm, "energy").mean()

    # Entropía de Shannon de la GLCM, normalizada por su máximo (log2 de nº de celdas)
    entropias = []
    for d in range(glcm.shape[2]):
        for a in range(glcm.shape[3]):
            p = glcm[:, :, d, a]
            p = p[p > 0]
            entropias.append(-(p * np.log2(p)).sum())
    entropia = float(np.mean(entropias)) / np.log2(NIVELES * NIVELES)

    indice = (contraste + entropia + (1 - homogeneidad) + (1 - energia)) / 4

    return {"contraste": float(contraste), "homogeneidad": float(homogeneidad),
            "energia": float(energia), "entropia": float(entropia),
            "indice_glcm": float(indice), "indice_bordes": indice_bordes}


def estimar_k(imagen_pre, embrion, verbose=True):
    """Devuelve la cantidad de células (2, 4 u 8) según el índice de textura."""
    m = medidas_textura(imagen_pre, embrion)
    if m["indice_bordes"] < UMBRAL_2_4:
        k = 2
    elif m["indice_bordes"] < UMBRAL_4_8:
        k = 4
    else:
        k = 8
    if verbose:
        print(f"  Textura: contraste={m['contraste']:.3f}  homogeneidad={m['homogeneidad']:.3f}  "
              f"energía={m['energia']:.3f}  entropía={m['entropia']:.3f}  "
              f"| bordes={m['indice_bordes']:.2f}  => {k} células")
    return k