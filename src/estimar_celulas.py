import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from skimage.feature import canny, graycomatrix, graycoprops

#estimar_celulas recibe la imagen preprocesada y la mascara del embrion y devuelve 2,4 u 8 celulas

#Se definen constantes y parametros de la GLCM (se fueron probando distintos y estos son los que mejor se ajustan)
NIVELES = 16                  # niveles de gris para la GLCM (32 es mucho y 8 se pierde detalle)
DISTANCIAS = (2, 4, 8)        # pares de píxeles
ANGULOS = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4) #se toman los pares de pixeles en 4 direcciones para que no dependa de la orientacion del embrion

SIGMA_BORDES = 1.5            #cuanto se suaviza la imagen antes de buscar bordes con Canny. Un valor mas alto ignora detalles finos
LARGO_MIN_REL = 0.6           # largo mínimo de un segmento de membrana, en radios del embrión

#Umbrales determinados experimentalmente para las 10 muestras que se utilizan
UMBRAL_COMPARTIMIENTO = 0.59  #
UMBRAL_4_8 = 7.0

#Busca el punto mas interno del embrion (mas lejos de la mascara) y recorta un cuadrado centrado que cabe entero dentro del embrion
#El recorte no incluye fondo ni membrana pelucida, solo lo usan las medidas de la GLCM
def _recorte_inscrito(imagen, mascara):
    dist = ndimage.distance_transform_edt(mascara) #calcula para cada pixel del embrion, la distancia al borde mas cercano de la mascara
    radio = dist.max()
    fila, col = np.unravel_index(dist.argmax(), dist.shape)
    medio = max(int(radio * np.sqrt(2) / 2), 8)
    f0, c0 = max(fila - medio, 0), max(col - medio, 0)
    return imagen[f0:fila + medio, c0:col + medio]
#La GLCM mide pares de pixeles vecinos, si el recorte incluyera fondo o zona pelucida, mediria el contraste embrion-fondo y no la textura


def _indice_membranas(imagen_pre, embrion): 
    r0 = np.sqrt(embrion.sum() / np.pi)

    #Binary openening borra lo que sobresale y es mas angosto que el disco (si el disco es grande) y se queda con la componente mas grande
    #En definitiva se limpia la mascara obtenida con watershed (restos, halo y pedazos que se agregaron al contorno)
    #hace una apertura con un disco de 0.35 radios y se queda con la componente conectada mas grande
    limpio = ndimage.binary_opening(embrion, structure=morphology.disk(int(0.35 * r0))) 
    etiquetas = measure.label(limpio)
    if etiquetas.max() > 0:
        limpio = etiquetas == np.argmax(np.bincount(etiquetas.ravel())[1:]) + 1
    else:
        limpio = embrion
    radio = np.sqrt(limpio.sum() / np.pi)

    #Binary erosion achicha la mascara limpia y mide solo adentro para no contar el borde del propio embrion como si fuera una membrana
    nucleo = ndimage.binary_erosion(limpio, iterations=max(3, int(0.08 * radio)))

    #Canny marca los pixeles donde la intensidad cambia mucho
    bordes = canny(imagen_pre, sigma=SIGMA_BORDES, mask=nucleo)

    #Se etiqueta cada segmento de borde conectado y se mide cuantos pixeles tiene, se conservan solo los segmentos de largo suficiente
    seg = measure.label(bordes, connectivity=2) #numera cada segmento de borde conectado y cuenta tambien las diagonales
    largos = np.bincount(seg.ravel()) #da el tamaño en pixeles de cada segmento
    largos[0] = 0 #anula la etiqueta del fondo que no es un borde
    membranas = largos[seg] >= LARGO_MIN_REL * radio

    #fraccion de pixeles del nucleo que son membrana, multiplicada por el radio para que no dependa del tamaño del embrion
    indice = float(membranas[nucleo].mean() * radio) 

    #Se engrosa las membranas para que cierren y se etiquetan las regiones de nucleo que quedan separadas
    #Devuelve la fraccion que ocupa la region mas grande
    barrera = ndimage.binary_dilation(membranas, iterations=1) #se usa la dilatacion minima que cierra las membranas sin fusionar regiones
    regiones = measure.label(nucleo & ~barrera)
    mayor = np.bincount(regiones.ravel())[1:].max() / nucleo.sum() if regiones.max() > 0 else 0.0
    return indice, float(mayor)

def medidas_textura(imagen_pre, embrion):
    #Calcula la GLCM
    #Primero valida la cantidad de pixeles, si son menos de 500 devuelve error
    embrion = embrion.astype(bool)
    if embrion.sum() < 500:
        raise ValueError("Máscara de embrión vacía o demasiado chica.")

    recorte = _recorte_inscrito(imagen_pre, embrion)
    indice_bordes, compartimiento = _indice_membranas(imagen_pre, embrion)

    #Se cuantiza con percentiles y se define el rango de grises, se lleva a 0-1 y se pasa a 16 niveles enteros
    lo, hi = np.percentile(recorte, [1, 99])
    q = np.clip((recorte - lo) / (hi - lo + 1e-9), 0, 1)
    q = (q * (NIVELES - 1)).astype(np.uint8)

    #Graycomatrix cuenta para cada par con que frecuencia aparece cada par de grises en pixeles vecinos
    glcm = graycomatrix(q, distances=list(DISTANCIAS), angles=list(ANGULOS),
                        levels=NIVELES, symmetric=True, normed=True)

    #Se toman 4 medidas de la GLCM y se promedian sobre distancias y angulos
    contraste = graycoprops(glcm, "contrast").mean() / (NIVELES - 1) ** 2 #se divide por niveles-1^2 para llevarlo a 0-1
    homogeneidad = graycoprops(glcm, "homogeneity").mean()
    energia = graycoprops(glcm, "energy").mean()

    # Entropía de Shannon de la GLCM, normalizada por su máximo (log2 de nº de celdas) para que quede en 0-1
    entropias = []
    for d in range(glcm.shape[2]):
        for a in range(glcm.shape[3]):
            p = glcm[:, :, d, a]
            p = p[p > 0]
            entropias.append(-(p * np.log2(p)).sum())
    entropia = float(np.mean(entropias)) / np.log2(NIVELES * NIVELES)

    #El indice hace el promedio de todos los atributos, principalmente para visualizacion
    indice = (contraste + entropia + (1 - homogeneidad) + (1 - energia)) / 4

    return {"contraste": float(contraste), "homogeneidad": float(homogeneidad),
            "energia": float(energia), "entropia": float(entropia),
            "indice_glcm": float(indice), "indice_bordes": indice_bordes,
            "compartimiento": compartimiento}


def estimar_k(imagen_pre, embrion, verbose=True):
    #Se plantea dos preguntas: ¿El compartimento mayor ocupa 0.59 o mas del nucleo? Entonces son dos celulas
    #Si la respuesta es no, ¿El indice de membranas es menor a 7? Entonces son 4; si no, 8

    m = medidas_textura(imagen_pre, embrion)
    if m["compartimiento"] >= UMBRAL_COMPARTIMIENTO:
        k = 2
    elif m["indice_bordes"] < UMBRAL_4_8:
        k = 4
    else:
        k = 8
    if verbose:
        print(f"  Textura: contraste={m['contraste']:.3f}  homogeneidad={m['homogeneidad']:.3f}  "
              f"energía={m['energia']:.3f}  entropía={m['entropia']:.3f}  "
              f"| membranas={m['indice_bordes']:.2f}  compartimiento={m['compartimiento']:.2f}  => {k} células")
    return k