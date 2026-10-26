from src import carga, preprocesamiento, visualizacion
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.feature import canny
from skimage.transform import hough_circle
from skimage import measure, morphology
from skimage import segmentation, filters
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour
from skimage.feature import peak_local_max

#Como el nivel de gris del fondo y células son muy similares, voy a segmentar por desvío estandar porque
#el fondo es liso pero las células NO, lo cual les da más desvío

#PASO 1.1 - SEGMENTACIÓN FONDO - EMBRIÓN con OTSU pero según DESVÍO
def mapa_desvio(imagen_pre, mascara_pocillo, tam=9):
    media_normal = ndimage.uniform_filter(imagen_pre, size=tam) #aplico kernel de 9x9 que hace la media
    media_cuadrado = ndimage.uniform_filter(imagen_pre**2, size=tam) #aplico lo mismo pero a la imagen al cuadrado
    varianza = media_cuadrado-media_normal**2 
    desvio = np.sqrt(np.maximum(varianza,0)) #hago el desvió que me da el mapa del tamaño de la imagen pero como tiene el borde claro qe lo detecta le hago una erosión para eliminarlo
    mascara_chica = ndimage.binary_erosion(mascara_pocillo, iterations=10)
    mapa = desvio * mascara_chica
    return mapa, mascara_chica

def otsu_embrion(mapa, mascara_chica):
    umbral = threshold_otsu(mapa[mascara_chica])*0.5 #defino la mitad del umbral para no tener problemas con la zona pelúcida y el contorno
    embrion_otsu = mapa > umbral #defino que la máscara sea las zonas que superan ese umbral
    embrion_otsu = ndimage.binary_closing(embrion_otsu, iterations=3) #le hago una erosión para elimina parte de la zona peúcida y el contorno blanco
    embrion_otsu = ndimage.binary_fill_holes(embrion_otsu) #hay zonas del interior que no superaban pero sí estaban dentor entonces los rellené
    embrion_otsu = ndimage.binary_opening(embrion_otsu, iterations=5) #le hago uan apertura para zonas dudosas
    etiquetas = measure.label(embrion_otsu) #me quiero quedar con las zonas más grandes (embrión) y NO las manchitas perdids por el fondo
    if etiquetas.max() > 0:
        areas = np.bincount(etiquetas.ravel())[1:]
        embrion_otsu = etiquetas == np.argmax(areas) + 1

    return embrion_otsu

def segmentacion_embrion(embrion_otsu, imagen_pre):
    embrion_segmentado_total = embrion_otsu * imagen_pre #multiplico para obtener las imágenes con el otsu 
    
    return embrion_segmentado_total


#PASO 1.2 - SEGMENTACIÓN con WATERSHED
def watershed_embrion(imagen_pre, mascara_pocillo, base, r_dentro=18, r_fuera=12): #elijo los radios de adentro y afuera respecto la máscara de otsu
    gradiente = filters.sobel(ndimage.gaussian_filter(imagen_pre, 1.5)) #elijo un sigma no my grande para suavizar bordes menores y NO los del embrión y luego le aplico sobel para ver el gradiente
    dentro = ndimage.binary_erosion(base, structure=morphology.disk(r_dentro)) #uso la erosión en forma de círculo que se extiende con esos radios para erosionar y luego dilatar
    fuera = ~ndimage.binary_dilation(base, structure=morphology.disk(r_fuera))
    marcadores = np.zeros(imagen_pre.shape, dtype=int)
    marcadores[fuera] = 1 #separo en fondo y embrión
    marcadores[dentro] = 2
    ws = segmentation.watershed(gradiente, marcadores) #aplico las zonas
    return (ws == 2) & mascara_pocillo #devuelve solo la zona 2 (embrión) y la máscara del pocillo

#PASO 1.3 - SEGMENTACIÓN con SNAKES
def snake_embrion(imagen_pre, mascara_pocillo, base, num_iter=40): #defino 40 a prueba y error 
    bordes = inverse_gaussian_gradient(imagen_pre, alpha=100, sigma=2) #uso invertido fondo y embrión. usamos alpha = 100 porque los bordes NO son tan fuertes, sigma = 2 porque los bordes del embrión son bastante finos
    snake = morphological_geodesic_active_contour(
        bordes, num_iter=num_iter, init_level_set=base,
        smoothing=2, balloon=-1, threshold="auto") #parto de la máscara base (otsu), ballon para achicarse o inflarse, smoothing para que no quede tan irregular la snake
    return snake.astype(bool) & mascara_pocillo #devuelve la zona booleana de 1 y 0 y la cruza con el pocillo

#PASO 2 - SEPARACIÓN DE CÉLULAS

#PASO 2.1 - REALCE DE BORDES CON CANNY
def bordes_canny(imagen_pre, embrion):
    mascara_embrion = ndimage.binary_erosion(embrion, iterations = 3)
    bordes = canny(imagen_pre, sigma = 2, mask = mascara_embrion)

    return bordes

#PASO 2.2 - HOUGH USANDO BORDES
def hough_embrion(bordes):
    radios = np.arange(24, 59, 2)
    puntaje = hough_circle(bordes, radios)
    
    return puntaje, radios

#PASO 2.3 - CALIFICACIÓN DE LOS CÍRCULOS DE HOUGH
def candidatos_circulos(puntaje, radios, embrion, umbral=0.15):
    candidatos = []
    for i, radio in enumerate(radios):
        picos = peak_local_max(puntaje[i], min_distance=6, threshold_abs=umbral, exclude_border=False)
        for fila, col in picos:
            if embrion[fila, col]:
                candidatos.append((puntaje[i][fila, col], fila, col, radio))
    candidatos.sort(reverse=True)
    return candidatos

#PASO 2.4 - ELEGIR LOS MEJORES CÍRCULOS
def elegir_circulos(candidatos, k=4, sep=1.0):
    elegidos = []
    for puntaje, fila, col, radio in candidatos:
        if all(np.hypot(fila - f2, col - c2) >= sep * max(radio, r2) for _, f2, c2, r2 in elegidos):
            elegidos.append((puntaje, fila, col, radio))
        if len(elegidos) == k:
            break
    return elegidos

#PASO 2.5 - MARCADORES A PARTIR DE LOS CÍRCULOS
def marcadores_circulos(forma, elegidos):
    marcadores = np.zeros(forma, dtype=int)
    filas, cols = np.indices(forma)
    for i, (puntaje, fila, col, radio) in enumerate(elegidos, 1):
        distancia = np.hypot(filas - fila, cols - col)
        marcadores[distancia <= 0.4 * radio] = i
    return marcadores

#PASO 2.6 - SEPARAR CÉLULAS CON WATERSHED
def separar_celulas(imagen_pre, embrion, elegidos, extension=1.2):
    relieve = (filters.sato(imagen_pre, sigmas=[3], black_ridges=False)
               + filters.sato(imagen_pre, sigmas=[3], black_ridges=True)
               + filters.sobel(ndimage.gaussian_filter(imagen_pre, 1.5)))
    marcadores = marcadores_circulos(imagen_pre.shape, elegidos)

    filas, cols = np.indices(imagen_pre.shape)
    permitido = np.zeros(imagen_pre.shape, dtype=bool)
    for puntaje, fila, col, radio in elegidos:
        permitido |= np.hypot(filas - fila, cols - col) <= extension * radio

    return segmentation.watershed(relieve, marcadores, mask=embrion & permitido)


def etapa_segmentacion(ruta):
    print("\n=== Etapa 2: segmentación ===")
    print("Embrión vs fondo: Otsu, Watershed y Snake. Células: Canny + Hough + Watershed.")

    # El k (cantidad de células) sale del nombre de la carpeta: 2cell, 4cell u 8cell
    k = None
    for parte in ruta.parts:
        if parte in carga.CELULAS_POR_ESTADIO:
            k = carga.CELULAS_POR_ESTADIO[parte]
    if k is None:
        k = int(input("  No pude deducir el estadio. ¿Cuántas células tiene? (2, 4 u 8): "))

    imagen = carga.cargar_imagen(ruta)
    imagen_pre, mascara_pocillo = preprocesamiento.preprocesar(imagen)

    # Embrión vs fondo: los tres métodos
    mapa, mascara_chica = mapa_desvio(imagen_pre, mascara_pocillo)
    base = otsu_embrion(mapa, mascara_chica)
    ws = watershed_embrion(imagen_pre, mascara_pocillo, base)
    snake = snake_embrion(imagen_pre, mascara_pocillo, base)

    # Figura 1: comparación de los tres métodos
    fig1, ejes1 = plt.subplots(1, 3, figsize=(15, 5))
    metodos = [("Otsu", base, "red"), ("Watershed", ws, "cyan"), ("Snake", snake, "yellow")]
    for eje, (nombre, mascara, color) in zip(ejes1, metodos):
        eje.imshow(imagen_pre, cmap="gray")
        eje.contour(mascara, [0.5], colors=color, linewidths=1.5)
        eje.set_title(nombre)
        eje.axis("off")
    fig1.suptitle(f"Embrión vs fondo: {ruta.name}")
    fig1.tight_layout()
    ruta1 = visualizacion.guardar_figura(fig1, f"segmentacion_embrion_{ruta.stem}")

    # Células (se usa el embrión del watershed)
    bordes = bordes_canny(imagen_pre, ws)
    puntaje, radios = hough_embrion(bordes)
    candidatos = candidatos_circulos(puntaje, radios, ws)
    elegidos = elegir_circulos(candidatos, k=k)
    celulas = separar_celulas(imagen_pre, ws, elegidos)

    # Figura 2: separación de células
    fig2, ejes2 = plt.subplots(1, 3, figsize=(15, 5))
    ejes2[0].imshow(imagen_pre, cmap="gray")
    ejes2[0].set_title("Preprocesada")
    ejes2[1].imshow(bordes, cmap="gray")
    ejes2[1].set_title("Bordes (Canny)")
    ejes2[2].imshow(imagen_pre, cmap="gray")
    ejes2[2].imshow(np.ma.masked_equal(celulas, 0), cmap="tab10", alpha=0.5, vmin=1, vmax=10)
    ejes2[2].set_title(f"Células separadas (k={k})")
    for eje in ejes2:
        eje.axis("off")
    fig2.suptitle(f"Separación de células: {ruta.name}")
    fig2.tight_layout()
    ruta2 = visualizacion.guardar_figura(fig2, f"segmentacion_celulas_{ruta.stem}")

    print(f"  Figuras guardadas en {ruta1} y {ruta2}")


if __name__ == "__main__":
    muestra = carga.cargar_muestra()

    fig, ejes = plt.subplots(2, 5, figsize=(25, 10))
    for eje, (ruta, estadio) in zip(ejes.ravel(), muestra):
        k = carga.CELULAS_POR_ESTADIO[estadio]
        imagen = carga.cargar_imagen(ruta)
        imagen_pre, mascara_pocillo = preprocesamiento.preprocesar(imagen)

        mapa, mascara_chica = mapa_desvio(imagen_pre, mascara_pocillo)
        base = otsu_embrion(mapa, mascara_chica)
        embrion = watershed_embrion(imagen_pre, mascara_pocillo, base)

        bordes = bordes_canny(imagen_pre, embrion)
        puntaje, radios = hough_embrion(bordes)
        candidatos = candidatos_circulos(puntaje, radios, embrion)
        elegidos = elegir_circulos(candidatos, k=k)
        celulas = separar_celulas(imagen_pre, embrion, elegidos)

        eje.imshow(imagen_pre, cmap="gray")
        eje.imshow(np.ma.masked_equal(celulas, 0), cmap="tab10", alpha=0.5, vmin=1, vmax=10)
        eje.set_title(f"{estadio}/{ruta.name} (k={k})")
        eje.axis("off")

    plt.tight_layout()
    plt.show()