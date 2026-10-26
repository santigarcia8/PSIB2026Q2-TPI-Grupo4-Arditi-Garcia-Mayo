from src import carga, preprocesamiento
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage import measure, morphology
from skimage import segmentation, filters
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour

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

if __name__ == "__main__":
    muestra = carga.cargar_muestra()

    fig, ejes = plt.subplots(3, 10, figsize=(40, 12))
    for col, (ruta, estadio) in enumerate(muestra):
        imagen = carga.cargar_imagen(ruta)
        imagen_pre, mascara_pocillo = preprocesamiento.preprocesar(imagen)
        mapa, mascara_chica = mapa_desvio(imagen_pre, mascara_pocillo)

        base = otsu_embrion(mapa, mascara_chica)
        ws = watershed_embrion(imagen_pre, mascara_pocillo, base)
        sn = snake_embrion(imagen_pre, mascara_pocillo, base)

        for fila, (mascara, nombre) in enumerate([(base, "Otsu"), (ws, "Watershed"), (sn, "Snake")]):
            eje = ejes[fila, col]
            eje.imshow(imagen_pre, cmap="gray")
            if mascara.any():
                eje.contour(mascara, [0.5], colors="red", linewidths=1.5)
            eje.set_title(f"{estadio}/{ruta.name}\n{nombre}", fontsize=10)
            eje.axis("off")

    plt.tight_layout()
    plt.show()