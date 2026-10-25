from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import measure

from src import carga, visualizacion

# Parámetros fijos, elegidos con la muestra
MARGEN_POCILLO = 0.95      # la máscara usa el 95 % del radio: deja afuera el halo brillante del borde
TAM_ROI = 15               # lado de la ventana lisa donde se mide el ruido
TAM_WIENER = 5             # ventana del Wiener: 5x5
CORTE_FOURIER = 0.01       # ciclos/píxel: variaciones de 100 píxeles o más (escala del sombreado del pocillo)
GANANCIA_BAJAS = 0.5       # las frecuencias más bajas se atenúan a la mitad
PERCENTILES_CONTRASTE = (0.5, 99.5)  # rango del pocillo que se lleva a [0, 1]


# ---------------------------------------------------------------------------
# Paso 1: máscara del pocillo
# ---------------------------------------------------------------------------

def _mayor_componente(mascara):
    """Se queda con la componente conectada más grande de una máscara (etiquetado de componentes)."""
    etiquetas = measure.label(mascara)
    if etiquetas.max() == 0:
        return mascara
    areas = np.bincount(etiquetas.ravel())[1:]
    return etiquetas == np.argmax(areas) + 1


def _ajustar_circulo(filas, cols):
    """
    Círculo que mejor pasa por los puntos, por cuadrados mínimos.
    (x - xc)² + (y - yc)² = R² se reescribe como 2·xc·x + 2·yc·y + c = x² + y², que es lineal en (xc, yc, c),
    con c = R² - xc² - yc². Se resuelve con np.linalg.lstsq.
    """
    diseno = np.column_stack([2 * cols, 2 * filas, np.ones_like(cols)])
    (xc, yc, c), *_ = np.linalg.lstsq(diseno, cols ** 2 + filas ** 2, rcond=None)
    return yc, xc, np.sqrt(c + xc ** 2 + yc ** 2)


def mascara_pocillo(imagen, margen=MARGEN_POCILLO):
    """
    Máscara bool del interior del pocillo. Devuelve (mascara, (fila_centro, col_centro, radio)).

    No sirve un umbral global (Otsu): la zona rayada de afuera del pocillo tiene el mismo gris que el fondo de
    adentro. Se usa que el pocillo es redondo. De su borde se ve:
      - abajo, el anillo brillante: el 3 % de píxeles más claros, su componente conectada más grande;
      - arriba, el límite con la zona negra de afuera: el borde de la componente oscura más grande
        (la componente menos su erosión).
    Se ajusta un círculo a esos puntos y se repite 5 veces descartando los que quedan a más de 8 píxeles del
    círculo (por ejemplo, el borde recto de la zona negra contra el marco de la imagen).
    """
    brillante = _mayor_componente(imagen >= np.percentile(imagen, 97))
    oscuro = _mayor_componente(imagen < 0.05)
    borde_oscuro = oscuro & ~ndimage.binary_erosion(oscuro)
    borde_oscuro[[0, -1], :] = False
    borde_oscuro[:, [0, -1]] = False

    filas, cols = np.nonzero(brillante | borde_oscuro)
    filas, cols = filas.astype(float), cols.astype(float)
    elegidos = np.ones(len(filas), dtype=bool)
    for _ in range(5):
        fc, cc, radio = _ajustar_circulo(filas[elegidos], cols[elegidos])
        elegidos = np.abs(np.hypot(filas - fc, cols - cc) - radio) < 8

    y, x = np.indices(imagen.shape)
    mascara = np.hypot(y - fc, x - cc) < margen * radio
    return mascara, (fc, cc, radio)


# ---------------------------------------------------------------------------
# Paso 2: reducción de ruido
# ---------------------------------------------------------------------------

def estimar_varianza_ruido(imagen, mascara, tam=TAM_ROI):
    """
    Varianza del ruido de la imagen, medida en la ventana más lisa del pocillo.
    1. Se busca la ventana de tam x tam con menor gradiente medio (Sobel): sin bordes adentro, lo que varía es ruido.
    2. A esa ventana se le resta el plano que mejor la ajusta, para que la iluminación (que cambia lento)
       no se cuente como ruido.
    3. La varianza de lo que queda es la del ruido.
    """
    gradiente = np.hypot(ndimage.sobel(imagen, axis=0), ndimage.sobel(imagen, axis=1))
    gradiente_medio = ndimage.uniform_filter(gradiente, size=tam)
    # Solo ventanas enteras dentro del pocillo
    adentro = ndimage.binary_erosion(mascara, iterations=tam // 2 + 1)
    fila, col = np.unravel_index(np.argmin(np.where(adentro, gradiente_medio, np.inf)), imagen.shape)
    r0, c0 = fila - tam // 2, col - tam // 2

    recorte = imagen[r0:r0 + tam, c0:c0 + tam].ravel()
    filas, cols = np.mgrid[0:tam, 0:tam]
    diseno = np.column_stack([np.ones(tam * tam), filas.ravel(), cols.ravel()])
    coeficientes, *_ = np.linalg.lstsq(diseno, recorte, rcond=None)
    return float(np.var(recorte - diseno @ coeficientes))


def filtro_wiener(imagen, varianza_ruido, tam=TAM_WIENER):
    """
    Wiener adaptativo: en cada píxel,
        salida = media_local + max(var_local - var_ruido, 0) / var_local · (imagen - media_local)
    En una zona lisa la varianza local es parecida a la del ruido, el factor da ~0 y el píxel se reemplaza por la
    media: suaviza. En un borde (una membrana) es mucho mayor, el factor da ~1 y el píxel queda igual.
    """
    media_local = ndimage.uniform_filter(imagen, size=tam)
    varianza_local = ndimage.uniform_filter(imagen ** 2, size=tam) - media_local ** 2
    ganancia = np.maximum(varianza_local - varianza_ruido, 0) / np.maximum(varianza_local, 1e-12)
    return media_local + ganancia * (imagen - media_local)


# ---------------------------------------------------------------------------
# Paso 3: corrección de iluminación
# ---------------------------------------------------------------------------

def corregir_iluminacion(imagen, mascara, corte=CORTE_FOURIER, ganancia_bajas=GANANCIA_BAJAS):
    """
    Pasa-altos gaussiano en el dominio de Fourier. El sombreado del pocillo cambia lento, así que vive en las
    frecuencias bajas; las membranas y la textura, en las altas. Pasos:
      1. Afuera del pocillo se rellena con la media de adentro, para que el salto en el borde no genere
         oscilaciones (ringing) al filtrar.
      2. FFT 2D y se multiplica por
             H(D) = g + (1 - g) · (1 - exp(-D² / (2·corte²)))
         donde D es la distancia al centro del espectro (la frecuencia) y g = ganancia_bajas. En D = 0 vale g
         (0,5: el sombreado se atenúa a la mitad, sin borrar el brillo medio) y lejos del centro vale 1
         (los detalles quedan igual, sin amplificar el ruido).
      3. FFT inversa (parte real).
    Devuelve la imagen corregida, sin reescalar (eso lo hace estirar_contraste).
    """
    relleno = np.where(mascara, imagen, imagen[mascara].mean())
    frec_filas = np.fft.fftfreq(imagen.shape[0])[:, None]
    frec_cols = np.fft.fftfreq(imagen.shape[1])[None, :]
    distancia2 = frec_filas ** 2 + frec_cols ** 2
    filtro = ganancia_bajas + (1 - ganancia_bajas) * (1 - np.exp(-distancia2 / (2 * corte ** 2)))
    return np.real(np.fft.ifft2(filtro * np.fft.fft2(relleno)))


# ---------------------------------------------------------------------------
# Paso 4: contraste
# ---------------------------------------------------------------------------

def estirar_contraste(imagen, mascara, percentiles=PERCENTILES_CONTRASTE):
    """
    Estiramiento lineal: el rango de grises del pocillo (entre los percentiles 0,5 y 99,5) se lleva a [0, 1].
    Se usan percentiles en lugar del mínimo y el máximo para que unos pocos píxeles extremos no definan la escala.
    Afuera del pocillo queda 0. No se ecualiza: la ecualización convertía las rayas del fondo en manchas.
    """
    bajo, alto = np.percentile(imagen[mascara], percentiles)
    return np.clip((imagen - bajo) / (alto - bajo), 0, 1) * mascara


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def preprocesar_con_pasos(imagen):
    """
    Aplica los 4 pasos en orden y devuelve cada paso intermedio, para poder mostrarlos.
    El Wiener va antes que la corrección de iluminación y el contraste porque esos dos realzan diferencias
    y también realzarían el ruido.
    Devuelve un dict con:
      - "mascara" y "circulo" (fila_centro, col_centro, radio): paso 1
      - "varianza_ruido": la medida en la imagen, que usa el Wiener
      - "wiener", "iluminacion" y "final": la imagen después de los pasos 2, 3 y 4
    """
    mascara, circulo = mascara_pocillo(imagen)
    varianza_ruido = estimar_varianza_ruido(imagen, mascara)
    sin_ruido = filtro_wiener(imagen, varianza_ruido)
    iluminacion_pareja = corregir_iluminacion(sin_ruido, mascara)
    final = estirar_contraste(iluminacion_pareja, mascara)
    return {"mascara": mascara, "circulo": circulo, "varianza_ruido": varianza_ruido,
            "wiener": sin_ruido, "iluminacion": iluminacion_pareja, "final": final}


def preprocesar(imagen):
    """Aplica los 4 pasos y devuelve solo (imagen preprocesada, máscara del pocillo). Es la que usan los otros módulos."""
    pasos = preprocesar_con_pasos(imagen)
    return pasos["final"], pasos["mascara"]


# ---------------------------------------------------------------------------
# Etapa interactiva (la que llama main.py)
# ---------------------------------------------------------------------------

def elegir_imagen(muestra):
    """
    Muestra la lista de imágenes de la muestra numeradas y le pide al usuario que elija una.
    También acepta la ruta de otra imagen (por ejemplo, los casos incógnita de la cátedra).
    Devuelve la ruta elegida.
    """
    print("\nImágenes de la muestra:")
    for numero, (ruta, estadio) in enumerate(muestra, start=1):
        print(f"  {numero:2d}. {estadio}/{ruta.name}")

    while True:
        respuesta = input(f"Elegí un número (1 a {len(muestra)}) o escribí la ruta de otra imagen: ").strip()
        if respuesta.isdigit() and 1 <= int(respuesta) <= len(muestra):
            return muestra[int(respuesta) - 1][0]
        if Path(respuesta).is_file():
            return Path(respuesta)
        print("  Opción no válida, probá de nuevo.")


def mostrar_preprocesamiento(imagen, pasos, nombre):
    """
    Muestra en orden la imagen original y cómo queda después de cada paso; la última es la preprocesada.
    Guarda la figura en resultados/ y devuelve su ruta.
    """
    mascara = pasos["mascara"]
    imagenes = [
        imagen,
        np.where(mascara, imagen, 0.3 * imagen),  # lo de afuera del pocillo, oscurecido
        pasos["wiener"],
        np.where(mascara, pasos["iluminacion"], np.nan),  # afuera tiene el relleno con la media: no se muestra
        pasos["final"],
    ]
    titulos = ["Original", "1. Máscara del pocillo", "2. Ruido (Wiener)",
               "3. Iluminación (pasa-altos)", "4. Contraste: preprocesada"]
    fig = visualizacion.mostrar_pasos(imagenes, titulos, f"Preprocesamiento de {nombre}")
    return visualizacion.guardar_figura(fig, f"preprocesamiento_{Path(nombre).stem}")


def etapa_preprocesamiento(muestra):
    """
    Etapa completa de preprocesamiento, la que llama main.py. Recibe la lista de (ruta, estadio) que devuelve
    carga.etapa_carga. El usuario elige una imagen y ve, en orden, cómo queda después de cada paso
    (la última es la preprocesada). Se repite hasta que el usuario quiera salir.
    """
    print("\n=== Etapa 1: preprocesamiento ===")
    print("Pasos: máscara del pocillo, reducción de ruido (Wiener), corrección de iluminación y contraste.")

    while True:
        ruta = elegir_imagen(muestra)
        imagen = carga.cargar_imagen(ruta)
        pasos = preprocesar_con_pasos(imagen)

        _, _, radio = pasos["circulo"]
        print(f"\n  {ruta.name}: radio del pocillo {radio:.0f} px, "
              f"ruido medido σ = {np.sqrt(pasos['varianza_ruido']) * 255:.2f} niveles de gris")
        ruta_figura = mostrar_preprocesamiento(imagen, pasos, ruta.name)
        print(f"  Figura guardada en {ruta_figura}")
        print("  (cerrá la ventana de la figura para seguir)")
        plt.show()

        respuesta = input("\n¿Querés preprocesar otra imagen? (s/n): ").strip().lower()
        if respuesta != "s":
            break


if __name__ == "__main__":
    # Prueba: python -m src.preprocesamiento (desde la carpeta del repo).
    # Guarda en resultados/ una figura con los pasos para cada imagen de la muestra.
    muestra = carga.cargar_muestra()
    titulos = ["original + pocillo", "2. Wiener", "3. iluminación", "4. contraste (final)"]
    fig, ejes = plt.subplots(len(muestra), len(titulos), figsize=(12, 3 * len(muestra)))
    for fila, (ruta, estadio) in enumerate(muestra):
        imagen = carga.cargar_imagen(ruta)
        pasos = preprocesar_con_pasos(imagen)
        mascara, (fc, cc, radio) = pasos["mascara"], pasos["circulo"]
        print(f"  {estadio}/{ruta.name}: radio del pocillo {radio:.0f} px, "
              f"ruido σ = {np.sqrt(pasos['varianza_ruido']) * 255:.2f} niveles de gris")

        # El paso 3 se muestra solo dentro del pocillo (afuera tiene el relleno con la media)
        paneles = [imagen, pasos["wiener"], np.where(mascara, pasos["iluminacion"], np.nan), pasos["final"]]
        for col, paso in enumerate(paneles):
            ejes[fila, col].imshow(paso, cmap="gray")
            ejes[fila, col].axis("off")
            if fila == 0:
                ejes[fila, col].set_title(titulos[col])
        angulo = np.linspace(0, 2 * np.pi, 200)
        ejes[fila, 0].plot(cc + MARGEN_POCILLO * radio * np.cos(angulo),
                           fc + MARGEN_POCILLO * radio * np.sin(angulo), color="#d67a2a", lw=1)
        ejes[fila, 0].set_xlim(0, imagen.shape[1])
        ejes[fila, 0].set_ylim(imagen.shape[0], 0)
        ejes[fila, 0].text(3, 15, f"{estadio}/{ruta.name}", color="white", fontsize=8)

    Path("resultados").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(Path("resultados") / "preprocesamiento_pasos.png", dpi=80)
    print("Figura guardada en resultados/preprocesamiento_pasos.png")
