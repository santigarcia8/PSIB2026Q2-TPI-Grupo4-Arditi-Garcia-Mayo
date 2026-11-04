from pathlib import Path

import matplotlib.pyplot as plt

CARPETA_RESULTADOS = Path("resultados")


def mostrar_pasos(imagenes, titulos, titulo_general=""):
    """
    Muestra varias imágenes lado a lado, en el orden en que se fueron aplicando los pasos.
    Recibe la lista de imágenes (2D, en gris) y la de sus títulos. Devuelve la figura (para poder guardarla).
    Los valores NaN se ven en blanco: sirven para mostrar solo una región (por ejemplo, el interior del pocillo).
    """
    fig, ejes = plt.subplots(1, len(imagenes), figsize=(4 * len(imagenes), 4.5))
    for eje, imagen, titulo in zip(ejes, imagenes, titulos):
        eje.imshow(imagen, cmap="gray")
        eje.set_title(titulo)
        eje.axis("off")
    if titulo_general:
        fig.suptitle(titulo_general)
        fig.tight_layout(rect=(0, 0, 1, 0.92))  # deja lugar arriba para el título general
    else:
        fig.tight_layout()
    return fig

def carpeta_imagen(nombre_imagen):
    """Devuelve resultados/<id de la imagen>/ (la crea si no existe)."""
    carpeta = CARPETA_RESULTADOS / Path(nombre_imagen).stem
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta

def guardar_figura(fig, nombre, imagen):
    """Guarda la figura en resultados/<imagen>/<nombre>.png. Devuelve la ruta."""
    ruta = carpeta_imagen(imagen) / f"{nombre}.png"
    fig.savefig(ruta, dpi=100)
    return ruta
