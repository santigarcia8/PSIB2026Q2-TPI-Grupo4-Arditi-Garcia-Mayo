"""
compresion.py
Módulo para la Etapa 5: Compresión final sin pérdidas (Lossless)
  - Huffman: para la imagen preprocesada (escala de grises, uint8).
  - RLE (Run-Length Encoding): para la máscara del pocillo (bool).
"""

from collections import Counter
import heapq
from pathlib import Path
import pickle
import numpy as np
from src import visualizacion

# ---------------------------------------------------------------------------
# Estrutura y Construcción del Árbol de Huffman
# ---------------------------------------------------------------------------

class NodoHuffman:
    def __init__(self, valor=None, freq=0):
        self.valor = valor
        self.freq = freq
        self.izq = None
        self.der = None

    def __lt__(self, otro):
        return self.freq < otro.freq


def _construir_arbol(frecuencias):
    heap = [NodoHuffman(val, freq) for val, freq in frecuencias.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        raiz = NodoHuffman()
        raiz.izq = heapq.heappop(heap)
        return raiz

    while len(heap) > 1:
        izq = heapq.heappop(heap)
        der = heapq.heappop(heap)
        padre = NodoHuffman(freq=izq.freq + der.freq)
        padre.izq = izq
        padre.der = der
        heapq.heappush(heap, padre)

    return heap[0]


def _generar_codigos(nodo, codigo_actual="", codigos=None):
    if codigos is None:
        codigos = {}
    if nodo is not None:
        if nodo.valor is not None:
            codigos[nodo.valor] = codigo_actual or "0"
        _generar_codigos(nodo.izq, codigo_actual + "0", codigos)
        _generar_codigos(nodo.der, codigo_actual + "1", codigos)
    return codigos


# ---------------------------------------------------------------------------
# Compresión Huffman (Imagen Preprocesada)
# ---------------------------------------------------------------------------

def comprimir_huffman(matriz_imagen: np.ndarray):
    if matriz_imagen.dtype != np.uint8:
        matriz_imagen = (np.clip(matriz_imagen, 0, 1) * 255).astype(np.uint8)

    shape_original = matriz_imagen.shape
    pixeles = matriz_imagen.flatten()

    frecuencias = Counter(pixeles)
    raiz = _construir_arbol(frecuencias)
    tabla_codigos = _generar_codigos(raiz)

    cadena_bits = "".join(tabla_codigos[p] for p in pixeles)

    padding = (8 - len(cadena_bits) % 8) % 8
    cadena_bits_padded = cadena_bits + "0" * padding

    bytes_comprimidos = bytearray()
    for i in range(0, len(cadena_bits_padded), 8):
        byte = cadena_bits_padded[i:i + 8]
        bytes_comprimidos.append(int(byte, 2))

    return {
        "bytes": bytes(bytes_comprimidos),
        "frecuencias": dict(frecuencias),
        "shape": shape_original,
        "padding": padding,
    }


# ---------------------------------------------------------------------------
# Compresión RLE (Máscara del Pocillo)
# ---------------------------------------------------------------------------

def comprimir_rle(mascara: np.ndarray):
    flat = mascara.flatten()
    n = len(flat)
    if n == 0:
        return {"runs": [], "shape": mascara.shape}

    runs = []
    val_actual = flat[0]
    longitud = 1

    for i in range(1, n):
        if flat[i] == val_actual:
            longitud += 1
        else:
            runs.append((int(val_actual), longitud))
            val_actual = flat[i]
            longitud = 1
    runs.append((int(val_actual), longitud))

    return {"runs": runs, "shape": mascara.shape}


# ---------------------------------------------------------------------------
# Etapa 5: Ejecución Principal de Compresión
# ---------------------------------------------------------------------------

def etapa_compresion(img_preprocesada: np.ndarray, mascara_pocillo: np.ndarray, nombre_imagen: str = "embrion"):
    """
    Etapa 5 final: realiza la compresión sin pérdidas de los resultados obtenidos
    durante el procesamiento de la imagen actual y guarda los paquetes comprimidos en resultados/.
    """
    print("\n=== Etapa 5: Compresión de resultados (Lossless) ===")

    # 1. Ejecutar algoritmos de compresión
    paquete_huffman = comprimir_huffman(img_preprocesada)
    paquete_rle = comprimir_rle(mascara_pocillo)

    # 2. Calcular tamaños y tasas de compresión
    tam_orig_img = img_preprocesada.nbytes
    tam_comp_img = len(paquete_huffman["bytes"])
    tasa_huffman = tam_orig_img / tam_comp_img if tam_comp_img > 0 else 0

    tam_orig_masc = mascara_pocillo.nbytes
    tam_comp_masc = len(paquete_rle["runs"]) * 5  # Estimación en bytes
    tasa_rle = tam_orig_masc / tam_comp_masc if tam_comp_masc > 0 else 0

    # 3. Guardar paquetes comprimidos en disco (resultados/)
    stem_nombre = Path(nombre_imagen).stem
    ruta_guardado = visualizacion.carpeta_imagen(nombre_imagen) / "comprimido.pkl"
    
    with open(ruta_guardado, "wb") as f:
        pickle.dump({"huffman": paquete_huffman, "rle": paquete_rle}, f)

    # 4. Reporte final en consola
    print(f"  [Huffman] Imagen preprocesada ({stem_nombre}):")
    print(f"    - Peso original: {tam_orig_img / 1024:.2f} KB | Peso comprimido: {tam_comp_img / 1024:.2f} KB")
    print(f"    - Tasa de compresión: {tasa_huffman:.2f}x (Ahorro del {(1 - 1/tasa_huffman)*100:.1f}%)")

    print(f"  [RLE] Máscara del pocillo:")
    print(f"    - Peso original: {tam_orig_masc / 1024:.2f} KB | Peso comprimido: ~{tam_comp_masc / 1024:.2f} KB")
    print(f"    - Tasa de compresión: {tasa_rle:.2f}x (Ahorro del {(1 - 1/tasa_rle)*100:.1f}%)")

    print(f"  Archivo comprimido guardado en: {ruta_guardado}")