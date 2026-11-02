
#Mide el tamaño y la forma de cada celula y del embrion
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import measure, morphology

from src import carga, preprocesamiento, segmentacion, estimar_celulas, visualizacion


CV_MAX = 0.5                     #coeficiente de variacion de las areas a partir del cual la simetria puntua 0
CORTES_FRAGMENTACION = [0.0, 0.10, 0.25, 0.50]   # fracción de fragmentación (consenso de Estambul)
PUNTAJES_FRAGMENTACION = [1.0, 1.0, 0.5, 0.0]    # puntaje que le corresponde a cada corte (arbitrarios)
PESOS = {"simetria": 0.35, "fragmentacion": 0.35, "forma_celular": 0.15, "forma_embrion": 0.15} #arbitrario, priorizamos simetria y fragmentacion por ser los criterios del consenso
CORTE_A, CORTE_B = 75, 50        # categorías: A >= 75, B >= 50, C < 50
FRACCION_APERTURA = 0.35         # misma limpieza de máscara que estimar_celulas
 #El indice de 0 a 100 que arman cvmax y pesos y los cortes es arbitrario, por eso es importante complementarlo con implantacion que tiene justificacion con papers

#Se repite la parte de segmentacion de la etapa 2

def embrion_limpio(embrion):
    #Hace una apertura morfologica con un disco de radio 0.35 del radio del embrion
    #La apertura borra todo lo que sea mas angosto que el disco y se queda con la componente conectada mas grande
    radio = np.sqrt(embrion.sum() / np.pi)
    limpio = ndimage.binary_opening(embrion, structure=morphology.disk(int(FRACCION_APERTURA * radio)))
    etiquetas = measure.label(limpio)
    if etiquetas.max() == 0:
        return embrion.astype(bool)
    return etiquetas == np.argmax(np.bincount(etiquetas.ravel())[1:]) + 1


def celulas_limpias(celulas, limpio):
    #Intersecta cada celula con la mascara limpia y conserva solo la pieza conexa mas grande de cada una
    #Las numera de 1 en adelante
    #Si una celula queda sin pixeles dentro de la mascara limpia, desaparece y despues analizar lo avisa como "se detectaron n de k"
    salida = np.zeros_like(celulas)
    nueva = 0
    for etiqueta in np.unique(celulas[celulas > 0]):
        piezas = measure.label((celulas == etiqueta) & limpio)
        if piezas.max() == 0:
            continue
        nueva += 1
        salida[piezas == np.argmax(np.bincount(piezas.ravel())[1:]) + 1] = nueva
    return salida


#Funciones auxiliares
def _perimetro(region):
    #Usa el perimetro de Crofton si esta en scikit-image 
    #El de Crofton estima el largo del contorno con mejor precision que contar pixeles de borde
    return float(getattr(region, "perimeter_crofton", region.perimeter))

def _eje(region, cual):
    #Devuelve el largo del eje mayor o menor, probando primero el nombre nuevo del atributo y despues el viejo
    nuevo, viejo = f"axis_{cual}_length", f"{cual}_axis_length"
    return float(getattr(region, nuevo) if hasattr(region, nuevo) else getattr(region, viejo))

def _circularidad(area, perimetro):
    #Vale 1 para un circulo perfecto y baja con formas alargadas o irregulares
    return float(min(4 * np.pi * area / perimetro ** 2, 1.0)) if perimetro > 0 else 0.0


def medidas_celulas(celulas, centro_embrion, radio_embrion, um_por_pixel=None):
    #Recorre las celulas con region props y arma un diccionario por celula
    lista = []
    for r in measure.regionprops(celulas):
        per = _perimetro(r)
        d = {
            "etiqueta": r.label,
            "area": float(r.area),
            "diametro_equiv": float(r.equivalent_diameter_area if hasattr(r, "equivalent_diameter_area")
                                    else r.equivalent_diameter),
            "perimetro": per,
            "circularidad": _circularidad(r.area, per),
            "excentricidad": float(r.eccentricity), #0 en un circulo, cerca de 1 en una forma alargada
            "solidez": float(r.solidity), #area sobre area de la envolvente convexa, baja si la celula tiene hendiduras o entrantes
            "centroide": (float(r.centroid[0]), float(r.centroid[1])), 
            "dist_centro_rel": float(np.hypot(r.centroid[0] - centro_embrion[0], #Distancia de la celula al centro del embrion, medida en radios del embrion para que no dependa del tamaño
                                              r.centroid[1] - centro_embrion[1]) / radio_embrion),
        }
        if um_por_pixel: #micrometros
            d["area_um2"] = d["area"] * um_por_pixel ** 2
            d["diametro_equiv_um"] = d["diametro_equiv"] * um_por_pixel
        lista.append(d)
    return lista


def asimetria(celulas_med):
    #Calcula dos numeros sobre las areas
    #CV area: desvio estandar dividido por la media. Es 0 si todas las celulas son iguales y crece con la dispersion
    #Razon min max: area de la menor sobre area de la mayor, devuelve 1 si son iguales
    #No depende de que tan grande se vea el embrion en la foto, es adimensional
    areas = np.array([c["area"] for c in celulas_med])
    if len(areas) < 2:
        return {"cv_area": 0.0, "razon_min_max": 1.0, "area_media": float(areas.mean()) if len(areas) else 0.0}
    return {"cv_area": float(areas.std() / areas.mean()),
            "razon_min_max": float(areas.min() / areas.max()),
            "area_media": float(areas.mean())}


def medidas_embrion(limpio, um_por_pixel=None):
    #Mide la forma global sobre la mascara limpia
    r = measure.regionprops(limpio.astype(int))[0]
    per = _perimetro(r)
    contornos = measure.find_contours(limpio.astype(float), 0.5)
    contorno = max(contornos, key=len)
    radial = np.hypot(contorno[:, 0] - r.centroid[0], contorno[:, 1] - r.centroid[1])
    d = {
        "area": float(r.area),
        "diametro_equiv": float(r.equivalent_diameter_area if hasattr(r, "equivalent_diameter_area")
                                else r.equivalent_diameter),
        "perimetro": per,
        "circularidad": _circularidad(r.area, per),
        "solidez": float(r.solidity),
        "excentricidad": float(r.eccentricity),
        "relacion_ejes": float(_eje(r, "major") / max(_eje(r, "minor"), 1e-9)), #eje mayor sobre eje menor de la elipse que tiene los mismos momentos que el embrion, 1 es redondo
        "irregularidad_contorno": float(radial.std() / radial.mean()), # se extrae el contorno con find_contours, 0 para un circulo perfecto y sube si el borde es lobulado
        "centroide": (float(r.centroid[0]), float(r.centroid[1])),
        "radio": float(np.sqrt(r.area / np.pi)),
    }
    if um_por_pixel:
        d["diametro_equiv_um"] = d["diametro_equiv"] * um_por_pixel
        d["area_um2"] = d["area"] * um_por_pixel ** 2
    return d


def fragmentacion(celulas):
   #Fraccion de la envolvente convexa de las celulas que ellas no cubren (huecos entre celulas)
   #No detecta fragmentos sueltos afuera de las celulas
    cubierto = celulas > 0
    return float(1 - cubierto.sum() / morphology.convex_hull_image(cubierto).sum())


def compactacion(celulas_med):
    #Distancia media de cada celula a su vecina mas cercana dividida por el diametro medio de las celulas
    #Dos celulas circulares tangentes del mismo diametro tienen los centroides a exactamente un diametro de distancia, asi que un valor cerca de 1 indica celulas en contacto
    #Es informacion relevante pero no se utiliza en el indice de calidad
    if len(celulas_med) < 2:
        return float("nan")
    pts = np.array([c["centroide"] for c in celulas_med])
    dist = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    np.fill_diagonal(dist, np.inf)
    return float(dist.min(axis=1).mean() / np.mean([c["diametro_equiv"] for c in celulas_med]))



def indice_calidad(asim, frag, celulas_med, emb):
    #Calcula cuatro puntajes entre 0 y 1
    #Los combina con los pesos y asigna categoria A,B o C. Es una forma de resumir las medidas, debe pasar despues por implantacion para tomar una decision
    puntajes = {
        "simetria": float(np.clip(1 - asim["cv_area"] / CV_MAX, 0, 1)),
        "fragmentacion": float(np.interp(frag, CORTES_FRAGMENTACION, PUNTAJES_FRAGMENTACION)), #interpolacion lineal con los cortes de Estambul
        "forma_celular": float(np.clip(np.mean([c["circularidad"] for c in celulas_med]), 0, 1)), #circularidad media
        "forma_embrion": float(np.clip((emb["circularidad"] + emb["solidez"]) / 2, 0, 1)), #promedio de circularidad y solidez del embrion
    }
    total = 100 * sum(PESOS[k] * v for k, v in puntajes.items())
    categoria = "A (buena)" if total >= CORTE_A else "B (regular)" if total >= CORTE_B else "C (baja)"
    return {"puntajes": puntajes, "indice": float(total), "categoria": categoria}



def analizar(embrion, celulas, k, um_por_pixel=None):
    #Mide embrion y celulas, calcula asimetria, fragmentacion y compactacion y arma el indice
    #Si se detectaron menos celulas que las estimadas avisa que la simetria es parcial
    #Si la celula mas chica mide menos del 15% del area de la mayor, avisa que puede ser un fragmento o una celula mal separada
    #Devuelve todo en un diccionario, incluidas las mascaras limpias para poder graficar
    limpio = embrion_limpio(embrion)
    cel = celulas_limpias(celulas, limpio)
    emb = medidas_embrion(limpio, um_por_pixel)
    cel_med = medidas_celulas(cel, emb["centroide"], emb["radio"], um_por_pixel)
    if not cel_med:
        raise ValueError("No hay células detectadas dentro del embrión.")
    asim = asimetria(cel_med)
    frag = fragmentacion(cel)
    calidad = indice_calidad(asim, frag, cel_med, emb)

    advertencias = []
    if len(cel_med) < k:
        advertencias.append(f"Se detectaron {len(cel_med)} de {k} células: la simetría es parcial.")
    if len(cel_med) > 1 and asim["razon_min_max"] < 0.15:
        advertencias.append("Hay una célula de menos del 15 % del área de la mayor: puede ser un fragmento "
                            "o una célula mal separada.")
    return {"k": k, "embrion": emb, "celulas": cel_med, "asimetria": asim, "fragmentacion": frag,
            "compactacion": compactacion(cel_med), "calidad": calidad, "advertencias": advertencias,
            "limpio": limpio, "celulas_limpias": cel}


def segmentar(imagen):
    #Reusa el pipeline de segmentar y devuelve la imagen preprocesada, el embrion, las celulas y k
    #De esta forma morfometria no depende de que se corra primero segmentacion
    imagen_pre, mascara_pocillo = preprocesamiento.preprocesar(imagen)
    embrion = segmentacion.embrion_watershed(imagen_pre, mascara_pocillo)
    k = estimar_celulas.estimar_k(imagen_pre, embrion, verbose=False)
    bordes = segmentacion.bordes_canny(imagen_pre, embrion)
    puntaje, radios = segmentacion.hough_embrion(bordes)
    candidatos = segmentacion.candidatos_circulos(puntaje, radios, embrion)
    elegidos = segmentacion.elegir_circulos(candidatos, k=k)
    celulas = segmentacion.separar_celulas(imagen_pre, embrion, elegidos)
    return imagen_pre, embrion, celulas, k

_CACHE_SEGMENTACION = {}

def segmentar_ruta(ruta):
    clave = str(ruta)
    if clave not in _CACHE_SEGMENTACION:
        _CACHE_SEGMENTACION[clave] = segmentar(carga.cargar_imagen(ruta))
    return _CACHE_SEGMENTACION[clave]


def imprimir(res, nombre=""):
    e, a, q = res["embrion"], res["asimetria"], res["calidad"]
    print(f"\n  --- Morfometría {nombre} ({res['k']} células estimadas, {len(res['celulas'])} medidas) ---")
    print("  Células:")
    for c in res["celulas"]:
        print(f"    #{c['etiqueta']}: área={c['area']:.0f} px  diám.={c['diametro_equiv']:.1f} px  "
              f"circularidad={c['circularidad']:.2f}  excentricidad={c['excentricidad']:.2f}  "
              f"solidez={c['solidez']:.2f}")
    print(f"  Asimetría: cv del área={a['cv_area']:.2f}  razón min/max={a['razon_min_max']:.2f}")
    print(f"  Embrión: diám.={e['diametro_equiv']:.0f} px  circularidad={e['circularidad']:.2f}  "
          f"solidez={e['solidez']:.2f}  relación de ejes={e['relacion_ejes']:.2f}  "
          f"irregularidad del contorno={e['irregularidad_contorno']:.3f}")
    print(f"  Fragmentación (huecos)={100 * res['fragmentacion']:.0f} %  compactación={res['compactacion']:.2f}")
    p = q["puntajes"]
    print(f"  Puntajes: simetría={p['simetria']:.2f}  fragmentación={p['fragmentacion']:.2f}  "
          f"forma celular={p['forma_celular']:.2f}  forma embrión={p['forma_embrion']:.2f}")
    print(f"  ÍNDICE DE CALIDAD: {q['indice']:.0f}/100  ->  categoría {q['categoria']}")
    for adv in res["advertencias"]:
        print(f"  Aviso: {adv}")


def figura(imagen_pre, res, titulo):
    """Tres paneles: células con su área, barras de áreas y resumen de la calidad."""
    cel = res["celulas_limpias"]
    fig, ejes = plt.subplots(1, 3, figsize=(16, 5))

    ejes[0].imshow(imagen_pre, cmap="gray")
    ejes[0].imshow(np.ma.masked_equal(cel, 0), cmap="tab10", alpha=0.5, vmin=1, vmax=10)
    ejes[0].contour(res["limpio"], [0.5], colors="white", linewidths=1)
    for c in res["celulas"]:
        ejes[0].text(c["centroide"][1], c["centroide"][0], f"{c['area']:.0f}", color="white", fontsize=9,
                     ha="center", va="center", fontweight="bold",
                     bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))
    ejes[0].set_title("Células y área (px)")
    ejes[0].axis("off")

    areas = [c["area"] for c in res["celulas"]]
    colores = plt.get_cmap("tab10")(np.arange(len(areas)) % 10)
    ejes[1].bar([f"#{c['etiqueta']}" for c in res["celulas"]], areas, color=colores)
    ejes[1].axhline(np.mean(areas), color="k", ls="--", lw=1, label="media")
    ejes[1].set_ylabel("área (px)")
    ejes[1].set_title(f"Asimetría: cv={res['asimetria']['cv_area']:.2f}, "
                      f"min/max={res['asimetria']['razon_min_max']:.2f}")
    ejes[1].legend()

    q, e = res["calidad"], res["embrion"]
    p = q["puntajes"]
    texto = (f"Índice de calidad: {q['indice']:.0f}/100\nCategoría {q['categoria']}\n\n"
             f"Simetría:       {p['simetria']:.2f}   (peso {PESOS['simetria']:.0%})\n"
             f"Fragmentación:  {p['fragmentacion']:.2f}   (peso {PESOS['fragmentacion']:.0%})\n"
             f"Forma celular:  {p['forma_celular']:.2f}   (peso {PESOS['forma_celular']:.0%})\n"
             f"Forma embrión:  {p['forma_embrion']:.2f}   (peso {PESOS['forma_embrion']:.0%})\n\n"
             f"Fragmentación (huecos): {100 * res['fragmentacion']:.0f} %\n"
             f"Circularidad embrión: {e['circularidad']:.2f}\n"
             f"Solidez embrión: {e['solidez']:.2f}\n"
             f"Relación de ejes: {e['relacion_ejes']:.2f}\n"
             f"Compactación: {res['compactacion']:.2f}")
    if res["advertencias"]:
        texto += "\n\nAviso: " + " ".join(res["advertencias"])
    ejes[2].text(0.0, 1.0, texto, va="top", family="monospace", fontsize=9, wrap=True)
    ejes[2].axis("off")
    ejes[2].set_title("Resumen")

    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


CAMPOS_CSV = ["imagen", "estadio_real", "k_estimado", "celulas_medidas", "cv_area", "razon_min_max",
              "circularidad_celular_media", "fragmentacion", "compactacion", "emb_diametro_px",
              "emb_circularidad", "emb_solidez", "emb_relacion_ejes", "emb_irregularidad",
              "p_simetria", "p_fragmentacion", "p_forma_celular", "p_forma_embrion", "indice", "categoria"]


def fila_csv(nombre, estadio, res):
    e, a, q = res["embrion"], res["asimetria"], res["calidad"]
    return {"imagen": nombre, "estadio_real": estadio, "k_estimado": res["k"],
            "celulas_medidas": len(res["celulas"]), "cv_area": round(a["cv_area"], 3),
            "razon_min_max": round(a["razon_min_max"], 3),
            "circularidad_celular_media": round(float(np.mean([c["circularidad"] for c in res["celulas"]])), 3),
            "fragmentacion": round(res["fragmentacion"], 3), "compactacion": round(res["compactacion"], 3),
            "emb_diametro_px": round(e["diametro_equiv"], 1), "emb_circularidad": round(e["circularidad"], 3),
            "emb_solidez": round(e["solidez"], 3), "emb_relacion_ejes": round(e["relacion_ejes"], 3),
            "emb_irregularidad": round(e["irregularidad_contorno"], 4),
            "p_simetria": round(q["puntajes"]["simetria"], 3),
            "p_fragmentacion": round(q["puntajes"]["fragmentacion"], 3),
            "p_forma_celular": round(q["puntajes"]["forma_celular"], 3),
            "p_forma_embrion": round(q["puntajes"]["forma_embrion"], 3),
            "indice": round(q["indice"], 1), "categoria": q["categoria"]}


def etapa_morfometria(ruta, um_por_pixel=None, mostrar=True):
    """Etapa para main.py: segmenta una imagen, mide, imprime y muestra la figura. Devuelve el dict de resultados."""
    print("\n=== Etapa 3: morfometría ===")
    imagen_pre, embrion, celulas, k = segmentar_ruta(ruta)
    res = analizar(embrion, celulas, k, um_por_pixel)
    imprimir(res, ruta.name)
    fig = figura(imagen_pre, res, f"Morfometría: {ruta.name}")
    ruta_fig = visualizacion.guardar_figura(fig, f"morfometria_{ruta.stem}")
    print(f"  Figura guardada en {ruta_fig}")
    if mostrar:
        plt.show(block=False)
        plt.pause(0.2)
        input("  Mirá la figura y presioná Enter para continuar... ")
    plt.close(fig)
    return res


if __name__ == "__main__":
    # python -m src.morfometria : mide toda la muestra y guarda resultados/morfometria.csv
    filas = []
    for ruta, estadio in carga.cargar_muestra():
        imagen_pre, embrion, celulas, k = segmentar(carga.cargar_imagen(ruta))
        res = analizar(embrion, celulas, k)
        imprimir(res, f"{estadio}/{ruta.name}")
        filas.append(fila_csv(f"{estadio}/{ruta.name}", estadio, res))
    Path("resultados").mkdir(exist_ok=True)
    with open(Path("resultados") / "morfometria.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS_CSV)
        escritor.writeheader()
        escritor.writerows(filas)
    print("\nTabla guardada en resultados/morfometria.csv")