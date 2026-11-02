
#Clasificación morfológica del embrión en estadio de clivaje para estimar su prioridad de transferencia.
#Recibe lo que ya midio morfometria y le asigna un grado 1,2 o 3 siguiendo la logica de consenso de Estambul

from src import carga, morfometria

FRAGMENTACION_G1 = 0.10          #Grado 1: fragmentación < 10 %
FRAGMENTACION_G2 = 0.25          #Grado 2: entre 10 y 25 %; grado 3: más de 25 %
DIFERENCIA_DIAMETRO_MAX = 0.25   #Mas de 25% de diferencia de diametro con la celula mas chica ya no es tamaño propio del estadio

def criterio_fragmentacion(fragmentacion):
    #Recibe la fraccion de huecos entre celulas que calculo morfometria y devuelve el grado
    #Esta fragmentacion mide solo los huecos entre celulas, no fragmentos sueltos alrededor
    #El grado es un minimo, puede haber mas fragmentacion de la que se ve
    if fragmentacion < FRAGMENTACION_G1:
        grado = 1
    elif fragmentacion <= FRAGMENTACION_G2: #el 0,10 exacto cae en grado 2 porque el conseso no aclara los bordes
        grado = 2
    else:
        grado = 3
    return {"nombre": "Fragmentación", "valor": f"{100 * fragmentacion:.0f} % (huecos entre células)",
            "grado": grado}


def criterio_simetria(celulas):
    #Compara cada celula contra la mayor para poder graduar en 1,2 y 3
    #Si usara solo el minimo contra el maximo no habria forma de distinguir un embrion con una celula atipica (grado 2) de uno donde la mayoria son atipicas (grado 3)

    #Si hay menos de 2 celulas, devuelve grado 3 y avisa que no se puede evaluar
    if len(celulas) < 2:
        return {"nombre": "Simetría", "valor": "una sola célula", "grado": 3,
                "nota": "No se puede evaluar la simetría."}

    #Toma el diametro de la celula mayor y calcula los diametros a partir del area
    d_max = max(c["diametro_equiv"] for c in celulas)

    #Cuenta cuantas celulas tienen una diferencia con la mayor de 25% o menos
    dentro = [c for c in celulas if (d_max - c["diametro_equiv"]) / d_max <= DIFERENCIA_DIAMETRO_MAX]
    todas = len(dentro) == len(celulas)
    mayoria = len(dentro) > len(celulas) / 2

    #Se determina grado 1 si estan todas dentro del rango, grado 2 si mas de la mitad, grado 3 si la mitad o menos
    grado = 1 if todas else 2 if mayoria else 3
    d_min = min(c["diametro_equiv"] for c in celulas)
    return {"nombre": "Simetría", "valor": f"{len(dentro)}/{len(celulas)} células de tamaño propio del estadio; "
                                           f"diferencia máx. de diámetro {100 * (d_max - d_min) / d_max:.0f} %",
            "grado": grado, "nota": "Foto 2D con células superpuestas: la asimetría puede estar sobreestimada."} #No es un metodo 100% confiable, es estimativo



GRADOS = {1: "Grado 1 (buena calidad): prioridad alta de transferencia",
          2: "Grado 2 (calidad intermedia): prioridad media",
          3: "Grado 3 (calidad baja): prioridad baja"}


def evaluar(res):
    #Corre los dos criterios y toma como grado final el peor de los dos, no hay una regla general sobre como se pondera cada criterio
    #devuelve un diccionario con los criterios, el grado y la cantidad de celulas estimada
    criterios = [criterio_simetria(res["celulas"]), criterio_fragmentacion(res["fragmentacion"])]
    grado = max(c["grado"] for c in criterios)
    return {"criterios": criterios, "grado": grado, "categoria": GRADOS[grado], "k": res["k"]}


def imprimir(ev, nombre=""): 
    #Solo muestra el resultado de la evaluacion
    print(f"\n  --- Clasificación morfológica {nombre} ({ev['k']} células estimadas) ---")
    for c in ev["criterios"]:
        print(f"    {c['nombre']}: {c['valor']} -> grado {c['grado']}")
        if c.get("nota"):
            print(f"      ({c['nota']})")
    print(f"  RESULTADO: {ev['categoria']}")


def etapa_implantacion(ruta):
    #Carga la imagen, la segmenta con morfometria.segmentar, mide con morfometria.analizar y analiza con evaluar
    #Repite la segmentacion que ya hizo la etapa 3, porque asi el modulo es independiente y funciona solo PERO se estaria procesando la imagen 2 veces
    print("\n=== Etapa 4: clasificación morfológica ===")
    _, embrion, celulas, k = morfometria.segmentar_ruta(ruta)
    res = morfometria.analizar(embrion, celulas, k)
    ev = evaluar(res)
    imprimir(ev, ruta.name)
    return res, ev


if __name__ == "__main__":
    print("grado por criterio: simetría / fragmentación")
    for ruta, estadio in carga.cargar_muestra():
        _, embrion, celulas, k = morfometria.segmentar(carga.cargar_imagen(ruta))
        ev = evaluar(morfometria.analizar(embrion, celulas, k))
        g = [c["grado"] for c in ev["criterios"]]
        print(f"{estadio}/{ruta.name:8s} k={k}  grados {g[0]}/{g[1]}  ->  {ev['categoria']}")