import faulthandler
import sys
import traceback

import matplotlib.pyplot as plt

from src import carga, preprocesamiento, segmentacion, morfometria, implantacion, compresion



if __name__ == "__main__":
    print("TPI - Segmentación de blastómeros en embriones de FIV")
    muestra = carga.etapa_carga()

    while True:
        ruta = preprocesamiento.elegir_imagen(muestra)

        try:
            # Etapa 1: preprocesamiento (con las funciones de tu compañera)
            print("\n=== Etapa 1: preprocesamiento ===")
            imagen = carga.cargar_imagen(ruta)
            pasos = preprocesamiento.preprocesar_con_pasos(imagen)
            ruta_figura = preprocesamiento.mostrar_preprocesamiento(imagen, pasos, ruta.name)
            print(f"  Figura guardada en {ruta_figura}")

            # Diagnóstico: si el cálculo de las etapas 2-4 tarda más de 60 s sin terminar, imprime en pantalla
            # en qué línea está (cada 60 s). Así se ve si es lentitud o un cuelgue real y dónde ocurre.
            faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)

            # Etapa 2: segmentación (guarda sus figuras en disco y no las muestra)
            segmentacion.etapa_segmentacion(ruta)
            plt.close("all")   # evita que se mezclen las figuras de la etapa 2 con las de la etapa 3

            # Etapa 3: morfometría
            morfometria.etapa_morfometria(ruta, mostrar=False)

            # Etapa 4: clasificación morfológica
            implantacion.etapa_implantacion(ruta)

            #Etapa 5: Compresion
            compresion.etapa_compresion(
                img_preprocesada=pasos["final"],
                mascara_pocillo=pasos["mascara"],
                nombre_imagen=ruta.name
            )
        except Exception:
            # Un error en una imagen no debe cerrar el programa: se informa y se sigue con el menú
            print("\n  ERROR procesando esta imagen:")
            traceback.print_exc()
        finally:
            faulthandler.cancel_dump_traceback_later()
            plt.close("all")

        respuesta = input("\n¿Querés usar otra imagen? (s/n): ").strip().lower()
        if respuesta != "s":
            break

    print("\nFin del programa.")