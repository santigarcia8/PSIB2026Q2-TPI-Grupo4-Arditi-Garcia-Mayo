import matplotlib.pyplot as plt

from src import carga, preprocesamiento, segmentacion

if __name__ == "__main__":
    print("TPI - Segmentación de blastómeros en embriones de FIV")
    muestra = carga.etapa_carga()

    while True:
        ruta = preprocesamiento.elegir_imagen(muestra)

        # Etapa 1: preprocesamiento (con las funciones de tu compañera)
        print("\n=== Etapa 1: preprocesamiento ===")
        imagen = carga.cargar_imagen(ruta)
        pasos = preprocesamiento.preprocesar_con_pasos(imagen)
        ruta_figura = preprocesamiento.mostrar_preprocesamiento(imagen, pasos, ruta.name)
        print(f"  Figura guardada en {ruta_figura}")
        print("  (cerrá la ventana de la figura para seguir)")
        plt.show()

        # Etapa 2: tu segmentación, sobre la misma imagen
        segmentacion.etapa_segmentacion(ruta)

        respuesta = input("\n¿Querés usar otra imagen? (s/n): ").strip().lower()
        if respuesta != "s":
            break

    print("\nFin del programa.")