from src import carga, preprocesamiento

if __name__ == "__main__":
    print("TPI - Segmentación de blastómeros en embriones de FIV")
    muestra = carga.etapa_carga()
    preprocesamiento.etapa_preprocesamiento(muestra)
    print("\nFin del programa.")