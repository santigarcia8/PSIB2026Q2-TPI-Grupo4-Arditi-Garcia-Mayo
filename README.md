# Algoritmo de procesamiento digital de imágenes para la segmentación de blastómeros, análisis morfométrico y evaluación de viabilidad embrionaria en etapas tempranas de Fertilización In Vitro (FIV) 

TPI - Procesamiento de Señales e Imágenes Biomédicas - ITBA - 2Q2026

El algoritmo permite realizar un análisis de embriones desarrollados por FIV mediante detección de cantidad de células, fragmentación, texturas y simetría. Este permite conocer el estadío en el que se encuentra para un monitoreo automatizado y control de la estabilidad del embrión pre implantación, asignando un grado de calidad morfológica.

## INTEGRANTES
- Arditi, Agustina Sara
- Garcia, Santiago
- Mayo, Olivia

## DATASET
- **HumanEmbryo2.0** (Zenodo, Record 14253170): imágenes reales
  de embriones humanos anotadas en 2, 4 y 8 células, mórula y blastocisto.
  https://zenodo.org/records/14253170

Del dataset se tomaron 10 imágenes de manera aleatoria que se encuentran en "muestra/", siendo tres de 2 células, tres de 4 células y cuatro de 8 células

## ESTRUCTURA
```
├── main.py                  # Programa principal (menú interactivo)
├── requirements.txt         # Dependencias
├── muestra/                 # Imágenes a analizar, agrupadas por estadio
├── resultados/              # Agrupados por carpeta según imagen
└── src/
    ├── carga.py             # Lectura y normalización de imágenes
    ├── preprocesamiento.py  # Máscara del pocillo, Wiener, iluminación, contraste
    ├── segmentacion.py      # Otsu, Watershed, Snake, Canny + Hough
    ├── estimar_celulas.py   # Estimación de k (2, 4 u 8) por textura (GLCM)
    ├── morfometria.py       # Área, circularidad, asimetría, fragmentación, índice
    ├── implantacion.py      # Clasificación en grados según el consenso de Estambul
    ├── compresion.py        # Huffman y RLE
    └── visualizacion.py     # Para graficar y guardar figuras
```

## PIPELINE
### 1. Preprocesamiento
Se detecta el pocillo del resto del fondo negro de la imagen microscópica. Se hace un ajuste por cuadrados mínimos del área circular que lo contiene, luego se elimina el ruido mediante un filtro Wiener y se mejora la iluminación mediante un filtro pasa altos en el dominio de Fourier. Finalmente, se realzan bordes y contraste

### 2. Segmentación
Inicialmente se segmenta la masa embrionaria del resto del fondo empleando Otsu sobre el desvío, Otsu + Watershed y Otsu + Snakes de forma comparativa. Luego, tomando los resultados del segundo método, se hace una estimación de la cantidad de células que debería contener ese embrión según membranas que separen a las células. Tomando dicho resultado, se emplea Canny + Transformada de Hough + Watershed para segmentar las células presentes

### 3. Morfometría
Mide área, circularidad, excentricidad y solidez de cada célula, y calcula la asimetría y la fragmentación

### 4. Clasificación
Asigna un grado de 1 a 3 según simetría y fragmentación.

### 5. Compresión
Huffman para la imagen preprocesada y RLE para la máscara, guardándolos como .pkl

## INSTALACIÓN
```bash
git clone https://github.com/santigarcia8/PSIB2026Q2-TPI-Grupo4-Arditi-Garcia-Mayo.git
cd PSIB2026Q2-TPI-Grupo4-Arditi-Garcia-Mayo
python -m venv venv
venv\Scripts\activate       
pip install -r requirements.txt
```

## USO

```bash
python main.py
```

Corriendo el main, se muestra una lista de las imágenes de "muestra/" con un input para seleccionar qué imagen se desea analizar. 
Una vez elegido, se ejecutan del paso 1 al 5 automáticamente, guardando los resultados en "resultados/". Cuando finaliza, vuelve a preguntar si se desea proceder con otra imagen o finalizar la ejecución.

## LIMITACIONES
- Los umbrales para segmentar la cantidad de células se determinaron en base a 10 imágenes
- No se pueden calcular medidas en mm^2 porque no hay referencia del tamaño del píxel
- Los radios de la transformada de Hough se determinaron experimentalmente