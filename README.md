# Predicción de la cobertura de nieve con un modelo NARX

## Primeros pasos

Lo primero será instalar los dos entornos necesarios para la ejecucion del proyecto con los siguientes ficheros: *'environment-hdf.yml'* y *'tf210_gpu.yml'*
Entornos para que funcione el proyecto correctamente, tanto para heatmaps.py como para algunas funcionalidades de limpieza_datos.py es necesario tener activo el entorno *environment-hdf.yml*. Para el resto usaremos *tf210_gpu.yml* ya que usará la versión 2.10 de TensorFlow (libreria para machine learning) y hará uso de la gpu (si el pc está configurado para ello) para procesar los datos más rapidamente. 

Para instalar el entorno simplemente habrá que ejecutar *conda env create -f tf210_gpu.yml*
*conda env list* para comprobar que el entorno se ha creado correctamente y *conda activate tf210_gpu* para activar nuestro nuevo entorno


## 1. Obtención de Datos <a name="id1"> </a>

Los datos MODIS se obtuvieron de [EarthData Search](https://search.earthdata.nasa.gov/search), la plataforma de NASA para la búsqueda de datos geoespaciales. Los pasos para la descarga personalizada fueron los siguientes:

1.  **Filtrado por Fecha y Área de Interés:**
    * Se filtraron los datos por el rango de fechas deseado y se definió el área de interés correspondiente a la basin Adda-Bormio mediante un archivo .zip que contiene el .shp para delimitar el perímetro.

2.  **Descarga Personalizada:**
    * Se seleccionó la opción de descarga personalizada para tener control sobre el formato y la proyección de los datos.
        * ![Primera opción](images/option1.png)
3.  **Reproyección geoespacial (Latitud/Longitud):**
    * Se solicitó que los datos fueran re proyectados al sistema de coordenadas geográficas (latitud/longitud). Por defecto, los datos MODIS se proporcionan en proyección sinusoidal, que no es adecuada para muchos análisis comunes.
        * ![Segunda opción](images/option2.png)
    * Es importante reproyectar los datos antes de la descarga para simplificar el procesamiento posterior.

    ## 🧠 Tipo de Modelo: NARX (Nonlinear AutoRegressive with eXogenous inputs)

Nuestro modelo se basa en la arquitectura **NARX (Nonlinear AutoRegressive with eXogenous inputs)**, que se implementa mediante **Redes Neuronales Recurrentes (RNN)** con **capas LSTM (Long Short-Term Memory)**.

## 2. Distribución de ficheros
El proyecto se divide en 2 grandes partes: el disco externo y el directorio en el que se encuentra este mismo README

Dentro del disco externo habrá dos directorios:
- **data:** en el que se encontrarán todos los CSVs y los archivos hdf descargados
    ->
- **models:** en el que habrá una carpeta por cada cuenca que contendrá el mejor modelo para esa cuenca asi como sus métricas, y gráficas relevantes con respecto a las predicciones:
    - *future_predictions*: se mostrarán las predicciones de los 5 modelos en los 4 escenarios posibles en cada cuenca 
    - *graphs_adda-bornio*: gráficas sobre el rendimiento del modelo comparado con los datos reales
    - *metrics.json*: metricas e hiperparámetros del modelo
    - *narx_model_adda-bornio.h5*: el modelos

![Ejemplo de la carpeta del modelo de adda-bornio](images/ejemplo-ficheros.png)

## 3. Ficheros útiles

### 5.1. limpieza_datos.py
Este fichero contiene un conjunto de funciones útiles para procesar los datos y crear diferentes csv, a continuación se detallarán brevemente las funciones contenidas

- **process_basin(basin):** función principal que calcula el area de la cuenca *'basin'*. Procesa cada uno de los archivos hdfs y calcula el area de nieve, el resultado se guarda en *EXTERNAL_DISK/data/csv/areas/*
- **process_var_exog(input_file, output_path, save=False):** coge el excel de series agregadas y lo convierte a csv separándolo y renombrando las columnas. Devuelve un csv con todas las variables exógenas y con una nueva columna 'cuenca' que idenfica a qué cuenca pertenece cada registro.
- **cleaning_future_series(input_data_path, output_data_path):** función que procesa el excel *EXTERNAL_DISK:\data\csv\Series_historicas-futuras.xlsx* y crea un csv con las varibles exógenas para cada escenario y cada modelo, en total saldrán 20 csv distintos
- **join_area_exog(exog_file, areas_path, output_path = './datasets', save=False):** función para obtener el el dataset final de cada cuenca, coge como parámetro el csv de variables exógenas, el csv de areas calculado anteriormente y los junta, preparado para entrenar al modelo
- **impute_outliers(df, cuenca, columna, save=False):** función que coge un dataframe, y quita los outliers de la columna especificada por parámetro. Se considerará outlier cualquier valor por encima de *1.5 * rango_intercuartilico*

### 5.2. models/best_params.py
Programa muy útil que hace uso de la librería optuna y se encarga de encontrar el mejor modelo para cada cuenca. Simplemente ejecutar el script y se pedirá al usuario la cuenca que se desea optimizar y el número de ensayos que se quiere realizar. Cada ensayo tarda bastante por lo que se recomienda no usar un número demasiado alto, ej: 10-20.
Se guardarán los hiperparámetros del mejor modelo encontrado en un json, el cual se mostrará la ruta por pantalla
La métrica que se usa para la optimización es el NSE (Nash Sutcliffe Efficiency)

### 5.3. models/create_load_model.py
Una vez se conocen la mejor configuración, para un modelo, este script creará o evaluará un nuevo modelo y se crearán gráficas para mejor visualización. Al igual que el fichero anterior, simplemente se ejecuta y el programa se encargará de pedir los datos al usuario

### 5.4. models/predictions.py
Este script pide al usuario el nombre de una cuenca, y el escenario del que se desea obtener la predicciones: obtiene los datasets de variables exógenas de cada modelo para ese escenario y genera una gráfica en la que se visualizan las diferentes predicciones y un csv con las predcicciones de cada modelo.

### 5.5. heatmaps.py
Se encarga de generar los mapas de probabilidad de que cada pixel esté cubierno o no de nieve, leyendo los archivos hdf.
¿Como usar?
Simplemente llamar a la funcion, save = True para guardar los resultados o False simplemente para mostrarlos por pantalla 

### 5.6. environment-hdf.yml & tf210_gpu.yml (IMPORTANTE)
Entornos para que funcione el proyecto correctamente, tanto para heatmaps.py como para algunas funcionalidades de limpieza_datos.py es necesario tener activo el entorno *environment-hdf.yml*. Para el resto usaremos *tf210_gpu.yml* ya que usará la versión 2.10 de TensorFlow (libreria para machine learning) y hará uso de la gpu (si el pc está configurado para ello) para procesar los datos más rapidamente. 

Para instalar el entorno simplemente habrá que ejecutar *conda env create -f tf210_gpu.yml*
*conda env list* para comprobar que el entorno se ha creado correctamente y *conda activate tf210_gpu* para activar nuestro nuevo entorno