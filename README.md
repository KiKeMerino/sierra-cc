# Predicción de la cobertura de nieve con un modelo NARX


## Descripción

Este proyecto tiene como objetivo crear conciencia sobre cómo va a ir cambiando el nivel de la capa de nieve en fechas futuras utilizando un modelo NARX (Non-linear Autoregressive with Exogenous Inputs).

La funcionalidad de este código es tomar un modelo de red neuronal recurrente (RNN) que has entrenado para predecir 'area_nieve' basándose en valores pasados de sí misma y de otras variables (exógenas), y utilizar ese modelo para predecir valores futuros de 'area_nieve' más allá de los datos que se utilizaron para entrenar y evaluar el modelo

## Estructura de los datos

Los datos utilizados que usaré en este proyecto contienen los siguientes datasets de la nasa del satélite MOD10A1F:

* CGF_NDSI_Snow_Cover: **Este será el que nos interesa**
    - long_name: 'cloud-gap-filled NDSI snow cover',
    - valid_range: [0, 100],
    - FillValue: 255,
    - Key:
        - **40 - 100 = NIEVE (1)**

        - 0-100 = NDSI snow,
        - 200 = missing data,
        - 201 = no decision,
        - 211 = night,
        - 237 = inland water,
        - 239 = ocean,
        - 250 = cloud,
        - 254 = detector saturated,
        - 255 = fill

* Cloud_Persistence
    - long_name:
    - cloud persistence for preceding days,
    - valid_range: [0, 254],
    - FillValue: 255,
    - Key:
        - count of consecutive preceding days of cloud cover

Cada dataset es un raster de datos diviendo la basin en píxeles con los valores arriba mencionados, en este proyecto se considerarán los valores entre 40 y 100 como nieve, y los demás como no nieve para simplificar el modelo

# Estructura del Dataset CGF_NDSI_Snow_Cover _(snow_cover)_
Este es el Dataset con el que trabajaremos, se trata de un xarray que contiene datos de cubierta de nieve derivados de imagenes MODIS, su estructura principal es la siguiente:
* **Dimensiones:**
    * `y`: Coordenadas de latitud
    * `x`: Coordenadas de longitud
    Se puede acceder a las coordenadas de latitud con `snow_cover.y.values` y a las de longitud con `snow_cover.x.values`.
* **Variable principal:**
    * `CGF_NDSI_Snow_Cover`: Representa el Índice de Nieve de Diferencia Normalizada (NDSI), indicando la fracción de cubierta de nieve en cada píxel.
    Los valores de cubierta de nieve se acceden directamente a través de `snow_cover["CGF_NDSI_Snow_Cover"]`.


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

## 2. Arquitectura del modelo

### ¿Qué es NARX?

**NARX** es una clase de modelos de series temporales que predice un valor futuro de una variable basándose en:
* **Auto-Regresivo (AR):** Valores pasados de la misma variable (en este caso, el área de nieve).
* **Variables Exógenas (X):** Valores pasados y/o presentes de otras variables externas que influyen en la variable objetivo. Para este proyecto, estas variables exógenas incluyen:
    * `temperatura`
    * `precipitacion`
    * `dias_sin_precip`
    * (`precipitacion_bool`, `year`, `month`) también presentes en los datos, en algunos modelos los incluiré y en otros no, para comprobar rendimiento y métricas
* **No Lineal (N):** Las relaciones entre las entradas y la salida no son lineales, lo que permite al modelo capturar dinámicas complejas. Las redes neuronales son la elección ideal para modelar estas relaciones no lineales.

### ¿Por qué LSTM?

Las **LSTM** son un tipo avanzado de capa de red neuronal recurrente. Su principal ventaja es su capacidad para aprender y recordar dependencias a largo plazo en secuencias de datos. Esto es crucial para la predicción de series temporales, donde el estado actual del sistema (área de nieve) puede depender de eventos que ocurrieron hace mucho tiempo. A diferencia de las RNN tradicionales, las LSTM superan el problema del "gradiente desvanecido" mediante "puertas" internas que controlan el flujo de información, permitiéndoles retener información relevante y descartar la irrelevante a lo largo de extensas secuencias.

### Arquitectura del Modelo Específico

Se entrena un modelo NARX-LSTM independiente para cada cuenca, lo que permite una especialización y adaptación a las características únicas de cada una. La arquitectura de cada modelo es la siguiente:

* **Capa de Entrada:** `input_shape = (n_lags_area, 1 + num_variables_exogenas)`
    * Recibe secuencias de longitud `n_lags_area` (ej. 3) de datos. Cada elemento de la secuencia es un vector que contiene el área de nieve escalada y las variables exógenas escaladas para un momento dado.
* **Capa LSTM:** `LSTM(n_units_lstm, activation='relu')` Una capa recurrente que procesa la secuencia de entrada. Aprende y extrae patrones temporales y relaciones no lineales. Se configura con 2 parámetros:
    *  n_units_lstm: numero de neuronas, cuanto mayor sea este número mayor capacidad para aprender patrones y relaciones complejas pero mayor coste computacional y mayor riesgo de 'overfitting'
    *  Funcion de activación, en este caso "relu" (Rectified Linear Unit): introduce no linealidad, permitiendo que la red aprenda y modele relaciones más complejas y no lineales en los datos.
* **Capa Densa de Salida:** Una capa totalmente conectada con una única neurona, que produce la predicción del área de nieve para el siguiente paso de tiempo ($t+1$).
    * `Dense(1)`

## 3. Preprocesamiento de los datos

En este apartado se juntarán todas las variables que nos intesan para nuestro modelo del apartado anterior y se analizará el dataset resultante, haremos lo siguiente:

1. EDA: exploracion de los datos
Lo primero será juntar los 2 csv creados anteriormente haciendo coincidir las fechas, el resultado será un dataframe de este estilo:
![Ejemplo genil-dilar](images/genil-dilar(head).png)

Este es el resultado de juntar los dos dataframes creados anteriormente, sin embargo añadiré una columna más para mejorar el modelo "dias_sin_precip" que contará los dias transcurridos desde la última precipitación
![Días sin precipitación](images/dias_sin_precip.png)

Ahora exploraremos cada variable para ver como se comporta:
# adda-bornio
![Estadisticas sobre Adda Bornio](images/genildilar-describe.png)
Este DataFrame contiene datos diarios para la basin "genil-dilar" a lo largo de varios años.

- **dia_sen:** Variable temporal normalizada utilizando la función seno. El valor medio es cercano a cero (-0.0026), con un rango que abarca desde -0.9999 hasta 0.9999, indicando la representación cíclica del año. La desviación estándar (0.7082) sugiere una dispersión considerable a lo largo del ciclo anual.

- **temperatura:** La temperatura media diaria presenta una media de 12.34 °C, con valores que oscilan entre un mínimo de -6.33 °C y un máximo de 29.82 °C. La desviación estándar de 7.49 °C indica una variabilidad significativa en la temperatura diaria.

- **precipitacion:** La precipitación diaria tiene una media de 1.60 litros/m², con un valor mínimo de 0 y un máximo de 54.79 litros/m². La desviación estándar de 3.99 litros/m² muestra una alta variabilidad en la cantidad de precipitación.

- **precipitacion_bool:** Variable binaria que indica la presencia (1) o ausencia (0) de precipitación. De media, llueve aproximadamente el 44% de los días.

- **area_nieve:** El área cubierta por nieve tiene una media de 262.36 km², con un rango que va desde 0 km² hasta un máximo de 477.57 km². La desviación estándar de 173.30 km² revela una gran variabilidad en la extensión de la capa de nieve.

- **dias_sin_precip:** El número de días transcurridos desde la última precipitación tiene una media de 2.37 días, con un máximo de 33 días sin lluvia registrada. La desviación estándar de 3.67 días indica una variabilidad en la frecuencia de las precipitaciones.

# genil-dilar
![Estadisticas sobre Genil Dilar](images/addabornio-describe.png)
Este DataFrame contiene datos diarios para la basin "adda-bornio" a lo largo de varios años.

- **dia_sen:** Similar a otras basins, esta variable temporal normalizada con la función seno tiene una media cercana a cero (-0.0088) y un rango completo de -0.9999 a 0.9999. La desviación estándar es de 0.7076.

- **temperatura:** La temperatura media diaria es de -0.79 °C, notablemente más baja que en la basin anterior. Los valores varían desde un mínimo extremo de -28.79 °C hasta un máximo de 16.91 °C. La desviación estándar de 8.48 °C indica una considerable variabilidad térmica.

- **precipitacion:** La precipitación diaria tiene una media de 2.93 litros/m², con un máximo registrado de 77.47 litros/m². La desviación estándar es de 5.59 litros/m².

- **precipitacion_bool:** La probabilidad de precipitación diaria es mayor en esta basin, con una media de 0.72, lo que indica que llueve aproximadamente el 72% de los días.

- **area_nieve:** El área de nieve presenta una media de 50.80 km², significativamente menor que en la basin "adda-bornio". El área varía desde 0 km² hasta un máximo de 583.21 km². La desviación estándar es de 67.23 km².

- **dias_sin_precip:** El promedio de días sin precipitación es bajo, de 0.81 días, con un máximo de 20 días consecutivos sin lluvia. La desviación estándar es de 1.98 días, lo que sugiere precipitaciones más frecuentes en comparación con la basin anterior.


# indrawati-melamchi
![Estadisticas sobre Indrawati Melamchi](images/indrawatimelamchi-describe.png)
Este DataFrame contiene datos diarios para la basin "indrawati-melamchi" a lo largo de varios años.

- **dia_sen:** La variable temporal normalizada con la función seno tiene una media cercana a cero (-0.0027) y un rango completo de -0.9999 a 0.9999. La desviación estándar es de 0.7083.

- **temperatura:** La temperatura media diaria es de -5.25 °C, similar a la basin "genil-dilar" y también más baja que la "adda-bornio". Los valores oscilan entre un mínimo de -23.50 °C y un máximo de 5.98 °C. La desviación estándar de 6.76 °C indica una variabilidad considerable.

- **precipitacion:** La precipitación diaria presenta una media de 5.25 litros/m², con un máximo registrado de 79.43 litros/m². La desviación estándar es de 7.48 litros/m², mostrando una alta variabilidad.

- **precipitacion_bool:** La presencia de precipitación es muy frecuente en esta basin, con una media de 0.89, lo que sugiere que llueve aproximadamente el 89% de los días.

- **area_nieve:** El área cubierta por nieve tiene una media de 61.61 km², con un rango de 0 km² hasta un máximo de 509.45 km². La desviación estándar de 65.42 km² indica una variabilidad notable en la extensión de la nieve.

- **dias_sin_precip:** El número de días sin precipitación es muy bajo, con una media de 0.31 días y un máximo de 20 días consecutivos sin lluvia. La desviación estándar de 1.28 días sugiere precipitaciones muy frecuentes.

# mapocho-almendros
![Estadisticas sobre mapocho Almendros](images/mapochoalmendros-describe.png)
Este DataFrame contiene datos diarios para la basin "mapocho-almendros" a lo largo de varios años.

- **dia_sen:** La variable temporal normalizada con la función seno presenta una media cercana a cero (-0.0027) y un rango completo de -0.9999 a 0.9999. La desviación estándar es de 0.7084.

- **temperatura:** La temperatura media diaria es de 4.00 °C, ubicándose entre las temperaturas medias de las basins anteriores. Los valores varían desde un mínimo de -16.60 °C hasta un máximo de 17.78 °C. La desviación estándar de 6.28 °C indica una variabilidad térmica considerable.

- **precipitacion:** La precipitación diaria tiene una media de 2.45 litros/m², con un máximo registrado de 94.46 litros/m². La desviación estándar es de 6.53 litros/m², mostrando una alta variabilidad.

- **precipitacion_bool:** La probabilidad de precipitación diaria es de aproximadamente el 59%, con una media de 0.59.

- **area_nieve:** El área cubierta por nieve tiene una media de 167.51 km², con un rango que va desde 0 km² hasta un máximo de 635.69 km². La desviación estándar de 179.31 km² revela una gran variabilidad en la extensión de la capa de nieve.

- **dias_sin_precip:** El promedio de días sin precipitación es de 1.40 días, con un máximo de 24 días consecutivos sin lluvia. La desviación estándar es de 2.61 días.

# nenskra-enguri
![Estadisticas sobre Nenskra Enguri](images/nenskraenguri-describe.png)
Este DataFrame contiene datos diarios para la basin "nenskra-enguri" a lo largo de varios años.

- **dia_sen:** La variable temporal normalizada con la función seno tiene una media cercana a cero (-0.0030) y un rango completo de -0.9999 a 0.9999. La desviación estándar es de 0.7082.

- **temperatura:** La temperatura media diaria es de 2.45 °C, similar a la basin "mapocho-almendros". Los valores oscilan entre un mínimo de -21.34 °C y un máximo de 22.05 °C. La desviación estándar de 8.90 °C indica una variabilidad térmica considerable.

- **precipitacion:** La precipitación diaria presenta una media de 4.45 litros/m², con un máximo registrado de 63.71 litros/m². La desviación estándar es de 6.91 litros/m², mostrando una alta variabilidad.

- **precipitacion_bool:** La presencia de precipitación es frecuente en esta basin, con una media de 0.76, lo que sugiere que llueve aproximadamente el 76% de los días.

- **area_nieve:** El área cubierta por nieve tiene una media de 320.70 km², con un rango de 0 km² hasta un máximo de 628.06 km². La desviación estándar de 224.81 km² revela una gran variabilidad en la extensión de la capa de nieve.

- **dias_sin_precip:** El número de días sin precipitación es bajo, con una media de 0.63 días y un máximo de 20 días consecutivos sin lluvia. La desviación estándar de 1.66 días sugiere precipitaciones frecuentes.

# uncompahgre-ridgway
![Estadisticas sobre Uncompahgre Ridgway](images/uncompahgreridgway-describe.png)
Este DataFrame contiene datos diarios para la basin "uncompahgre-ridgway" a lo largo de varios años.

- **dia_sen:** La variable temporal normalizada con la función seno tiene una media cercana a cero (-0.0027) y un rango completo de -0.9999 a 0.9999. La desviación estándar es de 0.7083.

- **temperatura:** La temperatura media diaria es de 1.64 °C. Los valores varían desde un mínimo de -23.79 °C hasta un máximo de 19.43 °C. La desviación estándar de 9.24 °C indica una variabilidad térmica considerable.

- **precipitacion:** La precipitación diaria presenta una media de 1.86 litros/m², con un máximo registrado de 35.79 litros/m². La desviación estándar es de 3.38 litros/m², mostrando una variabilidad moderada.

- **precipitacion_bool:** La probabilidad de precipitación diaria es de aproximadamente el 59%, con una media de 0.59.

- **area_nieve:** El área cubierta por nieve tiene una media de 243.62 km², con un rango que va desde 0 km² hasta un máximo de 639.85 km². La desviación estándar de 234.05 km² revela una gran variabilidad en la extensión de la capa de nieve.

- **dias_sin_precip:** El promedio de días sin precipitación es de 1.26 días, con un máximo de 20 días consecutivos sin lluvia. La desviación estándar es de 2.27 días.

## 4. Distribución de ficheros
El proyecto se divide en 2 grandes partes: el disco externo y el directorio en el que se encuentra este mismo README

Dentro del disco externo habrá dos directorios:
- **data:** en el que se encontrarán todos los CSVs y los archivos hdf descargados
- **models:** en el que habrá una carpeta por cada cuenca que contendrá el mejor modelo para esa cuenca asi como sus métricas, y gráficas relevantes con respecto a las predicciones
![Ejemplo de la carpeta del modelo de adda-bornio](images/ejemplo-ficheros.png)

## 5. Explicación de ficheros

### 5.1. limpieza_datos.py
Este fichero contiene un conjunto de funciones útiles para procesar los datos y crear diferentes csv, a continuación se detallarán brevemente las funciones contenidas

- **process_basin(basin):** función principal que calcula el area de la cuenca *'basin'*. Procesa cada uno de los archivos hdfs y calcula el area de nieve, el resultado se guarda en *EXTERNAL_DISK/data/csv/areas/*
- **process_var_exog(input_file, output_path, save=False):** coge el excel de series agregadas y lo convierte a csv separándolo y renombrando las columnas. Devuelve un csv con todas las variables exógenas y con una nueva columna 'cuenca' que idenfica a qué cuenca pertenece cada registro.
- **join_area_exog(exog_file, areas_path, output_path = './datasets', save=False):** función para obtener el el dataset final de cada cuenca, coge como parámetro el csv de variables exógenas, el csv de areas calculado anteriormente y los junta, preparado para entrenar al modelo
- **cleaning_future_series(input_data_path, output_data_path):** función que procesa el excel *EXTERNAL_DISK:\data\csv\Series_historicas-futuras.xlsx* y crea un csv con las varibles exógenas para cada escenario y cada modelo, en total saldrán 20 csv distintos
- **impute_outliers(df, cuenca, columna, save=False):** función que coge un dataframe, y quita los outliers de la columna especificada por parámetro. Se considerará outlier cualquier valor por encima de *1.5 * rango_intercuartilico*

### 5.2. models/best_params.py
### 5.3. models/create_load_model.py
### 5.4. models/predictions.py
### 5.5. graficas.py
### 5.6. environment-hdf.yml & environment-tf.yml




Actualiza la ventana de historial: elimina el valor más antiguo y añade la predicción actual (escalada).


METRICAS CON Random Forest:
- Evaluación en el conjunto de prueba (predicción directa):
    R2: 0.9467, MAE: 28.7578, NSE: 0.9467, KGE: 0.9404
- Evaluación en el conjunto de validación iterativa (modo predictivo):
    R2: 0.1556, MAE: 173.3554, NSE: 0.1556, KGE: 0.2602
- Métricas en todo el conjunto de datos (modo predictivo):
    R2: -0.5581, MAE: 177.8610, NSE: -0.5581, KGE: 0.0190




NUEVO MODELO NARX (Red Neuronal Auto-Regresiva con Variables Exógenas) implementada con capas LSTM(Long Short-Term Memory) de la libreria Keras (que se ejecuta sobre TensorFlow)
    - El LSTM es un tipo de capa de red neuronal recurrente
    - Para cada paspo en el tiempo, el modelo recibe como entrada una secuencia de los n_lags_area valores pasados del area de nieve
    - La capa LSTM procesa esta secuencia de entrada, aprendiendo las relaciones temporales entre el área de nieve pasada y las variables exógenas pasadas para entender cómo influyen en el valor futuro del área de nieve.






## 📊 Evaluación del Modelo

El rendimiento del modelo se evalúa utilizando cuatro métricas clave en diferentes conjuntos de datos para proporcionar una visión completa de su capacidad:

* **R2 (Coeficiente de Determinación):** Mide la proporción de la varianza en la variable dependiente que es predecible a partir de las variables independientes. Un valor cercano a 1.0 indica un buen ajuste.
* **MAE (Error Absoluto Medio):** Calcula la media de las diferencias absolutas entre las predicciones y los valores reales. Se expresa en las mismas unidades que la variable objetivo, facilitando su interpretación.
* **NSE (Eficiencia de Nash-Sutcliffe):** Una métrica hidrológica que evalúa qué tan bien las predicciones se ajustan a las observaciones. Un NSE de 1.0 indica un ajuste perfecto, mientras que valores negativos sugieren que el modelo es peor que usar la media de las observaciones.
* **KGE (Eficiencia de Kling-Gupta):** Mejora el NSE al considerar la correlación, la varianza relativa y la sesgo relativo entre las predicciones y las observaciones. Un KGE de 1.0 es el valor óptimo.

### Conjuntos de Evaluación

1.  **Métricas de Entrenamiento:**
    * **Dónde:** Calculadas sobre el `conjunto de entrenamiento`.
    * **Propósito:** Indican qué tan bien el modelo ha aprendido los patrones de los datos utilizados para su optimización. Un buen rendimiento es esencial, pero un rendimiento excesivamente alto puede indicar sobreajuste.

2.  **Métricas de Prueba:**
    * **Dónde:** Calculadas sobre el `conjunto de prueba`.
    * **Propósito:** Evalúan la capacidad de generalización del modelo en datos **nunca antes vistos** durante el entrenamiento. Son la medida más honesta del rendimiento del modelo fuera del proceso de aprendizaje.

3.  **Métricas de Validación (Predicción Paso a Paso):**
    * **Dónde:** Calculadas sobre el `conjunto de validación`.
    * **Propósito:** Simulan un escenario de predicción futura real. El modelo utiliza los `n_lags_area` datos históricos iniciales y, a partir de ahí, usa sus **propias predicciones anteriores** como entrada para los pasos subsiguientes, junto con los valores reales futuros de las variables exógenas.
    * **Importancia:** Revela la robustez del modelo y cómo se acumulan los errores de predicción a lo largo del tiempo. Las métricas aquí suelen ser las más bajas debido a la propagación de errores, lo cual es un comportamiento esperado.

4.  **Métricas en Todo el Conjunto de Datos:**
    * **Dónde:** Calculadas combinando las predicciones y valores reales de los `conjuntos de entrenamiento`, `prueba` y `validación`.
    * **Propósito:** Proporciona una visión global del rendimiento del modelo a lo largo de todo el período de datos disponibles, ofreciendo un resumen consolidado en modo predicción.


NARX_MODELS1
n_lags_area = 3
n_layers = 2
n_neuronas = 50
epochs = 30


NARX_MODELS2
n_lags_area = 5
n_layers = 2
n_neuronas = 10
epochs = 50


NARX_MODELS3
n_lags_area = 3
n_layers = 2
n_neuronas = 10
epochs = 50


NARX_MODELS4 
n_lags_area = 3
n_layers = 1
n_neuronas = 10
epochs = 30

NARX_MODELS5
n_lags_area = 4
n_layers = 3
n_neuronas = 20
epochs = 40