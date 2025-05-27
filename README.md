# Predicción de la cobertura de nieve con un modelo NARX

## Descripción

Este proyecto tiene como objetivo crear conciencia sobre cómo va a ir cambiando el nivel de la capa de nieve en fechas futuras utilizando un modelo NARX (Non-linear Autoregressive with Exogenous Inputs).

La funcionalidad de este código es tomar un modelo de red neuronal recurrente (RNN) que has entrenado para predecir 'area_nieve' basándose en valores pasados de sí misma y de otras variables (exógenas), y utilizar ese modelo para predecir valores futuros de 'area_nieve' más allá de los datos que se utilizaron para entrenar y evaluar el modelo

## Estructura de los datos

Los datos utilizados que usaré en este proyecto contienen los siguientes datasets de la NASA:

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

Cada dataset es un raster de datos diviendo la basin en píxeles con los valores arriba mencionados, yo consideraré  los valores entre 40 y 100 como nieve, y los demás como no nieve para simplificar el modelo

# Estructura del Dataset CGF_NDSI_Snow_Cover _(snow_cover)_
Este es el Dataset con el que trabajaremos, se trata de un xarray que contiene datos de cubierta de nieve derivados de imagenes MODIS, su estructura principal es la siguiente:
* **Dimensiones:**
    * `y`: Coordenadas de latitud
    * `x`: Coordenadas de longitud
    Se puede acceder a las coordenadas de latitud con `snow_cover.y.values` y a las de longitud con `snow_cover.x.values`.
* **Variable principal:**
    * `CGF_NDSI_Snow_Cover`: Representa el Índice de Nieve de Diferencia Normalizada (NDSI), indicando la fracción de cubierta de nieve en cada píxel.
    Los valores de cubierta de nieve se acceden directamente a través de `snow_cover["CGF_NDSI_Snow_Cover"]`.


## 1. Obtención de Datos

Los datos MODIS se obtuvieron de [EarthData Search](https://search.earthdata.nasa.gov/search), la plataforma de NASA para la búsqueda de datos geoespaciales. Los pasos para la descarga personalizada fueron los siguientes:

1.  **Filtrado por Fecha y Área de Interés:**
    * Se filtraron los datos por el rango de fechas deseado y se definió el área de interés correspondiente a la basin Adda-Bormio.

2.  **Descarga Personalizada:**
    * Se seleccionó la opción de descarga personalizada para tener control sobre el formato y la proyección de los datos.
        * ![Primera opción](img/option1.png)
3.  **Reproyección geoespacial (Latitud/Longitud):**
    * Se solicitó que los datos fueran re proyectados al sistema de coordenadas geográficas (latitud/longitud). Por defecto, los datos MODIS se proporcionan en proyección sinusoidal, que no es adecuada para muchos análisis comunes.
        * ![Segunda opción](img/option2.png)
    * Es importante reproyectar los datos antes de la descarga para simplificar el procesamiento posterior.

## 2. Limpieza de datos

Este apartado se centrará en leer tanto los archivos hdf descargados previamente como las series históricas agregadas y guardarlos en csv para su posterior procesamiento

1. Lectura de datos necesarios de archivos hdf
2. Cálculo del área de nieve para cada día y guardarlo en un csv con estas 2 columnas (fecha y area_nieve)
3. Lectura las series históricas agregadas: variables de temperatura y precipitación
4. Limpiar y normalizar los datos sobre las series agregadas: correción de formato, eliminación de columnas innecesarias, agregas columnas faltantes, etc...
5. Separar estas series agregadas en 6 csv (uno por cada basin)


## 3. Preprocesamiento de los datos

En este apartado se juntarán todas las variables que nos intesan para nuestro modelo del apartado anterior y se analizará el dataset resultante, haremos lo siguiente:

1. EDA: exploracion de los datos
Lo primero será juntar los 2 csv creados anteriormente haciendo coincidir las fechas, el resultado será un dataframe de este estilo:
![Ejemplo genil-dilar](img/genil-dilar(head).png)

Este es el resultado de juntar los dos dataframes creados anteriormente, sin embargo añadiré una columna más para mejorar el modelo "dias_sin_precip" que contará los dias transcurridos desde la última precipitación
![Días sin precipitación](img/dias_sin_precip.png)

Ahora exploraremos cada variable para ver como se comporta:
# adda-bornio
![Estadisticas sobre Adda Bornio](img/genildilar-describe.png)
Este DataFrame contiene datos diarios para la basin "genil-dilar" a lo largo de varios años.

- **dia_sen:** Variable temporal normalizada utilizando la función seno. El valor medio es cercano a cero (-0.0026), con un rango que abarca desde -0.9999 hasta 0.9999, indicando la representación cíclica del año. La desviación estándar (0.7082) sugiere una dispersión considerable a lo largo del ciclo anual.

- **temperatura:** La temperatura media diaria presenta una media de 12.34 °C, con valores que oscilan entre un mínimo de -6.33 °C y un máximo de 29.82 °C. La desviación estándar de 7.49 °C indica una variabilidad significativa en la temperatura diaria.

- **precipitacion:** La precipitación diaria tiene una media de 1.60 litros/m², con un valor mínimo de 0 y un máximo de 54.79 litros/m². La desviación estándar de 3.99 litros/m² muestra una alta variabilidad en la cantidad de precipitación.

- **precipitacion_bool:** Variable binaria que indica la presencia (1) o ausencia (0) de precipitación. De media, llueve aproximadamente el 44% de los días.

- **area_nieve:** El área cubierta por nieve tiene una media de 262.36 km², con un rango que va desde 0 km² hasta un máximo de 477.57 km². La desviación estándar de 173.30 km² revela una gran variabilidad en la extensión de la capa de nieve.

- **dias_sin_precip:** El número de días transcurridos desde la última precipitación tiene una media de 2.37 días, con un máximo de 33 días sin lluvia registrada. La desviación estándar de 3.67 días indica una variabilidad en la frecuencia de las precipitaciones.

# genil-dilar
![Estadisticas sobre Genil Dilar](img/addabornio-describe.png)
Este DataFrame contiene datos diarios para la basin "adda-bornio" a lo largo de varios años.

- **dia_sen:** Similar a otras basins, esta variable temporal normalizada con la función seno tiene una media cercana a cero (-0.0088) y un rango completo de -0.9999 a 0.9999. La desviación estándar es de 0.7076.

- **temperatura:** La temperatura media diaria es de -0.79 °C, notablemente más baja que en la basin anterior. Los valores varían desde un mínimo extremo de -28.79 °C hasta un máximo de 16.91 °C. La desviación estándar de 8.48 °C indica una considerable variabilidad térmica.

- **precipitacion:** La precipitación diaria tiene una media de 2.93 litros/m², con un máximo registrado de 77.47 litros/m². La desviación estándar es de 5.59 litros/m².

- **precipitacion_bool:** La probabilidad de precipitación diaria es mayor en esta basin, con una media de 0.72, lo que indica que llueve aproximadamente el 72% de los días.

- **area_nieve:** El área de nieve presenta una media de 50.80 km², significativamente menor que en la basin "adda-bornio". El área varía desde 0 km² hasta un máximo de 583.21 km². La desviación estándar es de 67.23 km².

- **dias_sin_precip:** El promedio de días sin precipitación es bajo, de 0.81 días, con un máximo de 20 días consecutivos sin lluvia. La desviación estándar es de 1.98 días, lo que sugiere precipitaciones más frecuentes en comparación con la basin anterior.


# indrawati-melamchi
![Estadisticas sobre Indrawati Melamchi](img/indrawatimelamchi-describe.png)
Este DataFrame contiene datos diarios para la basin "indrawati-melamchi" a lo largo de varios años.

- **dia_sen:** La variable temporal normalizada con la función seno tiene una media cercana a cero (-0.0027) y un rango completo de -0.9999 a 0.9999. La desviación estándar es de 0.7083.

- **temperatura:** La temperatura media diaria es de -5.25 °C, similar a la basin "genil-dilar" y también más baja que la "adda-bornio". Los valores oscilan entre un mínimo de -23.50 °C y un máximo de 5.98 °C. La desviación estándar de 6.76 °C indica una variabilidad considerable.

- **precipitacion:** La precipitación diaria presenta una media de 5.25 litros/m², con un máximo registrado de 79.43 litros/m². La desviación estándar es de 7.48 litros/m², mostrando una alta variabilidad.

- **precipitacion_bool:** La presencia de precipitación es muy frecuente en esta basin, con una media de 0.89, lo que sugiere que llueve aproximadamente el 89% de los días.

- **area_nieve:** El área cubierta por nieve tiene una media de 61.61 km², con un rango de 0 km² hasta un máximo de 509.45 km². La desviación estándar de 65.42 km² indica una variabilidad notable en la extensión de la nieve.

- **dias_sin_precip:** El número de días sin precipitación es muy bajo, con una media de 0.31 días y un máximo de 20 días consecutivos sin lluvia. La desviación estándar de 1.28 días sugiere precipitaciones muy frecuentes.

# mapocho-almendros
![Estadisticas sobre mapocho Almendros](img/mapochoalmendros-describe.png)
Este DataFrame contiene datos diarios para la basin "mapocho-almendros" a lo largo de varios años.

- **dia_sen:** La variable temporal normalizada con la función seno presenta una media cercana a cero (-0.0027) y un rango completo de -0.9999 a 0.9999. La desviación estándar es de 0.7084.

- **temperatura:** La temperatura media diaria es de 4.00 °C, ubicándose entre las temperaturas medias de las basins anteriores. Los valores varían desde un mínimo de -16.60 °C hasta un máximo de 17.78 °C. La desviación estándar de 6.28 °C indica una variabilidad térmica considerable.

- **precipitacion:** La precipitación diaria tiene una media de 2.45 litros/m², con un máximo registrado de 94.46 litros/m². La desviación estándar es de 6.53 litros/m², mostrando una alta variabilidad.

- **precipitacion_bool:** La probabilidad de precipitación diaria es de aproximadamente el 59%, con una media de 0.59.

- **area_nieve:** El área cubierta por nieve tiene una media de 167.51 km², con un rango que va desde 0 km² hasta un máximo de 635.69 km². La desviación estándar de 179.31 km² revela una gran variabilidad en la extensión de la capa de nieve.

- **dias_sin_precip:** El promedio de días sin precipitación es de 1.40 días, con un máximo de 24 días consecutivos sin lluvia. La desviación estándar es de 2.61 días.

# nenskra-enguri
![Estadisticas sobre Nenskra Enguri](img/nenskraenguri-describe.png)
Este DataFrame contiene datos diarios para la basin "nenskra-enguri" a lo largo de varios años.

- **dia_sen:** La variable temporal normalizada con la función seno tiene una media cercana a cero (-0.0030) y un rango completo de -0.9999 a 0.9999. La desviación estándar es de 0.7082.

- **temperatura:** La temperatura media diaria es de 2.45 °C, similar a la basin "mapocho-almendros". Los valores oscilan entre un mínimo de -21.34 °C y un máximo de 22.05 °C. La desviación estándar de 8.90 °C indica una variabilidad térmica considerable.

- **precipitacion:** La precipitación diaria presenta una media de 4.45 litros/m², con un máximo registrado de 63.71 litros/m². La desviación estándar es de 6.91 litros/m², mostrando una alta variabilidad.

- **precipitacion_bool:** La presencia de precipitación es frecuente en esta basin, con una media de 0.76, lo que sugiere que llueve aproximadamente el 76% de los días.

- **area_nieve:** El área cubierta por nieve tiene una media de 320.70 km², con un rango de 0 km² hasta un máximo de 628.06 km². La desviación estándar de 224.81 km² revela una gran variabilidad en la extensión de la capa de nieve.

- **dias_sin_precip:** El número de días sin precipitación es bajo, con una media de 0.63 días y un máximo de 20 días consecutivos sin lluvia. La desviación estándar de 1.66 días sugiere precipitaciones frecuentes.

# uncompahgre-ridgway
![Estadisticas sobre Uncompahgre Ridgway](img/uncompahgreridgway-describe.png)
Este DataFrame contiene datos diarios para la basin "uncompahgre-ridgway" a lo largo de varios años.

- **dia_sen:** La variable temporal normalizada con la función seno tiene una media cercana a cero (-0.0027) y un rango completo de -0.9999 a 0.9999. La desviación estándar es de 0.7083.

- **temperatura:** La temperatura media diaria es de 1.64 °C. Los valores varían desde un mínimo de -23.79 °C hasta un máximo de 19.43 °C. La desviación estándar de 9.24 °C indica una variabilidad térmica considerable.

- **precipitacion:** La precipitación diaria presenta una media de 1.86 litros/m², con un máximo registrado de 35.79 litros/m². La desviación estándar es de 3.38 litros/m², mostrando una variabilidad moderada.

- **precipitacion_bool:** La probabilidad de precipitación diaria es de aproximadamente el 59%, con una media de 0.59.

- **area_nieve:** El área cubierta por nieve tiene una media de 243.62 km², con un rango que va desde 0 km² hasta un máximo de 639.85 km². La desviación estándar de 234.05 km² revela una gran variabilidad en la extensión de la capa de nieve.

- **dias_sin_precip:** El promedio de días sin precipitación es de 1.26 días, con un máximo de 20 días consecutivos sin lluvia. La desviación estándar es de 2.27 días.



Para predecir el área de nieve en el día t+1:

Las entradas al modelo serán:
Las predicciones del área de nieve en los días t,t−1,t−2,...,t−n (donde n es el número de lags).
Los valores reales de las variables exógenas para el día t+1.
Para predecir el área de nieve en el día t+2:

Las entradas al modelo serán:
La predicción del área de nieve para el día t+1.
La predicción del área de nieve para los días t,t−1,...,t−(n−1).
Los valores reales de las variables exógenas para el día t+2.
Y así sucesivamente.


Esquema de la predicción iterativa en el conjunto de validación:

Selecciona la última ventana de n_lags_area datos del conjunto de entrenamiento (escalados) como el historial inicial.
Itera sobre el número de pasos que deseas predecir en el conjunto de validación iterativa.
En cada paso:
Toma la ventana de historial actual y los valores correspondientes de las variables exógenas del conjunto de validación iterativa.
Realiza la predicción con el modelo.
Desescala la predicción del área de nieve.
Almacena esta predicción.
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


Métricas conjunto de 'train' para adda-bornio: {'R2': 0.9841177450777601, 'MAE': 10.54553345032225, 'NSE': 0.9841177450777601, 'KGE': 0.9698584488088482}
Métricas conjunto de 'test' para adda-bornio: {'R2': 0.9858470369670861, 'MAE': 10.73206148625898, 'NSE': 0.9858470369670861, 'KGE': 0.9673529671629062}
Métricas conjunto de 'validation' (modo prediccion) para adda-bornio: {'R2': 0.7563844177771055, 'MAE': 72.30012986340141, 'NSE': 0.7563844177771055, 'KGE': 0.6762539448332815}
Métricas en todo el conjunto de datos (modo prediccion) para adda-bornio: R2=0.811, MAE=63.246, NSE=0.811, KGE=0.717

Métricas conjunto de 'train' para genil-dilar: {'R2': 0.9296348450461472, 'MAE': 7.3944132279862185, 'NSE': 0.9296348450461472, 'KGE': 0.8957372075123364}
Métricas conjunto de 'test' para genil-dilar: {'R2': 0.8951661809883981, 'MAE': 7.030168435421019, 'NSE': 0.8951661809883981, 'KGE': 0.8968190264881305}
Métricas conjunto de 'validation' (modo prediccion) para genil-dilar: {'R2': 0.24856697729693666, 'MAE': 28.169192683917696, 'NSE': 0.24856697729693666, 'KGE': 0.35094503823660084}
Métricas en todo el conjunto de datos (modo prediccion) para genil-dilar: R2=0.487, MAE=30.155, NSE=0.487, KGE=0.446

Métricas conjunto de 'train' para indrawati-melamchi: {'R2': 0.9248574420598481, 'MAE': 9.112193985405703, 'NSE': 0.9248574420598481, 'KGE': 0.9305120457255507}
Métricas conjunto de 'test' para indrawati-melamchi: {'R2': 0.9179178682963512, 'MAE': 10.536969623654578, 'NSE': 0.9179178682963512, 'KGE': 0.913456707965113}
Métricas conjunto de 'validation' (modo prediccion) para indrawati-melamchi: {'R2': 0.5623453093850601, 'MAE': 29.66755398277173, 'NSE': 0.5623453093850601, 'KGE': 0.44246558002169234}
Métricas en todo el conjunto de datos (modo prediccion) para indrawati-melamchi: R2=0.632, MAE=26.357, NSE=0.632, KGE=0.631

Métricas conjunto de 'train' para mapocho-almendros: {'R2': 0.969472524226134, 'MAE': 16.77330507876308, 'NSE': 0.969472524226134, 'KGE': 0.9725103476181967}
Métricas conjunto de 'test' para mapocho-almendros: {'R2': 0.9484543725445741, 'MAE': 17.46058055761427, 'NSE': 0.9484543725445741, 'KGE': 0.940657736292907}
Métricas conjunto de 'validation' (modo prediccion) para mapocho-almendros: {'R2': 0.2191022075862199, 'MAE': 99.29130034802725, 'NSE': 0.2191022075862199, 'KGE': 0.24717099755789784}
Métricas en todo el conjunto de datos (modo prediccion) para mapocho-almendros: R2=0.522, MAE=82.000, NSE=0.522, KGE=0.582

Métricas conjunto de 'train' para nenskra-enguri: {'R2': 0.987605172656895, 'MAE': 15.8249146683379, 'NSE': 0.987605172656895, 'KGE': 0.9557532036540093}
Métricas conjunto de 'test' para nenskra-enguri: {'R2': 0.9891162712771787, 'MAE': 15.696199299727915, 'NSE': 0.9891162712771787, 'KGE': 0.9565282885706232}
Métricas conjunto de 'validation' (modo prediccion) para nenskra-enguri: {'R2': 0.8829395871633829, 'MAE': 60.786487872681164, 'NSE': 0.8829395871633829, 'KGE': 0.7380100213134291}
Métricas en todo el conjunto de datos (modo prediccion) para nenskra-enguri: R2=0.893, MAE=62.420, NSE=0.893, KGE=0.739

Métricas conjunto de 'train' para uncompahgre-ridgway: {'R2': 0.9773677587032614, 'MAE': 20.512550422433243, 'NSE': 0.9773677587032614, 'KGE': 0.9446418074765338}
Métricas conjunto de 'test' para uncompahgre-ridgway: {'R2': 0.9688159511634374, 'MAE': 21.556014278926497, 'NSE': 0.9688159511634374, 'KGE': 0.94049778280746}
Métricas conjunto de 'validation' (modo prediccion) para uncompahgre-ridgway: {'R2': 0.8594611102424274, 'MAE': 56.796209398082276, 'NSE': 0.8594611102424274, 'KGE': 0.725263908874157}
Métricas en todo el conjunto de datos (modo prediccion) para uncompahgre-ridgway: R2=0.816, MAE=68.140, NSE=0.816, KGE=0.681








# Modelo de Predicción del Área de Nieve en Cuencas Hidrográficas

## ❄️ Visión General del Proyecto

Este proyecto desarrolla un modelo de predicción del área de nieve utilizando una Red Neuronal Auto-Regresiva con Variables Exógenas (NARX), implementada con capas Long Short-Term Memory (LSTM), para seis cuencas hidrográficas distintas.

## 🎯 Objetivo del Modelo

El objetivo principal de este modelo es predecir el área de nieve en diferentes cuencas hidrográficas en escenarios futuros. Aprovechando una amplia serie temporal de datos históricos de área de nieve y diversas variables meteorológicas (exógenas) clave, el modelo busca aprender los patrones complejos y las relaciones temporales para generar predicciones precisas y robustas.

Las predicciones de área de nieve son cruciales para la gestión de recursos hídricos, la previsión de inundaciones, la planificación agrícola y la mitigación de riesgos en regiones dependientes del deshielo.

## 🧠 Tipo de Modelo: NARX (Nonlinear AutoRegressive with eXogenous inputs)

Nuestro modelo se basa en la arquitectura **NARX (Nonlinear AutoRegressive with eXogenous inputs)**, que se implementa mediante **Redes Neuronales Recurrentes (RNN)** con **capas LSTM (Long Short-Term Memory)**.

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



Métricas conjunto de 'train' para adda-bornio: {'R2': 0.9848937273869058, 'MAE': 11.422625820357998, 'NSE': 0.9848937273869058, 'KGE': 0.9624158664471593}
Métricas conjunto de 'test' para adda-bornio: {'R2': 0.9808589956700315, 'MAE': 16.239473269537907, 'NSE': 0.9808589956700315, 'KGE': 0.9319888470111187}
Métricas conjunto de 'validation' (modo prediccion) para adda-bornio: {'R2': 0.96176474349375, 'MAE': 27.495538347091667, 'NSE': 0.96176474349375, 'KGE': 0.8498673087573136}
Métricas en todo el conjunto de datos (modo prediccion) para adda-bornio: R2=0.975, MAE=18.631, NSE=0.975, KGE=0.918

Métricas conjunto de 'train' para genil-dilar: {'R2': 0.9372259290592876, 'MAE': 8.388898751050093, 'NSE': 0.9372259290592876, 'KGE': 0.9406776634295443}
Métricas conjunto de 'test' para genil-dilar: {'R2': 0.860898100779585, 'MAE': 9.80432326462139, 'NSE': 0.860898100779585, 'KGE': 0.9200870924496527}
Métricas conjunto de 'validation' (modo prediccion) para genil-dilar: {'R2': 0.7303075303975746, 'MAE': 11.821146186536254, 'NSE': 0.7303075303975746, 'KGE': 0.8532709931938088}
Métricas en todo el conjunto de datos (modo prediccion) para genil-dilar: R2=0.880, MAE=12.508, NSE=0.880, KGE=0.885

Métricas conjunto de 'train' para indrawati-melamchi: {'R2': 0.9283167874697322, 'MAE': 8.958353898516156, 'NSE': 0.9283167874697322, 'KGE': 0.9481170501100785}
Métricas conjunto de 'test' para indrawati-melamchi: {'R2': 0.9210991782576825, 'MAE': 11.0904753786496, 'NSE': 0.9210991782576825, 'KGE': 0.9412663442053899}
Métricas conjunto de 'validation' (modo prediccion) para indrawati-melamchi: {'R2': 0.8882975583638146, 'MAE': 13.332796983608887, 'NSE': 0.8882975583638146, 'KGE': 0.9056586564238376}
Métricas en todo el conjunto de datos (modo prediccion) para indrawati-melamchi: R2=0.911, MAE=11.681, NSE=0.911, KGE=0.916

Métricas conjunto de 'train' para mapocho-almendros: {'R2': 0.9727379819072993, 'MAE': 17.60963005973425, 'NSE': 0.9727379819072993, 'KGE': 0.9615460278682817}
Métricas conjunto de 'test' para mapocho-almendros: {'R2': 0.9123884970799073, 'MAE': 24.618680508312146, 'NSE': 0.9123884970799073, 'KGE': 0.9419058778028343}
Métricas conjunto de 'validation' (modo prediccion) para mapocho-almendros: {'R2': 0.6282104632977037, 'MAE': 51.83852656097493, 'NSE': 0.6282104632977037, 'KGE': 0.7822081792919674}
Métricas en todo el conjunto de datos (modo prediccion) para mapocho-almendros: R2=0.927, MAE=26.784, NSE=0.927, KGE=0.936

Métricas conjunto de 'train' para nenskra-enguri: {'R2': 0.9893086966386203, 'MAE': 13.608720571033052, 'NSE': 0.9893086966386203, 'KGE': 0.9789563247566387}
Métricas conjunto de 'test' para nenskra-enguri: {'R2': 0.9748549207615304, 'MAE': 26.48993988811659, 'NSE': 0.9748549207615304, 'KGE': 0.9137027447412311}
Métricas conjunto de 'validation' (modo prediccion) para nenskra-enguri: {'R2': 0.861020025044487, 'MAE': 67.64532442089283, 'NSE': 0.861020025044487, 'KGE': 0.757474385856627}
Métricas en todo el conjunto de datos (modo prediccion) para nenskra-enguri: R2=0.969, MAE=26.365, NSE=0.969, KGE=0.931

Métricas conjunto de 'train' para uncompahgre-ridgway: {'R2': 0.9833959855837182, 'MAE': 17.978890962681497, 'NSE': 0.9833959855837182, 'KGE': 0.9792152268951854}
Métricas conjunto de 'test' para uncompahgre-ridgway: {'R2': 0.9632518023648534, 'MAE': 28.907755022417238, 'NSE': 0.9632518023648534, 'KGE': 0.9209165460237245}
Métricas conjunto de 'validation' (modo prediccion) para uncompahgre-ridgway: {'R2': 0.9069058683397624, 'MAE': 57.11412298191705, 'NSE': 0.9069058683397624, 'KGE': 0.7919199726871856}
Métricas en todo el conjunto de datos (modo prediccion) para uncompahgre-ridgway: R2=0.967, MAE=28.815, NSE=0.967, KGE=0.933



NARX_MODELS3
n_lags = 3
nº capas = 2
nº neuronas = 10
epochs = 50
exog_cols = []

Métricas conjunto de 'train' para adda-bornio: {'R2': 0.983896928781121, 'MAE': 9.984483335148319, 'NSE': 0.9838286063487955, 'KGE': 0.9906455358066028}
Métricas conjunto de 'test' para adda-bornio: {'R2': 0.9868272719726697, 'MAE': 11.037597709314099, 'NSE': 0.9863796193093345, 'KGE': 0.9741758082931852}
Métricas conjunto de 'validation' (modo prediccion) para adda-bornio: {'R2': 0.9810848261327887, 'MAE': 21.710144965615772, 'NSE': 0.9735983402015059, 'KGE': 0.9149646889945594}

Métricas conjunto de 'train' para genil-dilar: {'R2': 0.9306792392506685, 'MAE': 7.9688175711235605, 'NSE': 0.9281303393939707, 'KGE': 0.9072669403619928}
Métricas conjunto de 'test' para genil-dilar: {'R2': 0.8886086828869324, 'MAE': 7.90095319449694, 'NSE': 0.885240370788977, 'KGE': 0.8717586951125884}
Métricas conjunto de 'validation' (modo prediccion) para genil-dilar: {'R2': 0.83808972411395, 'MAE': 9.856877204207503, 'NSE': 0.8211626925405389, 'KGE': 0.7689465648331342}

Métricas conjunto de 'train' para indrawati-melamchi: {'R2': 0.925226360314113, 'MAE': 9.752870546073991, 'NSE': 0.9234139626139061, 'KGE': 0.9107661483885414}
Métricas conjunto de 'test' para indrawati-melamchi: {'R2': 0.9139605707866065, 'MAE': 12.163987529341346, 'NSE': 0.9104019289065044, 'KGE': 0.8874103412660206}
Métricas conjunto de 'validation' (modo prediccion) para indrawati-melamchi: {'R2': 0.8951894759943265, 'MAE': 13.769391539611773, 'NSE': 0.8857620633136292, 'KGE': 0.8398893330119777}
Métricas conjunto de 'train' para mapocho-almendros: {'R2': 0.9687613169315186, 'MAE': 17.41535955671052, 'NSE': 0.9683272712543375, 'KGE': 0.9821808789695055}
Métricas conjunto de 'test' para mapocho-almendros: {'R2': 0.9496353883982969, 'MAE': 19.19984458163843, 'NSE': 0.9493967711816299, 'KGE': 0.9724970658853499}
Métricas conjunto de 'validation' (modo prediccion) para mapocho-almendros: {'R2': 0.9170475630707088, 'MAE': 27.834043270863862, 'NSE': 0.9155256777639986, 'KGE': 0.9466252256910129}

Métricas conjunto de 'train' para nenskra-enguri: {'R2': 0.988450519604748, 'MAE': 13.740912433579512, 'NSE': 0.9880487647629842, 'KGE': 0.974271577981626}
Métricas conjunto de 'test' para nenskra-enguri: {'R2': 0.9899208347332757, 'MAE': 16.311982585026936, 'NSE': 0.9887155287547571, 'KGE': 0.9599238315088925}
Métricas conjunto de 'validation' (modo prediccion) para nenskra-enguri: {'R2': 0.9810525031739796, 'MAE': 22.991866949298153, 'NSE': 0.9787100869255537, 'KGE': 0.9418894257840128}

Métricas conjunto de 'train' para uncompahgre-ridgway: {'R2': 0.9787107197622111, 'MAE': 21.260291045045623, 'NSE': 0.9770844158181049, 'KGE': 0.948112210895693}
Métricas conjunto de 'test' para uncompahgre-ridgway: {'R2': 0.9692931936091792, 'MAE': 23.595801304086564, 'NSE': 0.9679568244811299, 'KGE': 0.9457212314151618}
Métricas conjunto de 'validation' (modo prediccion) para uncompahgre-ridgway: {'R2': 0.9644473528129811, 'MAE': 27.324829916991494, 'NSE': 0.9632503364089781, 'KGE': 0.9444897178026497}


NARX_MODELS4
n_lags = 5
nº capas = 2
nº neuronas = 10
epochs = 50

Métricas conjunto de 'train' para adda-bornio: {'R2': 0.9820846827872115, 'MAE': 12.893192004901827, 'NSE': 0.9820846827872115, 'KGE': 0.9534747566935462}
Métricas conjunto de 'test' para adda-bornio: {'R2': 0.9812962456932167, 'MAE': 15.915939906803802, 'NSE': 0.9812962456932167, 'KGE': 0.931203661399069}
Métricas conjunto de 'validation' (modo prediccion) para adda-bornio: {'R2': 0.9386193620371159, 'MAE': 36.05853805520503, 'NSE': 0.9386193620371159, 'KGE': 0.813688754477179}
Métricas conjunto de 'train' para genil-dilar: {'R2': 0.9289521307357114, 'MAE': 8.490814755384001, 'NSE': 0.9289521307357114, 'KGE': 0.9392181653395031}
Métricas conjunto de 'test' para genil-dilar: {'R2': 0.885534233784013, 'MAE': 8.746894863961257, 'NSE': 0.885534233784013, 'KGE': 0.9309578399512246}
Métricas conjunto de 'validation' (modo prediccion) para genil-dilar: {'R2': 0.790829718304547, 'MAE': 10.810780294886086, 'NSE': 0.790829718304547, 'KGE': 0.8902251067242253}
Métricas conjunto de 'train' para indrawati-melamchi: {'R2': 0.9231350814661871, 'MAE': 9.5669904235215, 'NSE': 0.9231350814661871, 'KGE': 0.9313539695531252}
Métricas conjunto de 'test' para indrawati-melamchi: {'R2': 0.9197610049607043, 'MAE': 11.532130098411919, 'NSE': 0.9197610049607043, 'KGE': 0.8979314014991887}
Métricas conjunto de 'validation' (modo prediccion) para indrawati-melamchi: {'R2': 0.8891431582559741, 'MAE': 12.373115100692113, 'NSE': 0.8891431582559741, 'KGE': 0.8235581944747713}
Métricas conjunto de 'train' para mapocho-almendros: {'R2': 0.9671307300472386, 'MAE': 19.20412861460158, 'NSE': 0.9671307300472386, 'KGE': 0.9412410697886805}
Métricas conjunto de 'test' para mapocho-almendros: {'R2': 0.9353768960834933, 'MAE': 23.31079145471893, 'NSE': 0.9353768960834933, 'KGE': 0.9166084632285502}
Métricas conjunto de 'validation' (modo prediccion) para mapocho-almendros: {'R2': 0.8514949213688723, 'MAE': 40.98742988767195, 'NSE': 0.8514949213688723, 'KGE': 0.8884135561344264}
Métricas conjunto de 'train' para nenskra-enguri: {'R2': 0.987589113533836, 'MAE': 15.563624099154744, 'NSE': 0.987589113533836, 'KGE': 0.9609391196072835}
Métricas conjunto de 'test' para nenskra-enguri: {'R2': 0.9875359562974575, 'MAE': 17.828940113225045, 'NSE': 0.9875359562974575, 'KGE': 0.9590691195970918}
Métricas conjunto de 'validation' (modo prediccion) para nenskra-enguri: {'R2': 0.9721606625437359, 'MAE': 27.21677833724709, 'NSE': 0.9721606625437359, 'KGE': 0.9371025660431993}
Métricas conjunto de 'train' para uncompahgre-ridgway: {'R2': 0.9761929957776784, 'MAE': 24.0877531291725, 'NSE': 0.9761929957776784, 'KGE': 0.9380140938568112}
Métricas conjunto de 'test' para uncompahgre-ridgway: {'R2': 0.9642104807892068, 'MAE': 28.98467881019145, 'NSE': 0.9642104807892068, 'KGE': 0.9135946558675531}
Métricas conjunto de 'validation' (modo prediccion) para uncompahgre-ridgway: {'R2': 0.9422694801174687, 'MAE': 42.53896359503632, 'NSE': 0.9422694801174687, 'KGE': 0.8452213790866914}