# Predicción de la cobertura de nieve con un modelo NARX

## Descripción

Este proyecto tiene como objetivo crear conciencia sobre cómo va a ir cambiando el nivel de la capa de nieve en fechas futuras utilizando un modelo NARX (Non-linear Autoregressive with Exogenous Inputs).

Ésta es solo una parte del proyecto, yo me centraré en el uso de la inteligencia artificial con redes neuronales autoregresivas para evaluar el impacto del cambio climático en la cobertura de nieve.

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

Cada dataset es un raster de datos diviendo la cuenca en píxeles con los valores arriba mencionados, yo consideraré  los valores entre 40 y 100 como nieve, y los demás como no nieve para simplificar el modelo

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
    * Se filtraron los datos por el rango de fechas deseado y se definió el área de interés correspondiente a la cuenca Adda-Bormio.

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
5. Separar estas series agregadas en 6 csv (uno por cada cuenca)


## 3. Preprocesamiento de los datos

En este apartado se juntarán todas las variables que nos intesan para nuestro modelo del apartado anterior y se analizará el dataset resultante, haremos lo siguiente:

1. EDA: exploracion de los datos
Lo primero será juntar los 2 csv creados anteriormente haciendo coincidir las fechas, el resultado será un dataframe de este estilo:
![Ejemplo genil-dilar](img/genil-dilar(head).png)

Este es el resultado de juntar los dos dataframes creados anteriormente, sin embargo añadiré una columna más para mejorar el modelo "dias_sin_precip" que contará los dias transcurridos desde la última precipitación
![Días sin precipitación](img/dias_sin_precip.png)

Ahora exploraremos cada variable para ver como se comporta:
# adda-bornio
![Estadisticas sobre Adda Bornio](img/addabornio-describe.png)

# genil-dilar
![Estadisticas sobre Genil Dilar](img/genildilar-describe.png)

# indrawati-melamchi
![Estadisticas sobre Indrawati Melamchi](img/indrawatimelamchi-describe.png)

# machopo-almendros
![Estadisticas sobre Machopo Almendros](img/machopoalmendros-describe.png)

# nenskra-enguri
![Estadisticas sobre Nenskra Enguri](img/nenskraenguri-describe.png)

# uncompahgre-ridgway
![Estadisticas sobre Uncompahgre Ridgway](img/uncompahgreridgway-describe.png)



2. Data cleaning: Limpiar datos en blanco, así como detectar outliers y errores lógicos de información
3. Visualización
4. Pre-procesing

