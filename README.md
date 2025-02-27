# Predicción de la cobertura de nieve con un modelo NARX

## Descripción

Este proyecto tiene como objetivo crear conciencia sobre cómo va a ir disminuyendo la capa de nieve en fechas futuras utilizando un modelo NARX (Non-linear Autoregressive with Exogenous Inputs).

Ésta es solo una parte del proyecto, yo me centraré en el uso de la inteligencia artificial para evaluar el impacto del cambio climático en la cobertura de nieve.

## Estructura de los datos

Los datos utilizados en este proyecto provienen de los siguientes datasets de la NASA:

* CGF_NDSI_Snow_Cover:
    - long_name: 'cloud-gap-filled NDSI snow cover',
    - valid_range: [0, 100],
    - FillValue: 255,
    - Key:
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

* Basic_QA
    - long_name: CGF snow cover general quality value,
    - valid_range: [0, 4],
    - FillValue: 255,
    - Key:
        - 0=best,
        - 1=good,
        - 2=ok,
        - 3=poor-not used,
        - 4=other-not used,
        - 211=night,
        - 239=ocean,
        - 255=unusable L1B data or no data

* Algorithm_Flags_QA
    - long_name: 'CGF algorithm bit flags',
    - format: 'bit flag',
    - Key: 'bit on means:
        - bit 0: inland water flag
        - bit 1: low visible screen failed, reversed snow detection\n    
        - bit 2: low NDSI screen failed, reversed snow detection\n
        - bit 3: combined temperature and height screen failed, snow reversed because too warm and too low
            This screen is also used to flag a high elevation too warm snow detection,
                in this case the snow detection is not changed but this bit is set.
        - bit 4: too high swir screen and applied at two thresholds: QA bit flag set if band6 TOA > 25% & band6 TOA <= 45%, indicative of unusual snow conditon, or snow commission error; now detection reversed if band6 TOA > 45%
        - bit 5: MOD35_L2 probably cloudy\n
        - bit 6: MOD35_L2 probably clear\n
        - bit 7: solar zenith screen, indicates increased uncertainty in results'

* MOD10A1_NDSI_Snow_Cover
    - long_name: 'NDSI snow cover from current day MOD10A1',
    - valid_range: [0, 100],
    - FillValue: 255,
    - Key:
        - 0-100 = NDSI snow
        - 200=missing data
        - 201=no decision
        - 211=night
        - 237=inland water
        - 239=ocean
        - 250=cloud
        - 254=detector saturated
        - 255=fill

Cada dataset es una matriz de 2400x2400 que contiene información sobre la cobertura de nieve y otras variables relevantes.
![Estructura de datos de cada dataset]("/img/estructura_datos.drawio.png")

## Proceso

El proceso de predicción de la cobertura de nieve se divide en los siguientes pasos:

1. Carga de datos
2. Preprocesamiento de datos
3. Creación del modelo NARX
4. Entrenamiento del modelo
5. Evaluación del modelo

## Resultados

Los resultados del proyecto se presentarán en un informe que incluirá:

* Métricas de evaluación del modelo
* Visualizaciones de la cobertura de nieve predicha
* Análisis de la importancia de las variables

## Herramientas utilizadas

* Python
* PyHDF
* Pandas
* Scikit-learn
* TensorFlow

## Próximos pasos

* Descargar datos adicionales (temperatura, precipitación, etc.)
* Combinar los datos en un solo DataFrame
* Implementar el modelo NARX
* Entrenar y evaluar el modelo
* Generar el informe final
