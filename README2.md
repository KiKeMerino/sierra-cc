# Predicción de la Cobertura de Nieve con Modelos NARX-LSTM

## ❄️ Visión General del Proyecto

Este proyecto se enfoca en la predicción del área de nieve en seis cuencas hidrográficas específicas utilizando modelos **NARX (Nonlinear Autoregressive with Exogenous Inputs)** implementados con capas **LSTM (Long Short-Term Memory)**. El objetivo es generar predicciones futuras de la capa de nieve para apoyar la gestión hídrica y la prevención de riesgos.

## 🎯 Objetivo del Modelo

La finalidad de este código es utilizar modelos de redes neuronales LSTM, previamente entrenados para cada cuenca, para predecir el `area_nieve`. Las predicciones se basan en el historial del `area_nieve` (componente auto-regresivo) y en variables meteorológicas externas (componente exógeno).

## 🧠 Tipo de Modelo: NARX con LSTM

Nuestros modelos utilizan la arquitectura **NARX** por su capacidad para predecir series temporales combinando:
* **Auto-Regresión (AR):** Dependencia de valores pasados del `area_nieve`.
* **Variables Exógenas (X):** Influencia de otras variables como `temperatura`, `precipitacion`, y `dias_sin_precip`.
* **No Linealidad (N):** Implementación con **capas LSTM** para capturar relaciones complejas y dependencias a largo plazo en los datos secuenciales, evitando problemas como el gradiente desvanecido de RNNs básicas.

### Arquitectura Común del Modelo

Cada cuenca tiene un modelo NARX-LSTM independiente. La arquitectura base es:

* **Capa de Entrada:** Recibe secuencias de `n_lags_area` (ej. 3 días) de `area_nieve` y variables exógenas escaladas.
    * Formato: `input_shape = (n_lags_area, 1 + num_variables_exogenas)`
* **Capa LSTM:** Procesa las secuencias, aprendiendo patrones temporales.
    * Configurable con `n_units_lstm` (número de neuronas, ej. 10, 20, 50) y activación `relu`.
* **Capa Densa de Salida:** Una única neurona que produce la predicción del `area_nieve` para el siguiente paso (`t+1`).

---

## 🗃️ Estructura de los Datos

Los datos son obtenidos del satélite **MOD10A1F de la NASA (EarthData Search)** y consisten principalmente en:

* **`CGF_NDSI_Snow_Cover`:** (Variable de interés) Índice de Nieve de Diferencia Normalizada. Valores entre `40` y `100` indican **nieve**. Otros valores son datos nulos, nubes, agua, etc.
* **`Cloud_Persistence`:** Conteo de días consecutivos con cobertura de nubes.

Ambos son datos ráster procesados para obtener el `area_nieve` diaria por cuenca y luego combinados con series temporales de variables meteorológicas.

### Procesamiento Inicial de Datos

1.  **Obtención:** Datos MODIS descargados de EarthData Search, filtrados por fecha y área (`.shp` de cuenca), reproyectados a latitud/longitud.
2.  **Limpieza:** Lectura de archivos `.hdf`, cálculo de `area_nieve` diaria, lectura de series históricas de temperatura y precipitación.
3.  **Preparación:** Normalización y limpieza de series agregadas, separación en CSVs individuales por cuenca.
4.  **Ingeniería de Características:** Adición de la columna `dias_sin_precip` para registrar el tiempo desde la última lluvia/nieve.

---

## 🚀 Esquema de Predicción Iterativa

El modelo predice el `area_nieve` un día a la vez, utilizando sus propias predicciones anteriores como entrada para los pasos futuros (excepto las variables exógenas, que son datos reales).

Para predecir el `area_nieve` en el día `t+1`, el modelo utiliza:
* Las predicciones de `area_nieve` de los días `t, t-1, ..., t - n_lags_area`.
* Los valores reales de las variables exógenas para el día `t+1`.

Este proceso se repite para cada día futuro, propagando las predicciones del `area_nieve`.

---

## 🧪 Tratamiento de Outliers y Mejora de Métricas

Se identificaron **outliers significativos** en la variable `precipitacion` para las cuencas **Indrawati-Melamchi** y **Genil-Dilar**. Para mitigar su impacto en el entrenamiento del modelo y mejorar la estabilidad de las predicciones, se aplicó un método de **winsorización**:

* Los valores de `precipitacion` que excedían el **percentil 99** de su distribución para cada cuenca fueron limitados a ese valor del percentil 99.

Este tratamiento resultó en una **mejora notable en las métricas de rendimiento** para los modelos de estas cuencas, indicando una mayor robustez frente a eventos extremos.

---

## 📊 Métricas de Rendimiento por Modelo (Archivos `.h5`)

Cada modelo entrenado está guardado en un archivo `.h5` y su rendimiento se evalúa con las siguientes métricas en diferentes conjuntos de datos:

* **R2 (Coeficiente de Determinación):** Proporción de la varianza explicada.
* **MAE (Error Absoluto Medio):** Error promedio en las mismas unidades que el `area_nieve`.
* **NSE (Eficiencia de Nash-Sutcliffe):** Métrica hidrológica (1.0 = ajuste perfecto).
* **KGE (Eficiencia de Kling-Gupta):** Métrica hidrológica mejorada (1.0 = óptimo).

Las métricas se presentan para:
1.  **Entrenamiento:** Rendimiento sobre los datos usados para entrenar.
2.  **Prueba (Directa):** Generalización sobre datos no vistos, prediciendo un paso adelante.
3.  **Validación (Iterativa/Paso a Paso):** Robustez del modelo en predicciones a futuro, usando sus propias salidas como entradas.
4.  **Todo el Conjunto (Predictivo):** Visión global del rendimiento en modo predicción.

A continuación, se detallan las métricas para cada modelo:

**Adda-Bormio (modelo_adda_bormio.h5):**
* **Prueba (Directa):**
    * R2: [Valor_R2_Adda_Prueba]
    * MAE: [Valor_MAE_Adda_Prueba]
    * NSE: [Valor_NSE_Adda_Prueba]
    * KGE: [Valor_KGE_Adda_Prueba]
* **Validación (Iterativa):**
    * R2: [Valor_R2_Adda_Val]
    * MAE: [Valor_MAE_Adda_Val]
    * NSE: [Valor_NSE_Adda_Val]
    * KGE: [Valor_KGE_Adda_Val]
* **Todo el Conjunto (Predictivo):**
    * R2: [Valor_R2_Adda_Total]
    * MAE: [Valor_MAE_Adda_Total]
    * NSE: [Valor_NSE_Adda_Total]
    * KGE: [Valor_KGE_Adda_Total]

**Genil-Dilar (modelo_genil_dilar.h5):**
* **Prueba (Directa):**
    * R2: [Valor_R2_Genil_Prueba]
    * MAE: [Valor_MAE_Genil_Prueba]
    * NSE: [Valor_NSE_Genil_Prueba]
    * KGE: [Valor_KGE_Genil_Prueba]
* **Validación (Iterativa):**
    * R2: [Valor_R2_Genil_Val]
    * MAE: [Valor_MAE_Genil_Val]
    * NSE: [Valor_NSE_Genil_Val]
    * KGE: [Valor_KGE_Genil_Val]
* **Todo el Conjunto (Predictivo):**
    * R2: [Valor_R2_Genil_Total]
    * MAE: [Valor_MAE_Genil_Total]
    * NSE: [Valor_NSE_Genil_Total]
    * KGE: [Valor_KGE_Genil_Total]

**Indrawati-Melamchi (modelo_indrawati_melamchi.h5):**
* **Prueba (Directa):**
    * R2: [Valor_R2_Indrawati_Prueba]
    * MAE: [Valor_MAE_Indrawati_Prueba]
    * NSE: [Valor_NSE_Indrawati_Prueba]
    * KGE: [Valor_KGE_Indrawati_Prueba]
* **Validación (Iterativa):**
    * R2: [Valor_R2_Indrawati_Val]
    * MAE: [Valor_MAE_Indrawati_Val]
    * NSE: [Valor_NSE_Indrawati_Val]
    * KGE: [Valor_KGE_Indrawati_Val]
* **Todo el Conjunto (Predictivo):**
    * R2: [Valor_R2_Indrawati_Total]
    * MAE: [Valor_MAE_Indrawati_Total]
    * NSE: [Valor_NSE_Indrawati_Total]
    * KGE: [Valor_KGE_Indrawati_Total]

**Mapocho-Almendros (modelo_mapocho_almendros.h5):**
* **Prueba (Directa):**
    * R2: [Valor_R2_Mapocho_Prueba]
    * MAE: [Valor_MAE_Mapocho_Prueba]
    * NSE: [Valor_NSE_Mapocho_Prueba]
    * KGE: [Valor_KGE_Mapocho_Prueba]
* **Validación (Iterativa):**
    * R2: [Valor_R2_Mapocho_Val]
    * MAE: [Valor_MAE_Mapocho_Val]
    * NSE: [Valor_NSE_Mapocho_Val]
    * KGE: [Valor_KGE_Mapocho_Val]
* **Todo el Conjunto (Predictivo):**
    * R2: [Valor_R2_Mapocho_Total]
    * MAE: [Valor_MAE_Mapocho_Total]
    * NSE: [Valor_NSE_Mapocho_Total]
    * KGE: [Valor_KGE_Mapocho_Total]

**Nenskra-Enguri (modelo_nenskra_enguri.h5):**
* **Prueba (Directa):**
    * R2: [Valor_R2_Nenskra_Prueba]
    * MAE: [Valor_MAE_Nenskra_Prueba]
    * NSE: [Valor_NSE_Nenskra_Prueba]
    * KGE: [Valor_KGE_Nenskra_Prueba]
* **Validación (Iterativa):**
    * R2: [Valor_R2_Nenskra_Val]
    * MAE: [Valor_MAE_Nenskra_Val]
    * NSE: [Valor_NSE_Nenskra_Val]
    * KGE: [Valor_KGE_Nenskra_Val]
* **Todo el Conjunto (Predictivo):**
    * R2: [Valor_R2_Nenskra_Total]
    * MAE: [Valor_MAE_Nenskra_Total]
    * NSE: [Valor_NSE_Nenskra_Total]
    * KGE: [Valor_KGE_Nenskra_Total]

**Uncompahgre-Ridgway (modelo_uncompahgre_ridgway.h5):**
* **Prueba (Directa):**
    * R2: [Valor_R2_Uncompahgre_Prueba]
    * MAE: [Valor_MAE_Uncompahgre_Prueba]
    * NSE: [Valor_NSE_Uncompahgre_Prueba]
    * KGE: [Valor_KGE_Uncompahgre_Prueba]
* **Validación (Iterativa):**
    * R2: [Valor_R2_Uncompahgre_Val]
    * MAE: [Valor_MAE_Uncompahgre_Val]
    * NSE: [Valor_NSE_Uncompahgre_Val]
    * KGE: [Valor_KGE_Uncompahgre_Val]
* **Todo el Conjunto (Predictivo):**
    * R2: [Valor_R2_Uncompahgre_Total]
    * MAE: [Valor_MAE_Uncompahgre_Total]
    * NSE: [Valor_NSE_Uncompahgre_Total]
    * KGE: [Valor_KGE_Uncompahgre_Total]