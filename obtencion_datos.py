# Este modulo de la librería pyhdf proporciona funciones para trabajar con archivos hdf en formato SD (Scientific Data)
from pyhdf.SD import SD, SDC
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from pyhdf.SD import SD, SDC

ruta_hdf = "data/Adda-Bormio_basin-2024/MOD10A1F_61-20250226_083107/MOD10A1F.A2024001.h18v04.061.2024005143528.hdf"
hdf_file = SD(ruta_hdf, SDC.READ)


gcf_ndsi_snow_cover = hdf_file.select("CGF_NDSI_Snow_Cover")
cloud_persistance = hdf_file.select("Cloud_Persistence")
basic_qa = hdf_file.select("Basic_QA")
algorithm_flags_qa = hdf_file.select("Algorithm_Flags_QA")
mod10a1_ndsi_snow_cover = hdf_file.select("MOD10A1_NDSI_Snow_Cover")

# Info del dataset: Returns:
# 5-element tuple holding:
#   - dataset name
#   - dataset rank (number of dimensions)
#   - dataset shape, that is a list giving the length of each dataset dimension; if the first dimension is unlimited, then
#       the first value of the list gives the current length of the unlimited dimension
#   - data type (one of the SDC.xxx values)
#   - number of attributes defined for the dataset

print(gcf_ndsi_snow_cover.info())
print(cloud_persistance.info())
print(basic_qa.info())
print(algorithm_flags_qa.info())
print(mod10a1_ndsi_snow_cover.info())

# print(data)

# df = pd.DataFrame(data)
# print(df.head())

# -------------------------------------------------------------------------------------------#

# Crea un modelo NARX básico con una capa oculta utilizando `keras`.
# Define el número de neuronas en la capa oculta (puedes empezar con un número igual o ligeramente superior al número de variables de entrada).
# Elige una función de activación adecuada (por ejemplo, ReLU).
# Define el número de retrasos (lags) para las variables de entrada y salida (puedes empezar con un retraso de 1 para ambas).

# # Define las entradas del modelo (con retrasos)
# input_cubierta_nieve = Input(shape=(lags_cubierta_nieve,))
# input_exogenas = Input(shape=(lags_exogenas,))

# # Combina las entradas
# merged_input = keras.layers.concatenate([input_cubierta_nieve, input_exogenas])

# # Capa oculta
# hidden = Dense(units=num_neuronas, activation='relu')(merged_input)

# # Capa de salida
# output = Dense(units=1)(hidden)

# # Crea el modelo
# model = Model(inputs=[input_cubierta_nieve, input_exogenas], outputs=output)


# -------------------------------------------------------------------------------------------#

# Compila el modelo con un optimizador (por ejemplo, Adam), una función de pérdida (por ejemplo, MSE) y métricas de evaluación (por ejemplo, RMSE).
# Entrena el modelo con los datos de entrenamiento.
# Utiliza `EarlyStopping` para detener el entrenamiento si el rendimiento del modelo en el conjunto de validación no mejora.

# model.compile(optimizer='adam', loss='mse', metrics=['rmse'])

# early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# model.fit(
#     x=[x_train_cubierta_nieve, x_train_exogenas],
#     y=y_train,
#     validation_data=([x_val_cubierta_nieve, x_val_exogenas], y_val),
#     epochs=100,
#     callbacks=[early_stopping]
# )

# -------------------------------------------------------------------------------------------#

# Evalúa el rendimiento del modelo en el conjunto de prueba.
# Si el rendimiento no es satisfactorio, ajusta la arquitectura del modelo (número de capas, neuronas, retrasos) o los parámetros de entrenamiento (optimizador, función de pérdida, épocas).
# Repite el entrenamiento y la evaluación hasta que obtengas un rendimiento aceptable.