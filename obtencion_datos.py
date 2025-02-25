# Este modulo de la librería pyhdf proporciona funciones para trabajar con archivos hdf en formato SD (Scientific Data)
from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Descarga los datos MOD10A1 o MYD10A1 de la cubierta de nieve del sitio web de la NASA.
# Utiliza `h5py` para leer los datos HDF y extraer la información relevante.
# Define tu área de estudio y filtra los datos.
# Convierte los datos a un formato adecuado para tu modelo NARX (por ejemplo, un DataFrame de pandas).
# Normaliza los datos utilizando `MinMaxScaler` para mejorar el rendimiento del modelo.
# Divide los datos en conjuntos de entrenamiento, validación y prueba.

ruta_hdf = "data/MOD10A1F.A2025048.h18v04.061.2025050043304.hdf"

hdf_file = SD(ruta_hdf, SDC.READ)
datasets = hdf_file.datasets()

# print(f"Datasets: {datasets}")
# print(type(datasets),"\n", len(datasets))

# for key, value in datasets.items():
#     print(f"Key: {key} ({len(value)} elements), Value: {value}\n")
#     print(type(value))
#     for elemento in value:
#         print(type(elemento))


capa_nieve = hdf_file.select("MOD10A1_NDSI_Snow_Cover")
data = capa_nieve.get()


print(data)

df = pd.DataFrame(data)
print(df.head())

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