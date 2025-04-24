import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler # Para escalar los datos
import pandas as pd

def create_lagged_data(data, n_lags_area, n_lags_exog):
    """
    Crea secuencias de datos con lags para la variable objetivo ('area_nieve')
    y las variables exógenas.
    """
    n_vars = 1 if type(data) is pd.Series else data.shape[1]
    df_shifted = pd.DataFrame(data)
    cols, names = list(), list()

    # Lags de la variable objetivo
    for i in range(1, n_lags_area + 1):
        cols.append(df_shifted.shift(i))
        names += [('area_nieve(t-%d)' % (i))]

    # Lags de las variables exógenas
    exog_cols = [col for col in data.columns if col != 'area_nieve']
    for col in exog_cols:
        for i in range(n_lags_exog + 1):  # Incluimos el valor actual (t-0)
            cols.append(df_shifted[col].shift(i))
            names += [(f'{col}(t-{i})')]

    # Juntamos todo
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg


df = pd.read_csv("adda_norm.csv")
del df['fecha']

n_lags_area = 3  # Número de lags de 'area_nieve' a usar
n_lags_exog = 2  # Número de lags de las variables exógenas a usar

df_lagged = create_lagged_data(df, n_lags_area, n_lags_exog)

# Separar variables independientes y dependiente del dataframe laggeado
X_lagged = df_lagged.drop('area_nieve(t-0)', axis=1).astype(float)
y_lagged = df_lagged['area_nieve(t-0)'].astype(float)

# Escalar los datos
scaler_x = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X_lagged)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_lagged.values.reshape(-1, 1))

# Separar en conjuntos de entrenamiento y prueba (manteniendo el orden temporal)
train_size = int(len(X_scaled) * 0.7)
X_train_nn, X_test_nn = X_scaled[:train_size], X_scaled[train_size:]
y_train_nn, y_test_nn = y_scaled[:train_size], y_scaled[train_size:]

# Reshape para la entrada de la red neuronal (samples, timesteps, features)
# En este caso, cada muestra tiene un solo "timestep" que contiene todos los lags
X_train_nn = X_train_nn.reshape((X_train_nn.shape[0], 1, X_train_nn.shape[1]))
X_test_nn = X_test_nn.reshape((X_test_nn.shape[0], 1, X_test_nn.shape[1]))