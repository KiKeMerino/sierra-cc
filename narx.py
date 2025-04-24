#%%
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler # Para escalar los datos
import pandas as pd

def create_lagged_data(data, n_lags_area, n_lags_exog):
    n_vars = data.shape[1]
    df_shifted = pd.DataFrame(data)
    cols, names = list(), list()

    # Lags de la variable objetivo
    for i in range(1, n_lags_area + 1):
        cols.append(df_shifted['area_nieve'].shift(i))
        names += [('area_nieve(t-%d)' % (i))]

    # Lags de las variables exógenas (incluyendo t-0)
    exog_cols = [col for col in data.columns if col != 'area_nieve']
    for col in exog_cols:
        for i in range(n_lags_exog + 1):
            cols.append(df_shifted[col].shift(i))
            names += [(f'{col}(t-{i})')]

    # Juntamos todo
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg
#%%

df = pd.read_csv("adda_norm.csv")
del df['fecha']

n_lags_area = 3
n_lags_exog = 2

df_lagged = create_lagged_data(df, n_lags_area, n_lags_exog)
df_lagged
#%%
# La variable dependiente es la 'area_nieve' en el tiempo t, alineada con los lags
y_lagged = df_lagged.index.map(df['area_nieve']).astype(float)
y_lagged

#%%
# Las variables independientes son todas las columnas de lags
X_lagged = df_lagged.values.astype(float)
X_lagged

#%%
# Escalar los datos
scaler_x = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X_lagged)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_lagged.values.reshape(-1, 1))

# Separar en conjuntos de entrenamiento y prueba (manteniendo el orden temporal)
train_size = int(len(X_scaled) * 0.7)
X_train_nn, X_test_nn = X_scaled[:train_size], X_scaled[train_size:]
y_train_nn, y_test_nn = y_scaled[:train_size], y_scaled[train_size:]

# Reshape para la entrada de la red neuronal
n_features = X_train_nn.shape[1]
X_train_nn = X_train_nn.reshape((X_train_nn.shape[0], 1, n_features))
X_test_nn = X_test_nn.reshape((X_test_nn.shape[0], 1, n_features))

y_train_nn