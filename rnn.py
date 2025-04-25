# SIMPLE RECURRENT NEURAL NETWORK
#%%
import numpy as np
from tensorflow import keras
from keras.layers import Input, Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler # Para escalar los datos
import pandas as pd
from sklearn import metrics


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

#%%
# Defino la arquitectura de la red neuronal
# En este caso una Red Neuronal Recurrente Simple (RNN)
n_timesteps = X_train_nn.shape[1]
n_features = X_train_nn.shape[2]

model_rnn = keras.Sequential([
    Input(shape=(n_timesteps, n_features)),
    SimpleRNN(units=32, activation='relu'), # 32 unidades en la capa RNN
    Dense(units=1) # Capa de salida con una única neurona para la predicción
])

#%%
# Compilo el modelo
model_rnn.compile(optimizer='adam', # optimizador comun y eficiente
                loss='mean_squared_error', # Funcion de perdida adecuada para regresión
                metrics=['mae']) # Error absoluto medio com ométrica adicional

# Entreno el modelo
history = model_rnn.fit(X_train_nn, y_train_nn,
                        epochs=100, # Numero de veces que el modelo recorre el conjunto de entrenamiento
                        batch_size=32, # Numero de muestras por lote durante el entrenamiento
                        validation_split=0.2, # Fracción de los datos de entrenamiento a usar como conjunto de validación
                        verbose=1)

# Evaluo el modelo
loss, mae = model_rnn.evaluate(X_test_nn, y_test_nn)
# print(f'Perdida (MSE) en el conjunto de prueba: {loss:.4f}')
# print(f'Error medio absoluto (MAE) en el conjunto de prueba: {mae:.4f}')

# Realizo predicciones
y_pred_scaled = model_rnn.predict(X_test_nn)
y_pred_nn = scaler_y.inverse_transform(y_pred_scaled)

# También necesitas invertir la escala de los valores reales del conjunto de prueba
y_test_nn_original = scaler_y.inverse_transform(y_test_nn)

# Comparo y_pred_nn con y_test_nn_original
df_resultados_nn = pd.DataFrame({'Actual': y_test_nn_original.flatten(), 'Predicted': y_pred_nn.flatten()})
print("\nResultados de la Red Neuronal:")
print(df_resultados_nn.head())

from sklearn import metrics
print("R2:", metrics.r2_score(y_test_nn_original, y_pred_nn))
print ("MAE:", metrics.mean_absolute_error(y_test_nn_original, y_pred_nn))