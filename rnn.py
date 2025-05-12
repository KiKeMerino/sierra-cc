# SIMPLE RECURRENT NEURAL NETWORK
#%%
import numpy as np
from tensorflow import keras
from keras.layers import Input, Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler # Para escalar los datos
import pandas as pd
from sklearn import metrics

#%%
def create_lagged_data(data, n_lags_area, n_lags_exog):
    n_vars = data.shape[1]
    df_shifted = pd.DataFrame(data)
    cols, names = list(), list()

    # Lags de la variable objetivo
    for i in range(1, n_lags_area + 1):
        cols.append(df_shifted['area_nieve'].shift(i))
        names += [f'area_nieve(t-{i}']

    # Lags de las variables exógenas (incluyendo t-0)
    exog_cols = [col for col in data.columns if col != 'area_nieve']
    for col in exog_cols:
        for i in range(n_lags_exog + 1):
            cols.append(df_shifted[col].shift(i))
            names += [f'{col}(t-{i})']

    # Juntamos todo
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg

def predict_future(model, initial_sequence, future_exog_values, scaler_x, scaler_y, n_lags_area, n_lags_exog, exog_cols):
    """
    Realiza predicciones futuras utilizando un modelo RNN autoregresivo con variables exógenas
    cuyos valores futuros son conocidos.

    Args:
        model (keras.Model): El modelo RNN entrenado.
        initial_sequence (np.array): La secuencia inicial de datos reales (escalados) con la forma (1, 1, n_features) que contiene los lags necesarios
            para la primera predicción. Debe incluir los lags de 'area_nieve' y los lags de las variables exógenas hasta el tiempo actual.
        future_exog_values (np.array): Un array con los valores futuros (escalados) de las variables exógenas para los `n_future_steps`. Debe tener
            la forma (n_future_steps, len(exog_cols)).
        scaler_x (MinMaxScaler): El escalador utilizado para las variables de entrada (X).
        scaler_y (MinMaxScaler): El escalador utilizado para la variable objetivo ('area_nieve').
        n_lags_area (int): El número de lags de 'area_nieve' utilizados en el entrenamiento.
        n_lags_exog (int): El número de lags de las variables exógenas utilizados en el entrenamiento (incluyendo t-0).
        exog_cols (list): Una lista con los nombres de las columnas de las variables exógenas en el DataFrame original.

    Returns:
        np.array: Un array con las predicciones desescaladas para los pasos futuros.

    Raises:
        ValueError: Si la longitud de `future_exog_values` no coincide con `n_future_steps`.
    """
    n_future_steps = future_exog_values.shape[0]
    future_predictions = []
    current_sequence = initial_sequence.copy()
    n_total_lags = n_lags_area + (len(exog_cols) * (n_lags_exog + 1))

    for i in range(n_future_steps):
        # 1. Realizar la predicción (escalada)
        predicted_scaled = model.predict(current_sequence)[0,0]
        future_predictions.append(predicted_scaled)

        # 2. Crear la nueva secuencia de entrada para la siguiente predicción
        #    - El valor predicho de 'area_nieve' se convierte en el nuevo lag más reciente.
        #    - Utilizar el valor futuro conocido de las variables exógenas para este paso.

        # Extraer los lags de 'area_nieve' de la secuencia actual
        current_area_lags = current_sequence[0,0, :n_lags_area].tolist()

        # Crear la nueva secuencia de lags de 'area_nieve'
        new_area_lags = [predicted_scaled] + current_area_lags[:-1]

        # Obtener los valores de las variables exógenas para el tiempo t+i
        future_exog_step = future_exog_values[i]

        # Crear la parte de las variables exógenas para la nueva secuencia
        new_exog_part = []
        for j in range(len(exog_cols)):
            # El valor actual (t+i) es el lag 0
            lags_exog = [future_exog_step[j]]
            # Los lags anteriores (hasta t+i - n_lags_exog) se toman de la secuencia actual
            # Necesitamos tener cuidado con los índices aquí.
            num_prev_lags = n_lags_exog
            start_index_exog = n_lags_area + j * (n_lags_exog + 1)
            end_index_exog = start_index_exog + num_prev_lags
            previous_exog_lags = current_sequence[0, 0, start_index_exog:end_index_exog].tolist()
            lags_exog.extend(previous_exog_lags)
            new_exog_part.extend(lags_exog)

        # Combinar los lags de 'area_nieve' y las variables exógenas
        new_sequence_list = new_area_lags + new_exog_part[-len(exog_cols) * (n_lags_exog + 1):]

        # Asegurarse de que la longitud sea correcta y reshape para la entrada del modelo
        current_sequence = np.array(new_sequence_list).reshape((1, 1, n_total_lags))

    # 3. Desescalar las predicciones
    future_predictions_descaled = scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    return future_predictions_descaled

#%%
df = pd.read_csv('csv_merged/adda-bornio_merged.csv')
df

#%%
# Definir los lags y las variables exógenas (deben coincidir con tu entrenamiento)
n_lags_area = 3
n_lags_exog = 2
exog_cols = [col for col in df.columns if col != 'area_nieve']
exog_cols

#%%
# Crear los datos lagged (igual que en tu script de entrenamiento)
df_lagged = create_lagged_data(df, n_lags_area, n_lags_exog)
y_lagged = df_lagged.index.map(df['area_nieve']).astype(float)
X_lagged = df_lagged.values.astype(float)

# Escalar los datos (igual que en tu script de entrenamiento)
scaler_x = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X_lagged)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_lagged.values.reshape(-1, 1))
#%%
y_scaled.shape
#%%
# --- Simular el modelo entrenado (reemplaza con la carga de tu modelo real) ---
n_features = X_scaled.shape[1]
model_rnn_sim = keras.Sequential([
    keras.layers.Input(shape=(1, n_features)),
    keras.layers.SimpleRNN(units=32, activation='relu'),
    keras.layers.Dense(units=1)
])
model_rnn_sim.compile(optimizer='adam', loss='mse')

 # --- Reshape los datos de entrenamiento para que tengan la forma (muestras, pasos_de_tiempo, características) ---
X_train_reshaped = X_scaled[:-10].reshape((-1, 1, n_features))
y_train_reshaped = y_scaled[:-10] # No es necesario reshape para y en este caso

model_rnn_sim.fit(X_train_reshaped, y_train_reshaped, epochs=10, verbose=1) # Entrenar brevemente para el ejemplo

# --- Preparar la secuencia inicial para la predicción ---
last_sequence_scaled = X_scaled[-1 - (n_lags_area + (len(exog_cols) * (n_lags_exog + 1))):-1]
if len(last_sequence_scaled) < (n_lags_area + (len(exog_cols) * (n_lags_exog + 1))):
    raise ValueError("Not enough data to create the initial sequence for prediction.")

if X_scaled.shape[0] > 0:
    last_row_scaled = X_scaled[-1]
    if len(last_row_scaled) == n_features:
        initial_sequence = last_row_scaled.reshape((1, 1, n_features))
    else:
        raise ValueError(f"La última fila de X_scaled tiene una longitud incorrecta ({len(last_row_scaled)}), se esperaba {n_features}.")
else:
    raise ValueError("X_scaled está vacío, no hay datos para la secuencia inicial.")

# --- Simular valores futuros CONOCIDOS de las variables exógenas (ESCALADOS) ---
# En un escenario real, tendrías estos valores de alguna fuente.
n_future_steps = 10
future_exog_scaled = np.random.rand(n_future_steps, len(exog_cols)) # Ejemplo aleatorio

# --- Realizar la predicción futura con valores exógenos conocidos ---
future_predictions = predict_future(model_rnn_sim, initial_sequence, future_exog_scaled, scaler_x, scaler_y, n_lags_area, n_lags_exog, exog_cols)

print("\nPredicciones futuras de 'area_nieve' (con valores exógenos conocidos):")
print(future_predictions)










def nash_sutcliffe_efficiency(observaciones, simulaciones):
    """Calcula el Nash-Sutcliffe Efficiency (NSE)."""
    numerador = np.sum((observaciones - simulaciones)**2)
    denominador = np.sum((observaciones - np.mean(observaciones))**2)
    nse = 1 - (numerador / denominador)
    return nse

def kling_gupta_efficiency(observaciones, simulaciones):
    """Calcula el Kling-Gupta Efficiency (KGE)."""
    # Correlación de Pearson
    r = np.corrcoef(observaciones, simulaciones)[0, 1]
    # Desviación estándar
    sigma_sim = np.std(simulaciones)
    sigma_obs = np.std(observaciones)
    # Media
    mu_sim = np.mean(simulaciones)
    mu_obs = np.mean(observaciones)

    # Componentes de KGE
    r_component = r
    beta_component = mu_sim / mu_obs
    gamma_component = sigma_sim / sigma_obs

    # KGE formula
    kge = 1 - np.sqrt((r_component - 1)**2 + (beta_component - 1)**2 + (gamma_component - 1)**2)
    return kge


# nse = nash_sutcliffe_efficiency(y_test_nn_original.flatten(), y_pred_nn.flatten())
# kge = kling_gupta_efficiency(y_test_nn_original.flatten(), y_pred_nn.flatten())

# print(f"R2: {r2:.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"NSE: {nse:.4f}")
# print(f"KGE: {kge:.4f}")

# df_resultados_nn.hist()