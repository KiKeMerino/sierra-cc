# Red Neuronal Recurrente Simple (RNN) - UNA CUENCA
#%%
import numpy as np
from tensorflow import keras
from keras.layers import Input, Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler # Para escalar los datos
import pandas as pd
from sklearn import metrics
import matplotlib as plt
from keras.utils import plot_model
import os
from sklearn.model_selection import train_test_split



def load_data(path, file):
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(os.path.join(path, file))

def create_lags(df, n_lags_area, n_lags_exog):
    """Crea las variables con lags."""
    df_shifted = pd.DataFrame(df)
    cols, names = list(), list()
    for i in range(1,n_lags_area + 1):
        cols.append(df_shifted['area_nieve'].shift(i))
        names += [f'area_nieve(t-d{i})']
    exog_cols = [col for col in df.columns if col != 'area_nieve']
    for col in exog_cols:
        for i in range(n_lags_exog + 1):
            cols.append(df_shifted[col].shift(i))
            names += [f'{col}(t-{i})']
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg

def preprocess_data(df_lagged):
    """Escala las variables independientes y la dependiente."""
    y = df_lagged['area_nieve(t-d1)'].astype(float)
    X = df_lagged.drop(columns=['area_nieve(t-d1)']).values.astype(float)
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    return X_scaled, y_scaled, scaler_x, scaler_y

def split_data(X_scaled, y_scaled, test_size=0.3, shuffle=False, random_state=42):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    return train_test_split(X_scaled, y_scaled, test_size=test_size, shuffle=shuffle, random_state=random_state)

def define_model(n_timesteps, n_features, units_rnn = 32):
    """Define la arquitectura del modelo SimpleRNN."""
    model = keras.Sequential([
        Input(shape=(n_timesteps, n_features)),
        SimpleRNN(units=units_rnn, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=None, verbose=1):
    """Entrena el modelo."""
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks, verbose=verbose)

def save_model(model, save_path, filename):
    """Guarda el modelo entrenado."""
    model.save(os.path.join(save_path, filename))
    print(f"Modelo guardado en: {os.path.join(save_path, filename)}")


def create_and_save_rnn_single(data_path, save_path, data_filename,
                               n_lags_area=3, n_lags_exog=2, test_size=0.3, epochs=100,
                               batch_size=32, units_rnn=32):
    """Función principal para crear, entrenar y guardar el modelo RNN para una sola cuenca."""
    # 1. Cargar datos
    df = load_data(data_path, data_filename)

    # 2. Crear lags
    df_lagged = create_lags(df.copy(), n_lags_area, n_lags_exog)

    # 3. Preprocesar datos
    X_scaled, y_scaled, scaler_x, scaler_y = preprocess_data(df_lagged)

    # 4. Dividir datos
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled, test_size=test_size)

    # 5. Reshape para RNN
    n_timesteps = 1
    n_features = X_train.shape[1]
    X_train_nn = X_train.reshape((X_train.shape[0], n_timesteps, n_features))
    X_test_nn = X_test.reshape((X_test.shape[0], n_timesteps, n_features))

    # 6. Definir modelo
    model_rnn = define_model(n_timesteps, n_features, units_rnn)

    # 7. Entrenar modelo
    history = train_model(model_rnn, X_train_nn, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

    # 8. Guardar modelo
    nombre_archivo_modelo = "simple_model.h5"
    save_model(model_rnn, save_path, nombre_archivo_modelo)
