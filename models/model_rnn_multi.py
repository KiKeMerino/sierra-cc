import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow import keras
from keras.layers import Input, Dense, SimpleRNN
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import os

def cargar_datos(ruta_datos, nombre_archivo):
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(os.path.join(ruta_datos, nombre_archivo))

def crear_lags(df, n_lags_area, n_lags_exog):
    """Crea las variables con lags incluyendo la columna 'cuenca'."""
    df_shifted = pd.DataFrame(df)
    cols, names = list(), list()
    for i in range(1, n_lags_area + 1):
        cols.append(df_shifted['area_nieve'].shift(i))
        names += [f'area_nieve(t-d{i})']
    exog_cols = [col for col in df.columns if col not in ['area_nieve', 'cuenca']]
    for col in exog_cols:
        for i in range(n_lags_exog + 1):
            cols.append(df_shifted[col].shift(i))
            names += [f'{col}(t-{i})']
    cols.append(df_shifted['cuenca'])
    names += ['cuenca(t-0)']
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg

def preprocesar_datos(df_lagged):
    """Escala las variables independientes y la dependiente, y aplica one-hot encoding a 'cuenca'."""
    y = df_lagged['area_nieve(t-d1)'].astype(float)
    X = df_lagged.drop(columns=['area_nieve(t-d1)'])
    cuenca_col = X['cuenca(t-0)'].values.reshape(-1, 1)
    exog_cols = [col for col in X.columns if col != 'cuenca(t-0)']
    X_exog = X[exog_cols].values.astype(float)
    scaler_x = MinMaxScaler().fit(X_exog)
    X_scaled_exog = scaler_x.transform(X_exog)
    encoder_cuenca = OneHotEncoder(sparse_output=False).fit(cuenca_col)
    X_scaled_cuenca = encoder_cuenca.transform(cuenca_col)
    X_scaled = np.concatenate((X_scaled_exog, X_scaled_cuenca), axis=1)
    scaler_y = MinMaxScaler().fit(y.values.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.values.reshape(-1, 1))
    return X_scaled, y_scaled, scaler_x, scaler_y, encoder_cuenca

def dividir_datos(X_scaled, y_scaled, test_size=0.3, shuffle=False, random_state=42):
    """Divide los datos en conjuntos de entrenamiento y prueba manteniendo el orden."""
    train_size = int(len(X_scaled) * (1 - test_size))
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
    return X_train, X_test, y_train, y_test

def definir_modelo(n_timesteps, n_features, unidades_rnn=64):
    """Define la arquitectura del modelo SimpleRNN."""
    model = keras.Sequential([
        Input(shape=(n_timesteps, n_features)),
        SimpleRNN(units=unidades_rnn, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def entrenar_modelo(model, X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=None, verbose=1):
    """Entrena el modelo."""
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=callbacks, verbose=verbose)

def guardar_modelo(model, ruta_guardado, nombre_archivo):
    """Guarda el modelo entrenado."""
    model.save(os.path.join(ruta_guardado, nombre_archivo))
    print(f"Modelo RNN multi-cuenca guardado en: {os.path.join(ruta_guardado, nombre_archivo)}")

def crear_y_guardar_rnn_multi(ruta_datos, ruta_guardado, nombre_archivo_datos,
                               n_lags_area=3, n_lags_exog=2, test_size=0.3, epochs=100,
                               batch_size=64, unidades_rnn=64):
    """Función principal para crear, entrenar y guardar el modelo RNN para todas las cuencas."""
    # 1. Cargar datos
    df = cargar_datos(ruta_datos, nombre_archivo_datos)

    # 2. Crear lags
    df_lagged = crear_lags(df.copy(), n_lags_area, n_lags_exog)

    # 3. Preprocesar datos
    X_scaled, y_scaled, scaler_x, scaler_y, encoder_cuenca = preprocesar_datos(df_lagged)

    # 4. Dividir datos
    X_train, X_test, y_train, y_test = dividir_datos(X_scaled, y_scaled, test_size=test_size)

    # 5. Reshape para RNN
    n_timesteps = 1
    n_features = X_train.shape[1]
    X_train_nn = X_train.reshape((X_train.shape[0], n_timesteps, n_features))
    X_test_nn = X_test.reshape((X_test.shape[0], n_timesteps, n_features))

    # 6. Definir modelo
    model_rnn = definir_modelo(n_timesteps, n_features, unidades_rnn)

    # 7. Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 8. Entrenar modelo
    history = entrenar_modelo(model_rnn, X_train_nn, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    # 9. Guardar modelo y scalers/encoder
    nombre_archivo_modelo = "simple_model_multicuenca.h5"
    guardar_modelo(model_rnn, ruta_guardado, nombre_archivo_modelo)
    # Puedes guardar los scalers y el encoder usando joblib si los necesitas para la evaluación
    from joblib import dump
    dump(scaler_x, os.path.join(ruta_guardado, 'scaler_x_rnn_multi.joblib'))
    dump(scaler_y, os.path.join(ruta_guardado, 'scaler_y_rnn_multi.joblib'))
    dump(encoder_cuenca, os.path.join(ruta_guardado, 'encoder_cuenca_rnn_multi.joblib'))

if __name__ == '__main__':
    RUTA_DATOS = './csv_merged/final/'
    RUTA_MODELOS = './models/'
    NOMBRE_ARCHIVO_DATOS = 'cuencas_all.csv'
    if not os.path.exists(RUTA_MODELOS):
        os.makedirs(RUTA_MODELOS)
    crear_y_guardar_rnn_multi(RUTA_DATOS, RUTA_MODELOS, NOMBRE_ARCHIVO_DATOS)