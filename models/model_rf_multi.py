import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from joblib import dump
import os
import numpy as np

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
    """Aplica one-hot encoding a 'cuenca' y escala las demás variables."""
    y = df_lagged['area_nieve(t-d1)'].astype(float)
    X = df_lagged.drop(columns=['area_nieve(t-d1)'])
    cuenca_col = X['cuenca(t-0)'].values.reshape(-1, 1)
    exog_cols = [col for col in X.columns if col != 'cuenca(t-0)']
    X_exog = X[exog_cols].values.astype(float)
    scaler_x = MinMaxScaler().fit(X_exog)
    X_scaled_exog = scaler_x.transform(X_exog)
    encoder_cuenca = OneHotEncoder(sparse_output=False).fit(cuenca_col)
    X_scaled_cuenca = encoder_cuenca.transform(cuenca_col)
    X_processed = np.concatenate((X_scaled_exog, X_scaled_cuenca), axis=1)
    return X_processed, y, scaler_x, encoder_cuenca

def dividir_datos(X_processed, y, test_size=0.3, shuffle=False, random_state=42):
    """Divide los datos en conjuntos de entrenamiento y prueba manteniendo el orden."""
    train_size = int(len(X_processed) * (1 - test_size))
    X_train, X_test = X_processed[:train_size], X_processed[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

def definir_modelo(n_estimators=100, random_state=42, n_jobs=-1):
    """Define el modelo Random Forest."""
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)

def entrenar_modelo(model, X_train, y_train):
    """Entrena el modelo Random Forest."""
    model.fit(X_train, y_train)
    return model

def guardar_modelo(model, ruta_guardado, nombre_archivo):
    """Guarda el modelo entrenado."""
    dump(model, os.path.join(ruta_guardado, nombre_archivo))
    print(f"Modelo Random Forest multi-cuenca guardado en: {os.path.join(ruta_guardado, nombre_archivo)}")

def crear_y_guardar_rf_multi(ruta_datos, ruta_guardado, nombre_archivo_datos,
                            n_lags_area=3, n_lags_exog=2, test_size=0.3,
                            n_estimators=100, random_state=42):
    """Función principal para crear, entrenar y guardar el modelo Random Forest para todas las cuencas."""
    # 1. Cargar datos
    df = cargar_datos(ruta_datos, nombre_archivo_datos)

    # 2. Crear lags
    df_lagged = crear_lags(df.copy(), n_lags_area, n_lags_exog)

    # 3. Preprocesar datos
    X_processed, y, scaler_x, encoder_cuenca = preprocesar_datos(df_lagged)

    # 4. Dividir datos
    X_train, X_test, y_train, y_test = dividir_datos(X_processed, y, test_size=test_size)

    # 5. Definir modelo
    model_rf = definir_modelo(n_estimators, random_state)

    # 6. Entrenar modelo
    model_rf = entrenar_modelo(model_rf, X_train, y_train)

    # 7. Guardar modelo y preprocesadores
    nombre_archivo_modelo = "random_forest_model.joblib"
    guardar_modelo(model_rf, ruta_guardado, nombre_archivo_modelo)
    dump(scaler_x, os.path.join(ruta_guardado, 'scaler_x_rf_multi.joblib'))
    dump(encoder_cuenca, os.path.join(ruta_guardado, 'encoder_cuenca_rf_multi.joblib'))

if __name__ == '__main__':
    RUTA_DATOS = './csv_merged/final/'
    RUTA_MODELOS = '../models/'
    NOMBRE_ARCHIVO_DATOS = 'cuencas_all.csv'
    if not os.path.exists(RUTA_MODELOS):
        os.makedirs(RUTA_MODELOS)
    crear_y_guardar_rf_multi(RUTA_DATOS, RUTA_MODELOS, NOMBRE_ARCHIVO_DATOS)