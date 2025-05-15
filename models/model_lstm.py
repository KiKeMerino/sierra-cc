import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from joblib import load
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr

def cargar_datos(ruta_datos, nombre_archivo):
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(os.path.join(ruta_datos, nombre_archivo), index_col=0)

def crear_lags(df, n_lags_area):
    """Crea los lags pasados del área de nieve y mantiene las variables exógenas y la cuenca."""
    df_shifted = df.copy()
    for i in range(1, n_lags_area + 1):
        df_shifted[f'area_nieve(t-{i})'] = df_shifted['area_nieve'].shift(i)
    df_shifted.dropna(inplace=True)
    print(f"Columnas de lags creadas: {df_shifted.columns}")
    return df_shifted

def preprocesar_datos(df_lagged):
    """Escala y codifica las variables para la entrada de la LSTM."""
    df_enconded = pd.get_dummies(df_lagged, columns=['cuenca'], prefix='cuenca', drop_first=False)
    X = df_enconded.drop(columns=['area_nieve'])
    y = df_enconded['area_nieve'].astype(float)
    exog_cols = [col for col in df_enconded.columns if col != 'area_nieve']
    scaler_X = MinMaxScaler().fit(X[exog_cols])
    X_scaled = scaler_X.transform(X[exog_cols])
    scaler_y = MinMaxScaler().fit(y.values.reshape(-1, 1))
    y_scaled = scaler_y.transform(y.values.reshape(-1, 1))
    return X_scaled, y_scaled, scaler_X, scaler_y, list(X.columns)

def dividir_datos_extendido(X_scaled, y_scaled, test_size=0.2, validation_iter_ratio=0.1, shuffle=False, random_state=42):
    """Divide los datos en conjuntos de entrenamiento, prueba y validación iterativa manteniendo el orden."""
    total_size = len(X_scaled)
    test_size_abs = int(total_size * test_size)
    val_iter_size_abs = int(test_size_abs * validation_iter_ratio)
    train_end_index = total_size - test_size_abs
    val_iter_start_index = total_size - val_iter_size_abs
    X_train = X_scaled[:train_end_index]
    y_train = y_scaled[:train_end_index]
    X_test = X_scaled[train_end_index:val_iter_start_index]
    y_test = y_scaled[train_end_index:val_iter_start_index]
    X_val_iter = X_scaled[val_iter_start_index:]
    y_val_iter = y_scaled[val_iter_start_index:]
    return X_train, X_test, y_train, y_test, X_val_iter, y_val_iter

def calcular_nse(y_true, y_pred):
    """Calcula el Coeficiente de Nash-Sutcliffe."""
    numerator = np.sum((y_pred - y_true)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (numerator/ denominator) if denominator != 0 else np.nan

def calcular_kge(y_true, y_pred):
    """Calcula el Coeficiente de Eficiencia de Kling-Gupta."""
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) != 0 else np.nan
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def evaluar_modelo(ruta_datos, ruta_modelos, nombre_archivo_datos, n_lags_area=7, test_size=0.2, validation_iter_ratio=0.1):
    """Evalúa un modelo LSTM previamente entrenado."""
    # 1. Cargar datos
    df = cargar_datos(ruta_datos, nombre_archivo_datos)

    # 2. Crear lags
    df_lagged = crear_lags(df.copy(), n_lags_area)

    # 3. Preprocesar datos
    X_scaled, y_scaled, scaler_X, scaler_y, feature_columns = preprocesar_datos(df_lagged)

    # 4. Dividir datos
    X_train, X_test, y_train, y_test, X_val_iter, y_val_iter = dividir_datos_extendido(X_scaled, y_scaled, test_size=test_size, validation_iter_ratio=validation_iter_ratio)

    # 5. Cargar modelo y scalers
    nombre_archivo_modelo = "lstm_model_multicuenca_fila.h5"
    model_lstm = keras.models.load_model(os.path.join(ruta_modelos, nombre_archivo_modelo))
    scaler_X_loaded = load(os.path.join(ruta_modelos, 'scaler_x_exog_lstm_multi_fila.joblib'))
    scaler_y_loaded = load(os.path.join(ruta_modelos, 'scaler_y_lstm_multi_fila.joblib'))

    n_features = X_train.shape[1]

    # 6. Evaluar en el conjunto de prueba (predicción directa)
    y_pred_scaled = model_lstm.predict(X_test.reshape(-1, 1, n_features))
    y_pred = scaler_y_loaded.inverse_transform(y_pred_scaled)
    y_true_test = scaler_y_loaded.inverse_transform(y_test.reshape(-1, 1))
    r2_test = r2_score(y_true_test, y_pred)
    mae_test = mean_absolute_error(y_true_test, y_pred)
    nse_test = calcular_nse(y_true_test, y_pred)
    kge_test = calcular_kge(y_true_test, y_pred)
    print(f"Métricas en el conjunto de prueba (predicción directa):")
    print(f"R2: {r2_test:.4f}, MAE: {mae_test:.4f}, NSE: {nse_test:.4f}, KGE: {kge_test:.4f}")

    # 7. Realizar predicción iterativa en el conjunto de validación
    n_val_iter = len(y_val_iter)
    if n_val_iter > n_lags_area:
        predicciones_iterativas_scaled = []
        historial_area_nieve_escalado = y_scaled[:-n_lags_area].flatten().tolist()

        for i in range(n_val_iter - n_lags_area):
            lags_area_nieve = np.array(historial_area_nieve_escalado[-n_lags_area:]).reshape(n_lags_area)
            exog_actual = X_val_iter[n_lags_area + i, :]
            entrada_prediccion_array = exog_actual.copy()
            for j in range(n_lags_area):
                if f'area_nieve(t-{n_lags_area - j})' in feature_columns:
                    idx_lag = feature_columns.index(f'area_nieve(t-{n_lags_area - j})')
                    entrada_prediccion_array[idx_lag] = lags_area_nieve[j]
            entrada_prediccion = entrada_prediccion_array.reshape(1, 1, n_features)

            prediccion_escalada = model_lstm.predict(entrada_prediccion, verbose=0)[0, 0]
            predicciones_iterativas_scaled.append(prediccion_escalada)
            historial_area_nieve_escalado.append(prediccion_escalada)

        y_pred_iter = scaler_y_loaded.inverse_transform(np.array(predicciones_iterativas_scaled).reshape(-1, 1))
        y_true_val_iter = scaler_y_loaded.inverse_transform(y_val_iter[n_lags_area:].reshape(-1, 1))

        if len(y_pred_iter) == len(y_true_val_iter):
            r2_val_iter = r2_score(y_true_val_iter, y_pred_iter)
            mae_val_iter = mean_absolute_error(y_true_val_iter, y_pred_iter)
            nse_val_iter = calcular_nse(y_true_val_iter, y_pred_iter)
            kge_val_iter = calcular_kge(y_true_val_iter, y_pred_iter)
            print(f"\nMétricas en el conjunto de validación iterativa ({len(y_true_val_iter)} puntos):")
            print(f"R2 (iterativo): {r2_val_iter:.4f}, MAE: {mae_val_iter:.4f}, NSE: {nse_val_iter:.4f}, KGE: {kge_val_iter:.4f}")
        else:
            print(f"Advertencia: La longitud de las predicciones iterativas ({len(y_pred_iter)}) no coincide con la longitud de los valores reales del conjunto de validación ({len(y_true_val_iter)}).")
    else:
        print("\nEl tamaño del conjunto de validación iterativa es menor o igual al número de lags.")

# Definir rutas y nombre de archivo
RUTA_DATOS = './'
RUTA_MODELOS = './models/'
NOMBRE_ARCHIVO_DATOS = 'df_all.csv'
N_LAGS_AREA = 7
TEST_SIZE = 0.2
VALIDATION_ITER_RATIO = 0.1

# Para ejecutar solo la evaluación (asumiendo que el modelo y los scalers ya están guardados):
evaluar_modelo(RUTA_DATOS, RUTA_MODELOS, NOMBRE_ARCHIVO_DATOS, N_LAGS_AREA, TEST_SIZE, VALIDATION_ITER_RATIO)