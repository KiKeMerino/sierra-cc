import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

def cargar_datos(archivo_datos):
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(archivo_datos, index_col=0)

def crear_lags(df, n_lags_area):
    """Crea los lags pasados del área de nieve y mantiene las variables exógenas y la cuenca."""
    df_shifted = df.copy()
    for i in range(1, n_lags_area + 1):
        df_shifted[f'area_nieve(t-{i})'] = df_shifted['area_nieve'].shift(i)
    df_shifted.dropna(inplace=True)
    print(f"Columnas de lags creadas: {df_shifted.columns}")
    return df_shifted

def preprocesar_datos_rf(df_lagged):
    """Prepara los datos para el modelo Random Forest."""
    df_encoded = pd.get_dummies(df_lagged, columns=['cuenca'], prefix='cuenca', drop_first=False)
    X = df_encoded.drop(columns=['area_nieve'])
    y = df_encoded['area_nieve'].astype(float)
    return X, y

def dividir_datos_rf(X, y, test_size=0.2, random_state=42):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False, random_state=random_state)
    return X_train, X_test, y_train, y_test

def entrenar_modelo_rf(X_train, y_train, n_estimators=100, random_state=42):
    """Entrena un modelo Random Forest Regressor."""
    model_rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model_rf.fit(X_train, y_train)
    return model_rf

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

def cargar_guardar_modelo(archivo_datos, n_lags_area, test_size, validation_size):
    # 1. Cargar datos
    df = cargar_datos(archivo_datos)

    # 2. Crear lags
    df_lagged = crear_lags(df.copy(), n_lags_area)

    # 3. Preprocesar datos
    X, y = preprocesar_datos_rf(df_lagged)

    # 4. Dividir datos para entrenamiento
    X_train, X_test, y_train, y_test = dividir_datos_rf(X, y, test_size=TEST_SIZE, random_state=42)

    # 5. Entrenar el modelo Random Forest
    model_rf = entrenar_modelo_rf(X_train, y_train, n_estimators=100, random_state=42)

    # 6. Guardar el modelo entrenado
    os.makedirs(RUTA_MODELOS, exist_ok=True)
    nombre_archivo_modelo = "rf_model_multicuenca.joblib"
    dump(model_rf, os.path.join(RUTA_MODELOS, nombre_archivo_modelo))
    print(f"\nModelo Random Forest guardado en: {os.path.join(RUTA_MODELOS, nombre_archivo_modelo)}")

def evaluar_modelo_rf(ruta_datos, ruta_modelos, nombre_archivo_datos, n_lags_area=3, test_size=0.2, validation_iter_ratio=0.1):
    """Evalúa un modelo Random Forest previamente entrenado con predicción iterativa total."""
    # 1. Cargar datos
    df = cargar_datos(os.path.join(ruta_datos, nombre_archivo_datos))

    # 2. Crear lags
    df_lagged = crear_lags(df.copy(), n_lags_area)

    # 3. Preprocesar datos
    X, y = preprocesar_datos_rf(df_lagged)
    feature_columns = X.columns

    # 4. Dividir datos (necesario para la evaluación tradicional)
    X_train_eval, X_test_eval, y_train_eval, y_test_eval = dividir_datos_rf(X, y, test_size)

    # 5. Cargar modelo
    nombre_archivo_modelo = "rf_model_multicuenca.joblib"
    model_rf_loaded = load(os.path.join(ruta_modelos, nombre_archivo_modelo))

    # 6. Evaluar en el conjunto de prueba (predicción directa)
    print("\nEvaluación en el conjunto de test:")
    y_pred_test = model_rf_loaded.predict(X_test_eval)
    r2_test = r2_score(y_test_eval, y_pred_test)
    mae_test = mean_absolute_error(y_test_eval, y_pred_test)
    nse_test = calcular_nse(y_test_eval, y_pred_test)
    kge_test = calcular_kge(y_test_eval, y_pred_test)
    print(f"R2: {r2_test:.4f}, MAE: {mae_test:.4f}, NSE: {nse_test:.4f}, KGE: {kge_test:.4f}")

    # 7. Realizar predicción iterativa en el conjunto de validación
    print("\nEvaluación en el conjunto de validación iterativa:")
    total_size = len(X)
    test_size_abs = int(total_size * test_size)
    val_iter_size_abs = int(test_size_abs * validation_iter_ratio)
    val_iter_start_index = total_size - val_iter_size_abs

    X_val_iter = X.iloc[val_iter_start_index:]
    y_val_iter = y.iloc[val_iter_start_index:]
    X_hist = X.iloc[:val_iter_start_index]
    y_hist = y.iloc[:val_iter_start_index]

    if len(y_val_iter) > n_lags_area:
        predicciones_iterativas = []
        historial_area_nieve = y_hist[-n_lags_area:].tolist()
        historial_exogenas = X_hist.iloc[-n_lags_area:].to_dict('records')

        for i in range(len(y_val_iter)):
            entrada_prediccion = {}
            for j in range(n_lags_area):
                entrada_prediccion[f'area_nieve(t-{n_lags_area - j})'] = historial_area_nieve[j]
            for key, value in historial_exogenas[-1].items():
                if 'area_nieve(t-' not in key:
                    entrada_prediccion[key] = value

            entrada_df = pd.DataFrame([entrada_prediccion])[feature_columns]
            prediccion = model_rf_loaded.predict(entrada_df)[0]
            predicciones_iterativas.append(prediccion)

            historial_area_nieve.append(prediccion)
            historial_area_nieve = historial_area_nieve[1:]
            if i < len(y_val_iter) - 1:
                historial_exogenas.append(X_val_iter.iloc[i].to_dict())
                historial_exogenas = historial_exogenas[-n_lags_area:]

        y_true_val_iter = y_val_iter.values

        if len(predicciones_iterativas) == len(y_true_val_iter):
            r2_val_iter = r2_score(y_true_val_iter, predicciones_iterativas)
            mae_val_iter = mean_absolute_error(y_true_val_iter, predicciones_iterativas)
            nse_val_iter = calcular_nse(y_true_val_iter, predicciones_iterativas)
            kge_val_iter = calcular_kge(y_true_val_iter, predicciones_iterativas)
            print(f"R2: {r2_val_iter:.4f}, MAE: {mae_val_iter:.4f}, NSE: {nse_val_iter:.4f}, KGE: {kge_val_iter:.4f}")
        else:
            print(f"Advertencia: La longitud de las predicciones iterativas ({len(predicciones_iterativas)}) no coincide con la longitud de los valores reales del conjunto de validación ({len(y_true_val_iter)}).")
    else:
        print("\nEl tamaño del conjunto de validación iterativa es menor o igual al número de lags.")

    # 8. Evaluar en todo el conjunto de datos (modo predictivo)
    predicciones_iterativas_total = []
    historial_area_nieve = y[:n_lags_area].tolist()
    historial_exogenas = X[:n_lags_area].to_dict('records')

    for i in range(len(y)):
        if i < n_lags_area:
            prediccion = historial_area_nieve[i]
            predicciones_iterativas_total.append(prediccion)
            continue

        entrada_prediccion = {}
        for j in range(n_lags_area):
            entrada_prediccion[f'area_nieve(t-{n_lags_area - j})'] = historial_area_nieve[j]
        for key, value in historial_exogenas[-1].items():
            if 'area_nieve(t-' not in key:
                entrada_prediccion[key] = value

        entrada_df = pd.DataFrame([entrada_prediccion])[feature_columns]
        prediccion = model_rf_loaded.predict(entrada_df)[0]
        predicciones_iterativas_total.append(prediccion)

        historial_area_nieve.append(prediccion)
        historial_area_nieve = historial_area_nieve[1:]
        if i < len(y) - 1:
            historial_exogenas.append(X.iloc[i].to_dict())
            historial_exogenas = historial_exogenas[-n_lags_area:]

    y_true_total = y.values

    if len(predicciones_iterativas_total) == len(y_true_total):
        r2_total_pred = r2_score(y_true_total, predicciones_iterativas_total)
        mae_total_pred = mean_absolute_error(y_true_total, predicciones_iterativas_total)
        nse_total_pred = calcular_nse(y_true_total, predicciones_iterativas_total)
        kge_total_pred = calcular_kge(y_true_total, predicciones_iterativas_total)
        print(f"\nMétricas en todo el conjunto de datos (modo predictivo):")
        print(f"R2: {r2_total_pred:.4f}, MAE: {mae_total_pred:.4f}, NSE: {nse_total_pred:.4f}, KGE: {kge_total_pred:.4f}")
    else:
        print("Error: La longitud de las predicciones iterativas no coincide con la longitud de los valores reales.")


# Modificamos la llamada a la función de evaluación

# Definir rutas y nombre de archivo
RUTA_DATOS = './'
RUTA_MODELOS = './models/'
NOMBRE_ARCHIVO_DATOS = 'df_all.csv'
N_LAGS_AREA = 7
TEST_SIZE = 0.2
VALIDATION_ITER_RATIO = 0.1

cargar_guardar_modelo(os.path.join(RUTA_DATOS,NOMBRE_ARCHIVO_DATOS), N_LAGS_AREA, TEST_SIZE, VALIDATION_ITER_RATIO)
evaluar_modelo_rf(RUTA_DATOS, RUTA_MODELOS, NOMBRE_ARCHIVO_DATOS, N_LAGS_AREA, TEST_SIZE, VALIDATION_ITER_RATIO)