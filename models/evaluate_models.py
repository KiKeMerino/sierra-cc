import pandas as pd
import numpy as np
from sklearn import metrics
from tensorflow import keras
from joblib import load
import os

# --- Parámetros ---
n_lags_area = 3
n_lags_exog = 2
test_size = 0.3
RUTA_MODELOS = './models/'
RUTA_DATOS = './csv_merged/final/'
NOMBRE_ARCHIVO_DATOS = 'cuencas_all.csv'

# --- Funciones de métricas ---
def nash_sutcliffe_efficiency(observaciones, simulaciones):
    numerador = np.sum((observaciones - simulaciones)**2)
    denominador = np.sum((observaciones - np.mean(observaciones))**2)
    nse = 1 - (numerador / denominador)
    return nse

def kling_gupta_efficiency(observaciones, simulaciones):
    r = np.corrcoef(observaciones, simulaciones)[0, 1]
    sigma_sim = np.std(simulaciones)
    sigma_obs = np.std(observaciones)
    mu_sim = np.mean(simulaciones)
    mu_obs = np.mean(observaciones)
    r_component = r
    beta_component = mu_sim / mu_obs
    gamma_component = sigma_sim / sigma_obs
    kge = 1 - np.sqrt((r_component - 1)**2 + (beta_component - 1)**2 + (gamma_component - 1)**2)
    return kge

def cargar_y_preprocesar_datos(ruta_datos, nombre_archivo, n_lags_area, n_lags_exog):
    """Carga y preprocesa los datos para los modelos."""
    df = pd.read_csv(os.path.join(ruta_datos, nombre_archivo))
    df_lagged = crear_lags(df.copy(), n_lags_area, n_lags_exog)
    y = df_lagged['area_nieve(t-d1)'].astype(float)
    X = df_lagged.drop(columns=['area_nieve(t-d1)'])
    cuenca_col = X['cuenca(t-0)'].values.reshape(-1, 1)
    exog_cols = [col for col in X.columns if col != 'cuenca(t-0)']
    X_exog = X[exog_cols].values.astype(float)
    scaler_x = load(os.path.join(RUTA_MODELOS, 'scaler_x_rnn_multi.joblib')) # Cargamos los scalers guardados
    X_scaled_exog = scaler_x.transform(X_exog)
    encoder_cuenca = load(os.path.join(RUTA_MODELOS, 'encoder_cuenca_rnn_multi.joblib')) # Cargamos el encoder guardado
    X_scaled_cuenca = encoder_cuenca.transform(cuenca_col)
    X_scaled = np.concatenate((X_scaled_exog, X_scaled_cuenca), axis=1)
    scaler_y = load(os.path.join(RUTA_MODELOS, 'scaler_y_rnn_multi.joblib')) # Cargamos el scaler_y guardado
    y_scaled = scaler_y.transform(y.values.reshape(-1, 1))
    
    train_size = int(len(X_scaled) * (1 - test_size))
    X_test = X_scaled[train_size:]
    y_test = y_scaled[train_size:]
    
    return X_test, y_test, scaler_y

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

def cargar_y_preprocesar_datos_rf(ruta_datos, nombre_archivo, n_lags_area, n_lags_exog):
    """Carga y preprocesa los datos para el modelo Random Forest."""
    df = pd.read_csv(os.path.join(ruta_datos, nombre_archivo))
    df_lagged = crear_lags(df.copy(), n_lags_area, n_lags_exog)
    y = df_lagged['area_nieve(t-d1)'].astype(float)
    X = df_lagged.drop(columns=['area_nieve(t-d1)'])
    cuenca_col = X['cuenca(t-0)'].values.reshape(-1, 1)
    exog_cols = [col for col in X.columns if col != 'cuenca(t-0)']
    X_exog = X[exog_cols].values.astype(float)
    scaler_x = load(os.path.join(RUTA_MODELOS, 'scaler_x_rf_multi.joblib')) # Cargamos el scaler_x guardado
    X_scaled_exog = scaler_x.transform(X_exog)
    encoder_cuenca = load(os.path.join(RUTA_MODELOS, 'encoder_cuenca_rf_multi.joblib')) # Cargamos el encoder guardado
    X_scaled_cuenca = encoder_cuenca.transform(cuenca_col)
    X_processed = np.concatenate((X_scaled_exog, X_scaled_cuenca), axis=1)
    
    train_size = int(len(X_processed) * (1 - test_size))
    X_test = X_processed[train_size:]
    y_test = y[train_size:]
    
    return X_test, y_test

# --- Evaluar modelo RNN para todas las cuencas ---
nombre_archivo_rnn_multi = 'simple_model_multicuenca.h5'
modelo_rnn_multi = keras.models.load_model(os.path.join(RUTA_MODELOS, nombre_archivo_rnn_multi))
X_test_rnn, y_test_rnn_scaled, scaler_y_rnn = cargar_y_preprocesar_datos(RUTA_DATOS, NOMBRE_ARCHIVO_DATOS, n_lags_area, n_lags_exog)
X_test_nn_multi = X_test_rnn.reshape((X_test_rnn.shape[0], 1, X_test_rnn.shape[1]))
y_pred_scaled_rnn_multi = modelo_rnn_multi.predict(X_test_nn_multi)
y_pred_rnn_multi = scaler_y_rnn.inverse_transform(y_pred_scaled_rnn_multi)
y_test_original_rnn_multi = scaler_y_rnn.inverse_transform(y_test_rnn_scaled)
r2_rnn_multi = metrics.r2_score(y_test_original_rnn_multi, y_pred_rnn_multi)
mae_rnn_multi = metrics.mean_absolute_error(y_test_original_rnn_multi, y_pred_rnn_multi)
nse_rnn_multi = nash_sutcliffe_efficiency(y_test_original_rnn_multi.flatten(), y_pred_rnn_multi.flatten())
kge_rnn_multi = kling_gupta_efficiency(y_test_original_rnn_multi.flatten(), y_pred_rnn_multi.flatten())

print("\n--- Resultados del modelo RNN (todas las cuencas) ---")
print(f"R2: {r2_rnn_multi:.4f}")
print(f"MAE: {mae_rnn_multi:.4f}")
print(f"NSE: {nse_rnn_multi:.4f}")
print(f"KGE: {kge_rnn_multi:.4f}")

# --- Evaluar modelo Random Forest para todas las cuencas ---
nombre_archivo_rf_multi = 'random_forest_model.joblib'
modelo_rf_multi = load(os.path.join(RUTA_MODELOS, nombre_archivo_rf_multi))
X_test_rf, y_test_rf = cargar_y_preprocesar_datos_rf(RUTA_DATOS, NOMBRE_ARCHIVO_DATOS, n_lags_area, n_lags_exog)
y_pred_rf_multi = modelo_rf_multi.predict(X_test_rf)
r2_rf_multi = metrics.r2_score(y_test_rf, y_pred_rf_multi)
mae_rf_multi = metrics.mean_absolute_error(y_test_rf, y_pred_rf_multi)
nse_rf_multi = nash_sutcliffe_efficiency(y_test_rf.flatten(), y_pred_rf_multi.flatten())
kge_rf_multi = kling_gupta_efficiency(y_test_rf.flatten(), y_pred_rf_multi.flatten())

print("\n--- Resultados del modelo Random Forest (todas las cuencas) ---")
print(f"R2: {r2_rf_multi:.4f}")
print(f"MAE: {mae_rf_multi:.4f}")
print(f"NSE: {nse_rf_multi:.4f}")
print(f"KGE: {kge_rf_multi:.4f}")