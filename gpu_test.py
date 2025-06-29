# IMPORTS 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error
import os
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import json
from keras.callbacks import EarlyStopping, ModelCheckpoint
import optuna
from functools import partial
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Permite el crecimiento de la memoria para que TensorFlow no reserve toda la VRAM de golpe.
        # Esto ayuda a que otras aplicaciones coexistan y TensorFlow solo use lo que necesita.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"Error configuring GPU memory growth: {e}")
else:
    print("No GPU devices found by TensorFlow.")

# --- FUNCIONES ---
def nash_sutcliffe_efficiency(y_true, y_pred):
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    if denominator == 0:
        return np.nan
    return 1 - (numerator / denominator)

def kling_gupta_efficiency(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    if np.std(y_true_flat) == 0 or np.std(y_pred_flat) == 0 or np.mean(y_true_flat) == 0:
        return -np.inf

    r_val, _ = pearsonr(y_true_flat, y_pred_flat)
    if np.isnan(r_val):
        return -np.inf

    r = r_val
    alpha = np.std(y_pred_flat) / np.std(y_true_flat)
    beta = np.mean(y_pred_flat) / np.mean(y_true_flat)
    return 1 - np.sqrt(((r - 1)**2) + ((alpha - 1)**2) + ((beta - 1)**2))

def preprocess_data(df_basin, exog_features, train_size=0.7, test_size=0.2):
    n_samples = len(df_basin)
    train_split = int(n_samples * train_size)
    test_split = int(n_samples * (train_size + test_size))

    train_idx = np.arange(train_split)
    test_idx = np.arange(train_split, test_split)
    val_idx = np.arange(test_split, n_samples)

    scaler_area = StandardScaler()
    scaler_exog = StandardScaler()

    if len(train_idx) == 0:
        raise ValueError("Training data subset is empty. Adjust train_size or data length.")

    scaler_area.fit(df_basin.iloc[train_idx]['area_nieve'].values.reshape(-1, 1))
    scaler_exog.fit(df_basin.iloc[train_idx][exog_features])

    df_basin.loc[:, 'area_nieve_scaled'] = scaler_area.transform(df_basin['area_nieve'].values.reshape(-1, 1))
    df_basin.loc[:, [f'{col}_scaled' for col in exog_features]] = scaler_exog.transform(df_basin[exog_features])

    basin_data = {
        'df': df_basin,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }
    basin_scalers = {
        'area': scaler_area,
        'exog': scaler_exog
    }
    return basin_data, basin_scalers

def create_sequences(data, n_lags, exog_cols_scaled, target_col_scaled='area_nieve_scaled'):
    X, y = [], []
    if len(data) < n_lags + 1:
        return np.array([]).reshape(0, n_lags, len(exog_cols_scaled) + 1), np.array([]).reshape(0, 1)

    for i in range(len(data) - n_lags):
        seq_area = data[target_col_scaled].iloc[i : i + n_lags].values
        seq_exog = data[exog_cols_scaled].iloc[i : i + n_lags].values
        seq = np.hstack((seq_area.reshape(-1, 1), seq_exog))
        X.append(seq)
        y.append(data[target_col_scaled].iloc[i + n_lags])
    
    if not X:
        return np.array([]).reshape(0, n_lags, len(exog_cols_scaled) + 1), np.array([]).reshape(0, 1)
        
    return np.array(X), np.array(y).reshape(-1, 1)

def create_narx_model(n_lags, n_layers, n_units_lstm, n_features, learning_rate, dropout_rate):
    model = Sequential()
    lstm_activation = 'tanh' # Prefer tanh for stability. Can be 'relu' if needed.

    if n_layers > 1:
        model.add(keras.layers.LSTM(n_units_lstm, activation=lstm_activation, input_shape=(n_lags, n_features), return_sequences=True))
    else:
        model.add(keras.layers.LSTM(n_units_lstm, activation=lstm_activation, input_shape=(n_lags, n_features)))

    for _ in range(1, n_layers):
        if _ == n_layers - 1:
            model.add(keras.layers.LSTM(n_units_lstm, activation=lstm_activation))
        else:
            model.add(keras.layers.LSTM(n_units_lstm, activation=lstm_activation, return_sequences=True))

    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5) # Stronger clipnorm
    model.compile(optimizer=optimizer, loss='mse')

    return model

def load_model_for_basin(cuenca_name, models_dir='models'):
    """Carga un único modelo para una cuenca específica."""
    model_path = os.path.join(models_dir, f'narx_model_{cuenca_name}.h5')
    if os.path.exists(model_path):
        loaded_model = keras.models.load_model(model_path)
        print(f"Modelo cargado para la cuenca: {cuenca_name} desde {model_path}")
        return loaded_model
    else:
        print(f"No se encontró el modelo para la cuenca: {cuenca_name} en {model_path}")
        return None

def evaluate_model(model, sequences, scaler_area):
    X = sequences['X']
    y_true = sequences['y']
    
    if X.shape[0] == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    y_pred_scaled = model.predict(X, verbose=0, batch_size=8)

    if np.any(np.isnan(y_pred_scaled)) or np.any(np.isinf(y_pred_scaled)):
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    y_pred_original = scaler_area.inverse_transform(y_pred_scaled)
    y_true_original = scaler_area.inverse_transform(y_true)

    r2 = pearsonr(y_true_original.flatten(), y_pred_original.flatten())
    r2 = r2.statistic**2
    mae = mean_absolute_error(y_true_original, y_pred_original)
    nse = nash_sutcliffe_efficiency(y_true_original, y_pred_original)
    kge = kling_gupta_efficiency(y_true_original, y_pred_original)

    return {'R2': r2, 'MAE': mae, 'NSE': nse, 'KGE': kge}, y_pred_original, y_true_original

def evaluate_validation(model, df_val_scaled, scaler_area, exog_cols, n_lags_area):
    n_exog_features = len(exog_cols)

    if len(df_val_scaled) < n_lags_area + 1:
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    y_val_true_original = df_val_scaled['area_nieve'].values[n_lags_area:].reshape(-1, 1)
    y_val_pred_scaled = []

    first_sequence_data = df_val_scaled[['area_nieve_scaled'] + [col + '_scaled' for col in exog_cols]].iloc[:n_lags_area].values
    last_sequence = first_sequence_data.reshape(1, n_lags_area, -1)

    for i in range(len(df_val_scaled) - n_lags_area):
        if np.any(np.isnan(last_sequence)) or np.any(np.isinf(last_sequence)):
            y_val_pred_scaled = [np.nan] * (len(df_val_scaled) - n_lags_area)
            break

        pred_scaled = model.predict(last_sequence, verbose=0, batch_size=8)

        if np.any(np.isnan(pred_scaled)) or np.any(np.isinf(pred_scaled)):
            y_val_pred_scaled.append(np.nan)
            y_val_pred_scaled.extend([np.nan] * (len(df_val_scaled) - n_lags_area - (i + 1)))
            break

        y_val_pred_scaled.append(pred_scaled[0, 0])

        next_area_scaled = pred_scaled.reshape(1, 1, 1)
        next_exog_scaled = df_val_scaled[[col + '_scaled' for col in exog_cols]].iloc[n_lags_area + i].values.reshape(1, 1, n_exog_features)

        updated_area_sequence = np.concatenate([last_sequence[:, 1:, 0].reshape(1, n_lags_area - 1, 1), next_area_scaled], axis=1)
        updated_exog_sequence = np.concatenate([last_sequence[:, 1:, 1:], next_exog_scaled], axis=1)
        last_sequence = np.concatenate([updated_area_sequence, updated_exog_sequence], axis=2)
    
    if not y_val_pred_scaled or np.any(np.isnan(y_val_pred_scaled)) or np.any(np.isinf(y_val_pred_scaled)):
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    y_val_pred_original = scaler_area.inverse_transform(np.array(y_val_pred_scaled).reshape(-1, 1))

    if np.any(np.isnan(y_val_pred_original)) or np.any(np.isinf(y_val_pred_original)):
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    r2_val = pearsonr(y_val_true_original.flatten(), y_val_pred_original.flatten())
    r2_val = r2_val.statistic**2
    mae_val = mean_absolute_error(y_val_true_original, y_val_pred_original)
    nse_val = nash_sutcliffe_efficiency(y_val_true_original, y_val_pred_original)
    kge_val = kling_gupta_efficiency(y_val_true_original, y_val_pred_original)

    return {'R2': r2_val, 'MAE': mae_val, 'NSE': nse_val, 'KGE': kge_val}, y_val_pred_original, y_val_true_original

def evaluate_full_dataset(model, df_full_scaled_cuenca, scaler_area, exog_cols_scaled, n_lags_area, graph=False, base_model_dir='./', cuenca_name=""):
    """
    Evalúa un modelo específico en el conjunto de datos completo (modo predicción walk-forward).
    Optimizado para un único modelo.
    """
    
    if model is None:
        print(f"Skipping full dataset evaluation for {cuenca_name}: No model provided.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    n_exog_features = len(exog_cols_scaled)

    if len(df_full_scaled_cuenca) < n_lags_area + 1:
        print(f"Warning: Full dataset for {cuenca_name} is too short for n_lags={n_lags_area}. Skipping full evaluation.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    y_full_true_original = df_full_scaled_cuenca['area_nieve'].values[n_lags_area:].reshape(-1, 1)
    y_full_pred_scaled = []

    first_sequence_full = df_full_scaled_cuenca[['area_nieve_scaled'] + exog_cols_scaled].iloc[:n_lags_area].values.reshape(1, n_lags_area, -1)
    last_sequence_full = first_sequence_full.copy()

    for i in range(len(df_full_scaled_cuenca) - n_lags_area):
        if np.any(np.isnan(last_sequence_full)) or np.any(np.isinf(last_sequence_full)):
            y_full_pred_scaled.extend([np.nan] * (len(df_full_scaled_cuenca) - n_lags_area - i))
            break

        pred_scaled = model.predict(last_sequence_full, verbose=0, batch_size=8)

        if np.any(np.isnan(pred_scaled)) or np.any(np.isinf(pred_scaled)):
            y_full_pred_scaled.append(np.nan)
            y_full_pred_scaled.extend([np.nan] * (len(df_full_scaled_cuenca) - n_lags_area - (i + 1)))
            break

        y_full_pred_scaled.append(pred_scaled[0, 0])

        next_area_scaled = pred_scaled.reshape(1, 1, 1)
        next_exog_scaled = df_full_scaled_cuenca[exog_cols_scaled].iloc[n_lags_area + i].values.reshape(1, 1, n_exog_features)

        updated_area_sequence = np.concatenate([last_sequence_full[:, 1:, 0].reshape(1, n_lags_area - 1, 1), next_area_scaled], axis=1)
        updated_exog_sequence = np.concatenate([last_sequence_full[:, 1:, 1:], next_exog_scaled], axis=1)
        last_sequence_full = np.concatenate([updated_area_sequence, updated_exog_sequence], axis=2)
            
    if not y_full_pred_scaled:
        print(f"Warning: No se generaron predicciones para {cuenca_name} en el conjunto completo.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    y_full_pred_original = scaler_area.inverse_transform(np.array(y_full_pred_scaled).reshape(-1, 1))

    # --- NUEVA COMPROBACIÓN DE LONGITUD Y VALORES ---
    if y_full_pred_original.shape[0] != y_full_true_original.shape[0]:
        print(f"Warning: Longitud de predicciones ({y_full_pred_original.shape[0]}) no coincide con valores reales ({y_full_true_original.shape[0]}) para {cuenca_name} en el conjunto completo.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
    
    if np.any(np.isnan(y_full_pred_original)) or np.any(np.isinf(y_full_pred_original)):
        print(f"Warning: Predicciones transformadas para {cuenca_name} en el conjunto completo contienen NaN/inf.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    r2_full = pearsonr(y_full_true_original.flatten(), y_full_pred_original.flatten())
    r2_full = r2_full.statistic**2
    mae_full = mean_absolute_error(y_full_true_original, y_full_pred_original)
    nse_full = nash_sutcliffe_efficiency(y_full_true_original, y_full_pred_original)
    kge_full = kling_gupta_efficiency(y_full_true_original, y_full_pred_original)

    full_metrics = {'R2': r2_full, 'MAE': mae_full, 'NSE': nse_full, 'KGE': kge_full}
    
    # --- GRÁFICOS ---
    if graph == True:
        # Aquí se asume que df_full_scaled_cuenca tiene una columna 'fecha'.
        real_plot = df_full_scaled_cuenca.iloc[n_lags_area:].copy()
        real_plot['area_nieve'] = y_full_true_original
        y_full_pred_df = pd.DataFrame(y_full_pred_original, columns=['area_nieve_pred'], index=real_plot.index)
        df_plot = pd.concat([real_plot, y_full_pred_df], axis=1)
        df_plot['fecha'] = pd.to_datetime(df_plot['fecha'], format='%Y-%m-%d')

        graph_types = ['per_day', 'per_month', 'all_days']
        for graph_type in graph_types:
            xlabel_text = ""
            groupby_col = 'fecha'
            title_suffix = ""

            if graph_type == 'per_day':
                df_plot['fecha_agrupada'] = df_plot['fecha'].dt.day_of_year
                xlabel_text = "Day of Year"
                groupby_col = 'fecha_agrupada'
                title_suffix = " (Average per day of the year)"
            elif graph_type == 'per_month':
                df_plot['fecha_agrupada'] = df_plot['fecha'].dt.month
                xlabel_text = 'Month'
                groupby_col = 'fecha_agrupada'
                title_suffix = " (Average per month)"
            elif graph_type == 'all_days': 
                xlabel_text = "Date"
                title_suffix = " (Serie temporal completa)"

            df_plot_grouped = df_plot.groupby(groupby_col).agg(
                area_nieve_real = ('area_nieve', 'mean'),
                area_nieve_pred=('area_nieve_pred', 'mean')
            ).reset_index()

            plt.figure(figsize=(15,6))
            if graph_type in ['per_day', 'per_month']:
                plt.xlim(left=min(df_plot_grouped[groupby_col]), right=(max(df_plot_grouped[groupby_col])))
            sns.lineplot(x=df_plot_grouped[groupby_col], y=df_plot_grouped.area_nieve_real, label='Real area')
            sns.lineplot(x=df_plot_grouped[groupby_col], y=df_plot_grouped.area_nieve_pred, label='Prediction')
            plt.title(f'Prediction vs Real {cuenca_name.upper()}{title_suffix}')
            plt.xlabel(xlabel_text)
            plt.ylabel("Snow area Km2")
            plt.legend()
            plt.grid(True)
            
            output_path = os.path.join(base_model_dir, f'graphs_{cuenca_name}')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.savefig(os.path.join(output_path, f'{graph_type}.png'))
            plt.close()
    
    return full_metrics


# Define tu create_narx_model para poder crear un modelo si no quieres cargar uno
def create_narx_model(n_lags, n_layers, n_units_lstm, n_features, learning_rate, dropout_rate):
    model = keras.models.Sequential()
    lstm_activation = 'tanh' # O 'relu' si usaste eso

    if n_layers > 1:
        model.add(keras.layers.LSTM(n_units_lstm, activation=lstm_activation, input_shape=(n_lags, n_features), return_sequences=True))
    else:
        model.add(keras.layers.LSTM(n_units_lstm, activation=lstm_activation, input_shape=(n_lags, n_features)))

    for _ in range(1, n_layers):
        if _ == n_layers - 1:
            model.add(keras.layers.LSTM(n_units_lstm, activation=lstm_activation))
        else:
            model.add(keras.layers.LSTM(n_units_lstm, activation=lstm_activation, return_sequences=True))

    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# --- Inicio del script de prueba ---
print("Iniciando prueba de rendimiento de evaluate_full_dataset...")

# Configura tus directorios y archivos
basins_data_root_dir = 'datasets_imputed/' # <--- RUTA A TUS DATOS CSV
models_root_dir = os.path.join("E:", "models") # <--- RUTA A TU DIRECTORIO DE MODELOS

specific_basin_name = 'adda-bornio' # Cuenca de prueba
specific_data_file_name = f'{specific_basin_name}.csv'

exog_cols = ["dia_sen", "temperatura", "precipitacion", "dias_sin_precip"]
exog_cols_scaled = [col + '_scaled' for col in exog_cols]

# Cargar y preprocesar los datos de la cuenca
try:
    df_basin = pd.read_csv(os.path.join(basins_data_root_dir, specific_data_file_name), index_col=0)
    df_basin.index = pd.to_datetime(df_basin.index) # Asegurar datetime para el preprocesamiento
    
    # Nota: preprocess_data necesita 'fecha' en el formato que usas para day_of_year etc.
    scaled_data_info, scalers_info = preprocess_data(df_basin.copy(), exog_cols, train_size=0.7, test_size=0.2)
    df_full_scaled_cuenca_test = scaled_data_info['df']
    scaler_area_test = scalers_info['area']
    print(f"Datos de '{specific_basin_name}' preprocesados para prueba.")
except Exception as e:
    print(f"Error al cargar/preprocesar datos para la prueba: {e}. Saliendo.")
    exit()

# Cargar un modelo ya entrenado (o crear uno simple para la prueba)
model_to_test = None
n_lags_area_test = 7 # Asume un valor típico o cárgalo de los params del modelo
try:
    # Intenta cargar un modelo existente para una prueba más realista
    model_path_test = os.path.join(models_root_dir, specific_basin_name, f'narx_model_best_{specific_basin_name}.keras')
    if not os.path.exists(model_path_test): # Fallback a .h5 si .keras no existe
        model_path_test = os.path.join(models_root_dir, specific_basin_name, f'narx_model_best_{specific_basin_name}.h5')
    
    # Se asume que CustomLSTM está definido y registrado globalmente para la carga de .h5
    # y que 'mse' como custom_object si es un .h5
    model_to_test = keras.models.load_model(model_path_test, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    n_lags_area_test = model_to_test.input_shape[1] # Obtener n_lags del modelo cargado
    print(f"Modelo cargado para prueba: {model_path_test} (n_lags={n_lags_area_test}).")
except Exception as e:
    print(f"Error al cargar modelo para prueba: {e}. Creando un modelo simple para continuar.")
    # Crea un modelo simple si la carga falla
    n_lags_area_test = 7
    n_features_test = 1 + len(exog_cols_scaled)
    model_to_test = create_narx_model(n_lags_area_test, 2, 10, n_features_test, 0.001, 0.2)
    print("Modelo simple creado para la prueba.")

# Ejecutar evaluate_full_dataset y medir el tiempo
print(f"\nEjecutando evaluate_full_dataset para '{specific_basin_name}' ({len(df_full_scaled_cuenca_test)} puntos de datos)...")
start_time = time.time()
metrics_test_run = evaluate_full_dataset(
    model_to_test,
    df_full_scaled_cuenca_test,
    scaler_area_test,
    exog_cols_scaled,
    n_lags_area_test,
    graph=False, # No generar gráficos durante esta prueba para centrarse en el tiempo
    base_model_dir=os.path.join(models_root_dir, specific_basin_name),
    cuenca_name=specific_basin_name
)
end_time = time.time()
print(f"Tiempo de ejecución de evaluate_full_dataset: {end_time - start_time:.2f} segundos.")
print(f"Métricas obtenidas: {metrics_test_run}")

print("\nPrueba de rendimiento finalizada. Revisa el uso de la GPU en el Administrador de Tareas durante la ejecución.")