import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import os
import json

# --- FUNCIONES DE SOPORTE ---

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
        return -np.inf # Devuelve un KGE muy malo si hay problemas con la desviación estándar o la media

    r_val, _ = pearsonr(y_true_flat, y_pred_flat)
    if np.isnan(r_val): # Si la correlación es NaN (por ejemplo, predicciones constantes)
        return -np.inf

    r = r_val
    alpha = np.std(y_pred_flat) / np.std(y_true_flat)
    beta = np.mean(y_pred_flat) / np.mean(y_true_flat)
    return 1 - np.sqrt(((r - 1)**2) + ((alpha - 1)**2) + ((beta - 1)**2))

def preprocess_data(df_basin, exog_features, train_size=0.7, test_size=0.2):
    """
    Función para escalar los datos de area_nieve y exog_features para UNA CUENCA.
    """
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

def evaluate_model(model, sequences, scaler_area):
    X = sequences['X']
    y_true = sequences['y']
    
    if X.shape[0] == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    y_pred_scaled = model.predict(X, verbose=0)

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
    
    # Calculate the expected length of predictions/true values *after* accounting for lags
    expected_pred_len = len(df_val_scaled) - n_lags_area

    if expected_pred_len <= 0:
        # If there's not enough data to even make one prediction sequence
        print(f"Warning (eval_val): Validation data ({len(df_val_scaled)}) is too short for n_lags={n_lags_area}. Expected pred len <= 0.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    # This is the target array, which has the correct expected length
    y_val_true_original = df_val_scaled['area_nieve'].values[n_lags_area:].reshape(-1, 1)
    
    y_val_pred_scaled = []

    # Initialize the first sequence from actual scaled data
    first_sequence_data = df_val_scaled[['area_nieve_scaled'] + [col + '_scaled' for col in exog_cols]].iloc[:n_lags_area].values
    last_sequence = first_sequence_data.reshape(1, n_lags_area, -1)

    # Loop to generate predictions
    for i in range(expected_pred_len):
        # Check if the current input sequence to the model is valid
        if np.any(np.isnan(last_sequence)) or np.any(np.isinf(last_sequence)):
            print(f"Warning (eval_val): Input sequence for prediction contains NaN/inf at step {i}. Stopping prediction loop.")
            # Pad the rest of the predictions with NaN if the input is invalid
            y_val_pred_scaled.extend([np.nan] * (expected_pred_len - i))
            break # Exit the loop early

        pred_scaled = model.predict(last_sequence, verbose=0)

        # Check if the prediction itself is valid
        if np.any(np.isnan(pred_scaled)) or np.any(np.isinf(pred_scaled)):
            print(f"Warning (eval_val): Model predicted NaN/inf at step {i}. Stopping prediction loop.")
            # Append the current NaN prediction, then pad the rest with NaN
            y_val_pred_scaled.append(np.nan)
            y_val_pred_scaled.extend([np.nan] * (expected_pred_len - (i + 1)))
            break # Exit the loop early

        y_val_pred_scaled.append(pred_scaled[0, 0])

        # Prepare the next sequence for the next prediction
        # Get the actual next exogenous values from df_val_scaled (future known exog data)
        next_exog_scaled = df_val_scaled[[col + '_scaled' for col in exog_cols]].iloc[n_lags_area + i].values.reshape(1, 1, n_exog_features)
        
        # The predicted snow area is used as the autoregressive part for the next step
        next_area_scaled = pred_scaled.reshape(1, 1, 1)

        # Update the sequence: drop oldest, add new prediction (for area_nieve) and new exog data
        updated_area_sequence = np.concatenate([last_sequence[:, 1:, 0].reshape(1, n_lags_area - 1, 1), next_area_scaled], axis=1)
        updated_exog_sequence = np.concatenate([last_sequence[:, 1:, 1:], next_exog_scaled], axis=1)
        last_sequence = np.concatenate([updated_area_sequence, updated_exog_sequence], axis=2)

    # IMPORTANT: Ensure y_val_pred_scaled has the exact expected length
    # If the loop finished normally, this won't change anything.
    # If it broke early, it's already padded.
    # If expected_pred_len is 0, y_val_pred_scaled might be empty, handle this at the start.
    
    # Convert predictions to original scale
    # Check if predictions are valid before inverse transform
    if not y_val_pred_scaled or np.any(np.isnan(y_val_pred_scaled)) or np.any(np.isinf(y_val_pred_scaled)):
        print(f"Warning (eval_val): Final scaled predictions contain NaN/inf. Returning NaN for metrics.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None
    
    # Ensure both arrays have the same length before inverse_transform.
    # The previous padding should handle the length, but double-check if lengths don't match exactly.
    # This assertion is primarily for debugging, the padding above should prevent it.
    if len(y_val_pred_scaled) != expected_pred_len:
         print(f"Error (eval_val): Predicted length mismatch: {len(y_val_pred_scaled)} vs expected {expected_pred_len}. This should not happen after padding.")
         return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    y_val_pred_original = scaler_area.inverse_transform(np.array(y_val_pred_scaled).reshape(-1, 1))

    # Final checks on original scale predictions before metrics
    if np.any(np.isnan(y_val_pred_original)) or np.any(np.isinf(y_val_pred_original)):
        print(f"Warning (eval_val): Inverse transformed predictions contain NaN/inf. Returning NaN for metrics.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None
    
    # Now, both y_val_true_original and y_val_pred_original should have the same length.
    # assert y_val_true_original.shape == y_val_pred_original.shape, "Arrays must have same shape for metrics"

    r2_val = pearsonr(y_val_true_original.flatten(), y_val_pred_original.flatten())
    r2_val = r2_val.statistic**2
    mae_val = mean_absolute_error(y_val_true_original, y_val_pred_original)
    nse_val = nash_sutcliffe_efficiency(y_val_true_original, y_val_pred_original)
    kge_val = kling_gupta_efficiency(y_val_true_original, y_val_pred_original)

    return {'R2': r2_val, 'MAE': mae_val, 'NSE': nse_val, 'KGE': kge_val}, y_val_pred_original, y_val_true_original

def evaluate_full_dataset(best_models_dict, all_basins_preprocessed_data, exog_cols_scaled, graph=False, base_model_dir='./'):
    full_metrics = {}

    for cuenca_name, model_info in best_models_dict.items():
        model = model_info['model']
        best_n_lags_area = model_info['best_n_lags_area']

        if model is None:
            full_metrics[cuenca_name] = {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
            continue

        scaler_area = all_basins_preprocessed_data[cuenca_name]['scalers']['area']
        df_full_scaled_cuenca = all_basins_preprocessed_data[cuenca_name]['data']['df'].copy()
        n_exog_features = len(exog_cols_scaled)

        if len(df_full_scaled_cuenca) < best_n_lags_area + 1:
            full_metrics[cuenca_name] = {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
            continue

        y_full_true_original = df_full_scaled_cuenca['area_nieve'].values[best_n_lags_area:].reshape(-1, 1)
        y_full_pred_scaled = []

        first_sequence_full = df_full_scaled_cuenca[['area_nieve_scaled'] + exog_cols_scaled].iloc[:best_n_lags_area].values.reshape(1, best_n_lags_area, -1)
        last_sequence_full = first_sequence_full.copy()

        for i in range(len(df_full_scaled_cuenca) - best_n_lags_area):
            if np.any(np.isnan(last_sequence_full)) or np.any(np.isinf(last_sequence_full)):
                y_full_pred_scaled.extend([np.nan] * (len(df_full_scaled_cuenca) - best_n_lags_area - i))
                break

            pred_scaled = model.predict(last_sequence_full, verbose=0)

            if np.any(np.isnan(pred_scaled)) or np.any(np.isinf(pred_scaled)):
                y_full_pred_scaled.append(np.nan)
                y_full_pred_scaled.extend([np.nan] * (len(df_full_scaled_cuenca) - best_n_lags_area - (i + 1)))
                break

            y_full_pred_scaled.append(pred_scaled[0, 0])

            next_area_scaled = pred_scaled.reshape(1, 1, 1)
            next_exog_scaled = df_full_scaled_cuenca[exog_cols_scaled].iloc[best_n_lags_area + i].values.reshape(1, 1, n_exog_features)

            updated_area_sequence = np.concatenate([last_sequence_full[:, 1:, 0].reshape(1, best_n_lags_area - 1, 1), next_area_scaled], axis=1)
            updated_exog_sequence = np.concatenate([last_sequence_full[:, 1:, 1:], next_exog_scaled], axis=1)
            last_sequence_full = np.concatenate([updated_area_sequence, updated_exog_sequence], axis=2)
                
        if not y_full_pred_scaled or np.any(np.isnan(y_full_pred_scaled)) or np.any(np.isinf(y_full_pred_scaled)):
            full_metrics[cuenca_name] = {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
            continue

        y_full_pred_original = scaler_area.inverse_transform(np.array(y_full_pred_scaled).reshape(-1, 1))

        if np.any(np.isnan(y_full_pred_original)) or np.any(np.isinf(y_full_pred_original)):
            full_metrics[cuenca_name] = {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
            continue

        r2_full = pearsonr(y_full_true_original.flatten(), y_full_pred_original.flatten())
        r2_full = r2_full.statistic**2
        mae_full = mean_absolute_error(y_full_true_original, y_full_pred_original)
        nse_full = nash_sutcliffe_efficiency(y_full_true_original, y_full_pred_original)
        kge_full = kling_gupta_efficiency(y_full_true_original, y_full_pred_original)

        full_metrics[cuenca_name] = {'R2': r2_full, 'MAE': mae_full, 'NSE': nse_full, 'KGE': kge_full}

    return full_metrics

# --- CONFIGURACIÓN DE RUTAS ---
# ¡IMPORTANTE! Ajusta estas rutas a las ubicaciones reales en tu sistema.
basins_dir = 'datasets/'  # Directorio que contiene los CSV de cada cuenca
models_dir = os.path.join("E:", "models_per_basin") # Directorio base donde guardaste los modelos por cuenca

# --- PARÁMETROS FIJOS ---
exog_cols = ["dia_sen", "temperatura", "precipitacion", "dias_sin_precip"]
exog_cols_scaled = [col + '_scaled' for col in exog_cols]

# --- 1. CARGAR Y PREPROCESAR DATOS DE LAS CUENCAS ---
print("Cargando y preprocesando datos de todas las cuencas...")
basin_files = [f for f in os.listdir(basins_dir) if f.endswith('.csv')]
cuencas = [os.path.splitext(f)[0] for f in basin_files]

all_basins_preprocessed_data = {}
for cuenca_name in cuencas:
    try:
        df_basin = pd.read_csv(os.path.join(basins_dir, f'{cuenca_name}.csv'), index_col=0)
        if 'fecha' in df_basin.columns:
            df_basin['fecha'] = df_basin['fecha'].astype(str)
        
        basin_data, basin_scalers = preprocess_data(df_basin.copy(), exog_cols)
        all_basins_preprocessed_data[cuenca_name] = {'data': basin_data, 'scalers': basin_scalers}
        print(f"Datos de '{cuenca_name}' preprocesados correctamente.")
    except Exception as e:
        print(f"Error al preprocesar datos para '{cuenca_name}': {e}. Esta cuenca será omitida.")
        continue

# Filtrar cuencas que no se pudieron preprocesar
cuencas_a_evaluar = list(all_basins_preprocessed_data.keys())

# --- 2. CARGAR MODELOS Y DEDUCIR PARÁMETROS DE ENTRADA ---
print("\nCargando modelos y deduciendo 'n_lags_area' de su forma de entrada...")
best_models_per_basin = {}
for cuenca_name in cuencas_a_evaluar:
    # Ruta al archivo .h5 del modelo
    model_path = os.path.join(models_dir, cuenca_name, f'narx_model_best_{cuenca_name}.h5')
    
    best_n_lags_area = None
    best_params_dict = {} # Se mantendrá vacío o con valores por defecto si no hay JSON

    if os.path.exists(model_path):
        try:
            model = keras.models.load_model(model_path)
            
            # Deducir n_lags_area del input_shape del modelo
            # El input_shape esperado es (None, n_lags, n_features)
            if hasattr(model, 'input_shape') and len(model.input_shape) >= 2:
                best_n_lags_area = model.input_shape[1]
                print(f"'{cuenca_name}': n_lags_area deducido del modelo: {best_n_lags_area}")
            else:
                print(f"Advertencia: No se pudo deducir n_lags_area del input_shape del modelo '{cuenca_name}'. Usando valor predeterminado de 7.")
                best_n_lags_area = 7 # Valor predeterminado si no se puede deducir
            
            # Intenta cargar cualquier otro parámetro si el archivo existiera (aunque la consigna es que no está)
            params_path = os.path.join(models_dir, cuenca_name, 'best_params.json')
            if os.path.exists(params_path):
                try:
                    with open(params_path, 'r') as f:
                        best_params_dict = json.load(f)
                    print(f"Se cargaron parámetros adicionales para '{cuenca_name}' del archivo JSON.")
                except Exception as e:
                    print(f"Error al cargar best_params.json para '{cuenca_name}': {e}. Se ignorarán los parámetros.")
            else:
                print(f"Nota: No se encontró best_params.json para '{cuenca_name}'. Los parámetros adicionales no estarán disponibles.")


            best_models_per_basin[cuenca_name] = {
                'model': model,
                'best_n_lags_area': best_n_lags_area,
                'best_params': best_params_dict # Contendrá los parámetros adicionales o estará vacío
            }
            print(f"Modelo cargado y n_lags_area obtenido para '{cuenca_name}'.")

        except Exception as e:
            print(f"Error al cargar el modelo .h5 para '{cuenca_name}': {e}. Esta cuenca será omitida de la evaluación.")
            best_models_per_basin[cuenca_name] = {'model': None, 'best_n_lags_area': None, 'best_params': None}
    else:
        print(f"No se encontró el archivo del modelo para '{cuenca_name}' en: {model_path}. Esta cuenca será omitida.")
        best_models_per_basin[cuenca_name] = {'model': None, 'best_n_lags_area': None, 'best_params': None}

# --- 3. RE-EVALUAR MÉTRICAS ---
print("\nRe-evaluando las métricas para cada modelo...")
all_final_metrics = {}

for cuenca_name in cuencas_a_evaluar:
    current_model_info = best_models_per_basin.get(cuenca_name)
    
    if current_model_info is None or current_model_info['model'] is None:
        print(f"Saltando la evaluación para '{cuenca_name}' (modelo no disponible/cargado).")
        all_final_metrics[cuenca_name] = {
            'best_params': current_model_info['best_params'] if current_model_info else None,
            'full_dataset': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
            'train': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
            'test': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
            'val': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
        }
        continue

    current_model = current_model_info['model']
    current_n_lags_area = current_model_info['best_n_lags_area']
    
    if cuenca_name not in all_basins_preprocessed_data:
        print(f"Saltando la evaluación para '{cuenca_name}' (datos preprocesados no disponibles).")
        all_final_metrics[cuenca_name] = {
            'best_params': current_model_info['best_params'],
            'full_dataset': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
            'train': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
            'test': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
            'val': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
        }
        continue

    current_scaler_area = all_basins_preprocessed_data[cuenca_name]['scalers']['area']
    basin_data_eval = all_basins_preprocessed_data[cuenca_name]['data']

    # Preparar secuencias para la evaluación
    sequences_for_eval = {
        'train': {'X': create_sequences(basin_data_eval['df'].iloc[basin_data_eval['train_idx']], current_n_lags_area, exog_cols_scaled)[0],
                  'y': create_sequences(basin_data_eval['df'].iloc[basin_data_eval['train_idx']], current_n_lags_area, exog_cols_scaled)[1]},
        'test': {'X': create_sequences(basin_data_eval['df'].iloc[basin_data_eval['test_idx']], current_n_lags_area, exog_cols_scaled)[0],
                 'y': create_sequences(basin_data_eval['df'].iloc[basin_data_eval['test_idx']], current_n_lags_area, exog_cols_scaled)[1]},
        'val_df': basin_data_eval['df'].iloc[basin_data_eval['val_idx']].copy()
    }

    metrics_train, _, _ = evaluate_model(current_model, sequences_for_eval['train'], current_scaler_area)
    metrics_test, _, _ = evaluate_model(current_model, sequences_for_eval['test'], current_scaler_area)
    metrics_val, _, _ = evaluate_validation(current_model, sequences_for_eval['val_df'], current_scaler_area, exog_cols, current_n_lags_area)
    
    metrics_full_dataset_result = evaluate_full_dataset({cuenca_name: current_model_info}, all_basins_preprocessed_data, exog_cols_scaled)

    all_final_metrics[cuenca_name] = {
        'best_params': current_model_info['best_params'],
        'full_dataset': metrics_full_dataset_result.get(cuenca_name, {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}),
        'train': metrics_train,
        'test': metrics_test,
        'val': metrics_val
    }
    print(f"Métricas re-calculadas para '{cuenca_name}'.")

# --- 4. GUARDAR MÉTRICAS EN JSON ---
output_json_path = os.path.join(models_dir, 'all_final_metrics.json')

# Función para convertir tipos de NumPy a tipos nativos de Python para JSON
def convert_numpy_to_python(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_to_python(elem) for elem in obj]
    return obj

print(f"\nGuardando todas las métricas finales en: {output_json_path}")
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(convert_numpy_to_python(all_final_metrics), f, indent=4)

print("Proceso completado. El archivo JSON ha sido generado.")