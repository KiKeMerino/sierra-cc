#%% Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import r2_score, mean_absolute_error
import os
import itertools # Para generar combinaciones de hiperparámetros
import json # Para guardar resultados
from joblib import dump, load

#%% --- Funciones de métricas ---
def nash_sutcliffe_efficiency(y_true, y_pred):
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    if denominator == 0:
        return np.nan
    return 1 - (numerator / denominator)

def kling_gupta_efficiency(y_true, y_pred):
    r = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    alpha = np.std(y_pred) / np.std(y_true)
    beta = np.mean(y_pred) / np.mean(y_true)
    return 1 - np.sqrt(((r - 1)**2) + ((alpha - 1)**2) + ((beta - 1)**2))

# --- Funciones de preprocesamiento ---
def preprocess_data(df, train_size=0.7, test_size=0.2):
    cuencas = df['cuenca'].unique()
    df_por_cuenca = {cuenca: df[df['cuenca'] == cuenca].copy() for cuenca in cuencas}
    scaled_data = {}
    scalers = {}

    for cuenca, df_cuenca in df_por_cuenca.items():
        n_samples = len(df_cuenca)
        train_split = int(n_samples * train_size)
        test_split = int(n_samples * (train_size + test_size))

        train_idx = np.arange(train_split)
        test_idx = np.arange(train_split, test_split)
        val_idx = np.arange(test_split, n_samples)

        scaler_area = StandardScaler()
        scaler_exog = StandardScaler()
        exog_features = ['temperatura', 'precipitacion', 'dias_sin_precip']

        # Ajustar escaladores solo en el conjunto de entrenamiento
        scaler_area.fit(df_cuenca.iloc[train_idx]['area_nieve'].values.reshape(-1, 1))
        scaler_exog.fit(df_cuenca.iloc[train_idx][exog_features])

        df_cuenca.loc[:, 'area_nieve_scaled'] = scaler_area.transform(df_cuenca['area_nieve'].values.reshape(-1, 1))
        df_cuenca.loc[:, [f'{col}_scaled' for col in exog_features]] = scaler_exog.transform(df_cuenca[exog_features])

        scaled_data[cuenca] = {
            'df': df_cuenca,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
        scalers[cuenca] = {
            'area': scaler_area,
            'exog': scaler_exog
        }
        # print(f"Cuenca: {cuenca}, Train: {len(train_idx)}, Test: {len(test_idx)}, Val: {len(val_idx)}")

    return scaled_data, scalers, cuencas

def create_sequences(data, n_lags, exog_cols_scaled, target_col_scaled='area_nieve_scaled'):
    X, y = [], []
    # Asegurarse de que data sea un DataFrame y usar .values para consistencia
    if isinstance(data, pd.DataFrame):
        data_target = data[target_col_scaled].values
        data_exog = data[exog_cols_scaled].values
    else: # Si ya es un array numpy
        data_target = data[:, 0] # Asumimos que la primera columna es la target
        data_exog = data[:, 1:] # Las demás son exógenas

    for i in range(len(data_target) - n_lags):
        seq_area = data_target[i : i + n_lags]
        seq_exog = data_exog[i : i + n_lags]
        # Asegurarse de que seq_area tenga la forma (n_lags, 1) para hstack
        seq = np.hstack((seq_area.reshape(-1, 1), seq_exog))
        X.append(seq)
        y.append(data_target[i + n_lags])
    return np.array(X), np.array(y).reshape(-1, 1)

def create_narx_model(n_lags, n_features, n_units_lstm, num_lstm_layers, dropout_rate, learning_rate=0.001, clipnorm=1.0):
    model = Sequential()

    # Primera capa LSTM
    if num_lstm_layers > 1:
        model.add(LSTM(n_units_lstm, activation='relu', input_shape=(n_lags, n_features), return_sequences=True))
    else:
        model.add(LSTM(n_units_lstm, activation='relu', input_shape=(n_lags, n_features)))

    # Capas LSTM adicionales
    for _ in range(1, num_lstm_layers):
        if _ == num_lstm_layers - 1: # Si es la última capa LSTM, no devuelve secuencias
            model.add(LSTM(n_units_lstm, activation='relu'))
        else:
            model.add(LSTM(n_units_lstm, activation='relu', return_sequences=True))

    # Capa Dropout (si se especifica)
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Capa de salida
    model.add(Dense(1))

    # Optimizador con recorte de gradientes
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    model.compile(optimizer=optimizer, loss='mse')

    return model

def train_model(model, X_train, y_train, epochs):
    history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_split=0.1)
    return history

def evaluate_model(model, X, y_true, scaler_area):
    y_pred_scaled = model.predict(X, verbose=0)
    y_pred_original = scaler_area.inverse_transform(y_pred_scaled)
    y_true_original = scaler_area.inverse_transform(y_true)

    r2 = r2_score(y_true_original, y_pred_original)
    mae = mean_absolute_error(y_true_original, y_pred_original)
    nse = nash_sutcliffe_efficiency(y_true_original, y_true_original) # Corrected: Use y_true_original for denominator
    kge = kling_gupta_efficiency(y_true_original, y_pred_original)

    return {'R2': r2, 'MAE': mae, 'NSE': nse, 'KGE': kge}, y_pred_original, y_true_original

def evaluate_validation(model, df_val_scaled, scaler_area, exog_cols, n_lags_area, n_exog_features):
    y_val_true_scaled = df_val_scaled['area_nieve_scaled'].values[n_lags_area:].reshape(-1, 1)
    y_val_pred_scaled = []

    # Usar los primeros 'n_lags_area' valores reales para iniciar la predicción
    initial_sequence = df_val_scaled[['area_nieve_scaled'] + [col + '_scaled' for col in exog_cols]].iloc[:n_lags_area].values.reshape(1, n_lags_area, -1)
    last_sequence_for_prediction = initial_sequence.copy()

    for i in range(len(df_val_scaled) - n_lags_area):
        pred_scaled = model.predict(last_sequence_for_prediction, verbose=0)
        y_val_pred_scaled.append(pred_scaled[0, 0])

        next_area_scaled = pred_scaled.reshape(1, 1, 1)
        next_exog_scaled = df_val_scaled[[col + '_scaled' for col in exog_cols]].iloc[n_lags_area + i].values.reshape(1, 1, n_exog_features)

        # Reconstruir la siguiente secuencia de entrada
        # Tomamos las últimas (n_lags_area - 1) entradas de la secuencia anterior
        # y las concatenamos con la nueva predicción (para area_nieve) y las nuevas exógenas.
        updated_area_seq = np.concatenate([last_sequence_for_prediction[:, 1:, 0].reshape(1, n_lags_area - 1, 1), next_area_scaled], axis=1)
        updated_exog_seq = np.concatenate([last_sequence_for_prediction[:, 1:, 1:], next_exog_scaled], axis=1)

        last_sequence_for_prediction = np.concatenate([updated_area_seq, updated_exog_seq], axis=2)

    y_val_pred_original = scaler_area.inverse_transform(np.array(y_val_pred_scaled).reshape(-1, 1))
    y_val_true_original = scaler_area.inverse_transform(y_val_true_scaled)

    r2_val = r2_score(y_val_true_original, y_val_pred_original)
    mae_val = mean_absolute_error(y_val_true_original, y_val_pred_original)
    nse_val = nash_sutcliffe_efficiency(y_val_true_original, y_val_pred_original)
    kge_val = kling_gupta_efficiency(y_val_true_original, y_val_pred_original)

    return {'R2': r2_val, 'MAE': mae_val, 'NSE': nse_val, 'KGE': kge_val}, y_val_pred_original, y_val_true_original

def evaluate_full_dataset(models, scaled_data, scalers, cuencas, n_lags_area, exog_cols_scaled):
    full_metrics = {}
    exog_cols_original = [col.replace('_scaled', '') for col in exog_cols_scaled]
    n_exog_features = len(exog_cols_scaled)

    for cuenca in cuencas:
        scaler_area = scalers[cuenca]['area']

        # Obtener y_true y y_pred para TRAIN
        X_train_full = sequences_data_global[cuenca]['X_train']
        y_train_true_full = sequences_data_global[cuenca]['y_train']
        y_train_pred_scaled = models[cuenca].predict(X_train_full, verbose=0)
        y_train_pred_original = scaler_area.inverse_transform(y_train_pred_scaled)
        y_train_true_original = scaler_area.inverse_transform(y_train_true_full)

        # Obtener y_true y y_pred para TEST
        X_test_full = sequences_data_global[cuenca]['X_test']
        y_test_true_full = sequences_data_global[cuenca]['y_test']
        y_test_pred_scaled = models[cuenca].predict(X_test_full, verbose=0)
        y_test_pred_original = scaler_area.inverse_transform(y_test_pred_scaled)
        y_test_true_original = scaler_area.inverse_transform(y_test_true_full)

        # Obtener y_true y y_pred para VALIDATION (paso a paso)
        df_val_scaled = scaled_data[cuenca]['df'].iloc[scaled_data[cuenca]['val_idx']].copy()
        _, y_val_pred_original, y_val_true_original = evaluate_validation(
            models[cuenca], df_val_scaled, scaler_area, exog_cols_original, n_lags_area, n_exog_features
        )

        # Concatenar todos los valores reales y predichos
        y_true_full_combined = np.vstack([y_train_true_original, y_test_true_original, y_val_true_original])
        y_pred_full_combined = np.vstack([y_train_pred_original, y_test_pred_original, y_val_pred_original])

        # Calcular métricas en todo el conjunto de datos
        r2_full = r2_score(y_true_full_combined, y_pred_full_combined)
        mae_full = mean_absolute_error(y_true_full_combined, y_pred_full_combined)
        nse_full = nash_sutcliffe_efficiency(y_true_full_combined, y_pred_full_combined)
        kge_full = kling_gupta_efficiency(y_true_full_combined, y_pred_full_combined)

        full_metrics[cuenca] = {'R2': r2_full, 'MAE': mae_full, 'NSE': nse_full, 'KGE': kge_full}
    return full_metrics


# --- Bucle de experimentación principal ---

df = pd.read_csv('df_all.csv', index_col=0)

# Parámetros fijos
exog_cols_scaled = ['temperatura_scaled', 'precipitacion_scaled', 'dias_sin_precip_scaled']
exog_cols_original = ['temperatura', 'precipitacion', 'dias_sin_precip'] # Para evaluate_validation
n_features = 1 + len(exog_cols_scaled)

#%% Hiperparámetros a probar
param_grid = {
    'n_lags_area': [3, 5],
    'n_units_lstm': [32, 50, 64],
    'num_lstm_layers': [1, 2],
    'dropout_rate': [0.1],
    'epochs': [50, 100],
    'learning_rate': [0.001]
}

#%% Preprocesar datos una sola vez
print("Preprocesando datos...")
scaled_data_global, scalers_global, cuencas_global = preprocess_data(df)
print("Datos preprocesados.")

# Crear secuencias para cada cuenca una sola vez (esto puede ser costoso)
print("Creando secuencias para todas las cuencas...")
sequences_data_global = {}
for cuenca, data_indices in scaled_data_global.items():
    train_data = data_indices['df'].iloc[data_indices['train_idx']]
    val_data = data_indices['df'].iloc[data_indices['val_idx']]
    test_data = data_indices['df'].iloc[data_indices['test_idx']]

    sequences_data_global[cuenca] = {
        'X_train': create_sequences(train_data, param_grid['n_lags_area'][0], exog_cols_scaled)[0], # Usar el primer lag para la creación de secuencias para todos los experimentos, esto es un punto a considerar. Si n_lags_area varía, las secuencias deben crearse dentro del bucle.
        'y_train': create_sequences(train_data, param_grid['n_lags_area'][0], exog_cols_scaled)[1],
        'X_val_initial': create_sequences(val_data, param_grid['n_lags_area'][0], exog_cols_scaled)[0], # Esto es solo para la forma, no para la predicción step-by-step
        'y_val_initial': create_sequences(val_data, param_grid['n_lags_area'][0], exog_cols_scaled)[1],
        'X_test': create_sequences(test_data, param_grid['n_lags_area'][0], exog_cols_scaled)[0],
        'y_test': create_sequences(test_data, param_grid['n_lags_area'][0], exog_cols_scaled)[1],
    }
print("Secuencias creadas.")

# Lista para almacenar los resultados de todos los experimentos
all_experiment_results = []
models_dir_base = 'experiment_models'

#%% Generar todas las combinaciones de hiperparámetros
keys = param_grid.keys()
combinations = list(itertools.product(*param_grid.values()))
keys
#%%
print(f"\nIniciando experimentos con {len(combinations)} combinaciones de hiperparámetros...")

for i, combo in enumerate(combinations):
    current_params = dict(zip(keys, combo))
    n_lags_current = current_params['n_lags_area'] # Usar el lag actual para la creación de secuencias
    print(f"\n--- Experimento {i+1}/{len(combinations)} ---")
    print(f"Parámetros actuales: {current_params}")

    current_experiment_results = {'params': current_params, 'metrics_by_cuenca': {}}
    experiment_models = {}

    for cuenca in cuencas_global:
        print(f"  Procesando cuenca: {cuenca}")
        
        # Crear secuencias específicas para el n_lags_area de este experimento
        train_data = scaled_data_global[cuenca]['df'].iloc[scaled_data_global[cuenca]['train_idx']]
        X_train, y_train = create_sequences(train_data, n_lags_current, exog_cols_scaled)
        
        test_data = scaled_data_global[cuenca]['df'].iloc[scaled_data_global[cuenca]['test_idx']]
        X_test, y_test = create_sequences(test_data, n_lags_current, exog_cols_scaled)


        # # Crear y entrenar el modelo para la configuración actual
        # model = create_narx_model(
        #     n_lags=n_lags_current,
        #     n_features=n_features,
        #     n_units_lstm=current_params['n_units_lstm'],
        #     num_lstm_layers=current_params['num_lstm_layers'],
        #     dropout_rate=current_params['dropout_rate'],
        #     learning_rate=current_params['learning_rate']
        # )
        
        # print(f"    Entrenando modelo para {cuenca}...")
        # _ = train_model(model, X_train, y_train, current_params['epochs'])
        # experiment_models[cuenca] = model

        # # Guardar el modelo
        models_dir_exp = os.path.join(models_dir_base, f"exp_{i+1}")
        # if not os.path.exists(models_dir_exp):
        #     os.makedirs(models_dir_exp)
        # model.save(os.path.join(models_dir_exp, f'narx_model_{cuenca}.h5'))

        model = load(os.path.join(models_dir_exp, f'narx_model_{cuenca}.h5'))

        # Evaluar en train y test
        train_metrics, _, _ = evaluate_model(model, X_train, y_train, scalers_global[cuenca]['area'])
        test_metrics, _, _ = evaluate_model(model, X_test, y_test, scalers_global[cuenca]['area'])

        # Evaluar en validación (modo predicción)
        df_val_scaled = scaled_data_global[cuenca]['df'].iloc[scaled_data_global[cuenca]['val_idx']].copy()
        validation_metrics, _, _ = evaluate_validation(
            model, df_val_scaled, scalers_global[cuenca]['area'], exog_cols_original, n_lags_current, len(exog_cols_scaled)
        )


        # Evaluar en todo el conjunto de datos (llamando a la función dedicada para esto)
        # Para evitar re-evaluaciones costosas, podemos recopilar todos los y_true y y_pred aquí
        # y luego llamar a evaluate_full_dataset una vez al final del experimento para todos los modelos.
        # Simplificación: Pasamos el modelo entrenado y los datos globales

        # full_metrics = evaluate_full_dataset({cuenca: model}, scaled_data_global, scalers_global, [cuenca], n_lags_current, exog_cols_scaled)

        current_experiment_results['metrics_by_cuenca'][cuenca] = {
            'train': train_metrics,
            'test': test_metrics,
            'validation': validation_metrics,
            # 'full_dataset': full_metrics[cuenca]
        }
        print(f"      Métricas para {cuenca}:")
        print(f"        Train: {train_metrics}")
        print(f"        Test: {test_metrics}")
        print(f"        Validation: {validation_metrics}")
        # print(f"        Full Dataset: {full_metrics[cuenca]}")

    all_experiment_results.append(current_experiment_results)

#%% Guardar los resultados de todos los experimentos en un archivo JSON
results_file = 'experiment_results.json'
with open(results_file, 'w') as f:
    json.dump(all_experiment_results, f, indent=4)

print(f"\nTodos los experimentos completados. Resultados guardados en '{results_file}'")

# --- Análisis de resultados (Ejemplo) ---
print("\n--- Resumen de los mejores resultados por cuenca (basado en KGE de validación) ---")
best_configs_by_cuenca = {}

for cuenca in cuencas_global:
    best_kge = -np.inf
    best_config = None
    best_metrics = None

    for exp_result in all_experiment_results:
        metrics = exp_result['metrics_by_cuenca'][cuenca]
        if metrics['validation']['KGE'] > best_kge:
            best_kge = metrics['validation']['KGE']
            best_config = exp_result['params']
            best_metrics = metrics

    best_configs_by_cuenca[cuenca] = {
        'best_params': best_config,
        'best_validation_kge': best_kge,
        'all_metrics': best_metrics
    }
    print(f"\nCuenca: {cuenca}")
    print(f"  Mejor KGE de Validación: {best_configs_by_cuenca[cuenca]['best_validation_kge']:.3f}")
    print(f"  Mejores Parámetros: {best_configs_by_cuenca[cuenca]['best_params']}")
    print(f"  Métricas Completas para la mejor configuración:")
    for key, val in best_configs_by_cuenca[cuenca]['all_metrics'].items():
        print(f"    {key}: {val}")