# IMPORTS (asegúrate de que todos los imports necesarios estén al principio del script completo)
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

# FUNCIONES (Estas funciones no necesitan cambios internos significativos)
# Asegúrate de que todas las funciones auxiliares (nash_sutcliffe_efficiency, kling_gupta_efficiency,
# preprocess_data, create_sequences, create_narx_model, load_model_for_basin, evaluate_model,
# evaluate_validation) estén definidas antes de esta sección.
# La función 'evaluate_full_dataset' necesita una pequeña modificación para recibir 'cuenca_name'.

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
    lstm_activation = 'tanh'

    if n_layers > 1:
        model.add(LSTM(n_units_lstm, activation=lstm_activation, input_shape=(n_lags, n_features), return_sequences=True))
    else:
        model.add(LSTM(n_units_lstm, activation=lstm_activation, input_shape=(n_lags, n_features)))

    for _ in range(1, n_layers):
        if _ == n_layers - 1:
            model.add(LSTM(n_units_lstm, activation=lstm_activation))
        else:
            model.add(LSTM(n_units_lstm, activation=lstm_activation, return_sequences=True))

    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5)
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

        pred_scaled = model.predict(last_sequence, verbose=0)

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

# Modificación en la función evaluate_full_dataset:
def evaluate_full_dataset(model, df_full_scaled_cuenca, scaler_area, exog_cols_scaled, n_lags_area, graph=False, output_dir='./', cuenca_name=""):
    """
    Evalúa un modelo específico en el conjunto de datos completo (modo predicción).
    Ahora toma 'output_dir' para guardar gráficos y el nombre de la cuenca.
    """
    full_metrics = {}

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
            print(f"Warning: Input sequence for prediction contains NaN/inf at step {i} for {cuenca_name}. Stopping full prediction.")
            y_full_pred_scaled.extend([np.nan] * (len(df_full_scaled_cuenca) - n_lags_area - i))
            break

        pred_scaled = model.predict(last_sequence_full, verbose=0)

        if np.any(np.isnan(pred_scaled)) or np.any(np.isinf(pred_scaled)):
            print(f"Warning: Model predicted NaN/inf at step {i} for {cuenca_name}. Stopping full prediction.")
            y_full_pred_scaled.append(np.nan)
            y_full_pred_scaled.extend([np.nan] * (len(df_full_scaled_cuenca) - n_lags_area - (i + 1)))
            break

        y_full_pred_scaled.append(pred_scaled[0, 0])

        next_area_scaled = pred_scaled.reshape(1, 1, 1)
        next_exog_scaled = df_full_scaled_cuenca[exog_cols_scaled].iloc[n_lags_area + i].values.reshape(1, 1, n_exog_features)

        updated_area_sequence = np.concatenate([last_sequence_full[:, 1:, 0].reshape(1, n_lags_area - 1, 1), next_area_scaled], axis=1)
        updated_exog_sequence = np.concatenate([last_sequence_full[:, 1:, 1:], next_exog_scaled], axis=1)
        last_sequence_full = np.concatenate([updated_area_sequence, updated_exog_sequence], axis=2)
    
    if not y_full_pred_scaled or np.any(np.isnan(y_full_pred_scaled)) or np.any(np.isinf(y_full_pred_scaled)):
        print(f"Warning: Final predictions for {cuenca_name} contain NaN/inf. Skipping metric calculation and plotting.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    y_full_pred_original = scaler_area.inverse_transform(np.array(y_full_pred_scaled).reshape(-1, 1))

    if np.any(np.isnan(y_full_pred_original)) or np.any(np.isinf(y_full_pred_original)):
        print(f"Warning: Inverse transformed predictions for {cuenca_name} contain NaN/inf. Skipping metric calculation and plotting.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    r2_full = pearsonr(y_full_true_original.flatten(), y_full_pred_original.flatten())
    r2_full = r2_full.statistic**2
    mae_full = mean_absolute_error(y_full_true_original, y_full_pred_original)
    nse_full = nash_sutcliffe_efficiency(y_full_true_original, y_full_pred_original)
    kge_full = kling_gupta_efficiency(y_full_true_original, y_full_pred_original)

    full_metrics = {'R2': r2_full, 'MAE': mae_full, 'NSE': nse_full, 'KGE': kge_full}
    print(f"Métricas en todo el conjunto de datos (modo prediccion) para {cuenca_name}: R2={r2_full:.3f}, MAE={mae_full:.3f}, NSE={nse_full:.3f}, KGE={kge_full:.3f}")

    if graph == True:
        graph_types = ['per_day', 'per_month', 'all_days']
        # Asegúrate de que el directorio de salida para los gráficos exista
        output_graph_path = os.path.join(output_dir, f'graphs_{cuenca_name}')
        os.makedirs(output_graph_path, exist_ok=True)

        for graph_type in graph_types:
            real_plot = df_full_scaled_cuenca.iloc[n_lags_area:].copy()
            real_plot['area_nieve'] = y_full_true_original
            y_full_pred_df = pd.DataFrame(y_full_pred_original, columns=['area_nieve_pred'], index=real_plot.index)
            df_plot = pd.concat([real_plot, y_full_pred_df], axis=1)
            df_plot['fecha'] = pd.to_datetime(df_plot['fecha'])

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
            plt.savefig(os.path.join(output_graph_path, f'{graph_type}.png'))
            plt.close()
    return full_metrics


# --- Optuna Objective Function (Optimizado para una sola cuenca) ---
def objective_single_basin(trial, basin_data, basin_scalers, exog_cols, exog_cols_scaled):
    # 1. Suggest Hyperparameters
    n_lags_area = trial.suggest_int('n_lags_area', 2, 7)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    n_units_lstm = trial.suggest_int('n_units_lstm', 5, 30)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    epochs = trial.suggest_int('epochs', 15, 120)

    # 2. Prepare data for the current trial
    n_features = 1 + len(exog_cols_scaled)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    # Extract data for the single basin
    train_data = basin_data['df'].iloc[basin_data['train_idx']]
    val_data = basin_data['df'].iloc[basin_data['val_idx']]
    scaler_area = basin_scalers['area']

    # Create sequences for the current basin with the trial's n_lags_area
    X_train, y_train = create_sequences(train_data, n_lags_area, exog_cols_scaled)
    
    # --- Try-except block for robust trial execution ---
    try:
        if X_train.shape[0] == 0:
            print(f"Warning: Insufficient training sequences for n_lags={n_lags_area}. Assigning -inf NSE.")
            return -np.inf # Return a very bad NSE if no sequences

        # Create and train the model
        model = create_narx_model(n_lags=n_lags_area, n_layers=n_layers,
                                    n_units_lstm=n_units_lstm, n_features=n_features,
                                    learning_rate=learning_rate, dropout_rate=dropout_rate)
        
        model.fit(X_train, y_train, epochs=epochs, verbose=0,
                    validation_split=0.1, callbacks=[early_stopping_callback]) # Use internal validation split for early stopping

        # Evaluate on the actual validation set (walk-forward)
        df_val_scaled = basin_data['df'].iloc[basin_data['val_idx']].copy()
        val_metrics, _, _ = evaluate_validation(model, df_val_scaled, scaler_area, exog_cols, n_lags_area)

        # If validation metrics contain NaN/inf, assign a very bad value
        if np.any(np.isnan(list(val_metrics.values()))) or np.any(np.isinf(list(val_metrics.values()))):
            return -np.inf
        else:
            return val_metrics['NSE']

    except (ValueError, tf.errors.InvalidArgumentError, tf.errors.ResourceExhaustedError, RuntimeError) as e:
        print(f"Trial {trial.number} failed with error: {e}. Returning -inf NSE.")
        return -np.inf # Assign a very low NSE to penalize this trial

# Función auxiliar para convertir tipos de NumPy a tipos de Python nativos
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

# --- Main execution ---
# Define the directory where your basin CSVs are located
basins_dir = 'datasets/'
models_dir = os.path.join("E:", "new_models")

exog_cols = ["dia_sen","temperatura","precipitacion", "dias_sin_precip"]
exog_cols_scaled = [col + '_scaled' for col in exog_cols]

# Discover all basin CSVs
basin_files = [f for f in os.listdir(basins_dir) if f.endswith('.csv')]
cuencas_all = [os.path.splitext(f)[0] for f in basin_files] # Extract basin names

# Preprocess all data upfront and store in a dictionary
all_basins_preprocessed_data = {}
for cuenca_name in cuencas_all:
    print(f"Preprocessing data for basin: {cuenca_name}")
    df_basin = pd.read_csv(os.path.join(basins_dir, f'{cuenca_name}.csv'), index_col=0)
    if 'fecha' in df_basin.columns:
        df_basin['fecha'] = df_basin['fecha'].astype(str)

    try:
        basin_data, basin_scalers = preprocess_data(df_basin.copy(), exog_cols)
        all_basins_preprocessed_data[cuenca_name] = {'data': basin_data, 'scalers': basin_scalers}
    except ValueError as e:
        print(f"Error during preprocessing for {cuenca_name}: {e}. Skipping this basin.")
        continue # Skip this basin if preprocessing fails

# Filter out basins that failed preprocessing
cuencas_to_process = list(all_basins_preprocessed_data.keys())

# Dictionary to store the best trial/model info for each basin
best_models_per_basin = {}
cuencas_to_process = ['genil-dilar']

# --- Loop through each basin for separate Optuna optimization and evaluation ---
for cuenca_name in cuencas_to_process:
    print(f"\n--- Starting Optuna Optimization for Cuenca: {cuenca_name} ---")

    # Define the output directory for this specific basin's best model and metrics
    basin_output_dir = os.path.join(models_dir, cuenca_name)
    model_path = os.path.join(basin_output_dir, f'narx_model_best_{cuenca_name}.h5')
    
    os.makedirs(basin_output_dir, exist_ok=True) # Asegura que el directorio de la cuenca exista

    # Create a partial objective function for the current basin
    objective_for_this_basin = partial(objective_single_basin,
                                       basin_data=all_basins_preprocessed_data[cuenca_name]['data'],
                                       basin_scalers=all_basins_preprocessed_data[cuenca_name]['scalers'],
                                       exog_cols=exog_cols,
                                       exog_cols_scaled=exog_cols_scaled)

    # Create an Optuna study for this specific basin
    study_basin = optuna.create_study(direction='maximize',
                                       sampler=optuna.samplers.TPESampler(),
                                       pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
                                       study_name=f"basin_optimization_{cuenca_name}")

    # Run the optimization for this basin
    n_trials_per_basin = 40 # Adjust as needed
    study_basin.optimize(objective_for_this_basin, n_trials=n_trials_per_basin, show_progress_bar=True)

    print(f"\n--- Optuna Optimization Results for {cuenca_name} ---")
    print(f"Number of finished trials: {len(study_basin.trials)}")
    
    current_basin_metrics = {} # Diccionario para las métricas de la cuenca actual

    if study_basin.best_trial is not None:
        best_trial_basin = study_basin.best_trial
        print(f"  Best Value (NSE): {best_trial_basin.value:.4f}")
        print("  Best Params: ")
        for key, value in best_trial_basin.params.items():
            print(f"    {key}: {value}")

        # Store best parameters and train the final model for this basin
        best_params_basin = best_trial_basin.params
        best_n_lags_area = best_params_basin['n_lags_area']
        best_n_layers = best_params_basin['n_layers']
        best_n_neuronas = best_params_basin['n_units_lstm']
        best_learning_rate = best_params_basin['learning_rate']
        best_dropout_rate = best_params_basin['dropout_rate']
        epochs_final_train = 100 # Max epochs for final training, EarlyStopping will take care

        print(f"\n--- Training final model for {cuenca_name} with best hyperparameters ---")
        basin_data_final = all_basins_preprocessed_data[cuenca_name]['data']
        X_train_final, y_train_final = create_sequences(basin_data_final['df'].iloc[basin_data_final['train_idx']],
                                                         best_n_lags_area, exog_cols_scaled)

        if X_train_final.shape[0] == 0:
            print(f"Skipping final training for {cuenca_name}: No training sequences for best_n_lags_area={best_n_lags_area}.")
            best_models_per_basin[cuenca_name] = {'model': None, 'best_n_lags_area': best_n_lags_area}
            
            # Asignar NaNs a las métricas si no hay entrenamiento
            current_basin_metrics = {
                'best_params': best_params_basin,
                'full_dataset': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
                'train': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
                'test': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
                'val': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
            }
            # Guardar el JSON para esta cuenca (incluso si tiene NaNs)
            json_output_path = os.path.join(basin_output_dir, 'metrics.json')
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(convert_numpy_to_python(current_basin_metrics), f, indent=4)
            print(f"Métricas (con NaNs) guardadas para {cuenca_name} en {json_output_path}")
            continue # Skip to next basin

        model_final = create_narx_model(n_lags=best_n_lags_area, n_layers=best_n_layers,
                                         n_units_lstm=best_n_neuronas, n_features=(1 + len(exog_cols_scaled)),
                                         learning_rate=best_learning_rate, dropout_rate=best_dropout_rate)
        
        # Use ModelCheckpoint to save the best model weights during training
        model_checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(basin_output_dir, f'narx_model_best_{cuenca_name}.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        early_stopping_callback_final = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )

        model_final.fit(X_train_final, y_train_final, epochs=epochs_final_train, verbose=0,
                         validation_split=0.1, callbacks=[early_stopping_callback_final, model_checkpoint_callback])

        # Load the best saved model if checkpoint was used, otherwise use the one from last epoch
        final_trained_model_path = os.path.join(basin_output_dir, f'narx_model_best_{cuenca_name}.h5')
        if os.path.exists(final_trained_model_path):
            final_trained_model = keras.models.load_model(final_trained_model_path)
            print(f"Cargado el mejor modelo guardado para {cuenca_name}.")
        else:
            final_trained_model = model_final # Use the model from the last epoch if no best was saved
            print(f"No se encontró el mejor modelo guardado para {cuenca_name}, usando el de la última época.")

        best_models_per_basin[cuenca_name] = {'model': final_trained_model,
                                               'best_n_lags_area': best_n_lags_area,
                                               'best_params': best_params_basin} # Store best params too

        # --- Evaluación del modelo para la cuenca actual ---
        current_scaler_area = all_basins_preprocessed_data[cuenca_name]['scalers']['area']
        
        # Prepara las secuencias para esta cuenca usando su mejor n_lags_area
        basin_data_eval = all_basins_preprocessed_data[cuenca_name]['data']
        sequences_for_eval = {
            'train': {'X': create_sequences(basin_data_eval['df'].iloc[basin_data_eval['train_idx']], best_n_lags_area, exog_cols_scaled)[0],
                      'y': create_sequences(basin_data_eval['df'].iloc[basin_data_eval['train_idx']], best_n_lags_area, exog_cols_scaled)[1]},
            'test': {'X': create_sequences(basin_data_eval['df'].iloc[basin_data_eval['test_idx']], best_n_lags_area, exog_cols_scaled)[0],
                     'y': create_sequences(basin_data_eval['df'].iloc[basin_data_eval['test_idx']], best_n_lags_area, exog_cols_scaled)[1]},
            'val_df': basin_data_eval['df'].iloc[basin_data_eval['val_idx']].copy() # Keep as DataFrame for evaluate_validation
        }

        metrics_train, _, _ = evaluate_model(final_trained_model, sequences_for_eval['train'], current_scaler_area)
        metrics_test, _, _ = evaluate_model(final_trained_model, sequences_for_eval['test'], current_scaler_area)
        metrics_val, _, _ = evaluate_validation(final_trained_model, sequences_for_eval['val_df'], current_scaler_area, exog_cols, best_n_lags_area)
        
        # Llama a evaluate_full_dataset y pasa el directorio de salida específico de la cuenca
        metrics_full = evaluate_full_dataset(final_trained_model, basin_data_eval['df'], current_scaler_area,
                                             exog_cols_scaled, best_n_lags_area, graph=True,
                                             output_dir=basin_output_dir, cuenca_name=cuenca_name)

        current_basin_metrics = {
            'best_params': best_params_basin,
            'full_dataset': metrics_full,
            'train': metrics_train,
            'test': metrics_test,
            'val': metrics_val
        }
        print(f"Métricas finales para {cuenca_name}:")
        print(f"  Full Dataset (modo prediccion): {current_basin_metrics['full_dataset']}")
        print(f"  Train: {current_basin_metrics['train']}")
        print(f"  Test: {current_basin_metrics['test']}")
        print(f"  Validation (modo prediccion): {current_basin_metrics['val']}")

        # Guarda las métricas de esta cuenca en su propio archivo JSON
        json_output_path = os.path.join(basin_output_dir, 'metrics.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_to_python(current_basin_metrics), f, indent=4)
        print(f"Métricas guardadas para {cuenca_name} en {json_output_path}")

    else:
        print(f"No successful trials for Cuenca: {cuenca_name}. Skipping model training and evaluation.")
        best_models_per_basin[cuenca_name] = {'model': None, 'best_n_lags_area': None}
        # Guardar un JSON con NaNs si no hay trials exitosos
        current_basin_metrics = {
            'best_params': None,
            'full_dataset': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
            'train': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
            'test': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
            'val': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
        }
        json_output_path = os.path.join(basin_output_dir, 'metrics.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_to_python(current_basin_metrics), f, indent=4)
        print(f"Métricas (con NaNs) guardadas para {cuenca_name} en {json_output_path}")

# El guardado de 'all_final_metrics.json' ya no es necesario si las métricas se guardan individualmente.
# Puedes eliminar las siguientes líneas:
# with open(archivo_json_total, 'w', encoding='utf-8') as f:
#     json.dump(convert_numpy_to_python(all_final_metrics), f, indent=4)
# print(f"\nMétricas finales para todas las cuencas guardadas en {archivo_json_total}")

print("\nProceso de optimización y evaluación por cuenca completado.")