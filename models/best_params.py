# IMPORTS (ensure all necessary imports are at the top)
import pandas as pd
from sklearn.preprocessing import StandardScaler # or MinMaxScaler
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

# FUNCIONES (no changes needed here, as the fix is in the objective)
def nash_sutcliffe_efficiency(y_true, y_pred):
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    if denominator == 0:
        return np.nan # Return NaN if denominator is zero
    return 1 - (numerator / denominator)

def kling_gupta_efficiency(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Handle cases where std or mean might be zero/NaN, or correlation is NaN
    if np.std(y_true_flat) == 0 or np.std(y_pred_flat) == 0 or np.mean(y_true_flat) == 0:
        return -np.inf # Return a very bad KGE if standard deviation or mean are problematic

    r_val, _ = pearsonr(y_true_flat, y_pred_flat)
    if np.isnan(r_val): # If correlation itself is NaN (e.g., due to constant predictions)
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

    # Use MinMaxScaler for better stability with LSTMs, scaling to (-1, 1) or (0, 1)
    # Using StandardScaler is also fine if the data is well-behaved, but MinMaxScaler is often more robust to outliers
    scaler_area = StandardScaler() # Changed back to StandardScaler as per user's original preference.
    scaler_exog = StandardScaler() # If NaN/inf issues persist, consider MinMaxScaler.

    # Ensure train_idx is not empty, which can happen with very small datasets
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
    print(f"Cuenca processed: Train: {len(train_idx)}, Test: {len(test_idx)}, Val: {len(val_idx)}")

    return basin_data, basin_scalers

def create_sequences(data, n_lags, exog_cols_scaled, target_col_scaled='area_nieve_scaled'):
    X, y = [], []
    # Ensure there's enough data for at least one sequence
    if len(data) < n_lags + 1:
        # print(f"Warning: Not enough data ({len(data)}) to create sequences with n_lags={n_lags}. Returning empty arrays.")
        return np.array([]).reshape(0, n_lags, len(exog_cols_scaled) + 1), np.array([]).reshape(0, 1)

    for i in range(len(data) - n_lags):
        seq_area = data[target_col_scaled].iloc[i : i + n_lags].values
        seq_exog = data[exog_cols_scaled].iloc[i : i + n_lags].values
        seq = np.hstack((seq_area.reshape(-1, 1), seq_exog))
        X.append(seq)
        y.append(data[target_col_scaled].iloc[i + n_lags])
    
    # Handle cases where X or y might be empty (e.g., if loop doesn't run)
    if not X:
        return np.array([]).reshape(0, n_lags, len(exog_cols_scaled) + 1), np.array([]).reshape(0, 1)
        
    return np.array(X), np.array(y).reshape(-1, 1)

# Modified create_narx_model with tanh activation and adjusted clipnorm
def create_narx_model(n_lags, n_layers, n_units_lstm, n_features, learning_rate):
    model = Sequential()

    # Use 'tanh' activation for LSTM layers for better stability
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

    model.add(Dropout(0.2)) # Increased dropout for stability. Can be an Optuna param.
    model.add(Dense(1))

    # Try a more aggressive clipnorm to prevent exploding gradients
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5) # Reduced clipnorm
    model.compile(optimizer=optimizer, loss='mse')

    return model

# create_train_models for final training (not for Optuna objective)
def create_train_models(sequences_data, n_lags_area, layers, units, epochs, exog_cols_scaled, cuencas_list, save, models_dir, learning_rate):
    models = {}
    history = {}
    n_features = 1 + len(exog_cols_scaled)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    for cuenca in cuencas_list:
        print(f"Training model for cuenca: {cuenca}")
        model = create_narx_model(n_lags = n_lags_area, n_layers=layers, n_units_lstm=units, n_features=n_features, learning_rate=learning_rate)
        X_train = sequences_data[cuenca]['X_train']
        y_train = sequences_data[cuenca]['y_train']

        if X_train.shape[0] == 0:
            print(f"Skipping training for {cuenca}: No training sequences available.")
            continue # Skip training if no sequences

        model_history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_split=0.1, callbacks=[early_stopping_callback])
        models[cuenca] = model
        history[cuenca] = model_history

        model_path = os.path.join(models_dir, f'narx_model_{cuenca}.h5')
        if save:
            model.save(model_path)
            print(f"Modelo entrenado y guardado para la cuenca: {cuenca} en {model_path}")
        else:
            print(f'Modelo entrenado para la cuenca {cuenca}')
    return models

def load_models(cuencas_list, models_dir='models'):
    loaded_models = {}
    for cuenca in cuencas_list:
        model_path = os.path.join(models_dir, f'narx_model_{cuenca}.h5')
        if os.path.exists(model_path):
            loaded_models[cuenca] = keras.models.load_model(model_path)
            print(f"Modelo cargado para la cuenca: {cuenca} desde {model_path}")
        else:
            print(f"No se encontró el modelo para la cuenca: {cuenca} en {model_path}")
    return loaded_models

def evaluate_model(model, sequences, scaler_area):
    X = sequences['X']
    y_true = sequences['y']
    
    if X.shape[0] == 0: # Handle empty sequences
        print("Warning: Empty sequences for evaluation.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    y_pred_scaled = model.predict(X, verbose=0)

    # Check for NaN/inf in predictions before inverse transform
    if np.any(np.isnan(y_pred_scaled)) or np.any(np.isinf(y_pred_scaled)):
        print("Warning: Model predicted NaN/inf values. Returning NaN for metrics.")
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

    # If validation data itself is too short for any sequence
    if len(df_val_scaled) < n_lags_area + 1:
        print(f"Warning: Validation data for walk-forward ({len(df_val_scaled)}) is too short for n_lags={n_lags_area}. Returning NaN.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    y_val_true_original = df_val_scaled['area_nieve'].values[n_lags_area:].reshape(-1, 1)
    y_val_pred_scaled = []

    first_sequence_data = df_val_scaled[['area_nieve_scaled'] + [col + '_scaled' for col in exog_cols]].iloc[:n_lags_area].values
    last_sequence = first_sequence_data.reshape(1, n_lags_area, -1)

    for i in range(len(df_val_scaled) - n_lags_area):
        # Check if last_sequence contains NaN/inf before prediction
        if np.any(np.isnan(last_sequence)) or np.any(np.isinf(last_sequence)):
            print(f"Warning: Input sequence for prediction contains NaN/inf at step {i}. Stopping prediction for this basin.")
            y_val_pred_scaled = [np.nan] * (len(df_val_scaled) - n_lags_area) # Fill remaining with NaN
            break # Exit the loop

        pred_scaled = model.predict(last_sequence, verbose=0)

        # Check for NaN/inf in the prediction itself
        if np.any(np.isnan(pred_scaled)) or np.any(np.isinf(pred_scaled)):
            print(f"Warning: Model predicted NaN/inf at step {i}. Stopping prediction for this basin.")
            y_val_pred_scaled.append(np.nan) # Append current NaN prediction
            y_val_pred_scaled.extend([np.nan] * (len(df_val_scaled) - n_lags_area - (i + 1))) # Fill rest with NaN
            break # Exit the loop

        y_val_pred_scaled.append(pred_scaled[0, 0])

        next_area_scaled = pred_scaled.reshape(1, 1, 1)
        next_exog_scaled = df_val_scaled[[col + '_scaled' for col in exog_cols]].iloc[n_lags_area + i].values.reshape(1, 1, n_exog_features)

        updated_area_sequence = np.concatenate([last_sequence[:, 1:, 0].reshape(1, n_lags_area - 1, 1), next_area_scaled], axis=1)
        updated_exog_sequence = np.concatenate([last_sequence[:, 1:, 1:], next_exog_scaled], axis=1)
        last_sequence = np.concatenate([updated_area_sequence, updated_exog_sequence], axis=2)

    # If y_val_pred_scaled is empty or contains NaNs, return NaN for metrics
    if not y_val_pred_scaled or np.any(np.isnan(y_val_pred_scaled)) or np.any(np.isinf(y_val_pred_scaled)):
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    y_val_pred_original = scaler_area.inverse_transform(np.array(y_val_pred_scaled).reshape(-1, 1))

    # Final check before calculating metrics
    if np.any(np.isnan(y_val_pred_original)) or np.any(np.isinf(y_val_pred_original)):
        print("Warning: Inverse transformed predictions contain NaN/inf. Returning NaN for metrics.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    r2_val = pearsonr(y_val_true_original.flatten(), y_val_pred_original.flatten())
    r2_val = r2_val.statistic**2
    mae_val = mean_absolute_error(y_val_true_original, y_val_pred_original)
    nse_val = nash_sutcliffe_efficiency(y_val_true_original, y_val_pred_original)
    kge_val = kling_gupta_efficiency(y_val_true_original, y_val_pred_original)

    return {'R2': r2_val, 'MAE': mae_val, 'NSE': nse_val, 'KGE': kge_val}, y_val_pred_original, y_val_true_original

def evaluate_full_dataset(models, scaled_data, scalers, cuencas_list, n_lags_area, exog_cols_scaled, graph=False, model_dir='./'):
    full_metrics = {}

    for cuenca in cuencas_list:
        model = models.get(cuenca) # Use .get() in case a model wasn't trained for this cuenca
        if model is None:
            print(f"Skipping full dataset evaluation for {cuenca}: No model found.")
            full_metrics[cuenca] = {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
            continue

        scaler_area = scalers[cuenca]['area']
        df_full_scaled_cuenca = scaled_data[cuenca]['df'].copy()
        n_exog_features = len(exog_cols_scaled)

        # Ensure enough data for full evaluation
        if len(df_full_scaled_cuenca) < n_lags_area + 1:
            print(f"Warning: Full dataset for {cuenca} is too short for n_lags={n_lags_area}. Skipping full evaluation.")
            full_metrics[cuenca] = {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
            continue

        y_full_true_original = df_full_scaled_cuenca['area_nieve'].values[n_lags_area:].reshape(-1, 1)
        y_full_pred_scaled = []

        first_sequence_full = df_full_scaled_cuenca[['area_nieve_scaled'] + exog_cols_scaled].iloc[:n_lags_area].values.reshape(1, n_lags_area, -1)
        last_sequence_full = first_sequence_full.copy()

        # Realizar la predicción paso a paso para todo el resto del DataFrame
        for i in range(len(df_full_scaled_cuenca) - n_lags_area):
            if np.any(np.isnan(last_sequence_full)) or np.any(np.isinf(last_sequence_full)):
                print(f"Warning: Input sequence for prediction contains NaN/inf at step {i} for {cuenca}. Stopping full prediction.")
                y_full_pred_scaled.extend([np.nan] * (len(df_full_scaled_cuenca) - n_lags_area - i))
                break

            pred_scaled = model.predict(last_sequence_full, verbose=0)

            if np.any(np.isnan(pred_scaled)) or np.any(np.isinf(pred_scaled)):
                print(f"Warning: Model predicted NaN/inf at step {i} for {cuenca}. Stopping full prediction.")
                y_full_pred_scaled.append(np.nan)
                y_full_pred_scaled.extend([np.nan] * (len(df_full_scaled_cuenca) - n_lags_area - (i + 1)))
                break

            y_full_pred_scaled.append(pred_scaled[0, 0])

            next_area_scaled = pred_scaled.reshape(1, 1, 1)
            next_exog_scaled = df_full_scaled_cuenca[exog_cols_scaled].iloc[n_lags_area + i].values.reshape(1, 1, n_exog_features)

            updated_area_sequence = np.concatenate([last_sequence_full[:, 1:, 0].reshape(1, n_lags_area - 1, 1), next_area_scaled], axis=1)
            updated_exog_sequence = np.concatenate([last_sequence_full[:, 1:, 1:], next_exog_scaled], axis=1)
            last_sequence_full = np.concatenate([updated_area_sequence, updated_exog_sequence], axis=2)
        
        # Check if predictions are valid before calculating metrics
        if not y_full_pred_scaled or np.any(np.isnan(y_full_pred_scaled)) or np.any(np.isinf(y_full_pred_scaled)):
            print(f"Warning: Final predictions for {cuenca} contain NaN/inf. Skipping metric calculation and plotting.")
            full_metrics[cuenca] = {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
            continue

        y_full_pred_original = scaler_area.inverse_transform(np.array(y_full_pred_scaled).reshape(-1, 1))

        # Final check before calculating metrics
        if np.any(np.isnan(y_full_pred_original)) or np.any(np.isinf(y_full_pred_original)):
            print(f"Warning: Inverse transformed predictions for {cuenca} contain NaN/inf. Skipping metric calculation and plotting.")
            full_metrics[cuenca] = {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
            continue

        r2_full = pearsonr(y_full_true_original.flatten(), y_full_pred_original.flatten())
        r2_full = r2_full.statistic**2
        mae_full = mean_absolute_error(y_full_true_original, y_full_pred_original)
        nse_full = nash_sutcliffe_efficiency(y_full_true_original, y_full_pred_original)
        kge_full = kling_gupta_efficiency(y_full_true_original, y_full_pred_original)

        full_metrics[cuenca] = {'R2': r2_full, 'MAE': mae_full, 'NSE': nse_full, 'KGE': kge_full}
        print(f"Métricas en todo el conjunto de datos (modo prediccion) para {cuenca}: R2={r2_full:.3f}, MAE={mae_full:.3f}, NSE={nse_full:.3f}, KGE={kge_full:.3f}")

        if graph == True:
            graph_types = ['per_day', 'per_month', 'all_days']
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
                plt.title(f'Prediction vs Real {cuenca.upper()}{title_suffix}')
                plt.xlabel(xlabel_text)
                plt.ylabel("Snow area Km2")
                plt.legend()
                plt.grid(True)
                output_path = os.path.join(model_dir, f'graphs_{cuenca}')
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                plt.savefig(os.path.join(output_path, f'{graph_type}.png'))
                plt.close()
    return full_metrics


# --- Optuna Objective Function (adapted for new data structure) ---
def objective(trial, basins_data_dict, exog_cols, exog_cols_scaled, list_of_basins_for_optuna):
    # 1. Suggest Hyperparameters
    n_lags_area = trial.suggest_int('n_lags_area', 5, 10)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    n_units_lstm = trial.suggest_int('n_units_lstm', 10, 50, step=5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2) # Broader, lower range for LR
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4) # Make dropout rate a hyperparameter
    epochs = 100

    # 2. Prepare data and train models for the current trial
    n_features = 1 + len(exog_cols_scaled)

    # Use a dummy callback for Optuna's internal training, as its EarlyStopping
    # is handled by the pruner and the objective function's return value.
    # We still use Keras's EarlyStopping for internal validation of model.fit
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0 # Set to 0 to avoid verbose output during hyperparameter search
    )

    validation_kges = []

    for cuenca_name in list_of_basins_for_optuna:
        # --- Try-except block for robust trial execution ---
        try:
            # Access the preprocessed data and scalers for the current basin
            cuenca_data = basins_data_dict[cuenca_name]['data']
            cuenca_scalers = basins_data_dict[cuenca_name]['scalers']

            train_data = cuenca_data['df'].iloc[cuenca_data['train_idx']]
            val_data = cuenca_data['df'].iloc[cuenca_data['val_idx']]

            # Create sequences for the current basin with the trial's n_lags_area
            X_train, y_train = create_sequences(train_data, n_lags_area, exog_cols_scaled)
            # If train sequences are empty, this basin can't be trained, assign a very bad value
            if X_train.shape[0] == 0:
                print(f"Warning: Skipping training for {cuenca_name} due to insufficient training sequences for n_lags={n_lags_area}. Assigning -inf KGE.")
                validation_kges.append(-np.inf)
                trial.report(-np.inf, step=list_of_basins_for_optuna.index(cuenca_name))
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                continue # Skip to next basin

            # Create and train the model for the current basin
            # Pass dropout_rate to the model function (you'll need to update create_narx_model)
            model = create_narx_model(n_lags=n_lags_area, n_layers=n_layers,
                                      n_units_lstm=n_units_lstm, n_features=n_features,
                                      learning_rate=learning_rate)
            # Temporarily modify the model to use the trial's dropout rate
            # A cleaner way is to pass dropout_rate to create_narx_model directly
            if 'Dropout' in [layer.__class__.__name__ for layer in model.layers]:
                for layer in model.layers:
                    if layer.__class__.__name__ == 'Dropout':
                        layer.rate = dropout_rate
                        # Recompile model if dropout rate is changed after compilation
                        # However, it's better to pass it directly to create_narx_model and define it there
                        # For now, let's just make sure create_narx_model accepts it
                        break
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5), loss='mse') # Recompile to ensure dropout change takes effect if done this way

            model.fit(X_train, y_train, epochs=epochs, verbose=0,
                      validation_split=0.1, callbacks=[early_stopping_callback])

            # Evaluate on the actual validation set (walk-forward)
            df_val_scaled = cuenca_data['df'].iloc[cuenca_data['val_idx']].copy()
            scaler_area = cuenca_scalers['area']

            val_metrics, _, _ = evaluate_validation(model, df_val_scaled, scaler_area, exog_cols, n_lags_area)

            # If validation metrics contain NaN/inf, assign a very bad value
            if np.any(np.isnan(list(val_metrics.values()))) or np.any(np.isinf(list(val_metrics.values()))):
                print(f"Warning: Validation metrics for {cuenca_name} contain NaN/inf. Assigning -inf KGE.")
                validation_kges.append(-np.inf)
            else:
                validation_kges.append(val_metrics['KGE'])

            # Report intermediate KGE to Optuna
            trial_step = list_of_basins_for_optuna.index(cuenca_name)
            trial.report(validation_kges[-1], step=trial_step) # Report the last KGE added
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        except (ValueError, tf.errors.InvalidArgumentError, tf.errors.ResourceExhaustedError, RuntimeError) as e:
            # Catch specific errors that indicate model instability or training issues
            print(f"Trial {trial.number} failed for basin {cuenca_name} with error: {e}")
            validation_kges.append(-np.inf) # Assign a very low KGE to penalize this trial
            # You might want to also report this specific trial as failed to Optuna if it's not already implicitly handled
            # by returning -np.inf.
            # Optuna's _run_trial will catch it and mark it as 'FAIL'
            # If a trial fails for one basin, and you're averaging, it might pull down the average for the trial.
            # If you want to strictly mark the trial as failed, you could re-raise an Optuna specific exception,
            # but returning -np.inf and letting it finish the rest of the basins for the trial is usually better.
            pass # Continue to the next basin in the current trial

    # 5. Return the aggregate metric to optimize (e.g., average KGE across basins)
    # If all KGEs are -inf, np.mean will return -inf, which is correct for pruning
    avg_kge = np.mean(validation_kges)
    return avg_kge


# --- Main execution (no changes here, as it calls the modified objective) ---
basins_dir = 'datasets/' 
exog_cols = ["dia_sen","temperatura","precipitacion", "dias_sin_precip"]
exog_cols_scaled = [col + '_scaled' for col in exog_cols]

basin_files = [f for f in os.listdir(basins_dir) if f.endswith('.csv')]
cuencas = [os.path.splitext(f)[0] for f in basin_files]

all_basins_preprocessed_data = {}
for cuenca_name in cuencas:
    print(f"Preprocessing data for basin: {cuenca_name}")
    df_basin = pd.read_csv(os.path.join(basins_dir, f'{cuenca_name}.csv'), index_col=0)
    if 'fecha' in df_basin.columns:
        df_basin['fecha'] = df_basin['fecha'].astype(str)

    try:
        basin_data, basin_scalers = preprocess_data(df_basin.copy(), exog_cols)
        all_basins_preprocessed_data[cuenca_name] = {'data': basin_data, 'scalers': basin_scalers}
    except ValueError as e:
        print(f"Error during preprocessing for {cuenca_name}: {e}. Skipping this basin.")
        # Do not add this basin to all_basins_preprocessed_data or cuencas list for Optuna
        # This will prevent it from being processed further
        cuencas.remove(cuenca_name) # Remove from the list that will be passed to Optuna
        continue


print("\n--- Starting Optuna Hyperparameter Optimization ---")

cuencas_for_optuna = cuencas # Optimize for all detected basins that were preprocessed successfully

objective_with_data = partial(objective,
                              basins_data_dict=all_basins_preprocessed_data,
                              exog_cols=exog_cols,
                              exog_cols_scaled=exog_cols_scaled,
                              list_of_basins_for_optuna=cuencas_for_optuna)

study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))

study.optimize(objective_with_data, n_trials=20, show_progress_bar=True)

print("\n--- Optuna Optimization Results ---")
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
best_trial = study.best_trial
print("  Value (Avg KGE): ", best_trial.value)
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

print("\n--- Training final models with best hyperparameters ---")

best_params = best_trial.params
best_n_lags_area = best_params['n_lags_area']
best_n_layers = best_params['n_layers']
best_n_neuronas = best_params['n_units_lstm']
best_learning_rate = best_params['learning_rate']
best_dropout_rate = best_params['dropout_rate'] # Get the best dropout rate

final_model_name = (f'best_model_L{best_n_lags_area}_N{best_n_neuronas}_H{best_n_layers}_LR{best_learning_rate:.6f}_D{best_dropout_rate:.2f}'
                    .replace('.', '_'))
final_model_dir = os.path.join("D:", "models", final_model_name)
os.makedirs(final_model_dir, exist_ok=True)

final_sequences_data = {}
for cuenca_name in cuencas: # Use all successfully preprocessed cuencas for final training
    cuenca_data = all_basins_preprocessed_data[cuenca_name]['data']
    train_data = cuenca_data['df'].iloc[cuenca_data['train_idx']]
    val_data = cuenca_data['df'].iloc[cuenca_data['val_idx']]
    test_data = cuenca_data['df'].iloc[cuenca_data['test_idx']]

    final_sequences_data[cuenca_name] = {
        'X_train': create_sequences(train_data, best_n_lags_area, exog_cols_scaled)[0],
        'y_train': create_sequences(train_data, best_n_lags_area, exog_cols_scaled)[1],
        'X_val': create_sequences(val_data, best_n_lags_area, exog_cols_scaled)[0],
        'y_val': create_sequences(val_data, best_n_lags_area, exog_cols_scaled)[1],
        'X_test': create_sequences(test_data, best_n_lags_area, exog_cols_scaled)[0],
        'y_test': create_sequences(test_data, best_n_lags_area, exog_cols_scaled)[1],
    }

# Create and train the final models using the best hyperparameters
# Modify create_narx_model to directly use the dropout_rate from best_params
def create_narx_model_final(n_lags, n_layers, n_units_lstm, n_features, learning_rate, dropout_rate):
    model = Sequential()
    lstm_activation = 'tanh' # Or 'relu'
    if n_layers > 1:
        model.add(LSTM(n_units_lstm, activation=lstm_activation, input_shape=(n_lags, n_features), return_sequences=True))
    else:
        model.add(LSTM(n_units_lstm, activation=lstm_activation, input_shape=(n_lags, n_features)))
    for _ in range(1, n_layers):
        if _ == n_layers - 1:
            model.add(LSTM(n_units_lstm, activation=lstm_activation))
        else:
            model.add(LSTM(n_units_lstm, activation=lstm_activation, return_sequences=True))
    model.add(Dropout(dropout_rate)) # Use the optimized dropout rate
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.5)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# create_train_models_final now expects the dropout_rate
def create_train_models_final(sequences_data, n_lags_area, layers, units, epochs, exog_cols_scaled, cuencas_list, save, models_dir, learning_rate, dropout_rate):
    models = {}
    history = {}
    n_features = 1 + len(exog_cols_scaled)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    for cuenca in cuencas_list:
        print(f"Training final model for cuenca: {cuenca}")
        # Use the _final version of create_narx_model
        model = create_narx_model_final(n_lags=n_lags_area, n_layers=layers, n_units_lstm=units, n_features=n_features, learning_rate=learning_rate, dropout_rate=dropout_rate)
        X_train = sequences_data[cuenca]['X_train']
        y_train = sequences_data[cuenca]['y_train']

        if X_train.shape[0] == 0:
            print(f"Skipping final training for {cuenca}: No training sequences available.")
            continue

        model_history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_split=0.1, callbacks=[early_stopping_callback])
        models[cuenca] = model
        history[cuenca] = model_history

        model_path = os.path.join(models_dir, f'narx_model_{cuenca}.h5')
        if save:
            model.save(model_path)
            print(f"Modelo entrenado y guardado para la cuenca: {cuenca} en {model_path}")
        else:
            print(f'Modelo entrenado para la cuenca {cuenca}')
    return models


# Train the final models
final_models = create_train_models_final(final_sequences_data,
                                         best_n_lags_area,
                                         best_n_layers,
                                         best_n_neuronas,
                                         epochs=100,
                                         exog_cols_scaled=exog_cols_scaled,
                                         cuencas_list=cuencas,
                                         save=True,
                                         models_dir=final_model_dir,
                                         learning_rate=best_learning_rate,
                                         dropout_rate=best_dropout_rate)


# --- Evaluate the final models ---
print("\n--- Evaluating Final Models ---")
final_metrics_results = {}
archivo_json = os.path.join(final_model_dir, 'metrics.json')

results_full_dataset = evaluate_full_dataset(final_models, all_basins_preprocessed_data,
                                              {k: v['scalers'] for k,v in all_basins_preprocessed_data.items()},
                                              cuencas, best_n_lags_area, exog_cols_scaled, True, final_model_dir)

for cuenca_name in cuencas:
    current_model = final_models.get(cuenca_name) # Use .get() in case model wasn't trained
    if current_model is None:
        print(f"Skipping evaluation for {cuenca_name}: Model not trained.")
        final_metrics_results[cuenca_name + ' (train)'] = {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
        final_metrics_results[cuenca_name + ' (test)'] = {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
        final_metrics_results[cuenca_name + ' (val)'] = {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
        final_metrics_results[cuenca_name + ' (full dataset)'] = {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
        continue

    current_scaler_area = all_basins_preprocessed_data[cuenca_name]['scalers']['area']

    train_sequences = {'X': final_sequences_data[cuenca_name]['X_train'], 'y': final_sequences_data[cuenca_name]['y_train']}
    metrics_train, _, _ = evaluate_model(current_model, train_sequences, current_scaler_area)
    final_metrics_results[cuenca_name + ' (train)'] = metrics_train
    print(f"Métricas conjunto de 'train' para {cuenca_name}: {metrics_train}")

    test_sequences = {'X': final_sequences_data[cuenca_name]['X_test'], 'y': final_sequences_data[cuenca_name]['y_test']}
    metrics_test, _, _ = evaluate_model(current_model, test_sequences, current_scaler_area)
    final_metrics_results[cuenca_name + ' (test)'] = metrics_test
    print(f"Métricas conjunto de 'test' para {cuenca_name}: {metrics_test}")

    df_val_scaled = all_basins_preprocessed_data[cuenca_name]['data']['df'].iloc[all_basins_preprocessed_data[cuenca_name]['data']['val_idx']].copy()
    metrics_val, _, _ = evaluate_validation(current_model, df_val_scaled, current_scaler_area, exog_cols, best_n_lags_area)
    final_metrics_results[cuenca_name + ' (val)'] = metrics_val
    print(f"Métricas conjunto de 'validation' (modo prediccion) para {cuenca_name}: {metrics_val}")

    # Ensure this key exists before trying to assign
    if cuenca_name in results_full_dataset:
        final_metrics_results[cuenca_name + ' (full dataset)'] = results_full_dataset[cuenca_name]
    else:
        final_metrics_results[cuenca_name + ' (full dataset)'] = {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan} # Assign NaN if not evaluated

# Save all final metrics to JSON
with open(archivo_json, 'w', encoding='utf-8') as f:
    json.dump(final_metrics_results, f, indent=4)

print(f"\nFinal metrics saved to {archivo_json}")