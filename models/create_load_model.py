# IMPORTS
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
import optuna # Aunque no se usa directamente para la creación manual, se mantiene por si se quiere integrar más Optuna.

# --- FUNCIONES (Copiadas de tu script original, no necesitan cambios internos) ---

# --- CLASE CUSTOM_LSTM PARA MANEJAR EL ERROR 'time_major' ---
# Este es el cambio clave para cargar tus modelos antiguos.
class CustomLSTM(keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        # Filtrar el argumento 'time_major' si está presente y no es reconocido
        if 'time_major' in kwargs:
            print(f"Advertencia: Ignorando el argumento 'time_major={kwargs['time_major']}' para la capa LSTM.")
            kwargs.pop('time_major')
        super().__init__(*args, **kwargs)

# Registrar la clase para que Keras pueda encontrarla al cargar el modelo
# Esto es importante para que Keras sepa cómo deserializar esta capa personalizada
tf.keras.utils.get_custom_objects()['LSTM'] = CustomLSTM

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

def load_model_and_params_for_basin(cuenca_name, models_dir='models'):
    """Carga un modelo y sus hiperparámetros para una cuenca específica,
    manejando el error 'time_major'."""
    basin_output_dir = os.path.join(models_dir, cuenca_name)
    model_path = os.path.join(basin_output_dir, f'narx_model_best_{cuenca_name}.h5')
    params_path = os.path.join(basin_output_dir, f'metrics.json') # o 'best_params.json' si lo tienes

    loaded_model = None
    loaded_params = None

    if os.path.exists(model_path):
        try:
            # Usar custom_objects para manejar la capa LSTM y la pérdida 'mse'
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    'LSTM': CustomLSTM, # Usar clase custom para LSTM
                    'mse': tf.keras.losses.MeanSquaredError() # Para el error 'mse' si aparece
                }
            )
            print(f"Modelo cargado para la cuenca: {cuenca_name} desde {model_path}")
        except Exception as e:
            print(f"Error al cargar el modelo de {cuenca_name}: {e}")

    # Cargar parámetros adicionales (aunque hayas dicho que metrics.json no existe)
    # Esto intentará cargarlo de todos modos, pero loaded_params será None si falla.
    if os.path.exists(params_path):
        try:
            with open(params_path, 'r', encoding='utf-8') as f:
                loaded_params = json.load(f)
            print(f"Hiperparámetros (o métricas) cargados para la cuenca: {cuenca_name} desde {params_path}")
        except Exception as e:
            print(f"Error al cargar los hiperparámetros de {cuenca_name}: {e}")
    
    return loaded_model, loaded_params

def evaluate_model(model, sequences, scaler_area):
    X = sequences['X']
    y_true = sequences['y']
    
    if X.shape[0] == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    y_pred_scaled = model.predict(X, verbose=0, batch_size=2)

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
            y_val_pred_scaled.extend([np.nan] * (len(df_val_scaled) - n_lags_area - i))
            break

        pred_scaled = model.predict(last_sequence, verbose=0, batch_size=2)

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
    Evalúa un modelo específico en el conjunto de datos completo (modo predicción).
    Modificado para evaluar un único modelo y generar gráficos para esa cuenca.
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

        pred_scaled = model.predict(last_sequence_full, verbose=0, batch_size=2)

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
            output_path = os.path.join(base_model_dir, f'graphs_{cuenca_name}')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.savefig(os.path.join(output_path, f'{graph_type}.png'))
            plt.close()
    return full_metrics

# --- Nuevas funciones para el script de carga/creación ---

def get_hyperparameters_from_user():
    """Solicita al usuario los hiperparámetros para un nuevo modelo."""
    print("\n--- Introduzca los hiperparámetros para el nuevo modelo ---")
    n_lags_area = int(input("Número de lags (n_lags_area, entero, ej: 5): "))
    n_layers = int(input("Número de capas LSTM (n_layers, entero, ej: 2): "))
    n_units_lstm = int(input("Número de unidades LSTM por capa (n_units_lstm, entero, ej: 20): "))
    learning_rate = float(input("Tasa de aprendizaje (learning_rate, flotante, ej: 0.001): "))
    dropout_rate = float(input("Tasa de Dropout (dropout_rate, flotante, ej: 0.2): "))
    epochs = int(input("Número de épocas para el entrenamiento (epochs, entero, ej: 100): "))
    
    return {
        'n_lags_area': n_lags_area,
        'n_layers': n_layers,
        'n_units_lstm': n_units_lstm,
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'epochs': epochs
    }

def convert_numpy_to_python(obj):
    """Convierte tipos de numpy a tipos nativos de Python para serialización JSON."""
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_to_python(elem) for elem in obj]
    return obj

# --- Main execution para el nuevo script ---

if __name__ == "__main__":
    basins_dir = 'datasets/'
    models_base_dir = os.path.join("E:", "models") # Directorio base para todos los modelos

    exog_cols = ["dia_sen","temperatura","precipitacion", "dias_sin_precip"]
    exog_cols_scaled = [col + '_scaled' for col in exog_cols]

    cuenca_name_input = input("Introduce el nombre de la cuenca (ej: 'cuenca1') o deja en blanco para ver las disponibles: ").strip()

    basin_files = [f for f in os.listdir(basins_dir) if f.endswith('.csv')]
    available_basins = [os.path.splitext(f)[0] for f in basin_files]

    if not cuenca_name_input:
        print("\nCuencas disponibles:")
        for basin in available_basins:
            print(f"- {basin}")
        cuenca_name_input = input("Por favor, introduce el nombre de la cuenca que deseas usar: ").lower().strip()
    
    if cuenca_name_input not in available_basins:
        print(f"Error: La cuenca '{cuenca_name_input}' no se encontró en el directorio '{basins_dir}'.")
        exit()

    cuenca_name = cuenca_name_input
    basin_output_dir = os.path.join(models_base_dir, cuenca_name)
    os.makedirs(basin_output_dir, exist_ok=True) # Asegura que el directorio de la cuenca exista

    # Preprocesar los datos de la cuenca
    df_basin = pd.read_csv(os.path.join(basins_dir, f'{cuenca_name}.csv'), index_col=0)
    if 'fecha' in df_basin.columns:
        df_basin['fecha'] = df_basin['fecha'].astype(str)

    try:
        basin_data, basin_scalers = preprocess_data(df_basin.copy(), exog_cols)
    except ValueError as e:
        print(f"Error durante el preprocesamiento de datos para {cuenca_name}: {e}")
        exit()

    model_to_evaluate = None
    params_to_use = None
    
    # Intentar cargar un modelo existente y sus parámetros
    loaded_model, loaded_params = load_model_and_params_for_basin(cuenca_name, models_base_dir)

    if loaded_model and loaded_params:
        print(f"\nSe encontró un modelo y sus hiperparámetros para '{cuenca_name}'.")
        choice = input("¿Deseas cargar el modelo existente (c) o crear un nuevo modelo (n)? [c/n]: ").lower()
        if choice == 'c':
            model_to_evaluate = loaded_model
            params_to_use = loaded_params.get('best_params')
            print("Cargando el modelo existente.")
        else:
            print("Creando un nuevo modelo.")
            params_to_use = get_hyperparameters_from_user()
            
            # Entrenar el nuevo modelo
            print(f"\n--- Entrenando el nuevo modelo para {cuenca_name} ---")
            X_train_final, y_train_final = create_sequences(basin_data['df'].iloc[basin_data['train_idx']],
                                                            params_to_use['n_lags_area'], exog_cols_scaled)

            if X_train_final.shape[0] == 0:
                print(f"Error: No hay secuencias de entrenamiento suficientes para n_lags={params_to_use['n_lags_area']}.")
                exit()
            
            n_features = 1 + len(exog_cols_scaled)
            model_to_evaluate = create_narx_model(n_lags=params_to_use['n_lags_area'], n_layers=params_to_use['n_layers'],
                                                 n_units_lstm=params_to_use['n_units_lstm'], n_features=n_features,
                                                 learning_rate=params_to_use['learning_rate'], dropout_rate=params_to_use['dropout_rate'])
            
            model_checkpoint_callback = ModelCheckpoint(
                filepath=os.path.join(basin_output_dir, f'narx_model_{cuenca_name}.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
            early_stopping_callback = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            )
            
            model_to_evaluate.fit(X_train_final, y_train_final, epochs=params_to_use['epochs'], verbose=0,
                                 validation_split=0.1, callbacks=[early_stopping_callback, model_checkpoint_callback])
            
            # Cargar el modelo guardado si se usó ModelCheckpoint
            final_trained_model_path = os.path.join(basin_output_dir, f'narx_model_{cuenca_name}.h5')
            if os.path.exists(final_trained_model_path):
                model_to_evaluate = keras.models.load_model(final_trained_model_path)
                print(f"Cargado el mejor modelo guardado para {cuenca_name}.")
            else:
                print(f"No se encontró el mejor modelo guardado para {cuenca_name}, usando el entrenado en la última época.")

            # Guardar los nuevos parámetros
            params_save_path = os.path.join(basin_output_dir, f'best_params_{cuenca_name}.json')
            with open(params_save_path, 'w', encoding='utf-8') as f:
                json.dump(convert_numpy_to_python(params_to_use), f, indent=4)
            print(f"Hiperparámetros guardados en {params_save_path}")

    else:
        print(f"\nNo se encontró un modelo existente para '{cuenca_name}'. Se procederá a crear uno nuevo.")
        params_to_use = get_hyperparameters_from_user()

        # Entrenar el nuevo modelo
        print(f"\n--- Entrenando el nuevo modelo para {cuenca_name} ---")
        X_train_final, y_train_final = create_sequences(basin_data['df'].iloc[basin_data['train_idx']],
                                                        params_to_use['n_lags_area'], exog_cols_scaled)
        
        if X_train_final.shape[0] == 0:
            print(f"Error: No hay secuencias de entrenamiento suficientes para n_lags={params_to_use['n_lags_area']}.")
            exit()

        n_features = 1 + len(exog_cols_scaled)
        model_to_evaluate = create_narx_model(n_lags=params_to_use['n_lags_area'], n_layers=params_to_use['n_layers'],
                                             n_units_lstm=params_to_use['n_units_lstm'], n_features=n_features,
                                             learning_rate=params_to_use['learning_rate'], dropout_rate=params_to_use['dropout_rate'])
        
        model_checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(basin_output_dir, f'narx_model_{cuenca_name}.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )

        model_to_evaluate.fit(X_train_final, y_train_final, epochs=params_to_use['epochs'], verbose=0,
                             validation_split=0.1, callbacks=[early_stopping_callback, model_checkpoint_callback])
        
        # Cargar el modelo guardado si se usó ModelCheckpoint
        final_trained_model_path = os.path.join(basin_output_dir, f'narx_model_{cuenca_name}.h5')
        if os.path.exists(final_trained_model_path):
            model_to_evaluate = keras.models.load_model(final_trained_model_path)
            print(f"Cargado el mejor modelo guardado para {cuenca_name}.")
        else:
            print(f"No se encontró el mejor modelo guardado para {cuenca_name}, usando el de la última época.")

        # Guardar los nuevos parámetros
        params_save_path = os.path.join(basin_output_dir, f'best_params_{cuenca_name}.json')
        with open(params_save_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_to_python(params_to_use), f, indent=4)
        print(f"Hiperparámetros guardados en {params_save_path}")

    if model_to_evaluate is None or params_to_use is None:
        print("No se pudo cargar o crear el modelo. Saliendo.")
        exit()

    # --- Evaluación del modelo ---
    print(f"\n--- Evaluando el modelo para la cuenca: {cuenca_name} ---")

    current_n_lags_area = params_to_use['n_lags_area']
    current_scaler_area = basin_scalers['area']

    sequences_for_eval = {
        'train': {'X': create_sequences(basin_data['df'].iloc[basin_data['train_idx']], current_n_lags_area, exog_cols_scaled)[0],
                  'y': create_sequences(basin_data['df'].iloc[basin_data['train_idx']], current_n_lags_area, exog_cols_scaled)[1]},
        'test': {'X': create_sequences(basin_data['df'].iloc[basin_data['test_idx']], current_n_lags_area, exog_cols_scaled)[0],
                 'y': create_sequences(basin_data['df'].iloc[basin_data['test_idx']], current_n_lags_area, exog_cols_scaled)[1]},
        'val_df': basin_data['df'].iloc[basin_data['val_idx']].copy()
    }

    metrics_train, _, _ = evaluate_model(model_to_evaluate, sequences_for_eval['train'], current_scaler_area)
    metrics_test, _, _ = evaluate_model(model_to_evaluate, sequences_for_eval['test'], current_scaler_area)
    metrics_val, _, _ = evaluate_validation(model_to_evaluate, sequences_for_eval['val_df'], current_scaler_area, exog_cols, current_n_lags_area)
    metrics_full = evaluate_full_dataset(model_to_evaluate, basin_data['df'], current_scaler_area, exog_cols_scaled, current_n_lags_area, graph=True, base_model_dir=basin_output_dir, cuenca_name=cuenca_name)

    print(f"\nMétricas finales para la cuenca {cuenca_name}:")
    print(f"  Hiperparámetros utilizados: {params_to_use}")
    print(f"  Train: {metrics_train}")
    print(f"  Test: {metrics_test}")
    print(f"  Validation (modo predicción): {metrics_val}")
    print(f"  Full Dataset (modo predicción): {metrics_full}")

    # Guardar todas las métricas finales en un JSON
    all_metrics_output_path = os.path.join(basin_output_dir, 'metrics.json')
    final_metrics_summary = {
        'basin_name': cuenca_name,
        'hyperparameters_used': params_to_use,
        'metrics': {
            'train': metrics_train,
            'test': metrics_test,
            'validation': metrics_val,
            'full_dataset': metrics_full
        }
    }
    with open(all_metrics_output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_to_python(final_metrics_summary), f, indent=4)
    print(f"\nMétricas detalladas guardadas en {all_metrics_output_path}")