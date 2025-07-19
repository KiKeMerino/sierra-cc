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
import matplotlib.ticker as mticker # Necesario para FixedLocator
import matplotlib.dates as mdates # Alias para fechas de matplotlib

plt.rcParams.update({'font.size': 18})
EXTERNAL_DISK = 'D:'

# --- CLASE CUSTOM_LSTM PARA MANEJAR EL ERROR 'time_major' ---
class CustomLSTM(keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        # Filtrar el argumento 'time_major' si está presente y no es reconocido
        if 'time_major' in kwargs:
            # print(f"Advertencia: Ignorando el argumento 'time_major={kwargs['time_major']}' para la capa LSTM.")
            kwargs.pop('time_major')
        super().__init__(*args, **kwargs)
tf.keras.utils.get_custom_objects()['LSTM'] = CustomLSTM

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

def evaluate_model(model, sequences, scaler_area):
    X = sequences['X']
    y_true = sequences['y']
    
    if X.shape[0] == 0:
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    y_pred_scaled = model.predict(X, verbose=0, batch_size=128)

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

        pred_scaled = model.predict(last_sequence, verbose=0, batch_size=16)

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

        pred_scaled = model.predict(last_sequence_full, verbose=0, batch_size=64)

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

    # Calculate metrics
    r2_full = pearsonr(y_full_true_original.flatten(), y_full_pred_original.flatten())
    r2_full = r2_full.statistic**2
    mae_full = mean_absolute_error(y_full_true_original, y_full_pred_original)
    nse_full = nash_sutcliffe_efficiency(y_full_true_original, y_full_pred_original)
    kge_full = kling_gupta_efficiency(y_full_true_original, y_full_pred_original)

    full_metrics = {'R2': r2_full, 'MAE': mae_full, 'NSE': nse_full, 'KGE': kge_full}
    print(f"Métricas en todo el conjunto de datos (modo prediccion) para {cuenca_name}: R2={r2_full:.3f}, MAE={mae_full:.3f}, NSE={nse_full:.3f}, KGE={kge_full:.3f}")

    if graph == True:
        graph_types = ['per_day', 'per_month', 'all_days']
        output_path = os.path.join(base_model_dir, f'graphs_{cuenca_name}')

        for graph_type in graph_types:
            if 'fecha' in df_full_scaled_cuenca:
                df_full_scaled_cuenca = df_full_scaled_cuenca.set_index('fecha')

            real_plot = df_full_scaled_cuenca.iloc[n_lags_area:].copy()
            real_plot['area_nieve'] = y_full_true_original
            
            y_full_pred_df = pd.DataFrame(y_full_pred_original, columns=['area_nieve_pred'], index=real_plot.index)
            y_full_pred_df.to_csv(os.path.join(output_path, 'full_dataset.csv'))
            
            df_plot = pd.concat([real_plot, y_full_pred_df], axis=1)

            xlabel_text = ""
            groupby_col = 'fecha' # Valor por defecto, se sobrescribirá si es necesario
            
            if not isinstance(df_plot.index, pd.DatetimeIndex):
                df_plot = df_plot.set_index(pd.to_datetime(df_plot.index))

            if graph_type == 'per_day':
                df_plot['fecha_agrupada'] = df_plot.index.day_of_year
                xlabel_text = "Day of the year"
                groupby_col = 'fecha_agrupada'
            elif graph_type == 'per_month':
                df_plot['fecha_agrupada'] = df_plot.index.month
                xlabel_text = 'Month of the year'
                groupby_col = 'fecha_agrupada'
            elif graph_type == 'all_days':
                # Para 'all_days', agrupamos por el índice (fecha)
                xlabel_text = "Date"
                groupby_col = df_plot.index # Agrupar por el DatetimeIndex directamente

            # El agrupamos para 'per_day' y 'per_month'
            if graph_type in ['per_day', 'per_month']:
                df_plot_grouped = df_plot.groupby(groupby_col).agg(
                    area_nieve_real = ('area_nieve', 'mean'),
                    area_nieve_pred=('area_nieve_pred', 'mean')
                ).reset_index()
                ylim_top = max(max(df_plot_grouped['area_nieve_pred']), max(df_plot_grouped['area_nieve_real']))

            else: # graph_type == 'all_days'
                df_plot_grouped = df_plot.copy()
                df_plot_grouped.reset_index(inplace=True) # El índice de fecha se convierte en columna 'fecha'
                df_plot_grouped = df_plot_grouped.rename(columns={'area_nieve':'area_nieve_real'})
                groupby_col = 'fecha' # Ahora la columna de fecha es 'fecha'
                ylim_top = max(max(df_plot_grouped['area_nieve_pred']), max(df_plot_grouped['area_nieve_real'])) + 200

            plt.figure(figsize=(12,8))
            
            # Plotear la serie real y la simulación
            sns.lineplot(x=df_plot_grouped[groupby_col], y=df_plot_grouped.area_nieve_real, label='Data', linewidth=2, color='black')
            sns.lineplot(x=df_plot_grouped[groupby_col], y=df_plot_grouped.area_nieve_pred, label='Simulation', linewidth=2, color='green')

            # 1. Establecer los límites del eje X exactamente al inicio y fin de tus datos
            min_date = min(df_plot_grouped[groupby_col])
            max_date = max(df_plot_grouped[groupby_col])

            plt.xlim(left=min_date, right=max_date)
            plt.ylim(bottom=0, top = ylim_top)
            
            if graph_type == 'all_days':
                metrics_text = (
                    f"R²: {full_metrics['R2']:.3f}\n"
                    f"MAE: {full_metrics['MAE']:.3f}\n"
                    f"NSE: {full_metrics['NSE']:.3f}\n"
                    f"KGE: {full_metrics['KGE']:.3f}"
                )
                plt.text(0.75, 0.97, metrics_text, transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')

                start_year = 2000
                end_year = 2023
                
                # Crear una lista de fechas para las marcas que quieres mostrar
                # Incluye el año de inicio, el año de fin, y años intermedios si deseas
                # Por ejemplo, cada 4 años desde 2000 hasta 2023.
                # Asegúrate de que las fechas sean objetos Timestamp
                years_to_show = list(range(start_year, end_year + 1, 4)) # Marcas cada 4 años, ajusta el '4' si quieres más/menos
                if end_year not in years_to_show: # Asegurarse de que el último año esté incluido si no es múltiplo de 4
                    years_to_show.append(end_year) 
                
                # Convertir los años a objetos Timestamp para el locator
                # Usamos el 1 de enero de cada año como referencia para la marca
                tick_dates = [pd.Timestamp(f'{year}-01-01') for year in years_to_show]
                
                # Convertir a formato numérico de Matplotlib para el FixedLocator
                tick_values_mpl = mdates.date2num(tick_dates)

                # 2. Forzar los límites del eje X para que incluyan exactamente el 2000 y el 2023
                # Establecemos los límites al 1 de enero del año de inicio y el 31 de diciembre del año de fin
                plt.xlim(pd.Timestamp(f'{start_year}-01-01'), pd.Timestamp(f'{end_year}-12-31'))
                
                # 3. Usar FixedLocator para establecer las marcas EXACTAS en los años deseados
                plt.gca().xaxis.set_major_locator(mticker.FixedLocator(tick_values_mpl))
                
                # 4. Formatear las etiquetas como solo el año
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                
                # 5. Rotar las etiquetas para que no se superpongan
                plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right') 
            

            plt.xlabel(xlabel_text)
            plt.ylabel("Snow cover area (km2)")
            plt.legend(loc='best', frameon=True)
            plt.tight_layout()
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.savefig(os.path.join(output_path, f'{graph_type}.png'))
            plt.close()
    return full_metrics

# --- Optuna Objective Function ---
def objective_single_basin(trial, basin_data, basin_scalers, exog_cols, exog_cols_scaled):
    # 1. Suggest Hyperparameters
    n_lags_area = trial.suggest_int('n_lags_area', 2, 9)
    n_layers = trial.suggest_int('n_layers', 1, 4)
    n_units_lstm = trial.suggest_int('n_units_lstm', 5, 30)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 64, 128])
    epochs = trial.suggest_int('epochs', 20, 120)

    # 2. Prepare data for the current trial
    n_features = 1 + len(exog_cols_scaled)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=20, # Paciencia para EarlyStopping durante el entrenamiento del trial
        restore_best_weights=True,
        verbose=0
    )

    # Extract data for the single basin
    train_data_df = basin_data['df'].iloc[basin_data['train_idx']]
    val_data_df = basin_data['df'].iloc[basin_data['val_idx']].copy() 
    scaler_area = basin_scalers['area']

    # Create sequences for the current basin with the trial's n_lags_area
    X_train, y_train = create_sequences(train_data_df, n_lags_area, exog_cols_scaled)
    
    # --- Try-except block for robust trial execution ---
    try:
        if X_train.shape[0] == 0:
            print(f"Warning: Insufficient training sequences for n_lags={n_lags_area}. Assigning -inf NSE.")
            return -np.inf # Return a very bad NSE if no sequences

        # Create and train the model for the current trial
        model = create_narx_model(n_lags=n_lags_area, n_layers=n_layers,
                                   n_units_lstm=n_units_lstm, n_features=n_features,
                                   learning_rate=learning_rate, dropout_rate=dropout_rate)
        
        # Entrenar el modelo con batch_size especificado para controlar el uso de VRAM
        model.fit(X_train, y_train, epochs=epochs, verbose=0,
                  validation_split=0.1, # Validación interna para EarlyStopping
                  callbacks=[early_stopping_callback],
                  batch_size=batch_size)

        # --- EVALUACIÓN PARA OPTUNA (NSE en el conjunto COMPLETO - full_dataset) ---
        # metrics = evaluate_full_dataset(
        #     model, 
        #     basin_data['df'],
        #     scaler_area, 
        #     exog_cols_scaled, 
        #     n_lags_area,
        #     graph=False,
        # )
         # --- EVALUACIÓN PARA OPTUNA (NSE en el conjunto de VALIDACIÓN) ---
        metrics, _, _ = evaluate_validation(
            model, 
            val_data_df,
            scaler_area,
            exog_cols, 
            n_lags_area
        )

        # Si las métricas del full_dataset contienen NaN/inf, asigna un valor muy malo
        if np.any(np.isnan(list(metrics.values()))) or np.any(np.isinf(list(metrics.values()))):
            print(f"Warning: Full dataset metrics for trial {trial.number} contain NaN/inf. Assigning -inf NSE.")
            return -np.inf
        else:
            # Optuna va a MAXIMIZAR este valor de NSE del conjunto COMPLETO
            return metrics['NSE']

    except (ValueError, tf.errors.InvalidArgumentError, tf.errors.ResourceExhaustedError, RuntimeError) as e:
        print(f"Trial {trial.number} failed with error: {e}. Returning -inf NSE.")
        # Limpiar la sesión de Keras para prevenir posibles problemas en el siguiente trial
        tf.keras.backend.clear_session()
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
basins_dir = 'datasets_imputed/' # Asegúrate de que esta ruta sea correcta para tus CSV de cuencas
models_dir = os.path.join(EXTERNAL_DISK, "new_models") # Directorio donde se guardarán/buscarán los modelos por cuenca

exog_cols = ["dia_sen","temperatura","precipitacion", "dias_sin_precip"]
exog_cols_scaled = [col + '_scaled' for col in exog_cols]

# Pedir al usuario la cuenca a optimizar 
available_basins = [f[:-4] for f in os.listdir(basins_dir)]
cuenca_name = ""
while cuenca_name not in available_basins:
        cuenca_name =  input("Introduce el nombre de la cuenca (ej: 'adda-bornio') que deseas optimizar o deja en blanco para ver las disponibles: ").lower().strip()
        if not cuenca_name or cuenca_name not in available_basins: # Si el usuario deja en blanco, imprimo las cuencas disponibles
            print('\n--- Cuencas disponibles ---')
            for basin in available_basins:
                print(f'- {basin}')
            print("-"*25)
            cuenca_name = input("\nPor favor, introduce el nombre de la cuenca que desesas usar: ").lower().strip()

# Pedir al usuario el numero de ensayos para la optimización
n_trials_per_basin = ""
while not (n_trials_per_basin.isdigit() and int(n_trials_per_basin) > 0 and int(n_trials_per_basin) < 100):
    n_trials_per_basin = input("Ingrese el número de ensayos (trials) para la optimización (ej. 20): ").strip()
    if not (n_trials_per_basin.isdigit() and int(n_trials_per_basin) > 0):
        print("Entrada inválida. Por favor, ingrese un número entero positivo.")

n_trials_per_basin = int(n_trials_per_basin)

# Preprocesando la cuenca para la busqueda del mejor modelo
print(f"Preprocessing data for basin: {cuenca_name}")
df_basin = pd.read_csv(os.path.join(basins_dir, f'{cuenca_name}.csv'), index_col=0)
if 'fecha' in df_basin.columns:
    df_basin['fecha'] = df_basin['fecha'].astype(str) # Asegurar tipo string para pd.to_datetime posterior

try:
    basin_data, basin_scalers = preprocess_data(df_basin.copy(), exog_cols)
    preprocessed_data = {'data': basin_data, 'scalers': basin_scalers}
except ValueError as e:
    print(f"Error during preprocessing for {cuenca_name}: {e}. Skipping this basin.")


# --- Optimización y evaluación de Optuna ---
print(f"\n--- Iniciando Optimización Optuna para Cuenca: {cuenca_name} ---")

# Definir el directorio de salida para el mejor modelo y métricas de esta cuenca específica
basin_output_dir = os.path.join(models_dir, cuenca_name)
os.makedirs(basin_output_dir, exist_ok=True)

# Crear una función objetivo parcial para la cuenca actual
objective_for_this_basin = partial(objective_single_basin,
                                    basin_data=preprocessed_data['data'],
                                    basin_scalers=preprocessed_data['scalers'],
                                    exog_cols=exog_cols,
                                    exog_cols_scaled=exog_cols_scaled)

# Crear un estudio Optuna para esta cuenca específica
study_basin = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10),
                                    study_name=f"basin_optimization_{cuenca_name}")

# Ejecutar la optimización para esta cuenca
study_basin.optimize(objective_for_this_basin, n_trials=n_trials_per_basin, show_progress_bar=True)

print(f"\n--- Resultados de Optimización Optuna para {cuenca_name} ---")
print(f"Número de trials completados: {len(study_basin.trials)}")

current_basin_metrics = {} # Diccionario para las métricas de la cuenca actual

if study_basin.best_trial is not None:
    best_trial_basin = study_basin.best_trial
    print(f"  Mejor Valor (NSE): {best_trial_basin.value:.4f}")
    print("  Mejores Parámetros: ")
    for key, value in best_trial_basin.params.items():
        print(f"    {key}: {value}")

    # Almacenar los mejores parámetros y entrenar el modelo final para esta cuenca
    best_params_basin = best_trial_basin.params
    json_output_path = os.path.join(basin_output_dir, 'best_params.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_to_python(best_params_basin), f, indent=4)
    print(f"Métricas guardadas para {cuenca_name} en {json_output_path}")

#     best_n_lags_area = best_params_basin['n_lags_area']
#     best_n_layers = best_params_basin['n_layers']
#     best_n_neuronas = best_params_basin['n_units_lstm']
#     best_learning_rate = best_params_basin['learning_rate']
#     best_dropout_rate = best_params_basin['dropout_rate']
#     epochs_final_train = 100 # Máximo de épocas para el entrenamiento final, EarlyStopping se encargará

#     print(f"\n--- Entrenando modelo final para {cuenca_name} con los mejores hiperparámetros ---")
#     basin_data_final = preprocessed_data['data']
#     X_train_final, y_train_final = create_sequences(basin_data_final['df'].iloc[basin_data_final['train_idx']],
#                                                     best_n_lags_area, exog_cols_scaled)

#     if X_train_final.shape[0] == 0:
#         print(f"Saltando entrenamiento final para {cuenca_name}: No hay secuencias de entrenamiento para best_n_lags_area={best_n_lags_area}.")
#         best_model = {'model': None, 'best_n_lags_area': best_n_lags_area}
        
#         # Asignar NaNs a las métricas si no hay entrenamiento
#         current_basin_metrics = {
#             'best_params': best_params_basin,
#             'full_dataset': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
#             'train': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
#             'test': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
#             'val': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
#         }
#         # Guardar el JSON para esta cuenca (incluso si tiene NaNs)
#         json_output_path = os.path.join(basin_output_dir, 'metrics.json')
#         with open(json_output_path, 'w', encoding='utf-8') as f:
#             json.dump(convert_numpy_to_python(current_basin_metrics), f, indent=4) # 'convert_numpy_to_python' debe estar definida
#         print(f"Métricas (con NaNs) guardadas para {cuenca_name} en {json_output_path}")


#     model_final = create_narx_model(n_lags=best_n_lags_area, n_layers=best_n_layers,
#                                         n_units_lstm=best_n_neuronas, n_features=(1 + len(exog_cols_scaled)),
#                                         learning_rate=best_learning_rate, dropout_rate=best_dropout_rate)
    

#     model_final_save_path = os.path.join(basin_output_dir, f'narx_model_{cuenca_name}.keras') # CAMBIADO a .keras
#     model_checkpoint_callback = ModelCheckpoint(
#         filepath=model_final_save_path,
#         monitor='val_loss',
#         save_best_only=True,
#         verbose=0
#     )
#     early_stopping_callback_final = EarlyStopping(
#         monitor='val_loss',
#         patience=20,
#         restore_best_weights=True,
#         verbose=1
#     )

#     model_final.fit(X_train_final, y_train_final, epochs=epochs_final_train, verbose=0,
#                     validation_split=0.1, callbacks=[early_stopping_callback_final, model_checkpoint_callback])

#     # Cargar el mejor modelo guardado si el checkpoint fue usado, de lo contrario usar el de la última época
#     final_trained_model = None
#     if os.path.exists(model_final_save_path): # Cargar el .keras
#         try:
#             final_trained_model = keras.models.load_model(model_final_save_path)
#             print(f"Cargado el mejor modelo guardado para {cuenca_name} desde .keras.")
#         except Exception as e_keras:
#             print(f"Error al cargar modelo .keras para {cuenca_name}: {e_keras}. Intentando cargar .h5...")
#             # Fallback a .h5 con custom_objects si .keras falla
#             h5_path = os.path.join(basin_output_dir, f'narx_model_{cuenca_name}.h5')
#             if os.path.exists(h5_path):
#                 try:
#                     final_trained_model = keras.models.load_model(h5_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
#                     print(f"Cargado el mejor modelo guardado para {cuenca_name} desde .h5 con custom_objects.")
#                 except Exception as e_h5:
#                     print(f"Error al cargar modelo .h5 para {cuenca_name} incluso con custom_objects: {e_h5}. Usando modelo de última época.")
#                     final_trained_model = model_final # Usar el modelo en memoria si no se pudo guardar/cargar
#             else:
#                 final_trained_model = model_final # Usar el modelo en memoria
#                 print(f"No se encontró ningún archivo de modelo guardado para {cuenca_name}, usando el de la última época.")
#     else: # Si no se guardó el .keras, intentar el .h5
#         h5_path = os.path.join(basin_output_dir, f'narx_model_{cuenca_name}.h5')
#         if os.path.exists(h5_path):
#             try:
#                 final_trained_model = keras.models.load_model(h5_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
#                 print(f"Cargado el mejor modelo guardado para {cuenca_name} desde .h5 con custom_objects.")
#             except Exception as e_h5_fallback:
#                 print(f"Error al cargar modelo .h5 para {cuenca_name} incluso con custom_objects: {e_h5_fallback}. Usando modelo de última época.")
#                 final_trained_model = model_final
#         else:
#             final_trained_model = model_final # Usar el modelo de la última época si no se encontró nada
#             print(f"No se encontró el mejor modelo guardado para {cuenca_name}, usando el de la última época.")


#     if final_trained_model is not None:
#         best_model = {'model': final_trained_model,
#                     'best_n_lags_area': best_n_lags_area,
#                     'best_params': best_params_basin} # Almacenar los mejores parámetros también

#         # --- Evaluación del modelo para la cuenca actual ---
#         current_scaler_area = preprocessed_data['scalers']['area']
        
#         # Prepara las secuencias para esta cuenca usando su mejor n_lags_area
#         basin_data_eval_df = preprocessed_data['data']['df']
#         train_data_eval_df = basin_data_eval_df.iloc[basin_data_final['train_idx']]
#         test_data_eval_df = basin_data_eval_df.iloc[basin_data_final['test_idx']]
#         val_data_eval_df = basin_data_eval_df.iloc[basin_data_final['val_idx']].copy()


#         sequences_for_eval = {
#             'train': {'X': create_sequences(train_data_eval_df, best_n_lags_area, exog_cols_scaled)[0],
#                         'y': create_sequences(train_data_eval_df, best_n_lags_area, exog_cols_scaled)[1]},
#             'test': {'X': create_sequences(test_data_eval_df, best_n_lags_area, exog_cols_scaled)[0],
#                         'y': create_sequences(test_data_eval_df, best_n_lags_area, exog_cols_scaled)[1]},
#             'val_df': val_data_eval_df # Pasar el DF para evaluate_validation
#         }

#         metrics_train, _, _ = evaluate_model(final_trained_model, sequences_for_eval['train'], current_scaler_area)
#         metrics_test, _, _ = evaluate_model(final_trained_model, sequences_for_eval['test'], current_scaler_area)
#         metrics_val, _, _ = evaluate_validation(final_trained_model, sequences_for_eval['val_df'], current_scaler_area, exog_cols, best_n_lags_area)
        
#         # Llama a evaluate_full_dataset y pasa el directorio de salida específico de la cuenca
#         metrics_full = evaluate_full_dataset(final_trained_model, preprocessed_data['data'],
#                                                 current_scaler_area,exog_cols_scaled, best_n_lags_area, 
#                                                 graph=True, base_model_dir=basin_output_dir, cuenca_name=cuenca_name)


#         current_basin_metrics = {
#             'best_params': best_params_basin,
#             'full_dataset': metrics_full,
#             'train': metrics_train,
#             'test': metrics_test,
#             'val': metrics_val
#         }
#         print(f"Métricas finales para {cuenca_name}:")
#         print(f"  Full Dataset (modo prediccion): {current_basin_metrics['full_dataset']}")
#         print(f"  Train: {current_basin_metrics['train']}")
#         print(f"  Test: {current_basin_metrics['test']}")
#         print(f"  Validation (modo prediccion): {current_basin_metrics['val']}")

#         # Guarda las métricas de esta cuenca en su propio archivo JSON
#         json_output_path = os.path.join(basin_output_dir, 'metrics.json')
#         with open(json_output_path, 'w', encoding='utf-8') as f:
#             json.dump(convert_numpy_to_python(current_basin_metrics), f, indent=4) # 'convert_numpy_to_python' debe estar definida
#         print(f"Métricas guardadas para {cuenca_name} en {json_output_path}")

#     else: # Si final_trained_model es None (no se pudo cargar/entrenar)
#         print(f"No se pudo cargar/entrenar el modelo final para Cuenca: {cuenca_name}. Saltando la evaluación.")
#         best_model = {'model': None, 'best_n_lags_area': None, 'best_params': None}
#         # Guardar un JSON con NaNs si no hay trials exitosos o modelo
#         current_basin_metrics = {
#             'best_params': None,
#             'full_dataset': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
#             'train': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
#             'test': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan},
#             'val': {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
#         }
#         json_output_path = os.path.join(basin_output_dir, 'metrics.json')
#         with open(json_output_path, 'w', encoding='utf-8') as f:
#             json.dump(convert_numpy_to_python(current_basin_metrics), f, indent=4)
#         print(f"Métricas (con NaNs) guardadas para {cuenca_name} en {json_output_path}")
else: # Si study_basin.best_trial is None (no hay trials exitosos)
    print(f"No successful trials for Cuenca: {cuenca_name}. Saltando entrenamiento y evaluación del modelo.")
    best_model = {'model': None, 'best_n_lags_area': None, 'best_params': None}
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

print("\nProceso de optimización y evaluación por cuenca completado.")