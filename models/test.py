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
from keras.callbacks import EarlyStopping
import optuna
# Instalar Optuna si no lo tienes


# --- Asegurarse del crecimiento de memoria de la GPU (si aplica) ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Crecimiento de memoria de GPU habilitado.")
    except RuntimeError as e:
        print(f"Error al configurar el crecimiento de memoria de GPU: {e}")

#%% FUNCIONES

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

def preprocess_data_single_basin(df_basin, exog_features, train_size=0.7, test_size=0.2):
    """
    Función para escalar los datos de area_nieve y exog_features para una sola cuenca.
    Retorna los DataFrames escalados y los scalers.
    """
    n_samples = len(df_basin)
    train_split = int(n_samples * train_size)
    test_split = int(n_samples * (train_size + test_size))

    train_idx = np.arange(train_split)
    val_idx = np.arange(test_split, n_samples) # Usaremos el conjunto de validación para Optuna
    test_idx = np.arange(train_split, test_split) # Todavía se usa para la división, pero no para Optuna

    scaler_area = StandardScaler()
    scaler_exog = StandardScaler()

    scaler_area.fit(df_basin.iloc[train_idx]['area_nieve'].values.reshape(-1, 1))
    scaler_exog.fit(df_basin.iloc[train_idx][exog_features])

    df_basin_scaled = df_basin.copy()
    df_basin_scaled.loc[:, 'area_nieve_scaled'] = scaler_area.transform(df_basin_scaled['area_nieve'].values.reshape(-1, 1))
    df_basin_scaled.loc[:, [f'{col}_scaled' for col in exog_features]] = scaler_exog.transform(df_basin_scaled[exog_features])

    return {
        'df_scaled': df_basin_scaled,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }, {
        'area': scaler_area,
        'exog': scaler_exog
    }

def create_sequences(data, n_lags, exog_cols_scaled, target_col_scaled='area_nieve_scaled'):
    X, y = [], []
    # Asegúrate de que haya suficientes datos para crear al menos una secuencia
    if len(data) <= n_lags:
        return np.array([]), np.array([])
        
    for i in range(len(data) - n_lags):
        seq_area = data[target_col_scaled].iloc[i : i + n_lags].values
        seq_exog = data[exog_cols_scaled].iloc[i : i + n_lags].values
        seq = np.hstack((seq_area.reshape(-1, 1), seq_exog))
        X.append(seq)
        y.append(data[target_col_scaled].iloc[i + n_lags])
    return np.array(X), np.array(y).reshape(-1, 1)

def create_narx_model(n_lags, n_layers, n_units_lstm, n_features, learning_rate=0.001):
    model = Sequential()
    if n_layers > 1:
        model.add(LSTM(n_units_lstm, activation='relu', input_shape=(n_lags, n_features), return_sequences=True))
        for _ in range(1, n_layers - 1): # Capas intermedias
            model.add(LSTM(n_units_lstm, activation='relu', return_sequences=True))
        model.add(LSTM(n_units_lstm, activation='relu')) # Última capa LSTM
    else:
        model.add(LSTM(n_units_lstm, activation='relu', input_shape=(n_lags, n_features)))

    model.add(Dropout(0.1))
    model.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=5.0)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def evaluate_validation(model, df_val_scaled, scaler_area, exog_cols, n_lags_area):
    n_exog_features = len(exog_cols)
    
    # Asegurarse de que haya suficientes datos en df_val_scaled para los lags
    if len(df_val_scaled) <= n_lags_area:
        print(f"Advertencia: Datos de validación insuficientes (solo {len(df_val_scaled)} puntos) para n_lags_area={n_lags_area}. Retornando NaN.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': -np.inf, 'KGE': np.nan}, None, None

    y_val_true = df_val_scaled['area_nieve'].values[n_lags_area:].reshape(-1, 1)
    y_val_pred_scaled = []

    first_sequence = df_val_scaled[['area_nieve_scaled'] + [col + '_scaled' for col in exog_cols]].iloc[:n_lags_area].values.reshape(1, n_lags_area, -1)
    last_sequence = first_sequence.copy()

    for i in range(len(df_val_scaled) - n_lags_area):
        pred_scaled = model.predict(last_sequence, verbose=0)
        y_val_pred_scaled.append(pred_scaled[0, 0])

        next_area_scaled = pred_scaled.reshape(1, 1, 1)
        # Asegurarse de que el índice para next_exog_scaled es válido
        if n_lags_area + i >= len(df_val_scaled):
            print(f"Advertencia: Índice fuera de límites al obtener next_exog_scaled. Se interrumpe la predicción autorregresiva.")
            break # Salir del bucle si no hay más datos exógenos

        next_exog_scaled = df_val_scaled[[col + '_scaled' for col in exog_cols]].iloc[n_lags_area + i].values.reshape(1, 1, n_exog_features)

        updated_area_sequence = np.concatenate([last_sequence[:, 1:, 0].reshape(1, n_lags_area - 1, 1), next_area_scaled], axis=1)
        updated_exog_sequence = np.concatenate([last_sequence[:, 1:, 1:], next_exog_scaled], axis=1)
        last_sequence = np.concatenate([updated_area_sequence, updated_exog_sequence], axis=2)

    try:
        y_val_pred_original = scaler_area.inverse_transform(np.array(y_val_pred_scaled).reshape(-1, 1))
    except ValueError as e:
        print(f"ERROR: {e} - Probablemente valores NaN/inf en las predicciones. Retornando NaN para las métricas.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': -np.inf, 'KGE': np.nan}, None, None

    if np.isnan(y_val_pred_original).any() or np.isinf(y_val_pred_original).any():
        print("ADVERTENCIA: Las predicciones originales (después de inverse_transform) contienen NaN/inf. Retornando NaN para las métricas.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': -np.inf, 'KGE': np.nan}, None, None

    # Asegurarse de que y_val_true y y_val_pred_original tienen la misma longitud después de cualquier interrupción
    min_len = min(len(y_val_true), len(y_val_pred_original))
    y_val_true = y_val_true[:min_len]
    y_val_pred_original = y_val_pred_original[:min_len]

    if min_len == 0: # Si no hay datos para comparar después de los lags/roturas
         return {'R2': np.nan, 'MAE': np.nan, 'NSE': -np.inf, 'KGE': np.nan}, None, None

    r2_val = pearsonr(y_val_true.flatten(), y_val_pred_original.flatten())
    r2_val = r2_val.statistic**2
    mae_val = mean_absolute_error(y_val_true, y_val_pred_original)
    nse_val = nash_sutcliffe_efficiency(y_val_true, y_val_pred_original)
    kge_val = kling_gupta_efficiency(y_val_true, y_val_pred_original)

    return {'R2': r2_val, 'MAE': mae_val, 'NSE': nse_val, 'KGE': kge_val}, y_val_pred_original, y_val_true


#%% Carga de datos
df_all = pd.read_csv('df_all.csv', index_col=0)

# Selecciona una cuenca para optimizar con Optuna (puedes hacer un bucle para todas las cuencas)
target_basin = 'adda-bornio' # O cualquier otra cuenca de tu DataFrame
df_target_basin = df_all[df_all['cuenca'] == target_basin].copy()

# Definir las características exógenas que vas a usar (estas se mantendrán fijas para la optimización actual)
# Si quieres optimizar también las características exógenas, tendrías que gestionarlo en la función objetivo.
base_exog_cols = ['dia_sen', 'temperatura', 'precipitacion', 'dias_sin_precip']

# Preprocesar los datos para la cuenca objetivo una única vez
scaled_data_basin, scalers_basin = preprocess_data_single_basin(df_target_basin, base_exog_cols)
df_scaled_basin = scaled_data_basin['df_scaled']
train_idx_basin = scaled_data_basin['train_idx']
val_idx_basin = scaled_data_basin['val_idx']
test_idx_basin = scaled_data_basin['test_idx'] # No usado directamente en Optuna, pero para referencia

scaler_area_basin = scalers_basin['area']
exog_cols_scaled_basin = [col + '_scaled' for col in base_exog_cols]

# Datos para el entrenamiento y validación
train_data_basin = df_scaled_basin.iloc[train_idx_basin]
val_data_basin = df_scaled_basin.iloc[val_idx_basin]


#%% FUNCIÓN OBJETIVO DE OPTUNA

def objective(trial):
    # 1. Sugerir hiperparámetros
    n_lags_area = trial.suggest_int('n_lags_area', 3, 10) # Rango de lags
    n_layers = trial.suggest_int('n_layers', 1, 3) # Número de capas LSTM
    n_units_lstm = trial.suggest_int('n_units_lstm', 5, 50) # Unidades LSTM
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True) # Tasa de aprendizaje
    
    # Se podría sugerir la combinación de exog_cols, pero es más complejo.
    # Por ahora, usamos las definidas globalmente (base_exog_cols)
    current_exog_cols_scaled = exog_cols_scaled_basin
    
    n_features = 1 + len(current_exog_cols_scaled) # area_nieve + exógenas

    # 2. Crear secuencias con los lags sugeridos
    X_train, y_train = create_sequences(train_data_basin, n_lags_area, current_exog_cols_scaled)
    # Validar que las secuencias no estén vacías
    if X_train.shape[0] == 0:
        print(f"Trial {trial.number}: No se pudieron crear secuencias de entrenamiento válidas para n_lags_area={n_lags_area}. Podando prueba.")
        raise optuna.TrialPruned() # Podar la prueba si no hay datos de entrenamiento

    # 3. Crear el modelo
    tf.keras.backend.clear_session() # Limpiar sesión de Keras para cada trial
    model = create_narx_model(n_lags=n_lags_area, 
                              n_layers=n_layers, 
                              n_units_lstm=n_units_lstm, 
                              n_features=n_features,
                              learning_rate=learning_rate)

    # Configurar EarlyStopping (puede ser con TrialPruned Callback para Optuna)
    # Optuna tiene su propio callback para EarlyStopping y pruning
    # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.TFKerasPruningCallback.html
    # Para simplificar, usamos el EarlyStopping de Keras por ahora.
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    # 4. Entrenar el modelo
    epochs = trial.suggest_int('epochs', 10, 60) # Puedes hacer que las épocas sean un hiperparámetro también si quieres
    try:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            verbose=0, # No imprimir el progreso de cada época
            validation_split=0.1, # Validación interna del fit para early stopping
            callbacks=[early_stopping_callback, 
                       optuna.integration.TFKerasPruningCallback(trial, 'val_loss')] # Callback de Optuna para poda
        )
    except Exception as e:
        print(f"Trial {trial.number}: Error durante el entrenamiento del modelo: {e}. Podando prueba.")
        tf.keras.backend.clear_session()
        del model
        gc.collect()
        raise optuna.TrialPruned() # Poda la prueba si hay un error en el entrenamiento

    # 5. Evaluar el modelo en el conjunto de validación (modo predicción autorregresiva)
    val_metrics, _, _ = evaluate_validation(model, val_data_basin, scaler_area_basin, base_exog_cols, n_lags_area)
    
    # Liberar recursos del modelo explícitamente después de la evaluación
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    # 6. Retornar la métrica a optimizar (Optuna minimiza por defecto, así que retornamos -NSE o 1 - NSE)
    if np.isnan(val_metrics['NSE']) or val_metrics['NSE'] == -np.inf:
        print(f"Trial {trial.number}: NSE es NaN o -inf. Podando prueba.")
        raise optuna.TrialPruned() # Poda la prueba si la métrica no es válida
    
    # Optimizaremos para maximizar el NSE, así que retornamos el negativo del NSE
    return -val_metrics['NSE']

#%% EJECUTAR OPTUNA
if __name__ == "__main__":
    import gc
    
    study_name = f"narx_hp_optimization_{target_basin.replace(' ', '_')}"
    storage_path = f"sqlite:///{study_name}.db" # Guarda el progreso en un archivo de base de datos SQLite

    # Crea un estudio de Optuna. Si el archivo de la base de datos ya existe, continuará desde donde se quedó.
    # direction='minimize' es el valor por defecto, pero lo especificamos para claridad ya que devolvemos -NSE
    study = optuna.create_study(study_name=study_name, storage=storage_path, direction='minimize', load_if_exists=True)

    print(f"Iniciando optimización para la cuenca: {target_basin}")
    print(f"Estudio de Optuna almacenado en: {storage_path}")

    # Ejecuta la optimización
    n_trials = 20 # Número de combinaciones a probar. Ajusta según tus recursos y tiempo.
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True) # gc_after_trial fuerza la recolección de basura

    print("\n--- Resultados de la optimización ---")
    print(f"Mejor prueba (Trial): {study.best_trial.number}")
    print(f"Mejor NSE: {-study.best_value}") # Invertimos el signo para mostrar el NSE real
    print("Mejores parámetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Opcional: Visualización de los resultados
    # Requiere instalar plotly y matplotlib
    # pip install plotly matplotlib
    try:
        import plotly
        import matplotlib.pyplot as plt

        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.show()

        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.show()

    except ImportError:
        print("Para visualizar los resultados, instala plotly y matplotlib: pip install plotly matplotlib")
    except Exception as e:
        print(f"Error al generar visualizaciones: {e}")

    # Puedes guardar el mejor modelo si lo necesitas, pero Optuna solo guarda los parámetros.
    # Para guardar el mejor modelo entrenado, tendrías que re-entrenarlo con `study.best_params`.
    print("\nPara entrenar el mejor modelo y guardarlo, puedes usar study.best_params y tu función create_narx_model.")