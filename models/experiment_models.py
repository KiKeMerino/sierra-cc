#%% IMPORTS
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
import gc

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

def preprocess_data(df, exog_features, train_size=0.7, test_size=0.2):
    """
        Función para escalar los datos de area_nieve y exog_features
    """
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
        # exog_features = ['temperatura', 'precipitacion', 'dias_sin_precip' , 'dia_sen', 'year', 'month']

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
        print(f"Cuenca: {cuenca}, Train: {len(train_idx)}, Test: {len(test_idx)}, Val: {len(val_idx)}")

    return scaled_data, scalers, cuencas

def create_sequences(data, n_lags, exog_cols_scaled, target_col_scaled='area_nieve_scaled'):
    X, y = [], []
    for i in range(len(data) - n_lags):
        seq_area = data[target_col_scaled].iloc[i : i + n_lags].values
        seq_exog = data[exog_cols_scaled].iloc[i : i + n_lags].values
        seq = np.hstack((seq_area.reshape(-1, 1), seq_exog))
        X.append(seq)
        y.append(data[target_col_scaled].iloc[i + n_lags])
    return np.array(X), np.array(y).reshape(-1, 1)

def create_narx_model(n_lags, n_layers, n_units_lstm, n_features):
    model = Sequential()

    if n_layers > 1:
        model.add(LSTM(n_units_lstm, activation='relu', input_shape=(n_lags, n_features), return_sequences=True))
    else:
        model.add(LSTM(n_units_lstm, activation='relu', input_shape=(n_lags, n_features)))

    # Capas LSTM adicionales
    for _ in range(1, n_layers):
        if _ == n_layers - 1: # Si es la última capa LSTM, no devuelve secuencias
            model.add(LSTM(n_units_lstm, activation='relu'))
        else:
            model.add(LSTM(n_units_lstm, activation='relu', return_sequences=True))

    model.add(Dropout(0.1)) # Añadimos una capa Dropout
    model.add(Dense(1))
    
    # Optimizador con recorte de gradientes
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=5.0) # clipnorm para el recorte
    model.compile(optimizer=optimizer, loss='mse')

    return model

def create_train_models(sequences_data, n_lags_area, layers, units, epochs, exog_cols_scaled, cuencas, save, models_dir='narx_models'):
    models = {}
    history = {}
    n_features = 1 + len(exog_cols_scaled)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    for cuenca in cuencas:
        model_path = os.path.join(models_dir, f'narx_model_{cuenca}.h5')

        # --- Lógica para cargar modelo existente (ya la tienes) ---
        if os.path.exists(model_path):
            try:
                # Es crucial liberar el modelo antes de cargar otro si la memoria es crítica
                if cuenca in models: # Si un modelo anterior de esta cuenca estaba en el diccionario
                    del models[cuenca]
                    tf.keras.backend.clear_session()
                    gc.collect()
                
                print(f"Cargando modelo existente para la cuenca: {cuenca} desde {model_path}")
                model = keras.models.load_model(model_path)
                models[cuenca] = model
                history[cuenca] = None 
                continue 
            except Exception as e:
                print(f"Error al cargar el modelo {model_path}: {e}. Re-entrenando.")
                # Si falla al cargar, procedemos a entrenar

        # Si el modelo no existe o hubo un error al cargar, entrenar
        model = create_narx_model(n_lags = n_lags_area, n_layers=layers, n_units_lstm=units, n_features=n_features)
        X_train = sequences_data[cuenca]['X_train']
        y_train = sequences_data[cuenca]['y_train']
        
        if X_train.shape[0] == 0:
            print(f"Advertencia: No hay suficientes datos para crear secuencias de entrenamiento para la cuenca {cuenca}. Saltando entrenamiento.")
            # Es importante asegurarse de que el modelo no se añade a 'models' si no se entrena
            if cuenca in models: del models[cuenca] # Limpiar si existe por algún error anterior
            continue 

        print(f"Entrenando modelo para la cuenca: {cuenca}...")
        model_history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_split=0.1, callbacks=[early_stopping_callback])
        models[cuenca] = model
        history[cuenca] = model_history

        if save:
            model.save(model_path)
            # print(f"Modelo entrenado y guardado para la cuenca: {cuenca} en {model_path}")
        
        # --- Liberar memoria después de cada entrenamiento/carga de modelo de cuenca ---
        del model # Elimina la referencia al objeto del modelo
        tf.keras.backend.clear_session() # Limpia el grafo de Keras
        gc.collect() # Fuerza la recolección de basura
        # Puedes añadir este print para ver la liberación
        # print(f"Memoria liberada después de procesar cuenca {cuenca}")

    return models


def load_models(cuencas, models_dir='models'):
    loaded_models = {}
    for cuenca in cuencas:
        model_path = os.path.join(models_dir, f'narx_model_{cuenca}.h5')
        if os.path.exists(model_path):
            loaded_models[cuenca] = keras.models.load_model(model_path)
            # print(f"Modelo cargado para la cuenca: {cuenca} desde {model_path}")
        else:
            print(f"No se encontró el modelo para la cuenca: {cuenca} en {model_path}")
    return loaded_models

def evaluate_model(model, sequences, scaler_area):
    X = sequences['X']
    y_true = sequences['y']
    y_pred_scaled = model.predict(X, verbose=0)
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
    y_val_true = df_val_scaled['area_nieve'].values[n_lags_area:].reshape(-1, 1)
    y_val_pred_scaled = []

    # Cojo las 3 primeras filas del dataframe real para empezar
    first_sequence = df_val_scaled[['area_nieve_scaled'] + [col + '_scaled' for col in exog_cols]].iloc[:n_lags_area].values.reshape(1, n_lags_area, -1)
    last_sequence = first_sequence.copy()

    for i in range(len(df_val_scaled) - n_lags_area):
        pred_scaled = model.predict(last_sequence, verbose=0)
        y_val_pred_scaled.append(pred_scaled[0, 0])

        next_area_scaled = pred_scaled.reshape(1, 1, 1)
        next_exog_scaled = df_val_scaled[[col + '_scaled' for col in exog_cols]].iloc[n_lags_area + i].values.reshape(1, 1, n_exog_features)

        updated_area_sequence = np.concatenate([last_sequence[:, 1:, 0].reshape(1, n_lags_area - 1, 1), next_area_scaled], axis=1)
        updated_exog_sequence = np.concatenate([last_sequence[:, 1:, 1:], next_exog_scaled], axis=1)
        last_sequence = np.concatenate([updated_area_sequence, updated_exog_sequence], axis=2)

    try:
        y_val_pred_original = scaler_area.inverse_transform(np.array(y_val_pred_scaled).reshape(-1, 1))
    except ValueError as e:
        # Si hay un error de ValueError, retornamos métricas NaN para indicar el fallo
        print(f"ERROR: {e} - Probablemente valores NaN/inf en las predicciones. Retornando NaN para las métricas.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': -np.inf, 'KGE': np.nan}, None, None # Usar -np.inf para NSE lo descarta en la búsqueda

    # También es una buena práctica verificar si las predicciones resultantes (después de inverse_transform)
    # aún contienen NaN o inf, lo que podría indicar un problema posterior o que el try-except no atrapó todo.
    if np.isnan(y_val_pred_original).any() or np.isinf(y_val_pred_original).any():
        print("ADVERTENCIA: Las predicciones originales (después de inverse_transform) contienen NaN/inf. Retornando NaN para las métricas.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': -np.inf, 'KGE': np.nan}, None, None


    r2_val = pearsonr(y_val_true.flatten(), y_val_pred_original.flatten())
    r2_val = r2_val.statistic**2
    mae_val = mean_absolute_error(y_val_true, y_val_pred_original)
    nse_val = nash_sutcliffe_efficiency(y_val_true, y_val_pred_original)
    kge_val = kling_gupta_efficiency(y_val_true, y_val_pred_original)

    return {'R2': r2_val, 'MAE': mae_val, 'NSE': nse_val, 'KGE': kge_val}, y_val_pred_original, y_val_true

def evaluate_full_dataset(models, scaled_data, scalers, cuencas, n_lags_area, exog_cols_scaled, graph=None, model_dir='./'):
    full_metrics = {}

    for cuenca in cuencas:
        model = models[cuenca]
        scaler_area = scalers[cuenca]['area']
        
        # Recuperar el DataFrame completo escalado para esta cuenca
        df_full_scaled_cuenca = scaled_data[cuenca]['df'].copy()

        n_exog_features = len(exog_cols_scaled)

        # Inicializar listas para almacenar las predicciones y los valores reales para toda la serie
        y_full_true_original = df_full_scaled_cuenca['area_nieve'].values[n_lags_area:].reshape(-1, 1)
        y_full_pred_scaled = []

        # Usar los primeros 'n_lags_area' valores reales de todo el conjunto de datos para iniciar la predicción
        first_sequence_full = df_full_scaled_cuenca[['area_nieve_scaled'] + exog_cols_scaled].iloc[:n_lags_area].values.reshape(1, n_lags_area, -1)
        last_sequence_full = first_sequence_full.copy()

        # Realizar la predicción paso a paso para todo el resto del DataFrame
        for i in range(len(df_full_scaled_cuenca) - n_lags_area):
            pred_scaled = model.predict(last_sequence_full, verbose=0)
            y_full_pred_scaled.append(pred_scaled[0, 0])

            # Crear la siguiente secuencia de entrada
            next_area_scaled = pred_scaled.reshape(1, 1, 1)
            next_exog_scaled = df_full_scaled_cuenca[exog_cols_scaled].iloc[n_lags_area + i].values.reshape(1, 1, n_exog_features)

            # Descartar el valor más antiguo del área de nieve y agregar la nueva predicción
            updated_area_sequence = np.concatenate([last_sequence_full[:, 1:, 0].reshape(1, n_lags_area - 1, 1), next_area_scaled], axis=1)

            # Descartar el valor más antiguo de las variables exógenas y agregar el nuevo valor
            updated_exog_sequence = np.concatenate([last_sequence_full[:, 1:, 1:], next_exog_scaled], axis=1)

            # Concatenar las secuencias actualizadas
            last_sequence_full = np.concatenate([updated_area_sequence, updated_exog_sequence], axis=2)

        # Invertir el escalado de las predicciones para todo el conjunto
        y_full_pred_original = scaler_area.inverse_transform(np.array(y_full_pred_scaled).reshape(-1, 1))

        # Calcular las métricas en todo el conjunto de datos (excepto los primeros n_lags)
        r2_full = pearsonr(y_full_true_original.flatten(), y_full_pred_original.flatten())
        r2_full = r2_full.statistic**2
        mae_full = mean_absolute_error(y_full_true_original, y_full_pred_original)
        nse_full = nash_sutcliffe_efficiency(y_full_true_original, y_full_pred_original)
        kge_full = kling_gupta_efficiency(y_full_true_original, y_full_pred_original)

        full_metrics[cuenca] = {'R2': r2_full, 'MAE': mae_full, 'NSE': nse_full, 'KGE': kge_full}
        # print(f"Métricas en todo el conjunto de datos (modo prediccion) para {cuenca}: R2={r2_full:.3f}, MAE={mae_full:.3f}, NSE={nse_full:.3f}, KGE={kge_full:.3f}")

        if graph != None:
            real_plot = df_full_scaled_cuenca.iloc[n_lags_area:][['fecha','area_nieve']]
            y_full_pred_df = pd.DataFrame(y_full_pred_original, columns=['area_nieve_pred'], index=real_plot.index)
            df_plot = pd.concat([real_plot, y_full_pred_df], axis=1)
            df_plot.fecha = pd.to_datetime(df_plot.fecha, format='%Y-%m-%d')
            if graph == 'per_day':
                df_plot.fecha = df_plot.fecha.dt.day_of_year
                plt.xlabel("Day")
                plt.xlim(right=366)
            elif graph == 'per_month':
                df_plot.fecha = df_plot.fecha.dt.month # Cambiado a mes
                plt.xlabel("Month")
                plt.xlim(right=12)
            df_plot_grouped = df_plot.groupby('fecha').agg(
                area_nieve_real = ('area_nieve', 'mean'),
                area_nieve_pred=('area_nieve_pred', 'mean')
            )

            sns.lineplot(x=df_plot_grouped.index, y=df_plot_grouped.area_nieve_real, label='Real area')
            sns.lineplot(x=df_plot_grouped.index, y=df_plot_grouped.area_nieve_pred, label='Prediction')
            plt.xlim(left=0)
            plt.title(f'Prediction vs Real {cuenca}')
            plt.ylabel("Snow area Km2")
            plt.legend()
            plt.grid(True)
            output_path = os.path.join(model_dir, graph)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.savefig(os.path.join(output_path, cuenca))
            plt.close()
    
    return full_metrics

## Bucle de Optimización de Hiperparámetros

#%% --- Main execution ---
df = pd.read_csv('df_all.csv', index_col=0)

# Definir las combinaciones de parámetros a probar
# Puedes ajustar estos rangos y valores según tus necesidades.
param_combinations = []

# Prueba con diferentes lags de área
for n_lags_area in [3, 5, 7]: 
    # Prueba con diferentes números de capas LSTM
    for n_layers in [1, 2]: 
        # Prueba con diferentes unidades LSTM
        for n_neuronas in [5, 10, 15, 30]: 
            # Prueba con diferentes conjuntos de características exógenas
            exog_options = [
                ['dia_sen', 'temperatura', 'precipitacion'],
                ['dia_sen', 'temperatura', 'precipitacion', 'dias_sin_precip'],
                ['dia_sen', 'temperatura', 'precipitacion', 'precipitacion_bool', 'dias_sin_precip']
            ]
            for exog_cols in exog_options:
                param_combinations.append({
                    'n_lags_area': n_lags_area,
                    'n_layers': n_layers,
                    'n_neuronas': n_neuronas,
                    'exog_cols': exog_cols
                })

epochs = 30 # Las épocas se mantienen fijas para esta búsqueda
base_model_dir = os.path.join("E:", "models")

# Diccionario para almacenar los mejores modelos y métricas por cuenca
best_models_per_cuenca = {cuenca: {'metrics': {'NSE': -np.inf}, 'params': None, 'path': None} for cuenca in df['cuenca'].unique()}

processed_combinations_file = os.path.join(base_model_dir, 'processed_combinations_log.json')
if os.path.exists(processed_combinations_file):
    with open(processed_combinations_file, 'r', encoding='utf-8') as f:
        processed_combinations_log = json.load(f)
    print(f"Cargado el progreso anterior desde {processed_combinations_file}")
else:
    processed_combinations_log = {}

# Si tienes un archivo best_models_summary.json previo, cárgalo también
output_best_models_path = os.path.join(base_model_dir, 'best_models_summary.json')
if os.path.exists(output_best_models_path):
    with open(output_best_models_path, 'r', encoding='utf-8') as f:
        loaded_best_models = json.load(f)
        # Reconstruir best_models_per_cuenca desde lo cargado
        for cuenca, data in loaded_best_models.items():
            # Asegúrate de que 'metrics' tenga 'NSE' y que 'path' sea una cadena si lo era antes
            if 'NSE' in data['metrics'] and data['metrics']['NSE'] is not None: # Verifica que NSE no sea null
                best_models_per_cuenca[cuenca] = {
                    'metrics': data['metrics'],
                    'params': data['params'],
                    'path': data['path']
                }
            else: # Si NSE es null o no existe, inicializa con -inf
                best_models_per_cuenca[cuenca] = {'metrics': {'NSE': -np.inf}, 'params': None, 'path': None}
    print(f"Cargado el resumen de mejores modelos anterior desde {output_best_models_path}")

for i, params in enumerate(param_combinations):
    n_lags_area = params['n_lags_area']
    n_layers = params['n_layers']
    n_neuronas = params['n_neuronas']
    exog_cols = params['exog_cols']

    param_name = f"lags_{n_lags_area}_layers_{n_layers}_units_{n_neuronas}_exog_{'_'.join(exog_cols)}"
    current_model_dir = os.path.join(base_model_dir, param_name)

    # --- Saltar si ya se procesó ---
    if param_name in processed_combinations_log:
        print(f"--- Saltando combinación {i+1}/{len(param_combinations)} ({param_name}): Ya procesada ---")
        continue # Pasa a la siguiente combinación

    print(f"\n--- Probando combinación {i+1}/{len(param_combinations)}: {param_name} ---")

    scaled_data, scalers, cuencas = preprocess_data(df, exog_cols)
    exog_cols_scaled = [col + '_scaled' for col in exog_cols]

    sequences_data = {}
    for cuenca_name in cuencas: # Usar cuenca_name para evitar conflicto con la variable global 'cuencas'
        train_data = scaled_data[cuenca_name]['df'].iloc[scaled_data[cuenca_name]['train_idx']]
        val_data = scaled_data[cuenca_name]['df'].iloc[scaled_data[cuenca_name]['val_idx']]
        test_data = scaled_data[cuenca_name]['df'].iloc[scaled_data[cuenca_name]['test_idx']]

        # Asegurarse de que hay suficientes datos para crear secuencias antes de intentar
        if len(train_data) <= n_lags_area or len(val_data) <= n_lags_area or len(test_data) <= n_lags_area:
            print(f"Advertencia: No hay suficientes datos para crear secuencias con n_lags_area={n_lags_area} para la cuenca {cuenca_name}. Saltando esta cuenca para esta combinación.")
            # Si no hay suficientes datos para una cuenca, registramos métricas inválidas para ella
            # y pasamos a la siguiente cuenca o combinación si todas fallan.
            sequences_data[cuenca_name] = { # Todavía necesitamos la entrada para la cuenca en sequences_data
                'X_train': np.array([]), 'y_train': np.array([]),
                'X_val': np.array([]), 'y_val': np.array([]),
                'X_test': np.array([]), 'y_test': np.array([])
            }
            continue # Pasa a la siguiente cuenca en este bucle interno

        sequences_data[cuenca_name] = {
            'X_train': create_sequences(train_data, n_lags_area, exog_cols_scaled)[0],
            'y_train': create_sequences(train_data, n_lags_area, exog_cols_scaled)[1],
            'X_val': create_sequences(val_data, n_lags_area, exog_cols_scaled)[0],
            'y_val': create_sequences(val_data, n_lags_area, exog_cols_scaled)[1],
            'X_test': create_sequences(test_data, n_lags_area, exog_cols_scaled)[0],
            'y_test': create_sequences(test_data, n_lags_area, exog_cols_scaled)[1],
        }

    # Entrenar y guardar los modelos para esta combinación
    models = create_train_models(sequences_data, n_lags_area, n_layers, n_neuronas, epochs, exog_cols_scaled, cuencas, True, models_dir=current_model_dir)

    # Evaluar los modelos en el conjunto de validación y actualizar los mejores por cuenca
    current_metrics = {}
    for cuenca_name in cuencas: # Usar cuenca_name para evitar conflicto
        # Si no hay modelo para la cuenca (ej. no se pudo entrenar por falta de datos)
        if cuenca_name not in models:
            current_metrics[cuenca_name] = {'R2': np.nan, 'MAE': np.nan, 'NSE': -np.inf, 'KGE': np.nan}
            print(f"Cuenca: {cuenca_name} - Modelo no disponible para esta combinación. Saltando evaluación.")
            continue

        model = models[cuenca_name]
        scaler_area = scalers[cuenca_name]['area']

        # Evaluar en el conjunto de validación (modo predicción autorregresiva)
        df_val_scaled = scaled_data[cuenca_name]['df'].iloc[scaled_data[cuenca_name]['val_idx']].copy()
        val_metrics, _, _ = evaluate_validation(model, df_val_scaled, scaler_area, exog_cols, n_lags_area)
        current_metrics[cuenca_name] = val_metrics

        # Verificar si las métricas son válidas antes de comparar
        if np.isnan(val_metrics['NSE']): # Si NSE es NaN, el modelo fue inestable
            print(f"Cuenca: {cuenca_name} - Modelo inestable (NaN/inf predicciones). Saltando evaluación.")
            continue # Salta esta cuenca para esta combinación, no la considerará como la mejor

        print(f"Cuenca: {cuenca_name} - Métricas (Val): R2={val_metrics['R2']:.3f}, MAE={val_metrics['MAE']:.3f}, NSE={val_metrics['NSE']:.3f}, KGE={val_metrics['KGE']:.3f}")

        # Comparar con el mejor modelo actual para esta cuenca (usando NSE como métrica principal)
        if val_metrics['NSE'] > best_models_per_cuenca[cuenca_name]['metrics']['NSE']:
            best_models_per_cuenca[cuenca_name]['metrics'] = val_metrics
            best_models_per_cuenca[cuenca_name]['params'] = params
            best_models_per_cuenca[cuenca_name]['path'] = os.path.join(current_model_dir, f'narx_model_{cuenca_name}.h5')
            print(f"  --> ¡Nueva mejor configuración para {cuenca_name}!")

    # --- Liberar memoria después de CADA COMBINACIÓN de parámetros ---
    tf.keras.backend.clear_session()
    gc.collect()
    print(f"Memoria liberada después de procesar la combinación: {param_name}")

    # Guardar las métricas de esta combinación para todas las cuencas
    # Esto indica que esta combinación ha sido evaluada (incluso si falló en alguna cuenca)
    processed_combinations_log[param_name] = current_metrics
    with open(processed_combinations_file, 'w', encoding='utf-8') as f:
        json.dump(processed_combinations_log, f, indent=4)

    # Guardar el resumen de los mejores modelos en cada iteración por si vuelve a fallar
    best_models_json_friendly = {}
    for c, data in best_models_per_cuenca.items():
        best_models_json_friendly[c] = {
            'metrics': data['metrics'],
            'params': data['params'],
            'path': data['path']
        }

    with open(output_best_models_path, 'w', encoding='utf-8') as f:
        json.dump(best_models_json_friendly, f, indent=4)


print("\n--- Resumen de los mejores modelos por cuenca ---")
for cuenca, data in best_models_per_cuenca.items():
    print(f"\nCuenca: {cuenca}")
    print(f"  Mejores Parámetros: {data['params']}")
    print(f"  Mejores Métricas (Val): {data['metrics']}")
    print(f"  Ruta del Modelo: {data['path']}")

# Opcional: Cargar los mejores modelos y generar gráficos para ellos
print("\n--- Generando gráficos para los mejores modelos ---")
for cuenca_name, data in best_models_per_cuenca.items():
    if data['path'] and os.path.exists(data['path']):
        print(f"Cargando y evaluando el mejor modelo para {cuenca_name} desde {data['path']}")

        params_best_cuenca = data['params']
        # Asegúrate de que los exog_cols existen en params_best_cuenca
        if 'exog_cols' not in params_best_cuenca or params_best_cuenca['exog_cols'] is None:
            print(f"No se pudieron encontrar las columnas exógenas para la cuenca {cuenca_name}. Saltando gráficos.")
            continue

        scaled_data_best, scalers_best, _ = preprocess_data(df, params_best_cuenca['exog_cols'], train_size=0.7, test_size=0.2)
        exog_cols_scaled_best = [col + '_scaled' for col in params_best_cuenca['exog_cols']]

        model_best = keras.models.load_model(data['path'])

        best_model_graph_dir = os.path.dirname(data['path'])
        # Asegúrate de que el directorio del gráfico exista antes de llamar
        if not os.path.exists(os.path.join(best_model_graph_dir, 'per_day')):
            os.makedirs(os.path.join(best_model_graph_dir, 'per_day'))

        evaluate_full_dataset(
            {cuenca_name: model_best}, 
            scaled_data_best, 
            scalers_best, 
            [cuenca_name], 
            params_best_cuenca['n_lags_area'], 
            exog_cols_scaled_best, 
            'per_day', 
            best_model_graph_dir
        )
        print(f"Gráfico generado para {cuenca_name} en {os.path.join(best_model_graph_dir, 'per_day', cuenca_name)}")
    else:
        print(f"No se encontró un mejor modelo guardado para la cuenca {cuenca_name} o la ruta no es válida.")