import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import os
import json
import matplotlib.pyplot as plt # Necesario si graficas
import seaborn as sns # Necesario si graficas

# --- FUNCIONES DE SOPORTE (Mantienen su funcionalidad original) ---

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
    test_split = int(n_samples * (train_size + test_size)) # Corregido: test_split es un índice, no un tamaño relativo

    train_idx = np.arange(train_split)
    test_idx = np.arange(train_split, test_split) # Corregido: indices para test
    val_idx = np.arange(test_split, n_samples) # Corregido: indices para validación

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

# Las funciones evaluate_model, evaluate_validation y evaluate_full_dataset
# no se necesitan directamente para la predicción futura aquí,
# pero las mantengo por si las usas para otras evaluaciones.
# He corregido las comprobaciones de longitud y NaN/Inf dentro de ellas para mayor robustez.

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

    # Añadir comprobación de longitud aquí también, aunque para train/test/val de create_sequences
    # las longitudes deberían coincidir por construcción.
    if y_pred_original.shape[0] != y_true_original.shape[0]:
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

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
            
    if not y_val_pred_scaled:
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    y_val_pred_original = scaler_area.inverse_transform(np.array(y_val_pred_scaled).reshape(-1, 1))

    # --- NUEVA COMPROBACIÓN DE LONGITUD Y VALORES ---
    if y_val_pred_original.shape[0] != y_val_true_original.shape[0]:
        print(f"Advertencia: Longitud de predicciones ({y_val_pred_original.shape[0]}) no coincide con valores reales ({y_val_true_original.shape[0]}).")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None
    
    if np.any(np.isnan(y_val_pred_original)) or np.any(np.isinf(y_val_pred_original)):
        print(f"Advertencia: Predicciones transformadas contienen NaN/inf.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}, None, None

    r2_val = pearsonr(y_val_true_original.flatten(), y_val_pred_original.flatten())
    r2_val = r2_val.statistic**2
    mae_val = mean_absolute_error(y_val_true_original, y_val_pred_original)
    nse_val = nash_sutcliffe_efficiency(y_val_true_original, y_val_pred_original)
    kge_val = kling_gupta_efficiency(y_val_true_original, y_val_pred_original)

    return {'R2': r2_val, 'MAE': mae_val, 'NSE': nse_val, 'KGE': kge_val}, y_val_pred_original, y_val_true_original

def evaluate_full_dataset(model, scaled_data, scaler_area, n_lags_area, exog_cols_scaled, graph=False, model_dir='./', cuenca_name=""):
    # Esta función se modificó para que acepte un solo modelo y sus datos
    full_metrics = {}

    df_full_scaled_cuenca = scaled_data['df'].copy()

    n_exog_features = len(exog_cols_scaled)

    if len(df_full_scaled_cuenca) < n_lags_area + 1:
        print(f"Advertencia: Datos completos de la cuenca ({len(df_full_scaled_cuenca)}) son muy cortos para n_lags={n_lags_area}.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    y_full_true_original = df_full_scaled_cuenca['area_nieve'].values[n_lags_area:].reshape(-1, 1)
    y_full_pred_scaled = []

    first_sequence_full = df_full_scaled_cuenca[['area_nieve_scaled'] + exog_cols_scaled].iloc[:n_lags_area].values.reshape(1, n_lags_area, -1)
    last_sequence_full = first_sequence_full.copy()

    for i in range(len(df_full_scaled_cuenca) - n_lags_area):
        if np.any(np.isnan(last_sequence_full)) or np.any(np.isinf(last_sequence_full)):
            y_full_pred_scaled.extend([np.nan] * (len(df_full_scaled_cuenca) - n_lags_area - i))
            break

        pred_scaled = model.predict(last_sequence_full, verbose=0)

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
        print(f"Advertencia: No se generaron predicciones para el conjunto completo.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    y_full_pred_original = scaler_area.inverse_transform(np.array(y_full_pred_scaled).reshape(-1, 1))

    # --- NUEVA COMPROBACIÓN DE LONGITUD Y VALORES ---
    if y_full_pred_original.shape[0] != y_full_true_original.shape[0]:
        print(f"Advertencia: Longitud de predicciones ({y_full_pred_original.shape[0]}) no coincide con valores reales ({y_full_true_original.shape[0]}) para el conjunto completo.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}
    
    if np.any(np.isnan(y_full_pred_original)) or np.any(np.isinf(y_full_pred_original)):
        print(f"Advertencia: Predicciones transformadas para el conjunto completo contienen NaN/inf.")
        return {'R2': np.nan, 'MAE': np.nan, 'NSE': np.nan, 'KGE': np.nan}

    r2_full = pearsonr(y_full_true_original.flatten(), y_full_pred_original.flatten())
    r2_full = r2_full.statistic**2
    mae_full = mean_absolute_error(y_full_true_original, y_full_pred_original)
    nse_full = nash_sutcliffe_efficiency(y_full_true_original, y_full_pred_original)
    kge_full = kling_gupta_efficiency(y_full_true_original, y_full_pred_original)

    full_metrics = {'R2': r2_full, 'MAE': mae_full, 'NSE': nse_full, 'KGE': kge_full} # Corregido: NSE no era KGE
    
    # --- GRÁFICOS (mantengo la funcionalidad de gráficos si graph=True) ---
    if graph == True:
        # Asegurarse de que df_full_scaled_cuenca tiene la columna 'fecha'
        real_plot = df_full_scaled_cuenca.iloc[n_lags_area:].copy()
        real_plot['area_nieve'] = y_full_true_original # Valores reales en escala original
        y_full_pred_df = pd.DataFrame(y_full_pred_original, columns=['area_nieve_pred'], index=real_plot.index)
        df_plot = pd.concat([real_plot, y_full_pred_df], axis=1)
        df_plot['fecha'] = pd.to_datetime(df_plot['fecha'], format='%Y-%m-%d') # Asegurar formato

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
            
            output_path = os.path.join(model_dir, f'graphs_{cuenca_name}')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.savefig(os.path.join(output_path, f'{graph_type}.png'))
            plt.close()
    
    return full_metrics


# --- NUEVA FUNCIÓN PARA HACER PREDICCIONES FUTURAS ---
def make_future_predictions(model, historical_df, future_exog_df, exog_features, n_lags_area, scaler_area, scaler_exog):
    """
    Genera predicciones futuras de area_nieve_scaled, inicializando
    los lags de area_nieve con la media histórica para el día del año correspondiente.

    Args:
        model (keras.Model): El modelo NARX entrenado.
        historical_df (pd.DataFrame): DataFrame con datos históricos para calcular medias.
                                      Debe contener 'fecha' y 'area_nieve'.
        future_exog_df (pd.DataFrame): DataFrame con fechas futuras y variables exógenas futuras.
                                      Debe contener 'fecha' y las exog_features.
        exog_features (list): Lista de nombres de columnas de variables exógenas.
        n_lags_area (int): Número de rezagos de 'area_nieve' que el modelo usa.
        scaler_area (StandardScaler): Escalador ajustado para 'area_nieve'.
        scaler_exog (StandardScaler): Escalador ajustado para las variables exógenas.

    Returns:
        pd.DataFrame: DataFrame con fechas futuras y las predicciones de area_nieve.
    """
    print("\n--- Iniciando predicciones futuras ---")
    
    # 1. Calcular la media histórica de area_nieve por día del año
    historical_df['fecha'] = pd.to_datetime(historical_df['fecha'])
    historical_df['day_of_year'] = historical_df['fecha'].dt.day_of_year
    daily_avg_area_nieve = historical_df.groupby('day_of_year')['area_nieve'].mean()

    # 2. Preparar el DataFrame de datos futuros
    future_exog_df_processed = future_exog_df.copy()
    future_exog_df_processed['fecha'] = pd.to_datetime(future_exog_df_processed['fecha'])
    future_exog_df_processed['day_of_year'] = future_exog_df_processed['fecha'].dt.day_of_year
    
    # Escalar las variables exógenas futuras
    exog_cols_scaled = [f'{col}_scaled' for col in exog_features]
    future_exog_df_processed.loc[:, exog_cols_scaled] = scaler_exog.transform(future_exog_df_processed[exog_features])

    # 3. Inicializar la primera secuencia de entrada para el modelo
    # Esto usa las medias históricas para los lags de area_nieve_scaled
    initial_area_nieve_lags_scaled = []
    for i in range(n_lags_area):
        # Obtener el día del año para la fecha correspondiente al lag
        # Asumimos que la primera predicción es para future_exog_df_processed.iloc[0]
        # Entonces el primer lag es para (fecha_futura[0] - n_lags_area días)
        # Esto requiere cuidado si la 'fecha' en future_exog_df_processed
        # es el punto de inicio de la serie futura.
        
        # Para el primer conjunto de predicciones (los primeros 'n_lags_area' días en el futuro),
        # usaremos las medias históricas de 'area_nieve'.
        # Para cada lag, necesitamos la media del 'day_of_year' correspondiente al lag.
        
        # Si la predicción comienza en future_exog_df_processed.iloc[0],
        # el primer lag es para la fecha de future_exog_df_processed.iloc[0] - 1 día
        # el segundo lag para future_exog_df_processed.iloc[0] - 2 días
        # y así sucesivamente.
        
        # Calculamos la fecha para cada lag
        lag_date = future_exog_df_processed['fecha'].iloc[0] - pd.Timedelta(days=(n_lags_area - 1 - i))
        lag_day_of_year = lag_date.day_of_year
        
        avg_val = daily_avg_area_nieve.get(lag_day_of_year, daily_avg_area_nieve.mean()) # Usar la media global si el día del año no existe
        initial_area_nieve_lags_scaled.append(scaler_area.transform(np.array([[avg_val]]))[0,0])

    initial_exog_lags_scaled = future_exog_df_processed[exog_cols_scaled].iloc[:n_lags_area].values
    
    # Asegúrate de que las longitudes coincidan
    if len(initial_area_nieve_lags_scaled) != n_lags_area or initial_exog_lags_scaled.shape[0] != n_lags_area:
        print("Error: No se pudo inicializar la secuencia de lags. Verifica la longitud de los datos futuros o n_lags_area.")
        return None

    # Concatenar para formar la primera secuencia de entrada del modelo
    last_sequence_input = np.hstack((np.array(initial_area_nieve_lags_scaled).reshape(-1, 1), initial_exog_lags_scaled))
    last_sequence_input = last_sequence_input.reshape(1, n_lags_area, -1) # Reshape a (1, n_lags, n_features)

    predicted_area_nieve_scaled = []

    # 4. Bucle de predicción recursiva (walk-forward para el futuro)
    for i in range(len(future_exog_df_processed)):
        if np.any(np.isnan(last_sequence_input)) or np.any(np.isinf(last_sequence_input)):
            print(f"Advertencia: Secuencia de entrada para predicción futura contiene NaN/inf en el paso {i}. Deteniendo predicciones futuras.")
            predicted_area_nieve_scaled.extend([np.nan] * (len(future_exog_df_processed) - i))
            break
        
        pred_scaled = model.predict(last_sequence_input, verbose=0)

        if np.any(np.isnan(pred_scaled)) or np.any(np.isinf(pred_scaled)):
            print(f"Advertencia: Modelo predijo NaN/inf para el paso futuro {i}. Deteniendo predicciones futuras.")
            predicted_area_nieve_scaled.append(np.nan)
            predicted_area_nieve_scaled.extend([np.nan] * (len(future_exog_df_processed) - (i + 1)))
            break

        predicted_area_nieve_scaled.append(pred_scaled[0, 0])

        # Preparar la secuencia para la siguiente predicción
        # Los lags de area_nieve se actualizan con la nueva predicción
        next_area_scaled = pred_scaled.reshape(1, 1, 1)
        
        # Los exógenos se toman del siguiente paso temporal del DataFrame de datos futuros
        if i + 1 < len(future_exog_df_processed):
            next_exog_scaled = future_exog_df_processed[exog_cols_scaled].iloc[i + 1].values.reshape(1, 1, len(exog_features))
        else:
            # Si no hay más datos exógenos futuros, el bucle debería terminar
            break 

        updated_area_sequence = np.concatenate([last_sequence_input[:, 1:, 0].reshape(1, n_lags_area - 1, 1), next_area_scaled], axis=1)
        updated_exog_sequence = np.concatenate([last_sequence_input[:, 1:, 1:], next_exog_scaled], axis=1)
        last_sequence_input = np.concatenate([updated_area_sequence, updated_exog_sequence], axis=2)

    # 5. Invertir el escalado de las predicciones
    predicted_area_nieve_original = scaler_area.inverse_transform(np.array(predicted_area_nieve_scaled).reshape(-1, 1))

    # Crear DataFrame de resultados
    future_predictions_df = pd.DataFrame({
        'fecha': future_exog_df_processed['fecha'].iloc[:len(predicted_area_nieve_original)],
        'area_nieve_pred': predicted_area_nieve_original.flatten()
    })
    
    print("Predicciones futuras generadas.")
    return future_predictions_df


# --- MAIN EXECUTION FOR FUTURE PREDICTION ---

# 1. Define paths and file names
cuenca = 'indrawati-melamchi' # Nombre de la cuenca
model_file_path = f'E:/models/{cuenca}/narx_model_best_{cuenca}.h5' # Tu modelo entrenado
historical_data_file_name = f'{cuenca}.csv' # Archivo con datos históricos para medias
future_data_file_name = f'E:/data/csv/series_futuras_clean/{cuenca}/Indrawati ssp 245 2051-2070_clean_processed.csv' # Archivo con variables exógenas futuras (sin area_nieve)

basins_data_dir = 'datasets/' # Directorio de tus datos históricos y futuros

# 2. Fixed parameters (exógenas)
exog_cols = ["dia_sen", "temperatura", "precipitacion", "dias_sin_precip"]

# 3. Cargar el modelo
model = None
n_lags_area = None
try:
    model = keras.models.load_model(model_file_path)
    print(f"Modelo cargado desde: {model_file_path}")
    
    if hasattr(model, 'input_shape') and len(model.input_shape) >= 2:
        n_lags_area = model.input_shape[1]
        print(f"n_lags_area deducido del modelo: {n_lags_area}")
    else:
        print(f"Advertencia: No se pudo deducir n_lags_area del input_shape del modelo. Usando valor predeterminado de 7.")
        n_lags_area = 7 

except Exception as e:
    print(f"Error fatal al cargar el modelo: {e}. Asegúrate de que la ruta es correcta y el archivo .h5 es válido.")
    exit()

# 4. Cargar datos históricos y preprocesar para obtener los escaladores
historical_df = None
scaler_area = None
scaler_exog = None
try:
    historical_df = pd.read_csv(os.path.join(basins_data_dir, historical_data_file_name), index_col=0)
    if 'fecha' in historical_df.columns:
        historical_df['fecha'] = historical_df['fecha'].astype(str) # Asegurar string para pd.to_datetime
    
    # Hacemos un preprocesamiento "parcial" solo para obtener los scalers ajustados
    # No necesitamos las divisiones train/test/val aquí para la predicción futura.
    temp_scaled_data, temp_scalers = preprocess_data(historical_df.copy(), exog_cols, train_size=0.7, test_size=0.2)
    scaler_area = temp_scalers['area']
    scaler_exog = temp_scalers['exog']
    print(f"Escaladores obtenidos de datos históricos de {cuenca}.")
except Exception as e:
    print(f"Error fatal al cargar o preprocesar datos históricos para escaladores: {e}.")
    exit()

# 5. Cargar datos futuros de variables exógenas
future_exog_df = None
try:
    future_exog_df = pd.read_csv(os.path.join(basins_data_dir, future_data_file_name))
    if 'fecha' in future_exog_df.columns:
        future_exog_df['fecha'] = future_exog_df['fecha'].astype(str)
    print(f"Datos exógenos futuros cargados de {future_data_file_name}.")
except Exception as e:
    print(f"Error fatal al cargar datos exógenos futuros: {e}.")
    exit()

# 6. Realizar las predicciones futuras
if model and historical_df is not None and future_exog_df is not None and scaler_area and scaler_exog:
    future_predictions_df = make_future_predictions(
        model,
        historical_df, # Pasamos el DF histórico para calcular las medias por día del año
        future_exog_df,
        exog_cols,
        n_lags_area,
        scaler_area,
        scaler_exog
    )

    if future_predictions_df is not None:
        print("\n--- Predicciones Futuras Generadas ---")
        print(future_predictions_df.head())
        print(future_predictions_df.tail())

        # Opcional: Guardar las predicciones en un CSV
        output_predictions_dir = os.path.join(os.path.dirname(model_file_path), 'future_predictions')
        os.makedirs(output_predictions_dir, exist_ok=True)
        predictions_output_path = os.path.join(output_predictions_dir, f'future_predictions_{cuenca}.csv')
        future_predictions_df.to_csv(predictions_output_path, index=False)
        print(f"Predicciones futuras guardadas en: {predictions_output_path}")

        try:
            plt.figure(figsize=(15, 6))
            sns.lineplot(x=future_predictions_df['fecha'], y=future_predictions_df['area_nieve_pred'], label='Predicción futura')
            plt.title(f'Predicciones futuras de Area de Nieve para {cuenca}')
            plt.xlabel('Fecha')
            plt.ylabel('Area de Nieve (Km2)')
            plt.grid(True)
            graph_output_path = os.path.join(output_predictions_dir, f'future_predictions_graph_{cuenca}.png')
            plt.savefig(graph_output_path)
            plt.close()
            print(f"Gráfico de predicciones futuras guardado en: {graph_output_path}")
        except Exception as e:
            print(f"Error al generar el gráfico de predicciones futuras: {e}")

else:
    print("No se pudieron generar las predicciones futuras debido a errores previos.")

print("\nProceso de predicción futura completado.")