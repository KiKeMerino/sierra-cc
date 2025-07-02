import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt # Necesario si graficas
import seaborn as sns # Necesario si graficas

# --- FUNCIONES DE SOPORTE (Mantienen su funcionalidad original) ---

class CustomLSTM(keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        # Filtrar el argumento 'time_major' si está presente y no es reconocido
        if 'time_major' in kwargs:
            print(f"Advertencia: Ignorando el argumento 'time_major={kwargs['time_major']}' para la capa LSTM.")
            kwargs.pop('time_major')
        super().__init__(*args, **kwargs)

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


# --- FUNCIÓN PARA HACER PREDICCIONES FUTURAS ---
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
    if 'fecha' in historical_df.columns:
        historical_df = historical_df.set_index(['fecha'])
    historical_df.index = pd.to_datetime(historical_df.index)
    historical_df['day_of_year'] = historical_df.index.day_of_year
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
    
    future_predictions_df.loc[future_predictions_df['area_nieve_pred'] < 0, :] = 0

    print("Predicciones futuras generadas.")

    return future_predictions_df

def print_menu():
    print("\n" + "="*60)
    print( "\t\t ----- CREACION DE PREDICCIONES FUTURAS -----")
    print("="*60 + "\n")

    basins_data_dir = 'datasets_imputed/'
    available_basins = [f[:-4] for f in os.listdir(basins_data_dir)]

    cuenca = ""
    while cuenca not in available_basins:
        cuenca =  input("Introduce el nombre de la cuenca (ej: 'adda-bornio') que deseas predecir o deja en blanco para ver las disponibles: ").lower().strip()
        if not cuenca or cuenca not in available_basins: # Si el usuario deja en blanco, imprimo las cuencas disponibles
            print('\n--- Cuencas disponibles ---')
            for basin in available_basins:
                print(f'- {basin}')
            print("-"*25)
            cuenca = input("\nPor favor, introduce el nombre de la cuenca que desesas usar: ").lower().strip()
        
    series_path = 'D:/data/csv/series_futuras_clean/' + cuenca

    # Selección de Escenario futuro
    scenarios_list = []
    try:
        scenarios_list = [s for s in os.listdir(series_path)
                        if os.path.isdir(os.path.join(series_path, s))]
    except FileNotFoundError:
        print(f"Error: La ruta de series futuras no fue encontrada para '{cuenca}': {series_path}")
    except Exception as e:
        print(f"Error al obtener escenarios: {e}")

    scenarios = {str(i + 1): name for i, name in enumerate(scenarios_list)}

    print("--- Escenarios disponibles ---")
    for key, value in scenarios.items():
        print(f"{key}. {value}")

    selected_scenario_key = ""
    while selected_scenario_key not in scenarios:
        selected_scenario_key = input("Ingrese el número del escenario deseado: ")
        if selected_scenario_key not in scenarios:
            print("Selección inválida. Por favor, intente de nuevo.")

    selected_scenario_name = scenarios[selected_scenario_key]
    selected_scenario_path = os.path.join(series_path, selected_scenario_name)


    future_data_path = os.path.join(selected_scenario_path)

    return cuenca, future_data_path, selected_scenario_name

# --- MAIN EXECUTION FOR FUTURE PREDICTION ---

# 1. Define paths and file names
EXTERNAL_DISK = 'D:/'
 # Directorio de datos históricos
base_model_path =  os.path.join(EXTERNAL_DISK, 'models/')
# future_data_file_name = os.path.join(scenario_path, 'dataset.csv')
# base_series_path = os.path.join(EXTERNAL_DISK, f'data/csv/series_futuras_clean/{cuenca}')

cuenca, scenario_path, scenario = print_menu()
historical_data_file_name = os.path.join('datasets_imputed', f'{cuenca}.csv') # Archivo con datos históricos para medias
model_file_path = os.path.join(base_model_path, cuenca, f'narx_model_{cuenca}.h5')

model_future_exogs = os.listdir(scenario_path)

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
    historical_df = pd.read_csv(historical_data_file_name, index_col=0)
    if 'fecha' in historical_df.columns:
        historical_df.index = historical_df.index.astype(str) # Asegurar string para pd.to_datetime

    temp_scaled_data, temp_scalers = preprocess_data(historical_df.copy(), exog_cols, train_size=0.7, test_size=0.2)
    scaler_area = temp_scalers['area']
    scaler_exog = temp_scalers['exog']
    print(f"Escaladores obtenidos de datos históricos de {cuenca}.")
except Exception as e:
    print(f"Error fatal al cargar o preprocesar datos históricos para escaladores: {e}.")
    exit()

# 5. Cargar datos futuros de variables exógenas
future_exog_dfs = {}
for model_exog in model_future_exogs:
    future_data_file_name = os.path.join(scenario_path, model_exog)
    df = None
    try:
        df = pd.read_csv(future_data_file_name)
        if 'fecha' in df.columns:
            df['fecha'] = df['fecha'].astype(str)

        model_name = model_exog[:-4]
        future_exog_dfs[model_name] = df
    except Exception as e:
        print(f"Error fatal al cargar datos exógenos futuros: {e}.")
        exit()

print(f"Datos exógenos futuros cargados para el escenario {scenario_path}.")

# 6. Realizar las predicciones futuras
future_predictions = {}
for model_name, future_exog_df in future_exog_dfs.items():
    if model and historical_df is not None and future_exog_df is not None and scaler_area and scaler_exog:
        future_predictions[model_name] = make_future_predictions(
            model,
            historical_df, # Pasamos el DF histórico para calcular las medias por día del año
            future_exog_df,
            exog_cols,
            n_lags_area,
            scaler_area,
            scaler_exog
        )
    else:
        print("No se pudieron generar las predicciones futuras debido a errores previos.")

if future_predictions is not None:
    print("\n--- Predicciones Futuras Completadas ---")

    # Directorio para guardar las predicciones CSV y las gráficas
    output_base_dir = os.path.dirname(model_file_path)
    output_path = os.path.join(output_base_dir, 'future_predictions', scenario) # Carpeta específica para la cuenca
    # output_graphs_dir = os.path.join(scenario_path, 'graphs') # Carpeta para las gráficas

    os.makedirs(output_path, exist_ok=True)
    # os.makedirs(output_graphs_dir, exist_ok=True) # Crear el directorio de gráficas

    for model_name, predictions_df in future_predictions.items():
        predictions_file_name = os.path.join(output_path, f'predictions_{model_name}.csv')

        predictions_df = predictions_df.set_index('fecha')

        predictions_df.to_csv(predictions_file_name)
        print(f"Predicciones futuras guardadas en: {output_path}")

        try:
            # Unir datos históricos y predicciones futuras para el plotting
            if 'fecha' in historical_df.columns:
                historical_df = historical_df.set_index('fecha')
            historical_df.index = pd.to_datetime(historical_df.index)
            if 'fecha' in predictions_df.columns:
                predictions_df = predictions_df.set_index('fecha')
            predictions_df.index = pd.to_datetime(predictions_df.index)

            # # 1. Gráfica de la Serie Temporal Completa (Histórico + Predicciones)
            # combined_series_real = historical_df['area_nieve'].copy()
            # combined_series_pred = future_predictions_df['area_nieve_pred'].copy()

            # df_plot_all_days = pd.concat([combined_series_real, combined_series_pred], axis=1)
            # df_plot_all_days.columns = ['area_nieve_real', 'area_nieve_pred'] 

            # plt.figure(figsize=(18, 7))
            # sns.lineplot(data = df_plot_all_days, x=df_plot_all_days.index, y='area_nieve_real', label='Real Historical Snow Area')
            # sns.lineplot(data = df_plot_all_days, x=df_plot_all_days.index, y='area_nieve_pred', label='Predicted Future Snow Area')

            # plt.title(f'Prediction vs Real {cuenca.upper()} (Complete Time Series)')
            # plt.xlabel("Date")
            # plt.ylabel("Snow area Km2")
            # plt.legend()
            # plt.grid(True)

            # plt.xticks(rotation=45, ha='right')
            # plt.tick_params(axis='x', which='major', labelsize=10)
            # plt.tight_layout()
            # graph_output_path_all = os.path.join(output_graphs_dir, f'full_time_series_{cuenca}.png')
            # plt.savefig(graph_output_path_all)
            # plt.close()
            # print(f"Gráfica de serie temporal completa guardada en: {graph_output_path_all}")

            # 2. Gráficas de Estacionalidad (Media por Día del Año y por Mes)
            historical_for_agg = historical_df['area_nieve'].to_frame()
            predictions_for_agg = predictions_df['area_nieve_pred'].to_frame()

            historical_for_agg.index = pd.to_datetime(historical_for_agg.index)
            predictions_for_agg.index = pd.to_datetime(predictions_for_agg.index)

            # Calcular promedios estacionales para históricos
            historical_for_agg['day_of_year'] = historical_for_agg.index.day_of_year
            historical_for_agg['month'] = historical_for_agg.index.month
            # Calcular promedios estacionales para predicciones
            predictions_for_agg['day_of_year'] = predictions_for_agg.index.day_of_year
            predictions_for_agg['month'] = predictions_for_agg.index.month
        
            avg_historical_per_day = historical_for_agg.groupby('day_of_year')['area_nieve'].mean().rename('area_nieve_real_avg')
            avg_historical_per_month = historical_for_agg.groupby('month')['area_nieve'].mean().rename('area_nieve_real_avg')

            avg_predictions_per_day = predictions_for_agg.groupby('day_of_year')['area_nieve_pred'].mean().rename('area_nieve_pred_avg')
            avg_predictions_per_month = predictions_for_agg.groupby('month')['area_nieve_pred'].mean().rename('area_nieve_pred_avg')

            # Tipos de gráficas estacionales
            graph_types = [
                {'name': 'per_day',     'groupby_col': 'day_of_year',   'xlabel': 'Day of Year',    'title_suffix': ' (Average per day of the year)',   'historical_avg': avg_historical_per_day, 'predictions_avg': avg_predictions_per_day},
                {'name': 'per_month',   'groupby_col': 'month',         'xlabel': 'Month',          'title_suffix': ' (Average per month)',             'historical_avg': avg_historical_per_month, 'predictions_avg': avg_predictions_per_month}
            ]

            for graph_info in graph_types:
                plt.figure(figsize=(15, 6))
                sns.lineplot(x=graph_info['historical_avg'].index, y=graph_info['historical_avg'], palette='pastel', label='Real Historical Average')
                sns.lineplot(x=graph_info['predictions_avg'].index, y=graph_info['predictions_avg'], label=f'Prediction model {model_name}')

                plt.title(f'Prediction vs Real {cuenca.upper()}{graph_info["title_suffix"]}')
                plt.xlabel(graph_info['xlabel'])
                plt.ylabel("Snow area Km2")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                graph_output_path = os.path.join(output_path, f'{graph_info["name"]}_{cuenca}.png')
                plt.savefig(graph_output_path)

        except Exception as e:
            print(f"Error al generar gráficas: {e}")

    print(f"Gráfica de estacionalidad por {graph_info['name']} guardada en: {graph_output_path}")
    plt.close()

else:
        print("No se generaron predicciones futuras, omitiendo la generación de gráficas.")


print(f"\nProceso de predicción futura completado para {cuenca}.")