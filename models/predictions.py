import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 18})

EXTERNAL_DISK = 'D:/'

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
    
    # historical_df = historical_df.iloc[:400]
    # future_exog_df = future_exog_df[:400]

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
        
        pred_scaled = model.predict(last_sequence_input, batch_size=64, verbose=0)

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
    
    future_predictions_df.loc[future_predictions_df['area_nieve_pred'] < 0, 'area_nieve_pred'] = 0

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
        
    series_path = os.path.join(EXTERNAL_DISK, 'data/csv/series_futuras_clean/', cuenca)

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


    model_colors_map = {
        'Full Dataset Prediction': 'black',
        'ACCESS-ESM1-5': 'blue',
        'CNRM-CM6-1': 'orange',
        'MPI-ESM1-2-LR': 'grey',
        'HadGEM3-GC31-LL': 'green',
        'MRI-ESM2-0': 'yellow'
    }

    # Preparar la predicción del full-dataset
    full_dataset_file = os.path.join(output_base_dir, f'graphs_{cuenca}', 'full_dataset.csv')
    full_dataset_for_agg = pd.read_csv(full_dataset_file)
    if 'fecha' in full_dataset_for_agg.columns:
        full_dataset_for_agg = full_dataset_for_agg.set_index('fecha')
    full_dataset_for_agg.index = pd.to_datetime(full_dataset_for_agg.index)
    full_dataset_for_agg['day_of_year'] = full_dataset_for_agg.index.day_of_year
    full_dataset_for_agg['month'] = full_dataset_for_agg.index.month

    avg_full_dataset_pred_per_day = full_dataset_for_agg.groupby('day_of_year')['area_nieve_pred'].mean().rename('Full Dataset Prediction')
    avg_full_dataset_pred_per_month = full_dataset_for_agg.groupby('month')['area_nieve_pred'].mean().rename('Full Dataset Prediction')

    # Acumular predicciones de los 5 modelos
    all_models_avg_predictions_per_day = {}
    all_models_avg_predictions_per_month = {}

    for model_name, predictions_df in future_predictions.items():
        predictions_file_name = os.path.join(output_path, f'predictions_{model_name}.csv')
        predictions_df = predictions_df.set_index('fecha')
        predictions_df.to_csv(predictions_file_name)
        print(f"Predicciones futuras de {model_name} guardadas en: {output_path}")

        predictions_for_agg = predictions_df[['area_nieve_pred']].copy()
        predictions_for_agg.index = pd.to_datetime(predictions_for_agg.index)
        predictions_for_agg['day_of_year'] = predictions_for_agg.index.day_of_year
        predictions_for_agg['month'] = predictions_for_agg.index.month

        all_models_avg_predictions_per_day[model_name] = predictions_for_agg.groupby('day_of_year')['area_nieve_pred'].mean()
        all_models_avg_predictions_per_month[model_name] = predictions_for_agg.groupby('month')['area_nieve_pred'].mean()

    # --- Generación de gráficos ---
    try:
        # Definir la información para ambos tipos de gráficos
        graph_configs = [
            {'name': 'per_day', 'xlabel': 'Day of the year', 'full_dataset_pred_avg': avg_full_dataset_pred_per_day, 'models_avg_data': all_models_avg_predictions_per_day},
            {'name': 'per_month', 'xlabel': 'Month of the year', 'full_dataset_pred_avg': avg_full_dataset_pred_per_month, 'models_avg_data': all_models_avg_predictions_per_month}
        ]

        for config in graph_configs:
            plt.figure(figsize=(12, 8))

            # Plotear la línea de la predicción en el full-dataset
            sns.lineplot(x=config['full_dataset_pred_avg'].index, y=config['full_dataset_pred_avg'],
                        label='historical', color=model_colors_map['Full Dataset Prediction'], linewidth=2)

            # Plotear las líneas de los 5 modelos
            for i, (model_name, avg_data) in enumerate(config['models_avg_data'].items()):
                color_to_use = model_colors_map.get(model_name, 'red')
                sns.lineplot(x=avg_data.index, y=avg_data, label=model_name, color=color_to_use, linewidth=2)


                plt.text(x=0.02, y=0.02, # (0,0 = abajo-izq; 1,1 = arriba-der)
                    s=scenario,
                    transform=plt.gca().transAxes,
                    fontweight='bold', va='bottom', ha='left'
                    #bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5') # Opcional: fondo para el texto
                )
                plt.xlim(left=min(config['full_dataset_pred_avg'].index), right=(max(config['full_dataset_pred_avg'].index)))
                plt.ylim(bottom=0)
                plt.xlabel(config['xlabel'])
                plt.ylabel("Snow cover area (km2)")

                if cuenca == 'adda-bornio' or cuenca == 'nenskra-enguri':
                    plt.legend(loc='lower left', bbox_to_anchor=(0,0.1), frameon=False)
                
                elif cuenca == 'mapocho-almendros':
                    plt.legend(loc='upper left', frameon=False)

                else:   # indrwawati-melamchi, genil-dilar, uncompahgre-ridgway
                    plt.legend(loc='upper center', frameon=False)

                plt.tight_layout()

            graph_output_path = os.path.join(output_path, f'seasonal_avg_{config["name"]}_{cuenca}.png')
            plt.savefig(graph_output_path)
            plt.close() # Cierra la figura para liberar memoria
            print(f"Gráfico de promedio {config['name']} guardado en: {graph_output_path}")

    except Exception as e:
        print(f"Error al generar gráficas: {e}")

else:
        print("No se generaron predicciones futuras, omitiendo la generación de gráficas.")


print(f"\nProceso de predicción futura completado para {cuenca}.")