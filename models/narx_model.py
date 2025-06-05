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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0007902151611033951, clipnorm=5.0) # clipnorm para el recorte
    model.compile(optimizer=optimizer, loss='mse')

    return model

def create_train_models(sequences_data, n_lags_area, layers, units, epochs, exog_cols_scaled, cuencas, save, models_dir='narx_models'):
    models = {}
    history = {}
    n_features = 1 + len(exog_cols_scaled)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Configurar EarlyStopping
    # monitor='val_loss': La métrica a monitorear para la mejora (pérdida en el conjunto de validación interno del fit)
    # patience=20: Número de épocas sin mejora después de las cuales el entrenamiento se detendrá
    # restore_best_weights=True: Restaura los pesos del modelo a los de la época con el mejor 'val_loss'
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,  # Puedes ajustar este valor. 20 es un buen punto de partida.
        restore_best_weights=True,
        verbose=1     # Muestra mensajes cuando el entrenamiento se detiene temprano
    )

    for cuenca in cuencas:
        model = create_narx_model(n_lags = n_lags_area, n_layers=layers, n_units_lstm=units, n_features=n_features)
        X_train = sequences_data[cuenca]['X_train']
        y_train = sequences_data[cuenca]['y_train']
        model_history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_split=0.1, callbacks=[early_stopping_callback])
        models[cuenca] = model
        history[cuenca] = model_history

        model_path = os.path.join(models_dir, f'narx_model_{cuenca}.h5')
        if save:
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            model.save(model_path)
            print(f"Modelo entrenado y guardado para la cuenca: {cuenca} en {model_path}")
        else:
            print(f'Modelo entrenado para la cuenca {cuenca}')
    return models

def load_models(cuencas, models_dir='models'):
    loaded_models = {}
    for cuenca in cuencas:
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

    y_val_pred_original = scaler_area.inverse_transform(np.array(y_val_pred_scaled).reshape(-1, 1))

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
        print(f"Métricas en todo el conjunto de datos (modo prediccion) para {cuenca}: R2={r2_full:.3f}, MAE={mae_full:.3f}, NSE={nse_full:.3f}, KGE={kge_full:.3f}")

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
                df_plot.fecha = df_plot.fecha.dt.day_of_year
                plt.xlabel("Month")
                plt.xlim(right=12)
            elif graph == 'all_days':
                pass
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


#%% --- Main execution ---
df = pd.read_csv('df_all.csv', index_col=0)
df
# PARÁMETROS DEL MODELO
name = 'narx_models_adda'
n_lags_area = 5
n_layers = 1
n_neuronas = 34
epochs = 40
exog_cols = ['dia_sen','temperatura','precipitacion', 'dias_sin_precip']

model_dir = os.path.join("D:", "models")
scaled_data, scalers, cuencas = preprocess_data(df, exog_cols)
exog_cols_scaled = [col + '_scaled' for col in exog_cols]

# Crear la secuencias
sequences_data = {}
for cuenca, data_indices in scaled_data.items():
    train_data = data_indices['df'].iloc[data_indices['train_idx']]
    val_data = data_indices['df'].iloc[data_indices['val_idx']]
    test_data = data_indices['df'].iloc[data_indices['test_idx']]

    sequences_data[cuenca] = {
        'X_train': create_sequences(train_data, n_lags_area, exog_cols_scaled)[0],
        'y_train': create_sequences(train_data, n_lags_area, exog_cols_scaled)[1],
        'X_val': create_sequences(val_data, n_lags_area, exog_cols_scaled)[0],
        'y_val': create_sequences(val_data, n_lags_area, exog_cols_scaled)[1],
        'X_test': create_sequences(test_data, n_lags_area, exog_cols_scaled)[0],
        'y_test': create_sequences(test_data, n_lags_area, exog_cols_scaled)[1],
    }



# Entrenar y guardar los modelos
# models = create_train_models(sequences_data, n_lags_area, n_layers, n_neuronas, epochs, exog_cols_scaled, cuencas, True, models_dir=os.path.join(model_dir, name))
# O cargar los modelos si ya están entrenados
models = load_models(cuencas, os.path.join(model_dir, name))


# Evaluar los modelos
metrics = {}
archivo_json = os.path.join(model_dir, name, 'metrics.json')

# Evaluar en todo el conjunto de datos
# results = evaluate_full_dataset(models, scaled_data, scalers, cuencas, n_lags_area, exog_cols_scaled, 'per_day', os.path.join(model_dir,name))

for cuenca, model in models.items():
    scaler_area = scalers[cuenca]['area']

    train_sequences = {'X': sequences_data[cuenca]['X_train'], 'y': sequences_data[cuenca]['y_train']}
    metrics[cuenca + ' (train)'], _, _ = evaluate_model(model, train_sequences, scaler_area)
    print(f"Métricas conjunto de 'train' para {cuenca}: {metrics[cuenca + ' (train)']}")

    test_sequences = {'X': sequences_data[cuenca]['X_test'], 'y': sequences_data[cuenca]['y_test']}
    metrics[cuenca + ' (test)'], _, _ = evaluate_model(model, test_sequences, scaler_area)
    print(f"Métricas conjunto de 'test' para {cuenca}: {metrics[cuenca + ' (test)']}")

    df_val_scaled = scaled_data[cuenca]['df'].iloc[scaled_data[cuenca]['val_idx']].copy()
    metrics[cuenca + ' (val)'], _, _ = evaluate_validation(model, df_val_scaled, scaler_area, exog_cols, n_lags_area)
    print(f"Métricas conjunto de 'validation' (modo prediccion) para {cuenca}: {metrics[cuenca + ' (val)']}")

    # metrics[cuenca + ' (full dataset)'] = results[cuenca]

    with open(archivo_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)


# Automatizar experimentos con un bucle
# Añadir gráfica dia a dia
# Seleccionar el mejor modelo para cada cuenca