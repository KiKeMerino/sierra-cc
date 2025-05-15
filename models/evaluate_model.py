import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from joblib import load
import os

def nash_sutcliffe_efficiency(targets, predictions):
    """Calcula la eficiencia de Nash-Sutcliffe."""
    mean_target = np.mean(targets)
    numerator = np.sum((targets - predictions)**2)
    denominator = np.sum((targets - mean_target)**2)
    return 1 - (numerator / denominator)

def kling_gupta_efficiency(targets, predictions):
    """Calcula la eficiencia de Kling-Gupta."""
    r = np.corrcoef(targets, predictions)[0, 1]
    alpha = np.std(predictions) / np.std(targets)
    beta = np.mean(predictions) / np.mean(targets)
    return 1 - np.sqrt(((r - 1)**2) + ((alpha - 1)**2) + ((beta - 1)**2))

def evaluar_predicciones_simuladas(ruta_datos_historicos, nombre_archivo_historico, cuenca, ruta_modelo_guardado, n_lags_area, n_lags_exog, proporcion_validacion=0.2):
    """
    Evalúa el modelo RNN en modo de predicción simulada (backtesting) utilizando datos históricos.
    Calcula R2, MAE, NSE y KGE.
    """
    modelo_rnn = keras.models.load_model(os.path.join(ruta_modelo_guardado, 'simple_model_multicuenca.h5'))
    scaler_x = load(os.path.join(ruta_modelo_guardado, 'scaler_x_rnn_multi.joblib'))
    scaler_y = load(os.path.join(ruta_modelo_guardado, 'scaler_y_rnn_multi.joblib'))
    encoder_cuenca = load(os.path.join(ruta_modelo_guardado, 'encoder_cuenca_rnn_multi.joblib'))
    n_cuencas = len(encoder_cuenca.categories_[0])

    df_historico = pd.read_csv(os.path.join(ruta_datos_historicos, nombre_archivo_historico), index_col = 0)
    # cuencas = df_historico['cuenca'].unique()
    resultados_metricas = {}

    
    df_cuenca = df_historico[df_historico['cuenca'] == cuenca].sort_values(by=df_historico.columns[0]).copy()
    n_datos = len(df_cuenca)
    split_index = int(n_datos * (1 - proporcion_validacion))
    train_df = df_cuenca[:split_index].reset_index(drop=True)
    val_df = df_cuenca[split_index:].reset_index(drop=True)

    predicciones_cuenca = []
    real_cuenca = val_df['area_nieve'].values

    hist_area_nieve = train_df['area_nieve'].iloc[-n_lags_area:].tolist()
    exog_cols_historico = [col for col in train_df.columns if col not in ['area_nieve', 'cuenca', df_historico.columns[0]]]
    hist_exog_dict = {col: train_df[col].iloc[-n_lags_area:].tolist() for col in exog_cols_historico}

    for i in range(len(val_df)):
        input_features = {}
        for lag in range(1, n_lags_area + 1):
            input_features[f'area_nieve(t-d{lag})'] = hist_area_nieve[-lag]

        for col_exog in exog_cols_historico:
            for lag in range(n_lags_exog + 1):
                lag_name = f'{col_exog}(t-{lag})'
                if lag == 0:
                    input_features[lag_name] = val_df[col_exog].iloc[i]
                elif len(train_df) > n_lags_area + lag - 1:
                    input_features[lag_name] = train_df[col_exog].iloc[-(n_lags_area + lag)]
                else:
                    input_features[lag_name] = 0

        input_features['cuenca(t-0)'] = cuenca

        input_df = pd.DataFrame([input_features])
        exog_cols_entrada = sorted([col for col in input_df.columns if col not in ['area_nieve(t-d1)', 'cuenca(t-0)']])
        X_exog_entrada = input_df[exog_cols_entrada].values.astype(float)

        # if X_exog_entrada.shape[1] == scaler_x.n_features_in_:
        X_scaled_exog = scaler_x.transform(X_exog_entrada)
        cuenca_col = input_df[['cuenca(t-0)']].values
        X_scaled_cuenca = encoder_cuenca.transform(cuenca_col)
        X_scaled = np.concatenate((X_scaled_exog, X_scaled_cuenca), axis=1)
        X_scaled_nn = X_scaled.reshape((1, 1, X_scaled.shape[1]))

        prediccion_escalada = modelo_rnn.predict(X_scaled_nn)[0, 0]
        prediccion_original = scaler_y.inverse_transform(np.array([[prediccion_escalada]]))[0, 0]
        predicciones_cuenca.append(prediccion_original)

        hist_area_nieve.append(prediccion_original)
        if len(hist_area_nieve) > n_lags_area:
            hist_area_nieve.pop(0)
        # Asegurar que hist_exog_dict esté actualizado correctamente
        for col_exog in exog_cols_historico:
            if col_exog in hist_exog_dict:
                hist_exog_dict[col_exog].append(val_df[col_exog].iloc[i])
                if len(hist_exog_dict[col_exog]) > n_lags_area:
                    hist_exog_dict[col_exog].pop(0)
            else:
                hist_exog_dict[col_exog] = [val_df[col_exog].iloc[i]] # Inicializar si no existe

        if len(real_cuenca) > 0 and len(predicciones_cuenca) == len(real_cuenca):
            r2 = r2_score(real_cuenca, predicciones_cuenca)
            mae = mean_absolute_error(real_cuenca, predicciones_cuenca)
            nse = nash_sutcliffe_efficiency(real_cuenca, predicciones_cuenca)
            kge = kling_gupta_efficiency(real_cuenca, predicciones_cuenca)
            resultados_metricas = {'R2': r2, 'MAE': mae, 'NSE': nse, 'KGE': kge}
        else:
            print(f"No se pudieron calcular las métricas para la cuenca {cuenca} debido a la falta de predicciones o datos reales.")

    return resultados_metricas

# Definir los lags (deben coincidir con los usados en el entrenamiento)
n_lags_area = 3
n_lags_exog = 2

# Rutas (¡Asegúrate de actualizarlas!)
ruta_datos_historicos = './'
nombre_archivo_historico = 'df_all.csv'
ruta_modelo_guardado = './models/'

# Evaluar el modelo en modo de predicción simulada
metricas_prediccion = evaluar_predicciones_simuladas(ruta_datos_historicos, nombre_archivo_historico, 'adda-bornio',ruta_modelo_guardado, n_lags_area, n_lags_exog)

# Imprimir las métricas
for nombre_metrica, valor in metricas_prediccion:
    print(f"{nombre_metrica}: {valor:.4f}")