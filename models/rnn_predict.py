import pandas as pd
import numpy as np
import os
from tensorflow import keras
from joblib import load

def rnn_predict(ruta_datos_historicos, nombre_archivo_historico, ruta_datos_futuros, ruta_modelo_guardado):
    """
    Realiza predicciones futuras del área de nieve para múltiples cuencas y modelos de predicción futura
    utilizando un modelo RNN autoregresivo con variables exógenas, asumiendo un archivo futuro unificado por cuenca.

    Args:
        ruta_datos_historicos (str): Ruta al directorio que contiene el archivo de datos históricos.
        nombre_archivo_historico (str): Nombre del archivo CSV con los datos históricos.
        ruta_datos_futuros (str): Ruta al directorio que contiene los archivos CSV futuros unificados por cuenca.
        ruta_modelo_guardado (str): Ruta al directorio donde se guardó el modelo RNN y los objetos de preprocesamiento.

    Returns:
        dict: Un diccionario donde las claves son los nombres de las cuencas y los valores son diccionarios.
              Dentro de cada diccionario de cuenca, las claves son los nombres de los modelos futuros y los valores
              son DataFrames con las predicciones futuras (fecha, area_nieve).
    """
    # 1. Cargar modelo y objetos de preprocesamiento
    modelo_rnn = keras.models.load_model(os.path.join(ruta_modelo_guardado, 'simple_model_multicuenca.h5'))
    scaler_x = load(os.path.join(ruta_modelo_guardado, 'scaler_x_rnn_multi.joblib'))
    scaler_y = load(os.path.join(ruta_modelo_guardado, 'scaler_y_rnn_multi.joblib'))
    encoder_cuenca = load(os.path.join(ruta_modelo_guardado, 'encoder_cuenca_rnn_multi.joblib'))

    # 2. Cargar datos históricos y obtener las cuencas únicas
    df_historico = pd.read_csv(os.path.join(ruta_datos_historicos, nombre_archivo_historico))
    cuencas = df_historico['cuenca'].unique()

    resultados_predicciones = {}

    # 3. Iterar sobre las cuencas
    for cuenca in cuencas:
        resultados_predicciones[cuenca] = {}
        df_historico_cuenca = df_historico[df_historico['cuenca'] == cuenca].copy()
        df_historico_cuenca.sort_values(by=df_historico_cuenca.columns[0], inplace=True) # Asegurar orden temporal

        # Obtener los últimos datos históricos para lags iniciales
        ultimo_historico = df_historico_cuenca.iloc[-n_lags_area:].to_dict('records')
        if len(ultimo_historico) < n_lags_area:
            print(f"Advertencia: No hay suficientes datos históricos para crear {n_lags_area} lags para la cuenca {cuenca}. Saltando esta cuenca.")
            continue

        # 4. Identificar y cargar el archivo de datos futuros para la cuenca actual
        nombre_archivo_futuro_cuenca = f'{cuenca.lower()}.csv'
        ruta_archivo_futuro_cuenca = os.path.join(ruta_datos_futuros, nombre_archivo_futuro_cuenca)

        if os.path.exists(ruta_archivo_futuro_cuenca):
            df_futuro_cuenca = pd.read_csv(ruta_archivo_futuro_cuenca)
            modelos_futuros = df_futuro_cuenca['modelo'].unique()

            # 5. Iterar sobre los modelos futuros dentro del DataFrame de la cuenca
            for modelo in modelos_futuros:
                df_futuro_modelo = df_futuro_cuenca[df_futuro_cuenca['modelo'] == modelo].copy()
                df_futuro_modelo.sort_values(by='fecha', inplace=True) # Asegurar orden temporal de las predicciones
                predicciones_modelo = []

                # Preparar los lags iniciales
                hist_area_nieve = [d['area_nieve'] for d in ultimo_historico]
                hist_exog = {col: [d[col] for d in ultimo_historico] for col in df_historico_cuenca.columns if col not in ['area_nieve', 'cuenca']}

                # 6. Predicción autoregresiva para cada paso de tiempo del modelo actual
                for index, row_futuro in df_futuro_modelo.iterrows():
                    # Crear el vector de entrada para la predicción
                    input_data = {}
                    # Lags del área de nieve
                    for i in range(1, n_lags_area + 1):
                        input_data[f'area_nieve(t-d{i})'] = hist_area_nieve[-i]

                    # Lags de las variables exógenas
                    for col_exog in hist_exog.keys():
                        for i in range(n_lags_exog + 1):
                            if i == 0:
                                input_data[f'{col_exog}(t-{i})'] = row_futuro[col_exog]
                            elif len(hist_exog[col_exog]) > i - 1:
                                input_data[f'{col_exog}(t-{i})'] = hist_exog[col_exog][-i]
                            else:
                                input_data[f'{col_exog}(t-{i})'] = 0 # Manejar la falta de lags históricos iniciales

                    # Variable 'cuenca'
                    input_data['cuenca(t-0)'] = cuenca

                    # Convertir a DataFrame y preprocesar
                    df_entrada = pd.DataFrame([input_data])
                    exog_cols = [col for col in df_entrada.columns if col not in ['area_nieve(t-d1)', 'cuenca(t-0)']]
                    X_exog = df_entrada[exog_cols].values.astype(float)
                    X_scaled_exog = scaler_x.transform(X_exog)
                    cuenca_col = df_entrada[['cuenca(t-0)']].values
                    X_scaled_cuenca = encoder_cuenca.transform(cuenca_col)
                    X_scaled = np.concatenate((X_scaled_exog, X_scaled_cuenca), axis=1)
                    X_scaled_nn = X_scaled.reshape((1, 1, X_scaled.shape[1]))

                    # Realizar la predicción
                    prediccion_escalada = modelo_rnn.predict(X_scaled_nn)[0, 0]
                    prediccion_original = scaler_y.inverse_transform(np.array([[prediccion_escalada]]))[0, 0]
                    predicciones_modelo.append((row_futuro['fecha'], prediccion_original))

                    # Actualizar los lags para la siguiente predicción
                    hist_area_nieve.append(prediccion_original)
                    if len(hist_area_nieve) > n_lags_area:
                        hist_area_nieve.pop(0)
                    for col_exog in hist_exog.keys():
                        hist_exog[col_exog].append(row_futuro[col_exog])
                        if len(hist_exog[col_exog]) > n_lags_area:
                            hist_exog[col_exog].pop(0)

                resultados_predicciones[cuenca][modelo] = pd.DataFrame(predicciones_modelo, columns=['fecha', 'area_nieve'])
        else:
            print(f"Advertencia: No se encontró el archivo de datos futuros para la cuenca {cuenca}.")

    return resultados_predicciones

# Definir los lags (deben coincidir con los usados en la creación del modelo)
n_lags_area = 3
n_lags_exog = 2

# Rutas de los datos y el modelo guardado (¡Asegúrate de actualizarlas!)
ruta_datos_historicos = './'
nombre_archivo_historico = 'historic_data.csv'
ruta_datos_futuros = './predicted_exog/'
ruta_modelo_guardado = './models/'

# Realizar las predicciones con la estructura de archivos unificada
predicciones_unificadas = rnn_predict(ruta_datos_historicos, nombre_archivo_historico, ruta_datos_futuros, ruta_modelo_guardado)

# Imprimir o guardar los resultados
for cuenca, modelos in predicciones_unificadas.items():
    print(f"\nPredicciones para la cuenca: {cuenca}")
    for modelo, df_predicciones in modelos.items():
        print(f"\n  Modelo futuro: {modelo}")
        print(df_predicciones)

# O puedes guardar los resultados en archivos CSV
# for cuenca, modelos in predicciones_unificadas.items():
#     for modelo, df_predicciones in modelos.items():
#         df_predicciones.to_csv(f'predicciones_unificada_{cuenca}_{modelo}.csv', index=False)

