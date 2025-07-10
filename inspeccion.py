#%%
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd

EXTERNAL_DISK = 'D:'
models_directory = os.path.join(EXTERNAL_DISK, "models")
future_exog = os.path.join(EXTERNAL_DISK, 'data/csv/series_futuras_clean')
#%%
exog_file = pd.read_csv(os.path.join(EXTERNAL_DISK, 'data', 'csv/v_exog_hist.csv'), index_col = 0)
exog_file.fecha = pd.to_datetime(exog_file.fecha)
exog_file = exog_file[exog_file['fecha'].dt.year >= 2000]

adda_historic_imputed = pd.read_csv('./datasets_imputed/adda-bornio.csv')
adda_predictions = pd.read_csv(os.path.join(future_exog, 'adda-bornio/Adda ssp 245 2051-2070','ACCESS-ESM1-5.csv'))
adda_og = exog_file[exog_file['cuenca'] == 'adda-bornio']

genil_historic =  pd.read_csv('./datasets/genil-dilar.csv')
genil_predictions = pd.read_csv(os.path.join(future_exog, 'genil-dilar/Genil ssp 245 2051-2070','ACCESS-ESM1-5.csv'))
genil_og = exog_file[exog_file['cuenca'] == 'genil-dilar']

#%%
print(f'Media de precipitacion de datos historicos adda-bornio: {adda_historic_imputed.precipitacion.mean()}')
print(f'Media de precipitacion de predicciones adda-bornio: {adda_predictions.precipitacion.mean()}')

#%%
print(f'Media de temperatura de datos historicos  adda-bornio: {adda_historic_imputed.temperatura.mean()}')
print(f'Media de temperatura de predicciones adda-bornio: {adda_predictions.temperatura.mean()}')
print(f'Media de temperatura de historico adda-bornio original: {adda_og.temperatura.mean()}')

#%%
print(f'Media de precipitacion de datos historicos genil-dilar: {genil_historic.precipitacion.mean()}')
print(f'Media de precipitacion de predicciones genil-dilar: {genil_predictions.precipitacion.mean()}')

#%%
print(f'Media de temperatura de datos historicos genil-dilar: {genil_historic.temperatura.mean()}')
print(f'Media de temperatura de predicciones genil-dilar: {genil_predictions.temperatura.mean()}')
print(f'Media de temperatura de historico genil-dilar original: {genil_og.temperatura.mean()}')

#%%
print(f'Media de precipitacion de datos historicos nenskra-enguri: {nenskra_historic.precipitacion.mean()}')
print(f'Media de precipitacion de datos historicos nenskra-enguri: {nenskra_historic_imputed.precipitacion.mean()}')
print(f'Media de precipitacion de predicciones nenskra-enguri: {nenskra_predictions.precipitacion.mean()}')
#%%
print(f'Media de temperatura de datos historicos nenskra-enguri: {nenskra_historic.temperatura.mean()}')
print(f'Media de temperatura de datos historicos nenskra-enguri: {nenskra_historic_imputed.temperatura.mean()}')
print(f'Media de temperatura de predicciones nenskra-enguri: {nenskra_predictions.temperatura.mean()}')

#%%
df = pd.read_csv('D:\data\csv\series_futuras_og/adda-bornio\Adda ssp 245 2051-2070.csv', header=1)
df
#%%
df['T'].mean()
#%%

#%%





#%%
cuencas_disponibles = [d for d in os.listdir(models_directory) if os.path.isdir(os.path.join(models_directory, d)) and not d.startswith('graphs') ]

if not cuencas_disponibles:
    print(f"No se encontraron directorios de cuencas en '{models_directory}'. Asegúrate de que la ruta sea correcta y contenga los subdirectorios de los modelos.")
else:
    print(f"Cuencas encontradas en '{models_directory}': {cuencas_disponibles}")
    for cuenca in cuencas_disponibles:
    # Intenta cargar y obtener la información para la primera cuenca encontrada como ejemplo
        example_cuenca = cuenca
        
        # Construye la ruta al archivo .h5 del modelo.
        # Necesitarás saber el formato de nombre que usaste al guardar el modelo.
        # Por ejemplo, si usaste 'narx_model_CUENCA_NAME.h5'
        model_filename = f'narx_model_{example_cuenca}.h5'
        model_path = os.path.join(models_directory, example_cuenca, model_filename)

        if not os.path.exists(model_path):
            print(f"\nNo se encontró el archivo del modelo en: {model_path}")
            print("Por favor, verifica el nombre del archivo del modelo y la estructura de directorios.")
        else:
            print(f"\nCargando modelo de ejemplo para la cuenca: {example_cuenca}")
            try:
                # Cargar el modelo
                loaded_model = keras.models.load_model(model_path)

                # Paso 2: Obtener la forma de entrada (input shape)
                # La propiedad input_shape de la primera capa del modelo (generalmente la de entrada)
                # te da las dimensiones esperadas.
                # Para modelos Sequential, loaded_model.input_shape es el input_shape del modelo completo.
                
                # El input_shape será (None, n_lags, n_features)
                # 'None' en la primera dimensión indica el tamaño del batch (puede ser cualquiera).
                # 'n_lags' es el número de pasos de tiempo pasados que el modelo espera.
                # 'n_features' es el número total de características (area_nieve + variables exógenas).
                
                print(f"\n--- Información de entrada del modelo {cuenca}---")
                print(f"Forma de entrada esperada por el modelo: {loaded_model.input_shape}")
                print(f"Número de parámetros (pesos y sesgos): {loaded_model.count_params()}")

                # Paso 3: Opcional - Imprimir un resumen completo del modelo
                # Esto te dará detalles sobre cada capa, sus parámetros y sus formas de entrada/salida.
                print("\n--- Resumen del modelo ---")
                loaded_model.summary()

                # A partir de loaded_model.input_shape, puedes deducir:
                # n_lags = loaded_model.input_shape[1]
                # n_features = loaded_model.input_shape[2]
                print(f"\nEl modelo espera {loaded_model.input_shape[1]} pasos de tiempo (lags) y {loaded_model.input_shape[2]} características por paso de tiempo.")

            except Exception as e:
                print(f"Ocurrió un error al cargar o inspeccionar el modelo: {e}")
                print("Asegúrate de que TensorFlow y Keras estén correctamente instalados y que el archivo .h5 no esté corrupto.")