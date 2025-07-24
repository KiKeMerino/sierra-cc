#%%
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

EXTERNAL_DISK = 'D:'
cuenca = 'mapocho-almendros'

#%%
models_directory = os.path.join(EXTERNAL_DISK, "models")
future_series = os.path.join(EXTERNAL_DISK, 'data/csv/series_futuras_clean', cuenca, 'Mapocho ssp 585 2081-2100')
full = pd.read_csv(os.path.join(EXTERNAL_DISK, models_directory, 'mapocho-almendros/graphs_mapocho-almendros/full_dataset.csv'))
h = pd.read_csv('datasets_imputed/mapocho-almendros.csv', index_col=0)
#%%
scenario4_path = os.path.join(models_directory, 'mapocho-almendros/future_predictions/Mapocho ssp 585 2081-2100/')
model1 = pd.read_csv(os.path.join(scenario4_path, 'predictions_ACCESS-ESM1-5.csv'))
model2 = pd.read_csv(os.path.join(scenario4_path, 'predictions_CNRM-CM6-1.csv'))
model3 = pd.read_csv(os.path.join(scenario4_path, 'predictions_HadGEM3-GC31-LL.csv'))
model4 = pd.read_csv(os.path.join(scenario4_path, 'predictions_MPI-ESM1-2-LR.csv'))
model5 = pd.read_csv(os.path.join(scenario4_path, 'predictions_MRI-ESM2-0.csv'))
#%%
df = pd.read_csv(os.path.join(future_series, 'HadGEM3-GC31-LL.csv'), index_col=0)
df.index = pd.to_datetime(df.index)
#%%
h = pd.read_csv('datasets_imputed/mapocho-almendros.csv', index_col=0)
h.set_index('fecha', inplace=True)
h.index = pd.to_datetime(h.index)

#%%
full = pd.read_csv(os.path.join(EXTERNAL_DISK, models_directory, 'mapocho-almendros/graphs_mapocho-almendros/full_dataset.csv'))
full.set_index('fecha', inplace=True)
full.index = pd.to_datetime(full.index)

prediction = pd.read_csv('merged_df-mapocho.csv', index_col=0)
prediction.set_index('fecha', inplace=True)
prediction.index = pd.to_datetime(prediction.index)

prediction.drop(['dia_sen', 'precipitacion_bool', 'dias_sin_precip'], axis=1, inplace=True)
#%%
año = 2092

plt.figure(figsize=(15,8))
plt.style.use('default')
sns.lineplot(prediction[prediction.index.year == año][['temperatura','precipitacion']],)
plt.title("area de nieve predicha")
plt.legend = True
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()
#%%
plt.close()
#%%
prediction_grouped = prediction.groupby(prediction.index.day_of_year).mean()
full = full.groupby(full.index.day_of_year).mean()

# pd.merge(left=prediction_grouped, right=full, how='left', on='fecha', suffixes=('_pred', '_real'))
#%%
columnas_a_escalar = ['precipitacion', 'temperatura', 'dia_sen', 'precipitacion_bool', 'dias_sin_precip', 'area_nieve_pred']
df_plot = pd.DataFrame()
scaler = MinMaxScaler()
df_plot[columnas_a_escalar] = scaler.fit_transform(prediction[columnas_a_escalar])
df_plot.index = prediction.index


#%%
(
    df
    .groupby(df.index.month)
    ['precipitacion']
    .mean()
    .plot(kind='bar')
)

#%%
(
    df
    .groupby(df.index.month)
    ['temperatura']
    .mean()
    .plot(kind='bar')
)

#%%
(
    h
    .groupby(h.index.month)
    ['temperatura']
    .mean()
    .plot(kind='bar')
)

#%%
(
    h
    .groupby(h.index.month)
    ['temperatura']
    .mean()
    .plot(kind='bar')
)
#%%
fig = plt.figure(figsize=(12,10))
axs = fig.subplots(2,2)

# Plotting on axs[0,0] for df temperature
monthly_temp_df = df.groupby(df.index.month)['temperatura'].mean()
axs[0,0].bar(monthly_temp_df.index, monthly_temp_df.values) # Use .bar()
axs[0,0].set_title('Temperatura del GEM3')
axs[0,0].set_xlabel('Month')
axs[0,0].set_ylabel('Temperature (°C)')
axs[0,0].set_xticks(monthly_temp_df.index) # Ensure all months are shown as ticks

# Plotting on axs[0,1] for df precipitation
monthly_prec_df = df.groupby(df.index.month)['precipitacion'].mean()
axs[0,1].bar(monthly_prec_df.index, monthly_prec_df.values) # Use .bar()
axs[0,1].set_title('Precipitacion del GEM3')
axs[0,1].set_xlabel('Month')
axs[0,1].set_ylabel('Precipitation (mm)')
axs[0,1].set_xticks(monthly_prec_df.index)

# Plotting on axs[1,0] for h temperature
monthly_temp_h = h.groupby(h.index.month)['temperatura'].mean()
axs[1,0].bar(monthly_temp_h.index, monthly_temp_h.values) # Use .bar()
axs[1,0].set_title('Temperatura real')
axs[1,0].set_xlabel('Month')
axs[1,0].set_ylabel('Temperature (°C)')
axs[1,0].set_xticks(monthly_temp_h.index)

# Plotting on axs[1,1] for h precipitation
monthly_prec_h = h.groupby(h.index.month)['precipitacion'].mean()
axs[1,1].bar(monthly_prec_h.index, monthly_prec_h.values) # Use .bar()
axs[1,1].set_title('Precipitacion real')
axs[1,1].set_xlabel('Month')
axs[1,1].set_ylabel('Precipitation (mm)')
axs[1,1].set_xticks(monthly_prec_h.index)

plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
plt.show()



#%%
full.groupby(full.index.month)['area_nieve_pred'].mean()

#%%
# model1 = model1.set_index(pd.to_datetime(model1['fecha'])).drop('fecha', axis=1)
model2 = model2.set_index(pd.to_datetime(model2['fecha'])).drop('fecha', axis=1)
model3 = model3.set_index(pd.to_datetime(model3['fecha'])).drop('fecha', axis=1)
model4 = model4.set_index(pd.to_datetime(model4['fecha'])).drop('fecha', axis=1)
model5 = model5.set_index(pd.to_datetime(model5['fecha'])).drop('fecha', axis=1)

#%%
print(f"Media enero y febrero de modelo 1: {model1[model1.index.month.isin([12,1])].mean().iloc[0]}")
print(f"Media enero y febrero de modelo 3: {model3[model3.index.month.isin([12,1])].mean().iloc[0]}")
print(f"Media enero y febrero de modelo 2: {model2[model2.index.month.isin([12,1])].mean().iloc[0]}")

#%%
access = pd.read_csv('D:\data\csv\series_futuras_clean\mapocho-almendros\Mapocho ssp 585 2081-2100\ACCESS-ESM1-5.csv', index_col=0)
gem3 = pd.read_csv('D:\data\csv\series_futuras_clean\mapocho-almendros\Mapocho ssp 585 2081-2100\HadGEM3-GC31-LL.csv', index_col=0)
access.index = pd.to_datetime(access.index)
gem3.index = pd.to_datetime(gem3.index)

#%% Temperatura
print(f"Media enero y febrero de modelo 1: {access.loc[access.index.month.isin([12,1]),'temperatura'].mean()}")
print(f"Media enero y febrero de modelo 3: {gem3.loc[gem3.index.month.isin([12,1]),'temperatura'].mean()}")

#%% precipitacion
print(f"Media enero y febrero de modelo 1: {access.loc[access.index.month.isin([12,1]),'precipitacion'].mean()}")
print(f"Media enero y febrero de modelo 3: {gem3.loc[gem3.index.month.isin([12,1]),'precipitacion'].mean()}")

#%% dias_sin_precip
print(f"Media enero y febrero de modelo 1: {access.loc[gem3.index.month.isin([12,1]),'dias_sin_precip'].mean()}")
print(f"Media enero y febrero de modelo 3: {gem3.loc[gem3.index.month.isin([12,1]),'dias_sin_precip'].mean()}")
















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