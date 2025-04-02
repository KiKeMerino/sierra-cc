#%%
import pandas as pd

#%%
areas = pd.read_csv("D:/data/csv/areas/uncompahgre-ridgway.csv")
variables = pd.read_csv("D:/data/csv/series_agregadas/uncompahgre-ridgway.csv")

#%%
# def procesar_cuenca(areas, series):
df = pd.merge(variables, areas, how='inner', left_on='fecha', right_on='fecha')
columns = ['fecha','dia_normalizado','temperatura','precipitacion','precipitacion_bool','area_nieve']
df.columns = columns
df['fecha'] = pd.to_datetime(df["fecha"])
#%%
# Calculo de dias transcurridos desde la última precipitacion
df['dias_sin_precip'] = 0
dias_transcurridos = 0

for index, row in df.iterrows():
    if row['precipitacion_bool'] == 1:
        dias_transcurridos = 0  # reinicia el contador si ha llovido 
    else:
        dias_transcurridos += 1

    df.loc[index, 'dias_sin_precip'] = dias_transcurridos

#%%

df.describe()

#%%
# Se define el target como la variable a predecir con el modelo
target = ["area_nieve"]