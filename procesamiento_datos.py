#%%
import pandas as pd

#%%
area_uc = pd.read_csv("E:/data/csv/areas/uncompahgre-ridgway.csv")
variable_uc = pd.read_csv("E:/data/csv/series_agregadas/uncompahgre-ridgway.csv")

area_ab = pd.read_csv("E:/data/csv/areas/adda-bornio.csv")
variable_ab = pd.read_csv("E:/data/csv/series_agregadas/adda-bornio.csv")

cuencas = ['adda-bornio','genil-dilar','indrawati-melamchi','machopo-almendros','nenskra-Enguri','uncompahgre-ridgway']
paths = {}

#%%
def procesar_cuenca(areas, series):
    df = pd.merge(series, areas, how='inner', left_on='fecha', right_on='fecha')
    # columns = ['fecha','dia_sen','temperatura','precipitacion','precipitacion_bool','area_nieve']
    # df.columns = columns
    df['fecha'] = pd.to_datetime(df["fecha"])
    # Calculo de dias transcurridos desde la última precipitacion
    df['dias_sin_precip'] = 0
    dias_transcurridos = 0

    for index, row in df.iterrows():
        if row['precipitacion_bool'] == 1:
            dias_transcurridos = 0  # reinicia el contador si ha llovido 
        else:
            dias_transcurridos += 1

        df.loc[index, 'dias_sin_precip'] = dias_transcurridos

    return df
#%%
adda = procesar_cuenca(area_ab, variable_ab)

#%%
corr = adda.corr(numeric_only=True)
corr.style.background_gradient(cmap="coolwarm")

#%%
# Se define el target como la variable a predecir con el modelo
target = ["area_nieve"]