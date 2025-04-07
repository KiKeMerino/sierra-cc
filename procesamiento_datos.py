#%%
import pandas as pd

#%%
csv_path = "E:/data/csv/"
cuencas = ['adda-bornio','genil-dilar','indrawati-melamchi','machopo-almendros','nenskra-Enguri','uncompahgre-ridgway']

area_ab = pd.read_csv(csv_path + "areas/" + cuencas[0] + ".csv")
variable_ab = pd.read_csv(csv_path + "series_agregadas/" + cuencas[0] + ".csv")

area_gd = pd.read_csv(csv_path + "areas/" + cuencas[1] + ".csv")
variable_gd = pd.read_csv(csv_path + "series_agregadas/" + cuencas[1] + ".csv")

area_im = pd.read_csv(csv_path + "areas/" + cuencas[2] + ".csv")
variable_im = pd.read_csv(csv_path + "series_agregadas/" + cuencas[2] + ".csv")

area_ma = pd.read_csv(csv_path + "areas/" + cuencas[3] + ".csv")
variable_ma = pd.read_csv(csv_path + "series_agregadas/" + cuencas[3] + ".csv")

area_ne = pd.read_csv(csv_path + "areas/" + cuencas[4] + ".csv")
variable_ne = pd.read_csv(csv_path + "series_agregadas/" + cuencas[4] + ".csv")

area_ur = pd.read_csv(csv_path + "areas/" + cuencas[5] + ".csv")
variable_ur = pd.read_csv(csv_path + "series_agregadas/" + cuencas[5] + ".csv")


#%%
def procesar_cuenca(areas, series):
    df = pd.merge(series, areas, how='inner', left_on='fecha', right_on='fecha')
    columns = ['fecha','dia_sen','temperatura','precipitacion','precipitacion_bool','area_nieve']
    df.columns = columns
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
#adda-bornio
adda = procesar_cuenca(area_ab, variable_ab)
adda.describe()

#%%
#genil-dilar
genil = procesar_cuenca(area_gd, variable_gd)
genil.describe()

#%%
#indrawati-melamchi
indrawati = procesar_cuenca(area_im, variable_im)
indrawati.describe()

#%%
# machopo-almendros
machopo = procesar_cuenca(area_ma, variable_ma)
machopo.describe()

#%%
#nenskra-enguri
nenskra = procesar_cuenca(area_ne, variable_ne)
nenskra.describe()

#%%
#uncompahgre-ridgway
uncompahgre = procesar_cuenca(area_ur, variable_ur)
uncompahgre.describe()

#%%
corr = adda.corr(numeric_only=True)
corr.style.background_gradient(cmap="coolwarm")

#%%
corr = genil.corr(numeric_only=True)
corr.style.background_gradient(cmap="coolwarm")

#%%
corr = indrawati.corr(numeric_only=True)
corr.style.background_gradient(cmap="coolwarm")

#%%
corr = machopo.corr(numeric_only=True)
corr.style.background_gradient(cmap="coolwarm")

#%%
corr = nenskra.corr(numeric_only=True)
corr.style.background_gradient(cmap="coolwarm")

#%%
corr = uncompahgre.corr(numeric_only=True)
corr.style.background_gradient(cmap="coolwarm")

#%%
# Se define el target como la variable a predecir con el modelo
target = ["area_nieve"]