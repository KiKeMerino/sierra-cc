#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

#%%
csv_path = "E:/data/csv/"
cuencas = ['genil-dilar','adda-bornio','indrawati-melamchi','mapocho-almendros','nenskra-Enguri','uncompahgre-ridgway']


#%%
def normalizar_cuenca(cuencas, area_path = 'E:/data/csv/areas/', series_path = 'E:/data/csv/series_agregadas/'):
    """!
    @brief Calcula las anomalías acumuladas de una columna específica en un DataFrame.

    Esta función toma un DataFrame, el nombre de una columna numérica y un tamaño de
    ventana, y calcula las anomalías acumuladas de los valores en esa columna con
    respecto a la media móvil de la ventana especificada.

    @param df Un pandas DataFrame que contiene la columna para la cual se calcularán
              las anomalías. Debe tener un índice temporal ordenado.
    @param columna Un string que especifica el nombre de la columna en `df` para la
                   cual se calcularán las anomalías. Esta columna debe contener datos numéricos.
    @param ventana Un entero que define el tamaño de la ventana móvil utilizada para
                   calcular la media móvil.

    @details
    La función realiza los siguientes pasos:
    - Calcula la media móvil de la columna especificada utilizando una ventana móvil centrada.
    - Calcula las anomalías restando la media móvil de los valores originales de la columna.
    - Calcula las anomalías acumuladas sumando secuencialmente las anomalías. Los valores
      NaN resultantes de la media móvil en los bordes del DataFrame se mantienen como NaN
      en las anomalías acumuladas.

    @return Un pandas Series que contiene las anomalías acumuladas de la columna especificada.
            El índice de la Serie coincidirá con el índice del DataFrame de entrada.

    @exception KeyError Si la `columna` especificada no existe en el DataFrame `df`.
    @exception TypeError Si la `columna` especificada no contiene datos numéricos.
    @exception ValueError Si el tamaño de la `ventana` no es un entero positivo.

    @note Se espera que el DataFrame de entrada tenga un índice temporal ordenado para que
          el cálculo de la media móvil y las anomalías acumuladas tenga sentido temporalmente.
    """
    if not isinstance(cuencas, list):
        cuencas = [cuencas]

    for cuenca in cuencas:
        try:
            area_file_path = area_path + cuenca + '.csv'
            serie_file_path = series_path + cuenca + '.csv'
            area = pd.read_csv(area_file_path)
            serie = pd.read_csv(serie_file_path)
        except FileNotFoundError:
            print(f"Error: files not found for basin {cuenca}")
            continue  # Move to the next basin if the file is not found

        df = pd.merge(serie, area, how='inner', left_on='fecha', right_on='fecha')
        columns = ['fecha','dia_sen','temperatura','precipitacion','precipitacion_bool','area_nieve']
        df.columns = columns
        df['fecha'] = pd.to_datetime(df["fecha"])
        # Calculo de dias transcurridos desde la última precipitacion
        df['dias_sin_precip'] = 0
        dias_transcurridos = 0

        del df['fecha']

        for index, row in df.iterrows():
            if row['precipitacion_bool'] == 1:
                dias_transcurridos = 0  # reinicia el contador si ha llovido
            else:
                dias_transcurridos += 1

            df.loc[index, 'dias_sin_precip'] = dias_transcurridos

        # MixMax Scaler y correlacion de adda-bornio
        lista_numericas = ['dia_sen', 'temperatura', 'precipitacion', 'dias_sin_precip']
        MinMax = MinMaxScaler()
        df[lista_numericas] = MinMax.fit_transform(df[lista_numericas]) # Transformamos las variables numéricas del dataset con MinMaxScaler

        df.to_csv(f'./csv_normalizados/{cuenca}_norm.csv', index=False)

#%%
normalizar_cuenca(cuencas)
#%%
df = pd.read_csv('csv_normalizados/adda-bornio_norm.csv')
df