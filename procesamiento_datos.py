#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os

#%%
csv_path = "E:/data/csv/"
basins = ['genil-dilar','adda-bornio','indrawati-melamchi','mapocho-almendros','nenskra-Enguri','uncompahgre-ridgway']

#%%
def merge_basin(basins, area_path='E:/data/csv/areas/', series_path='E:/data/csv/series_agregadas/', output_dir='./csv_merged/'):

    if not isinstance(basins, list):
        basins = [basins]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for basin in basins:
        try:
            area_file_path = os.path.join(area_path, f"{basin}.csv")
            serie_file_path = os.path.join(series_path, f"{basin}.csv")
            area = pd.read_csv(area_file_path)
            serie = pd.read_csv(serie_file_path)
        except FileNotFoundError:
            print(f"Error: files not found for basin {basin}")
            continue  # Move to the next basin if the file is not found

        df = pd.merge(serie, area, how='inner', on='fecha')
        del df['fecha']
        columns = ['dia_sen', 'temperatura', 'precipitacion', 'precipitacion_bool', 'area_nieve']
        df.columns = columns

        # Calculo de dias transcurridos desde la última precipitacion
        df['dias_sin_precip'] = 0
        dias_transcurridos = 0
        for index, row in df.iterrows():
            if row['precipitacion_bool'] == 1:
                dias_transcurridos = 0  # reinicia el contador si ha llovido
            else:
                dias_transcurridos += 1

            df.loc[index, 'dias_sin_precip'] = dias_transcurridos

        output_path = os.path.join(output_dir, f'{basin}_merged.csv')
        if os.path.exists(output_path):
            overwrite = input(f"Warning: File '{output_path}' already exists. Overwrite? (y/N): ".lower())
            if overwrite != 'y':
                print(f"Overwrite cancelled for basin '{basin}'")
                continue # Move to the next basin
        
        df.to_csv(output_path, index=False)
        print(f"Data for basin '{basin}' normalized and saved to '{output_path}'.")
            

#%%
merge_basin(basins)
