#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os

#%%
csv_path = "E:/data/csv/"
cuencas = ['genil-dilar','adda-bornio','indrawati-melamchi','mapocho-almendros','nenskra-Enguri','uncompahgre-ridgway']

#%%
def normalize_basin(cuencas, area_path='E:/data/csv/areas/', series_path='E:/data/csv/series_agregadas/'):
    """
    Normalizes numerical features (day of the year, temperature, precipitation,
    days since last precipitation) for specified basins by applying Min-Max scaling
    and saves the normalized data to a new CSV file.

    The function reads area and time series data from CSV files for each basin,
    merges them based on the 'fecha' column, calculates the number of days
    since the last precipitation event, applies Min-Max scaling to the specified
    numerical columns, and saves the resulting normalized DataFrame to a new
    CSV file named '{basin_name}_norm.csv' in the './csv_normalizados/' directory.

    Args:
        cuencas (str or list): A single basin name (string) or a list of
            basin names (strings). The function will look for area data in
            CSV files named '{basin_name}.csv' in the directory specified by
            the 'area_path' argument, and time series data in CSV files
            named '{basin_name}.csv' in the directory specified by the
            'series_path' argument.
        area_path (str, optional): The directory path where the area data CSV
            files are located. Defaults to 'E:/data/csv/areas/'.
        series_path (str, optional): The directory path where the time series
            data CSV files are located. Defaults to 'E:/data/csv/series_agregadas/'.

    Returns:
        None

    Raises:
        FileNotFoundError: If the area or time series CSV file for a specified
            basin is not found in the specified paths (though this is handled
            within the function, a message is printed to the console).

    Example:
        >>> normalize_basin('Guadalquivir')
        >>> normalize_basin(['Guadalquivir', 'Ebro'], area_path='/path/to/areas/', series_path='/path/to/series/')
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
        columns = ['fecha', 'dia_sen', 'temperatura', 'precipitacion', 'precipitacion_bool', 'area_nieve']
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
        df[lista_numericas] = MinMax.fit_transform(df[lista_numericas])  # Transformamos las variables numéricas del dataset con MinMaxScaler

        output_path = f'./csv_normalizados/{cuenca}_norm.csv'
        if os.path.exists(output_path):
            overwrite = input(f"Warning: File '{output_path}' already exists. Overwrite? (y/N): ".lower())
            if overwrite != 'y':
                print(f"Overwrite cancelled for basin '{cuenca}'")
                continue # Move to the next basin
        
        df.to_csv(output_path, index=False)
        print(f"Data for basin '{cuenca}' normalized and saved to '{output_path}'.")
            

#%%
# normalize_basin(cuencas)
