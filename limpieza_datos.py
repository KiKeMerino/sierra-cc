#%% IMPORTS
import os
import pandas as pd
# import geopandas as gpd
from pathlib import Path
import re
import datetime
# import rioxarray as rxr
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt


#%% DEFINICIÓN DE FUNCIONES
def calculate_area(snow_cover, basin):
    if basin == "adda-bornio":
        snow_cover = snow_cover.rio.reproject("EPSG:25832")
    elif basin == "genil-dilar":
        snow_cover = snow_cover.rio.reproject("EPSG:25830")
    elif basin == "indrawati-melamchi":
        snow_cover = snow_cover.rio.reproject("EPSG:32645")
    elif basin == "mapocho-almendros":
        snow_cover = snow_cover.rio.reproject("EPSG:32719")
    elif basin == "nenskra-Enguri":
        snow_cover = snow_cover.rio.reproject("EPSG:32638")
    elif basin == "uncompahgre-ridgway":
        snow_cover = snow_cover.rio.reproject("EPSG:32613")
    else:
        raise ValueError(f"Unsupported basin: {basin}. Supported basins are: "
                         "'adda-bornio', 'genil-dilar', 'indrawati-melamchi', "
                         "'mapocho-almendros', 'nenskra-Enguri', 'uncompahgre-ridgway'")

    df = pd.DataFrame(snow_cover["CGF_NDSI_Snow_Cover"])

    n_pixeles_nieve = ((df >= 40) & (df <= 100)).sum().sum()
    area_pixel_nieve = abs(snow_cover.rio.resolution()[0] * snow_cover.rio.resolution()[1])

    return (area_pixel_nieve * n_pixeles_nieve) / 1e6

# v_exog_hist.csv
def process_var_exog(input_file, output_path, save=False):
    """
        Coge el excel de series agregadas y lo convierte a csv separándolo y renombrando las columnas.

        Return -> devuelve un csv con todas las variables exógenas y con una nueva columna 'cuenca' que idenfica qué cuenca es
    """
    try:
        # Leo datos sobre variables exogenas: temperatura y precipitacion
        series_agregadas = pd.read_csv(input_file, delimiter=";")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return

    # Coreccion de formato
    series_agregadas['Fecha'] = pd.to_datetime(series_agregadas["Fecha"], format="%d/%m/%Y")
    columnas_numericas = ['T', 'T.1', 'T.2', 'T.3','T.4', 'T.5', 'P','P.1','P.2','P.3','P.4','P.5']
    for col in columnas_numericas:
        series_agregadas[col] = series_agregadas[col].apply(lambda x: x.replace(',','.')).apply(pd.to_numeric)

    # Añado P_bool.5
    series_agregadas['P_bool.5'] = np.where(series_agregadas['P.5']>0.1,1,0)

    # Pasar a dia juliano normalizado
    dia_juliano = series_agregadas['Fecha'].dt.strftime("%j")
    año = series_agregadas['Year']
    dias_año = año.apply(lambda x: 366 if x % 4 == 0 and x % 100 != 0 or x % 400 == 0 else 365)
    dia_normalizado = dia_juliano.astype(int) / dias_año
    dia_sen = np.sin(2 * np.pi * dia_normalizado)

    series_agregadas['dia_sen'] = dia_sen
    series_agregadas.rename(columns={'Fecha':'fecha'}, inplace=True)

    # Borrar columnas innecesarias
    columnas_innecesarias = ['Year','Mes']
    for col in columnas_innecesarias:
        del(series_agregadas[col])

    # Dividir columnas por basins
    columns = ['fecha', 'dia_sen', 'temperatura', 'precipitacion', 'precipitacion_bool']

    adda_bornio = series_agregadas[['fecha','dia_sen','T','P','P_bool']]
    adda_bornio.columns = columns
    adda_bornio['cuenca'] = 'adda-bornio'

    genil_dilar = series_agregadas[['fecha','dia_sen','T.1','P.1','P_bool.1']]
    genil_dilar.columns = columns
    genil_dilar['cuenca'] = 'genil-dilar' 

    indrawati_melamchi = series_agregadas[['fecha','dia_sen','T.2','P.2','P_bool.2']]
    indrawati_melamchi.columns = columns
    indrawati_melamchi['cuenca'] = 'indrawati-melamchi'

    mapocho_almendros = series_agregadas[['fecha','dia_sen','T.3','P.3','P_bool.3']]
    mapocho_almendros.columns = columns
    mapocho_almendros['cuenca'] = 'mapocho-almendros'

    nenskra_enguri = series_agregadas[['fecha','dia_sen','T.4','P.4','P_bool.4']]
    nenskra_enguri.columns = columns
    nenskra_enguri['cuenca'] = 'nenskra-enguri'

    uncompahgre_ridgway = series_agregadas[['fecha','dia_sen','T.5','P.5','P_bool.5']]
    uncompahgre_ridgway.columns = columns
    uncompahgre_ridgway['cuenca'] = 'uncompahgre-ridgway'

    df_final = pd.concat([adda_bornio, genil_dilar, indrawati_melamchi, mapocho_almendros, nenskra_enguri, uncompahgre_ridgway], axis=0)
    df_final.reset_index(drop=True, inplace=True)

    if save:
        df_final.to_csv(os.path.join(output_path, 'v_exog_hist.csv'))
    else:
        return df_final

# data/csv/areas/(6)
def process_hdf(basin, area, archivo):
    """
    Processes a single HDF file to extract snow cover area for a specific basin.

    The function reads a raster file, clips it to the area of the specified
    basin, extracts the snow cover data (NDSI), calculates the total snow
    cover area using the `calculate_area` function, and extracts the date
    from the filename.

    Args:
        basin (str): The name of the basin. This is used to pass to the
            `calculate_area` function for correct reprojection.
        area (geopandas.GeoDataFrame): A GeoDataFrame containing the geometry
            of the basin, used for clipping the raster data.
        archivo (str): The absolute path to the HDF raster file to be processed.
            The filename is expected to contain a date in the format
            '_AYYYYDDD_', where YYYY is the year and DDD is the day of the year
            (Julian day).

    Returns:
        dict: A dictionary containing the 'fecha' (datetime.date object)
            extracted from the filename and the 'area_nieve' (float) in km²
            calculated for the snow cover in the basin. Returns None if the
            date cannot be extracted from the filename.

    Example:
        >>> import geopandas as gpd
        >>> # Assuming 'area_guadalquivir.shp' and 'CGF_NDSI_Snow_Cover_A2023001_...' exist
        >>> area_gdf = gpd.read_file('area_guadalquivir.shp')
        >>> result = process_hdf('Guadalquivir', area_gdf.iloc[[0]], 'path/to/CGF_NDSI_Snow_Cover_A2023001_...')
        >>> if result:
        >>>     print(f"Date: {result['fecha']}, Snow Area: {result['area_nieve']:.2f} km²")
    """
    print(f"/r{basin}: procesando {archivo}...", end="")
    coincidencia = re.search(r"_A(/d{4})(/d{3})_", archivo)
    fecha = None
    if coincidencia:
        try:
            fecha = datetime.datetime.strptime(f"{coincidencia.group(1)}-{coincidencia.group(2)}", "%Y-%j").date()
        except ValueError:
            print(f"Warning: Could not parse date from filename '{archivo}'")

    snow_cover = rxr.open_rasterio(archivo, masked=True, variable="CGF_NDSI_Snow_Cover").rio.clip(
        area.geometry.to_list(), crs=area.crs, all_touched=False).squeeze()

    if fecha:
        return {'fecha': fecha, 'area_nieve': calculate_area(snow_cover, basin)}
    else:
        return None
def process_basin(basin):
    """
    Procesa los archivos hdf y crea un archivo de areas para cada cuenca
    """
    if basin not in ['adda-bornio', 'genil-dilar', 'indrawati-melamchi', 'mapocho-almendros', 'nenskra-Enguri', 'uncompahgre-ridgway']:
        raise ValueError(f"Unsupported basin: {basin}. Supported basins are: "
                         "'adda-bornio', 'genil-dilar', 'indrawati-melamchi', "
                         "'mapocho-almendros', 'nenskra-Enguri', 'uncompahgre-ridgway'")
    
    print(f"**Warning:** Processing basin '{basin}' can take more than 1 hour to execute.")
    confirmation = input("Are you sure you want to continue? (y/N): ").lower()
    if confirmation != 'y':
        print(f"Processing of basin '{basin}' cancelled by user.")
        return
    
    basin_path = Path(data_path, "hdfs", basin)
    archivos_hdf = [str(archivo) for archivo in basin_path.rglob("*.hdf")]
    archivos_shp = list(basin_path.glob("*.shp"))
    try:
        if not archivos_shp:
            raise FileNotFoundError(f"No shapefile (.shp) found in '{basin_path}'. ")
        area_path = str(archivos_shp[0])
        area = gpd.read_file(area_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find basin directory or shapefile for '{basin}'. {e}")
        return

    resultados = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_hdf, basin, area, archivo) for archivo in archivos_hdf]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                resultados.append(result)

    df_datos = pd.DataFrame(resultados)
    if not df_datos.empty:
        df_datos.set_index('fecha', inplace=True)
        df_datos.sort_index(inplace=True)
        output_filepath = f"{data_path}csv/areas/{basin}2.csv"
        if os.path.exists(output_filepath):
            overwrite = input(f"Warning: File '{output_filepath}' already exists. Overwrite? (y/N): ".lower())
            if overwrite != 'y':
                print(f"Overwrite cancelled for basin '{basin}'")
        df_datos.to_csv(output_filepath, index=False)

    else:
        print(f"/nNo valid snow cover data found for basin {basin}.")

# df_all.csv
def merge_areas_exog(areas_file, exog_file, save=False):
    areas = pd.read_csv(areas_file, index_col=0)
    exog = pd.read_csv(exog_file, index_col = 0)

    df = pd.merge(areas, exog, how='inner', on=['fecha', 'cuenca'])
    df['fecha'] = pd.to_datetime(df["fecha"])
    df['day'] = df['fecha'].dt.day_of_year

    # Calculo de dias transcurridos desde la última precipitacion
    df['dias_sin_precip'] = 0
    dias_transcurridos = 0
    for index, row in df.iterrows():
        if row['precipitacion_bool'] == 1:
            dias_transcurridos = 0  # reinicia el contador si ha llovido
        else:
            dias_transcurridos += 1

        df.loc[index, 'dias_sin_precip'] = dias_transcurridos
    if save:
        df.to_csv('./df_all.csv')
    else:
        return df

# ./datasets/(6)
def join_area_exog(exog_file, areas_path, output_path = './datasets', save=False):
    """
        Coge el archivo de areas, el archivo de variables exógenas y los junta, además de crear la nueva variable dias_sin_precip
    """
    exogs = pd.read_csv(exog_file, index_col=0)
    cuencas = exogs['cuenca'].unique()
    for cuenca in cuencas:
        df_area = pd.read_csv(os.path.join(areas_path, cuenca + '.csv'))
        dataset = pd.merge(left=df_area, right = exogs[exogs['cuenca'] == cuenca], how='inner', on='fecha')
        dataset.drop(columns=['cuenca'], inplace=True)

        dataset['dias_sin_precip'] = 0
        dias_transcurridos = 0
        for index, row in dataset.iterrows():
            if row['precipitacion_bool'] == 1:
                dias_transcurridos = 0  # reinicia el contador si ha llovido
            else:
                dias_transcurridos += 1

            dataset.loc[index, 'dias_sin_precip'] = dias_transcurridos

        if save:
            dataset.to_csv(os.path.join(output_path, f'{cuenca}.csv'))
        else:
            return dataset

def cleaning_future_series(input_data_path, output_data_path):
    """
    Procesa datos de cuencas, limpia y formatea archivos CSV, y guarda los resultados.

    Args:
        input_data_path (str): La ruta al directorio de entrada que contiene los datos de las cuencas.
        output_data_path (str): La ruta al directorio de salida donde se guardarán los archivos procesados.
        cuencas (list): Una lista de nombres de cuencas a procesar.
    """
    cuencas = os.listdir(input_data_path)
    for cuenca in cuencas:
        # Crea el directorio de salida si no existe
        os.makedirs(os.path.join(output_data_path, cuenca), exist_ok=True)
        
        escenarios = os.listdir(os.path.join(input_data_path, cuenca))
        for escenario in escenarios:
            try:
                df = pd.read_csv(os.path.join(input_data_path, cuenca, escenario))
            except FileNotFoundError:
                print(f"No se ha encontrado el archivo '{os.path.join(input_data_path, cuenca, escenario)}'")
                continue # Continúa con el siguiente escenario si el archivo no se encuentra

            df = df.loc[1:, :'Unnamed: 22']

            # Arreglamos la fecha con las 3 primeras columnas que tienen el formato AÑO -- DIA JULIANO -- MES
            # Y establecemos fecha como indice del dataset
            year = escenario.split('-')[0][-4:]
            df['fecha'] = df.apply(lambda x: str(int(x['Unnamed: 0']) + int(year)) + '-' + str(x['Unnamed: 2']) + '-' + str(x['Unnamed: 1']), axis=1)
            df.set_index(pd.to_datetime(df['fecha'], format='%Y-%m-%j'), inplace=True)
            df.drop(columns=['Unnamed: 0', 'Unnamed: 2', 'Unnamed: 1', 'fecha'], inplace=True)

            # Iteramos sobre cada modelo y dividimos los datasets quedandonos con las columnas de precipitacion y temperatura exclusivamente
            models = [col for col in df.columns if not col.startswith('Unnamed')]
            for model in models:
                start_index = df.columns.get_loc(model)
                df_model = df.iloc[:, start_index:start_index+4].copy()
                df_model = df_model.iloc[:,2:]
                df_model.columns = ['precipitacion', 'temperatura']

                # Hasta aqui tengo las variables de precipitacion y temperatura, faltan las de 'dia_sen', 'precipitacion_bool', y 'dias_sin_precip'
                # Pasar a dia juliano normalizado
                dia_juliano = df_model.index.strftime("%j")
                df_model['año'] = df_model.index.year
                año = df_model['año']
                dias_año = año.apply(lambda x: 366 if x % 4 == 0 and x % 100 != 0 or x % 400 == 0 else 365)
                dia_normalizado = dia_juliano.astype(int) / dias_año
                dia_sen = np.sin(2 * np.pi * dia_normalizado)
                df_model = df_model.drop('año', axis=1)

                df_model['dia_sen'] = dia_sen
                df_model.rename(columns={'Fecha':'fecha'}, inplace=True)

                file_name = escenario[:-4] + '_clean_processed.csv'
                df_model.to_csv(os.path.join(output_data_path, cuenca, file_name))

# Probar a imputar con bfil y ffil
def impute_outliers(df):
    df.index = pd.to_datetime(df.index)
    q1 = df.area_nieve.quantile(0.25)
    q3 = df.area_nieve.quantile(0.75)
    iqr = q3-q1
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = df.area_nieve > upper_bound

    tmp_df = df['area_nieve'].copy()
    tmp_df[outlier_mask] = np.nan

    valor_medio_estacional = tmp_df.groupby(
        [tmp_df.index.month, tmp_df.index.day]
    ).transform('mean')

    df_imputed = df.copy()
    df_imputed.loc[outlier_mask, 'area_nieve'] = valor_medio_estacional[outlier_mask]

    return df_imputed

#%% --- MAIN EXECUTION ---
# join_areas("E:/data/csv/areas", '', True)
# process_var_exog('E:/data/csv/Series_historicas_agregadas_ERA5Land.csv', '.')
# merge_areas_exog('areas_total.csv', './v_exog_hist.csv', save=True)


EXTERNAL_DISK = 'E:'
data_path = os.path.join(EXTERNAL_DISK, "data/")

exog_file = os.path.join(data_path, 'csv/v_exog_hist.csv')
areas_path = os.path.join(data_path, 'csv/areas/')
future_series_path_og = os.path.join(data_path, 'csv\series_futuras_og')
future_series_path_clean = os.path.join(data_path, 'csv\series_futuras_clean')

# df = pd.read_csv(os.path.join(areas_path, 'genil-dilar.csv'))
# df_imputed = impute_outliers(df)

# df_imputed.to_csv(os.path.join(areas_path, 'genil-dilar.csv'))
cleaning_future_series(future_series_path_og, future_series_path_clean)

join_area_exog(exog_file,areas_path,'datasets/', True)

# #%%
# df.dias_sin_precip.value_counts()
# # %%
# df.drop(columns=['cuenca', 'fecha'], inplace=True)
# df
# # %%
# sns.relplot(x="dia_sen", y = 'area_nieve', data = df)

# #%%
# corr_matrix = df.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".1f")
# plt.title('Correlation Matrix of Numerical Features')
# plt.show()