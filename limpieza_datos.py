#%% IMPORTS
import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import re
import datetime
import rioxarray as rxr
import concurrent.futures
import numpy as np

external_disk = "E:/"
data_path = os.path.join(external_disk, "data/")

#%% DEFINICIÓN DE FUNCIONES
def process_var_exog(input_file, output_path, save=False):

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
    
    basin_path = Path(data_path, "basins", basin)
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
        output_filepath = f"{data_path}csv/areas/{basin}.csv"
        if os.path.exists(output_filepath):
            overwrite = input(f"Warning: File '{output_filepath}' already exists. Overwrite? (y/N): ".lower())
            if overwrite != 'y':
                print(f"Overwrite cancelled for basin '{basin}'")
        df_datos.to_csv(output_filepath, index=False)

    else:
        print(f"/nNo valid snow cover data found for basin {basin}.")

def join_areas(areas_path, output_path='.', save=False):
    """
    Junta los 6 csv de areas para crear un unico dataframe con una nueva columna  identificando de que área se trata cada registro
    """
    dfs = []
    areas = os.listdir(areas_path)
    for area in areas:
        df = pd.read_csv(os.path.join(areas_path, area))
        df['cuenca'] = area[:-4]
        dfs.append(df)

    df_final = pd.concat(dfs, axis=0)
    if save:
        df_final.to_csv(os.path.join(output_path, 'areas_total.csv'))
    else:
        return df_final

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

def cleaning_exogenous_variables(excel_file):
    series_futuras = pd.read_excel(excel_file, sheet_name=None)
    adda_bornio = pd.concat(series_futuras['Genil ssp 245 2051-2070'], series_futuras['Genil ssp 245 2081-2100'])
    genil_dilar = pd.concat(series_futuras['Adda ssp 245 2051-2070'], series_futuras['Adda ssp 245 2081-2100'])
    indrawati_melamchi = pd.concat(series_futuras['Genil ssp 245 2051-2070'], series_futuras['Genil ssp 245 2081-2100'])
    mapocho_almendros = pd.concat(series_futuras['Genil ssp 245 2051-2070'], series_futuras['Genil ssp 245 2081-2100'])
    nenskra_enguri = pd.concat(series_futuras['Genil ssp 245 2051-2070'], series_futuras['Genil ssp 245 2081-2100'])
    uncompahgre_ridgway = pd.concat(series_futuras['Genil ssp 245 2051-2070'], series_futuras['Genil ssp 245 2081-2100'])

    adda_bornio.to_csv("./predicted_exog/adda-bornio.csv")
    genil_dilar.to_csv("./predicted_exog/genil-dilar.csv")
    indrawati_melamchi.to_csv("./predicted_exog/indrawati-melamchi.csv")
    mapocho_almendros.to_csv("./predicted_exog/mapocho-almendros.csv")
    nenskra_enguri.to_csv("./predicted_exog/nenskra-enguri.csv")
    uncompahgre_ridgway.to_csv("./predicted_exog/uncompahgre-ridgway.csv")


#%% --- MAIN EXECUTION ---
# join_areas("E:/data/csv/areas", '', True)
# process_var_exog('E:/data/csv/Series_historicas_agregadas_ERA5Land.csv', '.')
# merge_areas_exog('areas_total.csv', './v_exog_hist.csv', save=True)

df = pd.read_csv("df_all.csv", index_col=0)
df.head(50)

#%%
df.dias_sin_precip.value_counts()
# %%
df.drop(columns=['cuenca', 'fecha'], inplace=True)
df
# %%
sns.relplot(x="dia_sen", y = 'area_nieve', data = df)

#%%
corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".1f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()