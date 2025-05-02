#%%
import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import re
import datetime
import rioxarray as rxr
import concurrent.futures
import numpy as np

external_disk = "D:/"
data_path = os.path.join(external_disk, "data/")


def process_split_series(input_file, output_path):
    """
    Reads, processes, and splits the aggregated historical ERA5-Land time series
    data from a CSV file into separate CSV files for each basin.

    The function performs the following steps:
    1. Reads the CSV file specified by 'input_filepath' with semicolon delimiter.
    2. Corrects the format of the 'Fecha' column to datetime objects.
    3. Converts numerical columns ('T', 'T.1' to 'T.5', 'P', 'P.1' to 'P.5')
       by replacing commas with periods and then converting to numeric type.
    4. Creates a boolean precipitation column ('P_bool.5') based on a threshold
       for the 'P.5' column.
    5. Calculates the normalized day of the year (between 0 and 1) and its sine
       transformation ('dia_sen').
    6. Renames the 'Fecha' column to 'fecha'.
    7. Deletes the unnecessary 'Year' and 'Mes' columns.
    8. Splits the processed data into separate CSV files for each of the six
       basins (adda-bornio, genil-dilar, indrawati-melamchi, mapocho-almendros,
       nenskra-enguri, uncompahgre-ridgway) and saves them in the directory
       specified by 'output_dir'. Each output file includes the 'fecha', 'dia_sen',
       and the corresponding temperature ('T.x'), precipitation ('P.x'), and
       precipitation boolean ('P_bool.x') columns for that basin.

    Args:
        input_filepath (str): The absolute path to the input CSV file
            containing the aggregated historical ERA5-Land time series data.
        output_dir (str): The absolute path to the directory where the
            individual basin CSV files will be saved.

    Returns:
        None
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

    try:
        # Dividir columnas por cuencas
        series_agregadas[['fecha','dia_sen','T','P','P_bool']].to_csv(f"{output_path}/adda-bornio.csv", index=False)
        series_agregadas[['fecha','dia_sen','T.1','P.1','P_bool.1']].to_csv(f"{output_path}/genil-dilar.csv", index=False)
        series_agregadas[['fecha','dia_sen','T.2','P.2','P_bool.2']].to_csv(f"{output_path}/indrawati-melamchi.csv", index=False)
        series_agregadas[['fecha','dia_sen','T.3','P.3','P_bool.3']].to_csv(f"{output_path}/mapocho-almendros.csv", index=False)
        series_agregadas[['fecha','dia_sen','T.4','P.4','P_bool.4']].to_csv(f"{output_path}/nenskra-enguri.csv", index=False)
        series_agregadas[['fecha','dia_sen','T.5','P.5','P_bool.5']].to_csv(f"{output_path}/uncompahgre-ridgway.csv", index=False)

    except FileNotFoundError:
        print(f"Error: Output directory not found at '{output_path}'")

def calculate_area(snow_cover, cuenca):
    """
    Calculates the total snow cover area (in km²) for a given basin based on
    the provided snow cover raster data.

    The function reprojects the snow cover raster to a specific UTM coordinate
    system based on the basin name. It then identifies pixels with Normalized
    Difference Snow Index (NDSI) values between 40 and 100 (inclusive) as
    snow-covered and calculates the total area by multiplying the number of
    snow-covered pixels by the area of a single pixel.

    Args:
        snow_cover (rasterio.io.DatasetReader or xarray.Dataset): A raster dataset
            containing snow cover data, typically with an 'CGF_NDSI_Snow_Cover'
            variable. It should be readable by rasterio or xarray.
        cuenca (str): The name of the basin ('adda-bornio', 'genil-dilar',
            'indrawati-melamchi', 'machopo-almendros', 'nenskra-Enguri', or
            'uncompahgre-ridgway'). This determines the UTM projection to which
            the snow cover data will be reprojected.

    Returns:
        float: The total snow cover area in square kilometers (km²).

    Raises:
        ValueError: If the provided 'cuenca' name is not one of the supported basins.

    Example:
        >>> import rasterio
        >>> with rasterio.open('snow_cover_adda.tif') as src:
        >>>     area = calculate_area(src, 'adda-bornio')
        >>>     print(f"Snow cover area for adda-bornio: {area:.2f} km²")
    """
    if cuenca == "adda-bornio":
        snow_cover = snow_cover.rio.reproject("EPSG:25832")
    elif cuenca == "genil-dilar":
        snow_cover = snow_cover.rio.reproject("EPSG:25830")
    elif cuenca == "indrawati-melamchi":
        snow_cover = snow_cover.rio.reproject("EPSG:32645")
    elif cuenca == "machopo-almendros":
        snow_cover = snow_cover.rio.reproject("EPSG:32719")
    elif cuenca == "nenskra-Enguri":
        snow_cover = snow_cover.rio.reproject("EPSG:32638")
    elif cuenca == "uncompahgre-ridgway":
        snow_cover = snow_cover.rio.reproject("EPSG:32613")
    else:
        raise ValueError(f"Unsupported basin: {cuenca}. Supported basins are: "
                         "'adda-bornio', 'genil-dilar', 'indrawati-melamchi', "
                         "'machopo-almendros', 'nenskra-Enguri', 'uncompahgre-ridgway'")

    df = pd.DataFrame(snow_cover["CGF_NDSI_Snow_Cover"])

    n_pixeles_nieve = ((df >= 40) & (df <= 100)).sum().sum()
    area_pixel_nieve = abs(snow_cover.rio.resolution()[0] * snow_cover.rio.resolution()[1])

    return (area_pixel_nieve * n_pixeles_nieve) / 1e6

def process_hdf(cuenca, area, archivo):
    """
    Processes a single HDF file to extract snow cover area for a specific basin.

    The function reads a raster file, clips it to the area of the specified
    basin, extracts the snow cover data (NDSI), calculates the total snow
    cover area using the `calculate_area` function, and extracts the date
    from the filename.

    Args:
        cuenca (str): The name of the basin. This is used to pass to the
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
    print(f"\r{cuenca}: procesando {archivo}...", end="")
    coincidencia = re.search(r"_A(\d{4})(\d{3})_", archivo)
    fecha = None
    if coincidencia:
        try:
            fecha = datetime.datetime.strptime(f"{coincidencia.group(1)}-{coincidencia.group(2)}", "%Y-%j").date()
        except ValueError:
            print(f"Warning: Could not parse date from filename '{archivo}'")

    snow_cover = rxr.open_rasterio(archivo, masked=True, variable="CGF_NDSI_Snow_Cover").rio.clip(
        area.geometry.to_list(), crs=area.crs, all_touched=False).squeeze()

    if fecha:
        return {'fecha': fecha, 'area_nieve': calculate_area(snow_cover, cuenca)}
    else:
        return None

def process_basin(cuenca):
    """
    Processes all HDF files within a specified basin's directory to calculate
    daily snow cover area and saves the results to a CSV file.

    The function searches for all '.hdf' files within the subdirectory named
    after the basin inside the 'data_path/cuencas/' directory. It also locates
    the basin's shapefile ('.shp') in the same directory to define the spatial
    extent for processing. Each HDF file is processed using a thread pool
    to parallelize the calculation of snow cover area using the `process_hdf`
    function. The results are collected into a Pandas DataFrame, indexed by
    date, sorted chronologically, and saved to a new CSV file named
    '{basin_name}.csv' in the 'data_path/csv/areas/' directory.

    Args:
        cuenca (str): The name of the basin to process. The function expects
            a subdirectory with this name to exist within 'data_path/cuencas/'
            containing the HDF and SHP files.

    Returns:
        None

    Raises:
        FileNotFoundError: If the subdirectory for the specified basin or the
            basin's shapefile is not found in the expected locations.
        FileExistsError: If the output CSV file ('{basin_name}.csv') already
            exists in the 'data_path/csv/areas/' directory, preventing overwriting.

    Example:
        >>> data_path = '/path/to/your/data/'
        >>> process_basin('Guadalquivir')
        >>> process_basin('Ebro')
    """
    
    print(f"**Warning:** Processing basin '{cuenca}' can take more than 2 hours to execute.")
    confirmation = input("Are you sure you want to continue? (y/N): ").lower()
    if confirmation != 'y':
        print(f"Processing of basin '{cuenca}' cancelled by user.")
        return
    
    archivos_hdf = [str(archivo) for archivo in Path(data_path + "cuencas" + "/" + cuenca).rglob("*.hdf")]
    try:
        archivos_shp = [str(archivo) for archivo in Path(data_path + "cuencas" + "/" + cuenca).glob("*.shp")]
        area_path = archivos_shp[0]
        area = gpd.read_file(area_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find basin directory or shapefile for '{cuenca}'. {e}")
        return

    resultados = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_hdf, cuenca, area, archivo) for archivo in archivos_hdf]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                resultados.append(result)

    df_datos = pd.DataFrame(resultados)
    if not df_datos.empty:
        df_datos.set_index('fecha', inplace=True)
        df_datos.sort_index(inplace=True)
        output_filepath = f"{data_path}csv/areas/{cuenca}.csv"
        try:
            df_datos.to_csv(output_filepath, mode='x')
            print(f"\nCuenca {cuenca} procesada. Results saved to '{output_filepath}'.")
        except FileExistsError:
            print(f"\nError: Output file '{output_filepath}' already exists for basin {cuenca}. Skipping save.")
    else:
        print(f"\nNo valid snow cover data found for basin {cuenca}.")

cuencas = os.listdir(data_path + "cuencas/")
for cuenca in cuencas:
    process_basin(cuenca)