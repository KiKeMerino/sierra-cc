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

external_disk = "E:/"
data_path = os.path.join(external_disk, "data/")
#%%
# Leo datos sobre variables exogenas: temperatura y precipitacion
series_agregadas = pd.read_csv(data_path + "csv/" + "Series_historicas_agregadas_ERA5Land.csv", delimiter=";")

# Coreccion de formato
series_agregadas['Fecha'] = pd.to_datetime(series_agregadas["Fecha"], format="%d/%m/%Y")
columnas_numericas = ['T', 'T.1', 'T.2', 'T.3','T.4', 'T.5', 'P','P.1','P.2','P.3','P.4','P.5']
for col in columnas_numericas:
    series_agregadas[col] = series_agregadas[col].apply(lambda x: x.replace(',','.')).apply(pd.to_numeric)
# series_agregadas.info()

# Añado P_bool.5
series_agregadas['P_bool.5'] = np.where(series_agregadas['P.5']>0.1,1,0)
# series_agregadas.info()

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
# series_agregadas.head()

#
# Dividir columnas por cuencas
series_agregadas[['fecha','dia_sen','T','P','P_bool']].to_csv(f"{data_path}csv/series_agregadas/adda-bornio.csv", index=False)
series_agregadas[['fecha','dia_sen','T.1','P.1','P_bool.1']].to_csv(f"{data_path}csv/series_agregadas/genil-dilar.csv", index=False)
series_agregadas[['fecha','dia_sen','T.2','P.2','P_bool.2']].to_csv(f"{data_path}csv/series_agregadas/indrawati-melamchi.csv", index=False)
series_agregadas[['fecha','dia_sen','T.3','P.3','P_bool.3']].to_csv(f"{data_path}csv/series_agregadas/machopo-almendros.csv", index=False)
series_agregadas[['fecha','dia_sen','T.4','P.4','P_bool.4']].to_csv(f"{data_path}csv/series_agregadas/nenskra-enguri.csv", index=False)
series_agregadas[['fecha','dia_sen','T.5','P.5','P_bool.5']].to_csv(f"{data_path}csv/series_agregadas/uncompahgre-ridgway.csv", index=False)

#%%
def calcular_area_nieve(snow_cover, cuenca):
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

    df = pd.DataFrame(snow_cover["CGF_NDSI_Snow_Cover"])

    n_pixeles_nieve = ((df >= 40) & (df <= 100)).sum().sum()
    area_pixel_nieve = abs(snow_cover.rio.resolution()[0] * snow_cover.rio.resolution()[1])

    return (area_pixel_nieve * n_pixeles_nieve)/1e6

def procesar_archivo(cuenca, area, archivo):
    print(f"\r{cuenca}: procesando {archivo}...", end="")
    coincidencia = re.search(r"_A(\d{4})(\d{3})_", archivo)
    if coincidencia:
        fecha = datetime.datetime.strptime(f"{coincidencia.group(1)}-{coincidencia.group(2)}", "%Y-%j").date()

    snow_cover = rxr.open_rasterio(archivo, masked=True, variable="CGF_NDSI_Snow_Cover").rio.clip(
        area.geometry.to_list(), crs=area.crs, all_touched=False).squeeze()

    return {'fecha': fecha, 'area_nieve': calcular_area_nieve(snow_cover, cuenca)}

def procesar_cuenca(cuenca):
    archivos_hdf = [str(archivo) for archivo in Path(data_path + "cuencas" +"/" + cuenca).rglob("*.hdf")]
    archivos_hdf = archivos_hdf[:1000]
    archivos_shp = [str(archivo) for archivo in Path(data_path + "cuencas" +"/" + cuenca).glob("*.shp")]
    area_path = archivos_shp[0]
    area = gpd.read_file(area_path)

    resultados = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(procesar_archivo, cuenca, area, archivo) for archivo in archivos_hdf]
        for future in concurrent.futures.as_completed(futures):
            resultados.append(future.result())

    df_datos = pd.DataFrame(resultados)
    df_datos.set_index('fecha', inplace=True)
    df_datos.sort_index(inplace=True)
    df_datos.to_csv(f"{data_path}csv/areas/{cuenca}.csv", mode='x')
    print(f"\nCuenca {cuenca} procesada.")

cuencas = os.listdir(data_path + "cuencas/")
for cuenca in cuencas:
    procesar_cuenca(cuenca)