import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import re
import datetime
import rioxarray as rxr

external_disk = "D:/"
data_path = os.path.join(external_disk, "data/", "cuencas/")

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

# Lectura de datos
cuencas = os.listdir(data_path)
for cuenca in cuencas:
    # Lectura de archivos
    archivos_hdf = [str(archivo) for archivo in Path(data_path + "/" + cuenca).rglob("*.hdf")]
    archivos_shp = [str(archivo) for archivo in Path(data_path + "/" + cuenca).glob("*.shp")]
    area_path = archivos_shp[0]
    area = gpd.read_file(area_path)
    if len(archivos_hdf) < 1:
        print("No se han encontrado archivos hdf en el directorio ", data_path + "/" + cuenca)
        continue
    if len(archivos_shp) < 1:
        print("No se han encontrado archivos shp en el directorio ", data_path + "/" + cuenca)
        continue

    fechas = []
    resultados = []
    for archivo in archivos_hdf:
        print(f"\r{cuenca}: {len(resultados)/len(archivos_hdf)*100:.0f}%", end="")
        # Se busca la fecha en el nombre del fichero hdf
        coincidencia = re.search(r"_A(\d{4})(\d{3})_", archivo)
        if coincidencia:
            # Cambio de formato de fecha
            fecha = datetime.datetime.strptime(f"{coincidencia.group(1)}-{coincidencia.group(2)}", "%Y-%j").date()

        # snow_cover = open_bands_boundary(archivo, area)
        snow_cover = rxr.open_rasterio(archivo, masked=True, variable="CGF_NDSI_Snow_Cover").rio.clip(
            area.geometry.to_list(), crs=area.crs, all_touched=True).squeeze()

        resultados.append({'fecha': fecha, 'area_nieve': calcular_area_nieve(snow_cover, cuenca)})

    df_datos = pd.DataFrame(resultados)

    # Ordenar por fecha
    df_datos.set_index('fecha', inplace=True)
    df_datos.sort_index(inplace=True)
    df_datos.to_csv(cuenca + ".csv")