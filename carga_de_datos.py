import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import re
import datetime
import rioxarray as rxr
import time
from multiprocessing import Pool, cpu_count

external_disk = "D:/"
data_path = os.path.join(external_disk, "data/", "cuencas/")

def snow_mapping(array):
    nieve = (array >= 40) & (array <= 100)
    return np.where(nieve, 1, 0)

def procesar_archivo(archivo, area, cuenca):
    coincidencia = re.search(r"_A(\d{4})(\d{3})_", archivo)
    if coincidencia:
        fecha = datetime.datetime.strptime(f"{coincidencia.group(1)}-{coincidencia.group(2)}", "%Y-%j").date().strftime("%d/%m/%Y")
        snow_cover = rxr.open_rasterio(archivo, masked=True, chunks='auto').rio.clip(
            area.geometry.to_list(), crs=area.crs, all_touched=True).squeeze()
        snow_mapped = snow_mapping(snow_cover["CGF_NDSI_Snow_Cover"].values)
        n_ceros = np.sum(snow_mapped == 0)
        n_unos = np.sum(snow_mapped == 1)
        return {'fecha': fecha, cuenca: (n_ceros, n_unos)}
    return None

def procesar_cuenca(cuenca, archivos_hdf, area):
    try:
        resultados = []
        for archivo in archivos_hdf:
            res = procesar_archivo(archivo, area, cuenca)
            if res:
                resultados.append(res)
        if resultados: # Añadido esta comprobación.
            df_cuenca = pd.DataFrame(resultados)
            df_cuenca.set_index('fecha', inplace=True)
            return cuenca, df_cuenca
        else:
            print(f"No se encontraron datos de fecha para {cuenca}")
            return cuenca, pd.DataFrame() # devuelve un dataframe vacio.

    except FileNotFoundError:
        print(f"El directorio '{data_path}/{cuenca}' no fue encontrado.")
        return cuenca, pd.DataFrame()

def procesar_cuenca_wrapper(args):
    return procesar_cuenca(*args)

if __name__ == '__main__':
    tic_global = time.time()
    cuencas = os.listdir(data_path)
    resultados_cuencas = {}
    areas = {}
    archivos_hdf_cuencas = {}

    for cuenca in cuencas:
        archivos_hdf_cuencas[cuenca] = [str(archivo) for archivo in Path(data_path + "/" + cuenca).rglob("*.hdf")]
        archivos_shp = [str(archivo) for archivo in Path(data_path + "/" + cuenca).glob("*.shp")]
        area_path = archivos_shp[0]
        areas[cuenca] = gpd.read_file(area_path)

    args_list = [(cuenca, archivos_hdf_cuencas[cuenca], areas[cuenca]) for cuenca in cuencas]

    with Pool(cpu_count()) as pool:
        resultados = pool.map(procesar_cuenca_wrapper, args_list)
        for res in resultados:
            if res:
                cuenca, df_cuenca = res
                resultados_cuencas[cuenca] = df_cuenca

    df_adda_bornio = resultados_cuencas.get("adda-bornio", pd.DataFrame())
    df_genil_dilar = resultados_cuencas.get("genil-dilar", pd.DataFrame())
    df_indrawati_almendros = resultados_cuencas.get("indrawati-melamchi", pd.DataFrame())
    df_machopo_almendros = resultados_cuencas.get("machopo-almendros", pd.DataFrame())
    df_nenskra_Enguri = resultados_cuencas.get("nenskra-Enguri", pd.DataFrame())
    df_uncompahgre_ridgway = resultados_cuencas.get("uncompahgre-ridgway", pd.DataFrame())

    tac_global = time.time()
    print(f"Tiempo total {tac_global-tic_global :.4f} segundos.")