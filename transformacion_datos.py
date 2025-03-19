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

# Función para mapear todos los valores entre 40 y 100 (inclusive) a:
    # 0: no hay nieve
    # 1: hay nieve
def snow_mapping(array):
    nieve = (array >= 40) & (array <= 100)
    return np.where(nieve, 1, 0)

cuencas = os.listdir(data_path)

df_adda_bornio = pd.DataFrame()
df_genil_dilar = pd.DataFrame()
df_indrawati_almendros = pd.DataFrame()
df_machopo_almendros = pd.DataFrame()
df_nenskra_Enguri = pd.DataFrame()
df_uncompahgre_ridgway = pd.DataFrame()

df_datos = pd.DataFrame()

# Lectura de datos
for cuenca in cuencas:
    try:

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

            # Se busca la fecha en el nombre del fichero hdf
            coincidencia = re.search(r"_A(\d{4})(\d{3})_", archivo)
            if coincidencia:
                # Cambio de formato de fecha
                fecha = datetime.datetime.strptime(f"{coincidencia.group(1)}-{coincidencia.group(2)}", "%Y-%j").date().strftime("%d/%m/%Y")

                # snow_cover = open_bands_boundary(archivo, area)
                snow_cover = rxr.open_rasterio(archivo, masked=True, variable="CGF_NDSI_Snow_Cover").rio.clip(
                    area.geometry.to_list(), crs=area.crs, all_touched=True).squeeze()

                snow_mapped = snow_mapping(snow_cover["CGF_NDSI_Snow_Cover"].values)
                n_ceros = np.sum(snow_mapped == 0)
                n_unos = np.sum(snow_mapped == 1)
                resultados.append({'fecha': fecha, cuenca: (n_ceros, n_unos)})

        df_datos = pd.DataFrame(resultados)
        df_datos.set_index('fecha', inplace=True)

        if cuenca == "adda-bornio":
            df_adda_bornio = df_datos
        elif cuenca == "genil-dilar":
            df_genil_dilar = df_datos
        elif cuenca == "indrawati-melamchi":
            df_indrawati_almendros = df_datos
        elif cuenca == "machopo-almendros":
            df_machopo_almendros = df_datos
        elif cuenca == "nenskra-Enguri":
            df_nenskra_Enguri = df_datos
        elif cuenca == "uncompahgre-ridgway":
            df_uncompahgre_ridgway = df_datos

    except FileNotFoundError:
        print(f"El directorio '{data_path}/{cuenca}' no fue encontrado.")


# print(df_adda_bornio.to_string())
# print(df_genil_dilar.to_string())
# print(df_indrawati_almendros.to_string())
# print(df_machopo_almendros.to_string())
# print(df_nenskra_Enguri.to_string())
# print(df_uncompahgre_ridgway.to_string())