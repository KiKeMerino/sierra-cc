import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray
import rasterio as rio
from pathlib import Path
import re
import datetime
from carga_de_datos import open_bands_boundary


external_disk = "D:/"
data_path = os.path.join(external_disk, "data/", "cuencas/")

# Función para mapear todos los valores entre 40 y 100 (inclusive) a:
    # 0: no hay nieve
    # 1: hay nieve
def snow_mapping(df):
    nieve = (df >= 40) & (df <= 100)
    result = np.where(nieve, 1, 0)
    return result

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
        # archivos_hdf = archivos_hdf[:30]
        archivos_shp = [str(archivo) for archivo in Path(data_path + "/" + cuenca).glob("*.shp")]
        area_path = archivos_shp[0]

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
                fecha = datetime.datetime.strptime(f"{coincidencia.group(1)}-{coincidencia.group(2)}", "%Y-%j").date()
                fecha = fecha.strftime("%d/%m/%Y")

                snow_cover = open_bands_boundary(archivo, area_path)
                dataset = pd.DataFrame(snow_cover["CGF_NDSI_Snow_Cover"])
                n_ceros = ((snow_mapping(dataset) == 0).sum())
                n_unos = ((snow_mapping(dataset) == 1).sum())
                print(f"Cuenca: {cuenca} - {fecha} - {len(resultados)}")
                resultados.append({'fecha': fecha, cuenca: (n_ceros, n_unos)})
            
        if cuenca == "adda-bornio":
            df_adda_bornio = pd.DataFrame(resultados)
            df_adda_bornio.set_index('fecha', inplace=True)
        elif cuenca == "genil-dilar":
            df_genil_dilar = pd.DataFrame(resultados)
            df_genil_dilar.set_index('fecha', inplace=True)
        elif cuenca == "indrawati-melamchi":
            df_indrawati_almendros = pd.DataFrame(resultados)
            df_indrawati_almendros.set_index('fecha', inplace=True)
        elif cuenca == "machopo-almendros":
            df_machopo_almendros = pd.DataFrame(resultados)
            df_machopo_almendros.set_index('fecha', inplace=True)
        elif cuenca == "nenskra-Enguri":
            df_nenskra_Enguri = pd.DataFrame(resultados)
            df_nenskra_Enguri.set_index('fecha', inplace=True)
        elif cuenca == "uncompahgre-ridgway":
            df_uncompahgre_ridgway = pd.DataFrame(resultados)
            df_uncompahgre_ridgway.set_index('fecha', inplace=True)

    except FileNotFoundError:
        print(f"El directorio '{data_path}/{cuenca}' no fue encontrado.")

print(df_adda_bornio.to_string())
print(df_genil_dilar.to_string())
print(df_indrawati_almendros.to_string())
print(df_machopo_almendros.to_string())
print(df_nenskra_Enguri.to_string())
print(df_uncompahgre_ridgway.to_string())