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
# area_path = external_disk + "data/cuencas/adda-bornio/Adda-Bormio_basin.shp"
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
# df_genil_dilar.columns= ["n_ceros", "n_unos"]
df_indrawati_almendros = pd.DataFrame()
df_machopo_almendros = pd.DataFrame()
df_nenskra_Enguri = pd.DataFrame()
df_uncompahgre_ridgway = pd.DataFrame()

# Lectura de datos
for cuenca in ["genil-dilar"]:
    try:
        archivos_hdf = [str(archivo) for archivo in Path(data_path + "/" + cuenca).rglob("*.hdf")]
        archivos_hdf = archivos_hdf[0:50]
        archivos_shp = [str(archivo) for archivo in Path(data_path + "/" + cuenca).glob("*.shp")]
        area_path = archivos_shp[0]
        fechas = []

        resultados = []
        for archivo in archivos_hdf:
            coincidencia = re.search(r"_A(\d{4})(\d{3})_", archivo)
            if coincidencia:
                fecha = datetime.datetime.strptime(f"{coincidencia.group(1)}-{coincidencia.group(2)}", "%Y-%j").date()
                fecha = fecha.strftime("%d/%m/%Y")

                snow_cover = open_bands_boundary(archivo, area_path)
                dataset = pd.DataFrame(snow_cover["CGF_NDSI_Snow_Cover"])

                n_ceros, n_unos = ((snow_mapping(dataset) == 0).sum(), (snow_mapping(dataset) == 1).sum())
                resultados.append({'fecha': fecha, 'n_ceros': n_ceros, 'n_unos': n_unos})

        if cuenca == "adda-bornio":
            df_adda_bornio[fecha] = snow_mapping(dataset)
        elif cuenca == "genil-dilar":
            df_genil_dilar = pd.DataFrame(resultados)
            df_genil_dilar.set_index('fecha', inplace=True)
        elif cuenca == "indrawati-melamchi":
            df_indrawati_almendros[fecha] = snow_mapping(dataset)
        elif cuenca == "machopo-almendros":
            df_machopo_almendros[fecha] = snow_mapping(dataset)
        elif cuenca == "nenskra-Enguri":
            df_nenskra_Enguri[fecha] = snow_mapping(dataset)
        elif cuenca == "uncompahgre-ridgway":
            df_uncompahgre_ridgway[fecha] = snow_mapping(dataset)

    except FileNotFoundError:
        print(f"El directorio '{data_path}/{cuenca}' no fue encontrado.")

print(df_genil_dilar)