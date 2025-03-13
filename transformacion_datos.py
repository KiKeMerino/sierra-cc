import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray
import rasterio as rio
from carga_de_datos import open_bands_boundary


external_disk = "E:\\"
area_path = external_disk + "data/cuencas/adda-bornio/Adda-Bormio_basin.shp"
ruta_adda_bornio = external_disk + "data"
print(area_path)
# Lectura de datos
try:
    ficheros_hdf = os.listdir(ruta_adda_bornio)
    ficheros_hdf = [os.path.join(ruta_adda_bornio, fichero) for fichero in ficheros_hdf if fichero.endswith('.hdf') and os.path.isfile(os.path.join(ruta_adda_bornio, fichero))]

except FileNotFoundError:
    print(f"El directorio '{ruta_adda_bornio}' no fue encontrado.")

# Función para mapear todos los valores entre 40 y 100 (inclusive) a:
    # 0: no hay nieve
    # 1: hay nieve
def snow_mapping(df):
    nieve = (df >= 40) & (df <= 100)
    result = np.where(nieve, 1, 0)
    return result


for ruta in ficheros_hdf:
    snow_cover = open_bands_boundary(ruta, area_path)
    dataset = pd.DataFrame(snow_cover["CGF_NDSI_Snow_Cover"])
    df = snow_mapping(dataset)

    print(f"Numero de ceros: {(df == 0).sum()}")
    print(f"Numero de unos: {(df == 1).sum()}")