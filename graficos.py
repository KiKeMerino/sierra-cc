#%%
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import earthpy.plot as ep
import os
from rasterio.plot import plotting_extent
import rioxarray as rxr
import numpy as np
import pandas as pd

#%%
# Obtener los datos
archivo = "ejemplos/adda-bornio.hdf"
area = gpd.read_file("D:/data/cuencas/adda-bornio/Adda-Bormio_basin.shp")
snow_cover = rxr.open_rasterio(archivo, masked=True, variable="CGF_NDSI_Snow_Cover").rio.clip(
                    area.geometry.to_list(), crs=area.crs, all_touched=True).squeeze()
# Reproyecto a WGS84
snow_cover = snow_cover.rio.reproject("EPSG:4326")


#%%
data = pd.DataFrame(snow_cover["CGF_NDSI_Snow_Cover"])
datos = pd.read_csv("adda-bornio50.csv", index_col='fecha')


#%%
datos.index = pd.to_datetime(datos.index, format="%d/%m/%Y")
datos_agrupados = datos.groupby(pd.Grouper(freq='M'))['nieve (40-100)'].sum().reset_index()
datos

#%%
data.count().sum()

#%%
# Crear un histograma de los datos de la variable "CGF_NDSI_Snow_Cover"
plt.hist(data, bins=5)
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.title("Histograma de CGF_NDSI_Snow_Cover")
plt.show()
# plt.savefig("img/hist-CGF_NDSI_Snow_Cover")

#%%
# # Crear un mapa de calor de los datos de la variable "CGF_NDSI_Snow_Cover"
heatmap = sns.lineplot(datos, x='fecha', y="nieve (40-100)")

# plt.savefig("img/heatmap-CGF_NDSI_Snow_Cover")

