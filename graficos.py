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
# Crear un histograma de los datos de la variable "CGF_NDSI_Snow_Cover"
plt.hist(snow_cover["CGF_NDSI_Snow_Cover"].values.flatten(), bins=50)
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.title("Histograma de CGF_NDSI_Snow_Cover")
plt.show()
plt.savefig("img/hist-CGF_NDSI_Snow_Cover")

#%%
# # Crear un mapa de calor de los datos de la variable "CGF_NDSI_Snow_Cover"
sns.heatmap(snow_cover["CGF_NDSI_Snow_Cover"])
plt.xlabel("Columna")
plt.ylabel("Fila")
plt.title("Mapa de calor de CGF_NDSI_Snow_Cover")
plt.show()
plt.savefig("img/heatmap-CGF_NDSI_Snow_Cover")

#%%
cubierta = snow_cover["CGF_NDSI_Snow_Cover"]
# # Crear una figura y ejes
fig, ax = plt.subplots()
#Mostrar snow_cover en los ejes.
cubierta.plot(ax = ax)
#Añadir un titulo
ax.set_title("Cubierta de nieve")
plt.show()
