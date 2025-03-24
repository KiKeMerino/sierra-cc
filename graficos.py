import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import earthpy.plot as ep
import os
from rasterio.plot import plotting_extent
import rioxarray as rxr

# Obtener los datos
archivo = "ejemplos/adda-bornio.hdf"
area = gpd.read_file("D:/data/cuencas/adda-bornio/Adda-Bormio_basin.shp")
snow_cover = rxr.open_rasterio(archivo, masked=True, variable="CGF_NDSI_Snow_Cover").rio.clip(
                    area.geometry.to_list(), crs=area.crs, all_touched=True).squeeze()

# Crear un histograma de los datos de la variable "CGF_NDSI_Snow_Cover"
plt.hist(snow_cover["CGF_NDSI_Snow_Cover"].values.flatten(), bins=50)
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.title("Histograma de CGF_NDSI_Snow_Cover")
# plt.savefig("img/hist-CGF_NDSI_Snow_Cover")

# Crear un mapa de calor de los datos de la variable "CGF_NDSI_Snow_Cover"
sns.heatmap(snow_cover["CGF_NDSI_Snow_Cover"])
plt.xlabel("Columna")
plt.ylabel("Fila")
plt.title("Mapa de calor de CGF_NDSI_Snow_Cover")
# plt.savefig("img/heatmap-CGF_NDSI_Snow_Cover")



# # Obtener latitud y longitud del DataArray
latitudes = snow_cover.y.values
longitudes = snow_cover.x.values

modis_ext = plotting_extent(snow_cover.to_array().values[0],
                            snow_cover.rio.transform())
f, ax = plt.subplots()
ep.plot_bands(  snow_cover.to_array().values[0],
                ax=ax,
                extent=modis_ext,
                title="Cubierta de nieve")

area.plot( ax=ax,
          color=(1, 0, 0, 0.2)) # RGB - transparencia

plt.show()