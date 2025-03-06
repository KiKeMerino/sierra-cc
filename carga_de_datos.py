import os
import warnings

import matplotlib.pyplot as plt
import numpy.ma as ma
import xarray as xr
import rioxarray as rxr
from shapely.geometry import mapping, box
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import rasterio as rio

warnings.simplefilter('ignore')

modis_path = './data/ejemplo7(david).hdf'
modis_path = "./data/ejemplo8.hdf"
modis_path = "./data/ejemplo9.hdf"
modis_path = "./data/ejemplo_adda.hdf"


modis = rxr.open_rasterio(modis_path, masked=True)

# print(modis["CGF_NDSI_Snow_Cover"])

# Imprimir informacion completa sobre el hdf
# modis.info()

# Imprimir el nombre de los datasets
with rio.open(modis_path) as groups:
    for name in groups.subdatasets:
        pass # print(name)

# Imprimir los datos de la "banda"
modis_bands = rxr.open_rasterio(modis_path, masked=True, variable="CGF_NDSI_Snow_Cover").squeeze()
modis_rgb_xr = modis_bands.to_array()

# ep.plot_bands(modis_rgb_xr)
# plt.show()

area_path = "data/cuencas/adda-bornio/Adda-Bormio_basin.shp"
area = gpd.read_file(area_path)

crop_bound_box = [box(*area.total_bounds)]

desired_bands = ["CGF_NDSI_Snow_Cover"]


modis_pre_clip = rxr.open_rasterio(modis_path,
                                   masked=True,
                                   variable=desired_bands).rio.clip(crop_bound_box,
                                                                    crs=area.crs,
                                                                    # Include all pixels even partial pixels
                                                                    all_touched=True,
                                                                    from_disk=True).squeeze()
# The final clipped data
# For demonstration purpposes  i'm creating a plotting extent
from rasterio.plot import plotting_extent

modis_ext = plotting_extent(modis_pre_clip.to_array().values[0],
                            modis_pre_clip.rio.transform())

# View cropped data
f, ax = plt.subplots()
ep.plot_bands(modis_pre_clip.to_array().values[0],
              ax=ax,
              extent=modis_ext,
              title="Plot of data clipped to the crop box (extent)")
area.plot(ax=ax,
                    color="green")
plt.show()