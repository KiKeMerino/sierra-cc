import warnings

import matplotlib.pyplot as plt

import rioxarray as rxr
from shapely.geometry import mapping, box
import geopandas as gpd

import earthpy.plot as ep

from rasterio.plot import plotting_extent

warnings.simplefilter('ignore')


modis_path = "./data/ejemplo_adda.hdf"
area_path = "data/cuencas/adda-bornio/Adda-Bormio_basin.shp"

# modis = rxr.open_rasterio(modis_path, masked=True)

# # Imprimir informacion completa sobre el hdf
# # modis.info()

# # Imprimir el nombre de los datasets
# with rio.open(modis_path) as groups:
#     for name in groups.subdatasets:
#         pass #print(name)

# # Imprimir los datos de la "banda"
# modis_bands = modis["CGF_NDSI_Snow_Cover"].squeeze()
# # ep.plot_bands(modis_bands)
# # plt.show()


def open_bands_boundary(modis_path, area_path, variable="CGF_NDSI_Snow_Cover"):
    """Open, subset and crop a MODIS h4 file.

    Parametros
    -----------
    modis_path : string 
        La ruta con los datos.
    area : geopandas GeoDataFrame con el shp de la cuenca
        Utilizado para recortar los datos del ráster mediante rioxarray.clip
    variable : List
        La banda que deseamos abrir, en este caso CGF_NDSI_Snow_Cover

    Returns
    -----------
    band : xarray DataArray
        Retorna el xarray recortado de la cuenca
    """

    # area.total_bounds contiene una tupla con las coordenadas del bounding box (minx, miny, maxx, maxy)
    # El operador * se utiliza para desempaquetar esta tupla, pasando las 4 coordenadas separados a la función box()
    area = gpd.read_file(area_path)
    crop_bound_box = [box(*area.total_bounds)]

    # rio.clip se usa para recortar el raster a una extensión geográfica específica
    band = rxr.open_rasterio(modis_path, masked=True, variable=variable).rio.clip(
        crop_bound_box,  crs=area.crs, all_touched=True, from_disk=True).squeeze()

    return band


# snow_cover = open_bands_boundary(modis_path, area_path)

# # Obtener latitud y longitud del DataArray
# latitudes = snow_cover.y.values
# longitudes = snow_cover.x.values

# modis_ext = plotting_extent(snow_cover.to_array().values[0],
#                             snow_cover.rio.transform())
# f, ax = plt.subplots()
# ep.plot_bands(  snow_cover.to_array().values[0],
#                 ax=ax,
#                 extent=modis_ext,
#                 title="Cubierta de nieve")
# area = gpd.read_file(area_path)
# area.plot( ax=ax,
#           color=(1, 0, 0, 0.2)) # RGB - transparencia

# plt.show()