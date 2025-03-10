import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray
import rasterio as rio
from carga_de_datos import open_bands_boundary

modis_path = "./data/ejemplo_adda.hdf"
area_path = "data/cuencas/adda-bornio/Adda-Bormio_basin.shp"
snow_cover = open_bands_boundary(modis_path, area_path)

nieve = (snow_cover["CGF_NDSI_Snow_Cover"].data > 40) & (snow_cover["CGF_NDSI_Snow_Cover"].data < 100)
print(type(nieve))
df = pd.DataFrame(snow_cover["CGF_NDSI_Snow_Cover"])
nieve = (df > 40) & (df < 100) 
print(type(nieve))

df = np.where(nieve, 1, 0)


print(df)
print(f"Numero de ceros: {(df == 0).sum()}")
print(f"Numero de unos: {(df == 1).sum()}")