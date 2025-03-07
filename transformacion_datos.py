import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray
import rasterio as rio
from osgeo import gdal

f='data/cuencas/machopo-almendros/231750394/MOD10A1F_A2015001_h12v12_061_2021316063557_HEGOUT.hdf'
f = "data/ejemplo.hdf"
ds = gdal.Open(f)
informacion = gdal.GDALInfoOptions(f)
print(informacion)
