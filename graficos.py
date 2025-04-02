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
from pandasql import sqldf

#%%
# Obtener los datos
external_disk = "D:/"
data_path = os.path.join(external_disk, "data/", "csv/")

adda_bornio = pd.read_csv(os.path.join(data_path, 'adda-bornio.csv'), index_col='fecha')
adda_bornio.index = pd.to_datetime(adda_bornio.index)

#%%
query = """
    SELECT strftime('%Y', fecha), AVG(area_nieve)
    FROM adda_bornio
    WHERE CAST(strftime('%Y', fecha) AS INTEGER) < 2025
    GROUP BY strftime('%Y', fecha)
"""
df = sqldf(query, locals())

sns.lineplot(df)

#%%
query = """
    SELECT strftime('%m', fecha), AVG(area_nieve)
    FROM adda_bornio
    WHERE strftime('%Y', fecha) = '2010'
    GROUP BY strftime('%m', fecha);
"""
df = sqldf(query, locals())

sns.lineplot(df)

#%%
query = """
    SELECT strftime('%m', fecha), AVG(area_nieve)
    FROM adda_bornio
    GROUP BY strftime('%m', fecha)
"""
df = sqldf(query, locals())

sns.lineplot(df)

#%%
query = """
    SELECT strftime('%Y', fecha), AVG(area_nieve) 
    FROM adda_bornio
    WHERE strftime('%m', fecha) = '12'
    GROUP BY strftime('%Y', fecha)
"""
df = sqldf(query, locals())

sns.lineplot(df)