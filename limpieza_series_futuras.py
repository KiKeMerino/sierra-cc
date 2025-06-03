#%%
import pandas as pd
import os

#%%
data_path = 'D:/data/csv/series_futuras/genil-dilar/'
archivo = os.path.join(data_path, 'Genil ssp 245 2051-2070.csv')

df = pd.read_csv(archivo)
# %%
df