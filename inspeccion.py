#%% IMPORTS 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

#%% DATA
df_all = pd.read_csv('csv\df_all.csv',index_col=0)
df_genil = pd.read_csv('csv\csv_split\genil-dilar_merged.csv')
#%%
areas = pd.read_csv('csv/areas_total.csv', index_col=0)
areas = areas[areas.cuenca=='genil-dilar']
#%%
data = pd.DataFrame(df_all.loc[df_all.cuenca=='adda-bornio']['area_nieve'])
#%%
areas.set_index('fecha')

#%%
plt.figure(figsize=(15,6))
areas['area_nieve'].plot()
#%%
areas
# %%
corr = df_all.corr(numeric_only=True)
corr['area_nieve'].sort_values()

