#%% IMPORTS 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

#%% DATA
df_all = pd.read_csv('df_all.csv',index_col=0)
df_all.head()
#%%
areas = pd.read_csv('areas_total.csv', index_col=0)
areas = areas[areas.cuenca=='adda-bornio']
#%%
data = pd.DataFrame(df_all.loc[df_all.cuenca=='adda-bornio']['area_nieve'])
#%%
areas

#%%
data
# %%
corr = df_all.corr(numeric_only=True)
corr['area_nieve'].sort_values()