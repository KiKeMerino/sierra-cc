#%% IMPORTS 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

#%% DATA
df_all = pd.read_csv('df_all.csv',index_col=0)
df_all.head()
#%%
data = pd.DataFrame(df_all.loc[df_all.cuenca=='adda-bornio']['area_nieve'])
#%%
area_adda.iloc[7372]

#%%
data.iloc[7372]