#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#%%
df = pd.read_csv('E:/data/csv/areas/mapocho-almendros_mod.csv')
df
df = df.set_index('fecha')
df.index = pd.to_datetime(df.index)
#%%
df = pd.read_csv('E:/data/csv/areas/mapocho-almendros.csv')
df = df.set_index('fecha')
df.index = pd.to_datetime(df.index)
df
#%%
condicion = (df.index.month == 6) & (df.index.day > 26) #& (df['area_nieve'] < 50)
df[condicion]
#%%
# Reemplazo los valores incorrectos (0) de 'area_nieve' para el 1 de julio con la media de los valores del día anterior y posterior.
df_grouped = df.groupby(df.index.dayofyear)['area_nieve'].mean()
sns.lineplot(df_grouped)
plt.show()

#%%
plt.close()
#%%
inicio_junio = pd.to_datetime('2025-07-01').dayofyear
fin_julio = pd.to_datetime('2025-07-2').dayofyear

# Obtener el día del año del índice
dia_del_año = df.index.dayofyear

# Crear la condición para filtrar
condicion = (dia_del_año >= inicio_junio) & (dia_del_año <= fin_julio)

pd.set_option('display.max_rows', None)

df2 = df[condicion]
df2

#%%
pd.set_option('display.max_rows', None)
#%%
pd.reset_option('display.max_rows')
#%%
np.set_printoptions(threshold=np.inf)