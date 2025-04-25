#%%
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt

data_path ='D:/data/csv/areas'

cuencas = ['genil-dilar','adda-bornio','indrawati-melamchi','machopo-almendros','nenskra-Enguri','uncompahgre-ridgway']

adda_bornio = pd.read_csv(data_path + "/adda-bornio.csv")

#%%
adda_bornio['fecha'] = pd.to_datetime(adda_bornio['fecha'], format='%Y-%m-%d')
adda_bornio.head(10)

#%%
adda_bornio['fecha'] = adda_bornio['fecha'].dt.day_of_year

#%%
adda_bornio_grouped = adda_bornio.groupby('fecha')['area_nieve'].mean()
sns.lineplot(adda_bornio_grouped)

plt.xlim(left=0)
plt.xlabel('Day of the year')
plt.ylabel('km2')
plt.title('adda-bornio')
plt.show()