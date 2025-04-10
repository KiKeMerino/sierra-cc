#%%
import rioxarray as rxr
from pathlib import Path
import geopandas as gpd
import os
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt


external_disk = "D:/"
data_path = os.path.join(external_disk, "data/")
cuencas = ['genil-dilar','adda-bornio','indrawati-melamchi','machopo-almendros','nenskra-Enguri','uncompahgre-ridgway']
for cuenca in cuencas:
    archivos_hdf = [str(archivo) for archivo in Path(data_path + "cuencas" + "/" + cuenca).rglob("*.hdf")]
    archivos_shp = [str(archivo) for archivo in Path(data_path + "cuencas" + "/" + cuenca).glob("*.shp")]
    area_path = archivos_shp[0]
    area = gpd.read_file(area_path)

    # Lista para almacenar los arrays booleanos de cobertura de nieve
    snow_presence_list = []

    for archivo in archivos_hdf:
        try:
            snow_cover = rxr.open_rasterio(archivo, masked=True, variable="CGF_NDSI_Snow_Cover").rio.clip(
                area.geometry.to_list(), crs=area.crs, all_touched=False).squeeze()

            snow_presence = (snow_cover > 40) and (snow_cover < 100).astype(int)
            snow_presence_list.append(snow_presence)

        except Exception as e:
            print(f"Error al procesar el archivo {archivo}: {e}")

    if snow_presence_list:
        snow_presence_stacked = xr.concat(snow_presence_list, dim='time')
        probability_snow = snow_presence_stacked.mean(dim='time')

        # Convertir a DataFrame especificando el nombre de la variable en .to_dataframe()
        probability_df = probability_snow.to_dataframe().reset_index().rename(columns={'CGF_NDSI_Snow_Cover': 'probability'})

        probability_pivot = probability_df.pivot(index='y', columns='x', values='probability')

        plt.figure(figsize=(10, 8))
        sns.heatmap(probability_pivot, cmap='mako', cbar_kws={'label': 'Probabilidad de Nieve'})
        plt.title(f'Probabilidad de Nieve por Píxel - {cuenca}')
        plt.xlabel('Longitud (x)')
        plt.ylabel('Latitud (y)')
        plt.xticks([])  # Eliminar marcas del eje x
        plt.yticks([])  # Eliminar marcas del eje y
        plt.gca().invert_yaxis()
        plt.savefig(f"img/heatmaps/{cuenca}-probability.png")
        plt.close()

    else:
        print("No se encontraron datos de cobertura de nieve para procesar.")
