import rioxarray as rxr
from pathlib import Path
import geopandas as gpd
import os
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


data_path = os.path.join("E:/data/")

cuencas = ['genil-dilar','indrawati-melamchi','machopo-almendros','nenskra-Enguri','uncompahgre-ridgway']
for cuenca in cuencas:

    archivos_hdf = [str(archivo) for archivo in Path(data_path + "cuencas" + "/" + cuenca).rglob("*.hdf")]
    archivos_shp = [str(archivo) for archivo in Path(data_path + "cuencas" + "/" + cuenca).glob("*.shp")]
    area_path = archivos_shp[0]
    area = gpd.read_file(area_path)

    # Lista para almacenar los arrays booleanos de cobertura de nieve
    snow_presence_list = []

    for archivo in archivos_hdf:
        print(f"\r{cuenca} -> {archivos_hdf.index(archivo)/len(archivos_hdf)*100:.2f}% ...", end="")
        snow_cover = rxr.open_rasterio(archivo, masked=True, variable="CGF_NDSI_Snow_Cover").rio.clip(
            area.geometry.to_list(), crs=area.crs, all_touched=False).squeeze()


        snow_cover = snow_cover.to_dataframe().dropna().reset_index()
        snow_cover['nieve'] = ( (snow_cover.CGF_NDSI_Snow_Cover > 40) & (snow_cover.CGF_NDSI_Snow_Cover < 100) ).astype(int)
        snow_cover.drop(columns=['band', 'spatial_ref', 'CGF_NDSI_Snow_Cover'], inplace = True)

        snow_presence_list.append(snow_cover)

    if snow_presence_list:

        df_media = pd.concat(snow_presence_list, ignore_index=True)
        df_media = df_media.groupby(['x', 'y'])['nieve'].mean().reset_index()

        probability_pivot = df_media.pivot(index='y', columns='x', values='nieve')

        plt.figure(figsize=(10, 6))
        sns.heatmap(probability_pivot, cmap='Spectral', cbar_kws={'label': 'Snow probability'})
        plt.title(f'Snow probability by pixel - {cuenca}')
        plt.xlabel('Longitude (x)')
        plt.ylabel('Latitude (y)')
        plt.xticks([])  # Eliminar marcas del eje x
        plt.yticks([])  # Eliminar marcas del eje y
        plt.gca().invert_yaxis()
        plt.savefig(f"img/heatmaps/{cuenca}-probability.png")
        # plt.show()
        plt.close()

    else:
        print("No snow cover data was.")
