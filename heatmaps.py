#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray as rxr
from pathlib import Path
import geopandas as gpd
import os

def plot_heatmap_prob(basins, save=False, path='./images/heatmaps', data_path='E:/data/hdfs/'): # Added data_path as an argument for clarity
    """
    Generates and displays (or saves) a heatmap showing the probability of snow
    presence per pixel for specified basins.

    The function reads HDF files containing NDSI snow cover data within each
    basin's shapefile boundary and calculates the frequency of snow presence
    (NDSI between 40 and 100) for each pixel across all available HDF files.

    Args:
        basins (str or list): A single basin name (string) or a list of
            basin names (strings). The function will look for HDF files
            within subdirectories named after each basin inside the
            'data_path/basins/' directory, and shapefiles named
            '{basin_name}.shp' in the 'data_path/basins/' directory.
        save (bool, optional): If True, the generated heatmap will be saved
            as a PNG file to the directory specified by the 'path' argument.
            If False, the heatmap will be displayed using matplotlib's
            `plt.show()`. Defaults to False.
        path (str, optional): The directory path where the generated heatmap
            will be saved if `save` is True. Defaults to './image/heatmaps'.
        data_path (str, optional): The base directory where the 'basins'
            folder is located. Defaults to './data/'.

    Returns:
        None

    Raises:
        FileNotFoundError: If the subdirectory for a basin or the basin's
            shapefile is not found in the expected locations (though this is
            handled within the function, a message is printed to the console).

    Example:
        >>> data_path = '/path/to/your/data/'
        >>> plot_heatmap_prob(['Guadalquivir'], data_path=data_path)
        >>> plot_heatmap_prob(['Guadalquivir', 'Ebro'], save=True, path='./output_heatmaps', data_path=data_path)
    """
    # Set a base font size for all text elements
    plt.rcParams.update({'font.size': 14}) # Adjust this value as needed

    # Ensure basins is always a list for consistent iteration
    if isinstance(basins, str):
        basins = [basins]

    for basin in basins:

        # Lectura de datos
        try:
            archivos_hdf = [str(archivo) for archivo in Path(data_path + basin).rglob("*.hdf")]
            if not archivos_hdf:
                print(f"No HDF files found in '{data_path + basin}'. Skipping basin.")
                continue
        except FileNotFoundError:
            print(f"Error: Basin directory not found '{data_path + basin}'. Skipping basin.")
            continue

        try:
            archivos_shp = [str(archivo) for archivo in Path(data_path + basin).glob("*.shp")]
            if not archivos_shp:
                print(f"No SHP files found in '{data_path + basin}'. Skipping basin.")
                continue
            area_path = archivos_shp[0]
        except FileNotFoundError:
            print(f"Error: Shapefile directory not found '{data_path}/{basin}'. Skipping basin.")
            continue

        area = gpd.read_file(area_path)
        # Lista para almacenar los arrays booleanos de cobertura de nieve
        snow_presence_list = []

        for archivo in archivos_hdf:
            print(f"\r{basin} -> {archivos_hdf.index(archivo)/len(archivos_hdf)*100:.2f}% ...", end="")
            try:
                snow_cover = rxr.open_rasterio(archivo, masked=True, variable="CGF_NDSI_Snow_Cover").rio.clip(
                    area.geometry.to_list(), crs=area.crs, all_touched=False).squeeze()
            except Exception as e:
                print(f"\nError processing {archivo}: {e}. Skipping this file.")
                continue


            snow_cover = snow_cover.to_dataframe().dropna().reset_index()
            snow_cover['nieve'] = ((snow_cover.CGF_NDSI_Snow_Cover > 40) & (snow_cover.CGF_NDSI_Snow_Cover < 100)).astype(int)
            snow_cover.drop(columns=['band', 'spatial_ref', 'CGF_NDSI_Snow_Cover'], inplace=True)

            snow_presence_list.append(snow_cover)

        if snow_presence_list:

            df_media = pd.concat(snow_presence_list, ignore_index=True)
            df_media = df_media.groupby(['x', 'y'])['nieve'].mean().reset_index()

            probability_pivot = df_media.pivot(index='y', columns='x', values='nieve') * 100 # Multiply by 100 for percentage

            plt.figure(figsize=(12, 8))
            sns.heatmap(probability_pivot, cmap='Spectral',
                        cbar_kws={'label': 'Snow probability (%)', 'format': '%.0f'}, # Set legend label and format
                        vmin=0, vmax=100) # Set color bar limits
            plt.title(f'Snow probability by pixel - {basin}', fontsize=16) # Increased title font size
            plt.xlabel('Longitude (x)', fontsize=14) # Increased axis label font size
            plt.ylabel('Latitude (y)', fontsize=14) # Increased axis label font size
            plt.gca().set_aspect('equal') # plt.gca() es como una referencia a ax (axes)
            plt.xticks([])  # Eliminar marcas del eje x
            plt.yticks([])  # Eliminar marcas del eje y
            plt.gca().invert_yaxis()
            

            # Ensure the output directory exists
            if save:
                Path(path).mkdir(parents=True, exist_ok=True)
                plt.savefig(f"{path}/{basin}-probability.png", bbox_inches='tight') # bbox_inches='tight' prevents labels from being cut off
                print(f"\nHeatmap saved to {path}/{basin}-probability.png")
            else:
                plt.show()

            plt.close()

        else:
            print(f"\nNo snow cover data found for basin: {basin}.")


data_path ='E:/data/csv/areas/'
basins = ['genil-dilar','adda-bornio','indrawati-melamchi','mapocho-almendros','nenskra-Enguri','uncompahgre-ridgway']

plot_heatmap_prob(basins, save=True)