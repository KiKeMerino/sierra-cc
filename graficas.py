#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray as rxr
from pathlib import Path
import geopandas as gpd
import os

data_path ='E:/data/csv/areas/'

basins = ['genil-dilar','adda-bornio','indrawati-melamchi','mapocho-almendros','nenskra-Enguri','uncompahgre-ridgway']


def plot_area_perday(basins, save=False, path='./image/lineplots/per_day'):
    """
    Generates and displays (or saves) a line plot of the average snow cover area
    per day of the year for specified basins.

    Args:
        basins (str or list): A single basin name (string) or a list of
            basin names (strings). The function will look for CSV files
            named '{basin_name}.csv' in the directory specified by
            the global variable 'data_path'.
        save (bool, optional): If True, the plot will be saved as a PNG file
            to the directory specified by the 'path' argument. If False,
            the plot will be displayed using matplotlib's `plt.show()`.
            Defaults to False.
        path (str, optional): The directory path where the generated plot
            will be saved if `save` is True. Defaults to './image/lineplots/per_day'.

    Returns:
        None

    Raises:
        FileNotFoundError: If a CSV file for a specified basin is not found
            in the expected location (though this is handled within the
            function, a message is printed to the console).

    Example:
        >>> data_path = '/path/to/your/data/'
        >>> plot_area_perday('Guadalquivir')
        >>> plot_area_perday(['Guadalquivir', 'Ebro'], save=True, path='./output_plots')
    """
    if not isinstance(basins, list):
        basins = [basins]

    data_path = os.path.join('E:','data', 'csv', 'areas/')

    basins_path = [data_path + elemento + '.csv' for elemento in basins]

    for i, basin_file in enumerate(basins_path):

        try:
            df = pd.read_csv(basin_file)
        except FileNotFoundError:
            print(f"Error: file not found '{basin_file}'")
            continue  # Move to the next basin if the file is not found

        df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
        df['fecha'] = df['fecha'].dt.day_of_year
        df_grouped = df.groupby('fecha')['area_nieve'].mean()

        sns.lineplot(df_grouped)
        plt.xlim(left=0)
        plt.xlim(right=365)
        plt.xlabel('Day')
        plt.ylabel('km2')
        plt.title(f"Snow cover area - {basins[i]}")
        if save:
            plt.savefig(f"{path}/{basins[i]}-area.png") # Save with basin name
        else:
            plt.show()
        plt.close()

def plot_area_permonth(basins, save=False, path='./image/lineplots/per_month'):
    """
    Generates and displays (or saves) a line plot of the average snow cover area
    per month of the year for specified basins.

    Args:
        basins (str or list): A single basin name (string) or a list of
            basin names (strings). The function will look for CSV files
            named '{basin_name}.csv' in the directory specified by
            the global variable 'data_path'.
        save (bool, optional): If True, the plot will be saved as a PNG file
            to the directory specified by the 'path' argument. If False,
            the plot will be displayed using matplotlib's `plt.show()`.
            Defaults to False.
        path (str, optional): The directory path where the generated plot
            will be saved if `save` is True. Defaults to './image/lineplots/per_month'.

    Returns:
        None

    Raises:
        FileNotFoundError: If a CSV file for a specified basin is not found
            in the expected location (though this is handled within the
            function, a message is printed to the console).

    Example:
        >>> data_path = '/path/to/your/data/'
        >>> plot_area_permonth('Guadalquivir')
        >>> plot_area_permonth(['Guadalquivir', 'Ebro'], save=True, path='./output_plots')
    """
    if not isinstance(basins, list):
        basins = [basins]
    basins_path = [data_path + elemento + '.csv' for elemento in basins]

    for i, basin_file in enumerate(basins_path):

        try:
            df = pd.read_csv(basin_file)
        except FileNotFoundError:
            print(f"Error: file not found '{basin_file}'")
            continue  # Move to the next basin if the file is not found

        df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
        df['fecha'] = df['fecha'].dt.month
        df_grouped = df.groupby('fecha')['area_nieve'].mean()

        sns.lineplot(df_grouped)
        plt.xlim(left=0)
        plt.xlim(right=12)
        plt.xlabel('Month')
        plt.ylabel('km2')
        plt.title(f"Snow cover area - {basins[i]}")
        if save:
            plt.savefig(f"{path}/{basins[i]}-area.png") # Save with basin name
        else:
            plt.show()
        plt.close()

def plot_heatmap_prob(basins, save=False, path='./image/heatmaps', data_path='E:/data/hdfs/'): # Added data_path as an argument for clarity
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

            plt.figure(figsize=(12, 8)) # Increased figure size for better readability
            sns.heatmap(probability_pivot, cmap='Spectral',
                        cbar_kws={'label': 'Probabilidad de Nieve (%)', 'format': '%.0f'}, # Set legend label and format
                        vmin=0, vmax=100) # Set color bar limits
            plt.title(f'Probabilidad de Nieve por Píxel - {basin}', fontsize=16) # Increased title font size
            plt.xlabel('Longitud (x)', fontsize=14) # Increased axis label font size
            plt.ylabel('Latitud (y)', fontsize=14) # Increased axis label font size
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

plot_heatmap_prob(basins, save=True)