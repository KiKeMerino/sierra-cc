#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray as rxr
from pathlib import Path
import geopandas as gpd

data_path ='E:/data/csv/areas/'

cuencas = ['genil-dilar','adda-bornio','indrawati-melamchi','mapocho-almendros','nenskra-Enguri','uncompahgre-ridgway']


def plot_area_perday(cuencas, save=False, path = './img/lineplots/per_day'):
    """!
    @brief Plots the average snow cover area throughout the year for specified basins.

    This function takes a basin name or a list of basin names, reads corresponding
    CSV files containing snow cover data, calculates the average snow cover area
    for each day of the year, and generates a line plot for each basin.

    @param cuencas A string representing a single basin name or a list of strings
                   where each string is a basin name. The function expects CSV files
                   named `<basin_name>.csv` to be present in the `data_path` directory
                   (which should be defined globally).
    @param save (optional) A boolean indicating whether to save the plot as a PNG file
                in the 'img/' subdirectory (True) or display it (False). Defaults to False.

    @details
    The function performs the following steps for each specified basin:
    - Ensures the `cuencas` parameter is a list.
    - Constructs the file path for the basin's CSV file.
    - Reads the CSV file into a pandas DataFrame, expecting a 'fecha' column (YYYY-MM-DD)
      and an 'area_nieve' column (snow cover area in km2).
    - Converts the 'fecha' column to datetime objects and extracts the day of the year.
    - Groups the DataFrame by the day of the year and calculates the mean snow cover area.
    - Generates a line plot showing the average snow cover area against the day of the year.
    - Sets the x-axis limits to start from day 0.
    - Labels the x and y axes and sets the plot title.
    - If `save` is True, saves the plot as a PNG file in the 'img/' directory.
    - Otherwise, displays the plot.
    - Closes the plot to free up resources.

    @note The global variable `data_path` is expected to be defined and point to the
          directory containing the basin CSV files.

    @exception FileNotFoundError If the CSV file for a specified basin is not found
                                 in the expected location.
    """
    if not isinstance(cuencas, list):
        cuencas = [cuencas]
    cuencas_path = [data_path + elemento + '.csv' for elemento in cuencas]

    for i, cuenca_file in enumerate(cuencas_path):

        try:
            df = pd.read_csv(cuenca_file)
        except FileNotFoundError:
            print(f"Error: file not found '{cuenca_file}'")
            continue  # Move to the next basin if the file is not found

        df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
        df['fecha'] = df['fecha'].dt.day_of_year
        df_grouped = df.groupby('fecha')['area_nieve'].mean()

        sns.lineplot(df_grouped)
        plt.xlim(left=0)
        plt.xlabel('Day')
        plt.ylabel('km2')
        plt.title(f"Snow cover area - {cuencas[i]}")
        if save:
            plt.savefig(f"{path}/{cuencas[i]}-area.png") # Save with basin name
        else:
            plt.show()
        plt.close()

def plot_area_permonth(cuencas, save=False, path = './img/lineplots/per_month'):
    
    if not isinstance(cuencas, list):
        cuencas = [cuencas]
    cuencas_path = [data_path + elemento + '.csv' for elemento in cuencas]

    for i, cuenca_file in enumerate(cuencas_path):

        try:
            df = pd.read_csv(cuenca_file)
        except FileNotFoundError:
            print(f"Error: file not found '{cuenca_file}'")
            continue  # Move to the next basin if the file is not found

        df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
        df['fecha'] = df['fecha'].dt.month
        df_grouped = df.groupby('fecha')['area_nieve'].mean()

        sns.lineplot(df_grouped)
        plt.xlim(left=0)
        plt.xlabel('Month')
        plt.ylabel('km2')
        plt.title(f"Snow cover area - {cuencas[i]}")
        if save:
            plt.savefig(f"{path}/{cuencas[i]}-area.png") # Save with basin name
        else:
            plt.show()
        plt.close()

def plot_heatmap_prob(cuencas, save = False, path = './img/heatmaps' ):
    """!
    @brief Generates and displays (or saves) a heatmap of snow probability for specified basins.

    This function iterates through a list of basins, reads snow cover data from HDF files
    and area boundaries from SHP files within each basin's directory, calculates the
    snow probability per pixel, and visualizes it as a heatmap.

    @param cuencas A list of strings, where each string is the name of a basin directory.
    @param path The base path where the 'cuencas' subdirectory containing basin data is located.
    @param save (optional) A boolean indicating whether to save the plot as a PNG file (True)
                or display it (False). Defaults to False.

    @details
    The function performs the following steps for each basin:
    - Reads all HDF files containing snow cover data.
    - Reads the area boundary SHP file.
    - Clips the snow cover data to the basin's area.
    - Determines snow presence based on NDSI values (between 40 and 100).
    - Calculates the average snow presence probability for each pixel across all HDF files.
    - Generates a heatmap visualizing the snow probability.
    - Optionally saves the heatmap as a PNG file.

    @exception FileNotFoundError If the basin directory or required HDF/SHP files are not found.
    """
    for cuenca in cuencas:

        # Lectura de datos
        try:
            archivos_hdf = [str(archivo) for archivo in Path(data_path + "cuencas/" + cuenca).rglob("*.hdf")]
        except FileNotFoundError:
            print(f"Error: file not found '{data_path + 'cuencas/' + cuenca}'")

        try:
            archivos_shp = [str(archivo) for archivo in Path(data_path + "cuencas" + "/" + cuenca).glob("*.shp")]
            area_path = archivos_shp[0]
        except FileNotFoundError:
            print(f"Error: file not found '{data_path} cuencas/ {cuenca}'")

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

            if save:
                plt.savefig(f"{path}/{cuenca}-probability.png")
            else:
                plt.show()

            plt.close()

        else:
            print("No snow cover data found.")

plot_area_permonth(cuencas, save=True)
plot_area_perday(cuencas, save=True)