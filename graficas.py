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
    @brief Genera y muestra (o guarda) un gráfico de la superficie de nieve promedio por día del año para las cuencas especificadas.

    Esta función toma un nombre de cuenca o una lista de nombres de cuencas, lee los archivos CSV correspondientes que contienen datos de superficie de nieve,
    calcula la superficie de nieve promedio para cada día del año y genera un gráfico de líneas para cada cuenca.

    @param cuencas Una cadena que representa el nombre de una única cuenca o una lista de cadenas donde cada cadena es el nombre de una cuenca.
                   La función espera que los archivos CSV nombrados `<nombre_de_cuenca>.csv` estén presentes en el directorio especificado por la variable global `data_path`.
    @param save (opcional) Un booleano que indica si guardar el gráfico como un archivo PNG en el subdirectorio especificado por `path` (True) o mostrarlo (False). El valor predeterminado es False.
    @param path (opcional) Una cadena que especifica la ruta al directorio donde se guardarán los gráficos si `save` es True. El valor predeterminado es './img/lineplots/per_day'.

    @details
    La función realiza los siguientes pasos para cada cuenca especificada:
    - Asegura que el parámetro `cuencas` sea una lista.
    - Construye la ruta del archivo CSV para la cuenca.
    - Lee el archivo CSV en un DataFrame de pandas, esperando una columna 'fecha' (AAAA-MM-DD) y una columna 'area_nieve' (superficie de nieve en km2).
    - Convierte la columna 'fecha' a objetos datetime y extrae el día del año.
    - Agrupa el DataFrame por el día del año y calcula la superficie de nieve promedio.
    - Genera un gráfico de líneas que muestra la superficie de nieve promedio en función del día del año.
    - Establece el límite inferior del eje x en 0.
    - Etiqueta los ejes x e y y establece el título del gráfico.
    - Si `save` es True, guarda el gráfico como un archivo PNG en la ruta especificada, con el nombre de archivo incluyendo el nombre de la cuenca.
    - De lo contrario, muestra el gráfico.
    - Cierra el gráfico para liberar recursos.

    @note Se espera que la variable global `data_path` esté definida y apunte al directorio que contiene los archivos CSV de las cuencas.

    @exception FileNotFoundError Si no se encuentra el archivo CSV para una cuenca especificada en la ubicación esperada.
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
    """!
    @brief Genera y muestra (o guarda) un gráfico de la superficie de nieve promedio por mes para las cuencas especificadas.

    Esta función toma un nombre de cuenca o una lista de nombres de cuencas, lee los archivos CSV correspondientes que contienen datos de superficie de nieve,
    calcula la superficie de nieve promedio para cada mes del año y genera un gráfico de líneas para cada cuenca.

    @param cuencas Una cadena que representa el nombre de una única cuenca o una lista de cadenas donde cada cadena es el nombre de una cuenca.
                   La función espera que los archivos CSV nombrados `<nombre_de_cuenca>.csv` estén presentes en el directorio especificado por la variable global `data_path`.
    @param save (opcional) Un booleano que indica si guardar el gráfico como un archivo PNG en el subdirectorio especificado por `path` (True) o mostrarlo (False). El valor predeterminado es False.
    @param path (opcional) Una cadena que especifica la ruta al directorio donde se guardarán los gráficos si `save` es True. El valor predeterminado es './img/lineplots/per_month'.

    @details
    La función realiza los siguientes pasos para cada cuenca especificada:
    - Asegura que el parámetro `cuencas` sea una lista.
    - Construye la ruta del archivo CSV para la cuenca.
    - Lee el archivo CSV en un DataFrame de pandas, esperando una columna 'fecha' (AAAA-MM-DD) y una columna 'area_nieve' (superficie de nieve en km2).
    - Convierte la columna 'fecha' a objetos datetime y extrae el mes del año.
    - Agrupa el DataFrame por el mes del año y calcula la superficie de nieve promedio.
    - Genera un gráfico de líneas que muestra la superficie de nieve promedio en función del mes del año.
    - Establece el límite inferior del eje x en 0.
    - Etiqueta los ejes x e y y establece el título del gráfico.
    - Si `save` es True, guarda el gráfico como un archivo PNG en la ruta especificada, con el nombre de archivo incluyendo el nombre de la cuenca.
    - De lo contrario, muestra el gráfico.
    - Cierra el gráfico para liberar recursos.

    @note Se espera que la variable global `data_path` esté definida y apunte al directorio que contiene los archivos CSV de las cuencas.

    @exception FileNotFoundError Si no se encuentra el archivo CSV para una cuenca especificada en la ubicación esperada.
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