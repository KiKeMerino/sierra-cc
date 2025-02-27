# Este modulo de la librería pyhdf proporciona funciones para trabajar con archivos hdf en formato SD (Scientific Data)
from pyhdf.SD import SD, SDC
import pandas as pd

# Se le pasa la ruta de un archivo hdf y devuelve un diccionario con los 5 datasets del archivo
def obtener_datos(ruta_hdf):

    hdf_file = SD(ruta_hdf, SDC.READ)

    cgf_ndsi_snow_cover = hdf_file.select("CGF_NDSI_Snow_Cover").get()
    cloud_persistence = hdf_file.select("Cloud_Persistence").get()
    basic_qa = hdf_file.select("Basic_QA").get()
    algorithm_flags_qa = hdf_file.select("Algorithm_Flags_QA").get()
    mod10a1_ndsi_snow_cover = hdf_file.select("MOD10A1_NDSI_Snow_Cover").get()

    # Crear un DataFrame para cada dataset
    df_cgf_ndsi_snow_cover = pd.DataFrame(cgf_ndsi_snow_cover)
    df_cloud_persistence = pd.DataFrame(cloud_persistence)
    df_basic_qa = pd.DataFrame(basic_qa)
    df_algorithm_flags_qa = pd.DataFrame(algorithm_flags_qa)
    df_mod10a1_ndsi_snow_cover = pd.DataFrame(mod10a1_ndsi_snow_cover)

    datos = {
        "CGF_NDSI_Snow_Cover": df_cgf_ndsi_snow_cover,
        "Cloud_Persistence": df_cloud_persistence,
        "Basic_QA": df_basic_qa,
        "Algorithm_Flags_QA": df_algorithm_flags_qa,
        "MOD10A1_NDSI_Snow_Cover": df_mod10a1_ndsi_snow_cover
    }

    return datos

datos = obtener_datos("data/Adda-Bormio_basin-2024/MOD10A1F_61-20250226_083107/MOD10A1F.A2024001.h18v04.061.2024005143528.hdf")

