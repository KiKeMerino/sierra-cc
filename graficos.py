import matplotlib.pyplot as plt
import seaborn as sns
from obtencion_datos import obtener_datos

# Obtener los datos
datos = obtener_datos("data/Adda-Bormio_basin-2024/MOD10A1F_61-20250226_083107/MOD10A1F.A2024001.h18v04.061.2024005143528.hdf")

# Crear un histograma de los datos de la variable "CGF_NDSI_Snow_Cover"
plt.hist(datos["CGF_NDSI_Snow_Cover"].values.flatten(), bins=50)
plt.xlabel("Valor")
plt.ylabel("Frecuencia")
plt.title("Histograma de CGF_NDSI_Snow_Cover")
plt.savefig("img/hist-CGF_NDSI_Snow_Cover")

# Crear un mapa de calor de los datos de la variable "CGF_NDSI_Snow_Cover"
sns.heatmap(datos["CGF_NDSI_Snow_Cover"])
plt.xlabel("Columna")
plt.ylabel("Fila")
plt.title("Mapa de calor de CGF_NDSI_Snow_Cover")
plt.savefig("img/heatmap-CGF_NDSI_Snow_Cover")