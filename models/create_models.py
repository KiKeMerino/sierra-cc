import os
from models.model_rnn_multi import crear_y_guardar_rnn_multi
from models.model_rf_multi import crear_y_guardar_rf_multi

RUTA_MODELOS = '../modelos/'
RUTA_DATOS = '../data/csv_merged/'
NOMBRE_ARCHIVO_DATOS = 'cuencas_all.csv'

if not os.path.exists(RUTA_MODELOS):
    os.makedirs(RUTA_MODELOS)

# Crear y guardar el modelo RNN para todas las cuencas
crear_y_guardar_rnn_multi(RUTA_DATOS, RUTA_MODELOS, NOMBRE_ARCHIVO_DATOS)

# Crear y guardar el modelo Random Forest para todas las cuencas
crear_y_guardar_rf_multi(RUTA_DATOS, RUTA_MODELOS, NOMBRE_ARCHIVO_DATOS)

print("Proceso de creación y guardado de modelos completado.")