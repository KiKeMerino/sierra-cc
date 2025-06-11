import tensorflow as tf
from tensorflow import keras
import os

# Define el directorio donde guardaste tus modelos
# Asegúrate de que esta ruta sea correcta
models_directory = os.path.join("E:", "models_per_basin")

# Suponiendo que conoces el nombre de una de tus cuencas, por ejemplo 'adda-bornio'
# y el nombre del archivo del modelo dentro de su subdirectorio.
# Si tus modelos están en subdirectorios, necesitas construir la ruta completa.
# Por ejemplo, si un modelo está en D:\models_per_basin\cuenca_1\narx_model_best_cuenca_1.h5

# Paso 1: Encuentra la ruta de un modelo
# Puedes listar las cuencas para ver cuáles están disponibles
cuencas_disponibles = [d for d in os.listdir(models_directory) if os.path.isdir(os.path.join(models_directory, d)) and not d.startswith('graphs') ]

if not cuencas_disponibles:
    print(f"No se encontraron directorios de cuencas en '{models_directory}'. Asegúrate de que la ruta sea correcta y contenga los subdirectorios de los modelos.")
else:
    print(f"Cuencas encontradas en '{models_directory}': {cuencas_disponibles}")
    for cuenca in cuencas_disponibles:
    # Intenta cargar y obtener la información para la primera cuenca encontrada como ejemplo
        example_cuenca = cuenca
        
        # Construye la ruta al archivo .h5 del modelo.
        # Necesitarás saber el formato de nombre que usaste al guardar el modelo.
        # Por ejemplo, si usaste 'narx_model_best_CUENCA_NAME.h5'
        model_filename = f'narx_model_best_{example_cuenca}.h5'
        model_path = os.path.join(models_directory, example_cuenca, model_filename)

        if not os.path.exists(model_path):
            print(f"\nNo se encontró el archivo del modelo en: {model_path}")
            print("Por favor, verifica el nombre del archivo del modelo y la estructura de directorios.")
        else:
            print(f"\nCargando modelo de ejemplo para la cuenca: {example_cuenca}")
            try:
                # Cargar el modelo
                loaded_model = keras.models.load_model(model_path)

                # Paso 2: Obtener la forma de entrada (input shape)
                # La propiedad input_shape de la primera capa del modelo (generalmente la de entrada)
                # te da las dimensiones esperadas.
                # Para modelos Sequential, loaded_model.input_shape es el input_shape del modelo completo.
                
                # El input_shape será (None, n_lags, n_features)
                # 'None' en la primera dimensión indica el tamaño del batch (puede ser cualquiera).
                # 'n_lags' es el número de pasos de tiempo pasados que el modelo espera.
                # 'n_features' es el número total de características (area_nieve + variables exógenas).
                
                print(f"\n--- Información de entrada del modelo {cuenca}---")
                print(f"Forma de entrada esperada por el modelo: {loaded_model.input_shape}")
                print(f"Número de parámetros (pesos y sesgos): {loaded_model.count_params()}")

                # Paso 3: Opcional - Imprimir un resumen completo del modelo
                # Esto te dará detalles sobre cada capa, sus parámetros y sus formas de entrada/salida.
                print("\n--- Resumen del modelo ---")
                loaded_model.summary()

                # A partir de loaded_model.input_shape, puedes deducir:
                # n_lags = loaded_model.input_shape[1]
                # n_features = loaded_model.input_shape[2]
                print(f"\nEl modelo espera {loaded_model.input_shape[1]} pasos de tiempo (lags) y {loaded_model.input_shape[2]} características por paso de tiempo.")

            except Exception as e:
                print(f"Ocurrió un error al cargar o inspeccionar el modelo: {e}")
                print("Asegúrate de que TensorFlow y Keras estén correctamente instalados y que el archivo .h5 no esté corrupto.")