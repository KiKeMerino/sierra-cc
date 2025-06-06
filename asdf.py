import json

with open('D:\\models\\processed_combinations_log.json', 'r') as file:
    data = json.load(file)

# Diccionario para almacenar todos los modelos por cuenca
all_basin_models = {}

# Recopilar todos los modelos para cada cuenca
for config_name, config_data in data.items():
    for basin_name, metrics in config_data.items():
        if basin_name not in all_basin_models:
            all_basin_models[basin_name] = []
        
        # Crear un diccionario para el modelo actual, incluyendo el nombre de la configuración
        model_info = {
            "config": config_name,
            **metrics  # Añadir todas las métricas R2, MAE, NSE, KGE
        }
        all_basin_models[basin_name].append(model_info)

# Diccionario para almacenar los mejores 3 modelos para cada cuenca
best_models_output = {}

# Para cada cuenca, encontrar los 3 mejores modelos según NSE
for basin_name, models_list in all_basin_models.items():
    # Ordenar los modelos en orden descendente según la métrica NSE
    sorted_models = sorted(models_list, key=lambda x: x.get("NSE", -float('inf')), reverse=True)
    
    # Seleccionar los 3 mejores modelos
    best_models_output[basin_name] = sorted_models[:3]

# Escribir la salida a un archivo JSON
output_file_name = "best_models.json"
with open(output_file_name, 'w') as f:
    json.dump(best_models_output, f, indent=4)

print(f"Archivo '{output_file_name}' generado exitosamente con los 3 mejores modelos por cuenca.")