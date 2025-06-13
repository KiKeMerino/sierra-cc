import pandas as pd
import os

input_data_path = 'D:/data/csv/series_futuras_og/'
output_data_path = 'D:/data/csv/series_futuras_clean/'

cuencas = os.listdir(input_data_path)
for cuenca in cuencas:
    escenarios = os.listdir(os.path.join(input_data_path, cuenca))
    for escenario in escenarios:
        try:
            df = pd.read_csv(os.path.join(input_data_path, cuenca, escenario))
        except FileNotFoundError:
            print(f"No se ha encontrado el archivo '{os.path.join(input_data_path, cuenca, escenario)}'")

        df = df.loc[1:, :'Unnamed: 22']

        # Arreglamos la fecha con las 3 primeras columnas que tienen el formato AÑO -- DIA JULIANO -- MES
        # Y establecemos fecha como indice del dataset
        year = escenario.split('-')[0][-4:]
        df['fecha'] = df.apply(lambda x: str(int(x['Unnamed: 0']) + int(year)) + '-' + str(x['Unnamed: 2']) + '-' + str(x['Unnamed: 1']), axis=1)
        # df['fecha'] = str(year + int(df['Unnamed: 0'])) + '-' + str(df['Unnamed: 2']) + '-' + str(df['Unnamed: 1'])
        df.set_index(pd.to_datetime(df['fecha'], format='%Y-%m-%j'), inplace=True)
        df.drop(columns=['Unnamed: 0', 'Unnamed: 2', 'Unnamed: 1', 'fecha'], inplace=True)

        # Iteramos sobre cada modelo y dividimos los datasets quedandonos con las columnas de precipitacion y temperatura exclusivamente
        models = [col for col in df.columns if not col.startswith('Unnamed')]
        for model in models:
            start_index = df.columns.get_loc(model)
            df_model = df.iloc[:, start_index:start_index+4].copy()
            df_model = df_model.iloc[:,2:]
            df_model.columns = ['precipitacion', 'temperatura']

            file_name = escenario[:-4] + '_clean.csv'
            df_model.to_csv(os.path.join(output_data_path, cuenca, file_name))