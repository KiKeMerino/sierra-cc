#%%
from sklearn.model_selection import train_test_split # Librería para el separar los datasets de train y test
from sklearn.linear_model import LinearRegression # Librería para generar un modelo de regresión lineal
import pandas as pd
from sklearn import metrics # Importamos métricas para validar el modelo

#%%
df = pd.read_csv("adda_norm.csv")
del df['fecha']
df

X = df.drop(['area_nieve'],axis=1).astype(float) # VARIABLES INDEPENDIENTES
y = df['area_nieve'] # VARIABLE DEPENDIENTE (A PREDECIR)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=23) # Separamos los datasets de train y test
lin_reg = LinearRegression() # Modelo de regresión lineal
lin_reg.fit(X_train,y_train) # Entrenamos el modelo con los datasets de train
y_pred = lin_reg.predict(X_test) # Predecimos con el modelo usando los datasets de test

#%%
df_resultados = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred}) # Creamos un dataset con el precio predecido y el precio real del dataset de test

print("R2:", metrics.r2_score(y_test, y_pred))
print ("MAE:", metrics.mean_absolute_error(y_test, y_pred))

#%%
df_resultados['diff'] = df_resultados['Predicted'] - df_resultados['Actual']
df_resultados.sort_values(by='diff', inplace=True)
df_resultados

#%%
df_resultados.hist("diff",range=[-200,200])