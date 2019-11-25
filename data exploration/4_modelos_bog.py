##############################################################
#### Modelos Para El Aeropuerto Internacional el Dorado ######
##############################################################

#####################
# Importar librerías

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from sqlalchemy import create_engine
import seaborn as sns


###########################
#### Importar datos Limpios de Bogota
##########################

df = pd.read_csv('data exploration/bog_clean.csv.gz',low_memory=False,index_col=0,parse_dates=['day_hour',]).reset_index(drop=True)

#######################################################
#### MODELOS PARA LA VISIBILIDAD HORIZONTAL (vsby) ####
#######################################################

###############
#### Ordenar los datos de Menos a Mas Reciente Para Crear el Lag

df = df.sort_values(by=['day_hour'],ascending=True).reset_index(drop=True)
df.head()

###############
#### Quedarse con Y (Dependiente) y la Fecha
Y = df[['day_hour','vsby']]

###############
### Seleccionar Solo las variables del modelo
numeric_cols = ['tmpf','dwpf','relh','drct','sknt','alti','skyl1']
df_num = df[numeric_cols]

###############
### Rezagar la Info hasta 12 horas en intervalos de 2 horas para crear un data frame 

lagged_lists = []

for i in [1,2,4,6,8,10,12]:
    lag = df_num.shift(periods=1)
    lag.columns = [col+'_{}'.format(i) for col in lag.columns] 
    lagged_lists.append(lag)


##############
### Consolidar datos

df_fin = pd.concat([Y]+lagged_lists,axis=1)
df_fin = df_fin.sort_values(by=['day_hour'],ascending=False).reset_index(drop=True)
df_fin.dropna(inplace=True)

## Extraer Año Mes Dia Hora de la fecha
df_fin['year']=df_fin['day_hour'].dt.year
df_fin['month']=df_fin['day_hour'].dt.month
df_fin['day']=df_fin['day_hour'].dt.day 
df_fin['hour']=df_fin['day_hour'].dt.hour

## Recategorizar año
years = np.linspace(2017,2030,num=13,dtype='int')
years_dict = {}
for i,year in enumerate(years):
    years_dict[year]=i+1

df_fin['year']=df_fin['year'].map(years_dict)


###################
#### Seleccionar Set de Entrenamiento y Test

## Set de entrenamiento (tomar todo menos el ultimo mes de datos)
df.day_hour.max()
train = df_fin[df_fin['day_hour']<='2019-10-16']
test = df_fin[df_fin['day_hour']>'2019-10-16']

### Matriz X & y de entrenamiento
X_train = train.drop(['day_hour','vsby'],axis=1)
Y_train = train['vsby']

##### Matriz X & Y de test
Y_test = test['vsby']
X_test = test.drop(['day_hour','vsby'],axis=1)


#######################
### RANDOM FOREST #####

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import auc, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

#### Parametrizacion del Modelo y Grilla de Validacion Cruzada
estim = RandomForestRegressor(n_estimators=100,criterion='mse',n_jobs=1,random_state=123)
forest = GridSearchCV(estimator=estim,cv=5,param_grid={'max_depth':np.arange(16)+1},verbose=2)

##### Entrenar el modelo sobre la grilla de validacion
forest.fit(X=X_train,y=Y_train)

##### Mejor Parámetro
forest.best_params_
### {'max_depth': 8}
###  nuevo: {'max_depth': 7}

#### Mejor Error Cuadratico Medio
forest.best_score_
## 0.23321001075392683
###  nuevo: 0.21969718284717787

##### Re calibración del modelo con los mejores parámetros
forest = RandomForestRegressor(n_estimators=100,criterion='mse',n_jobs=1,random_state=123,max_depth=7)
forest.fit(X=X_train,y=Y_train)

#### Prediccion 
Y_pred = forest.predict(X_test)

#### MSE de train 0.32294936345991776
train_mse = mean_squared_error(Y_train,forest.predict(X_train))

#### MSE de test 0.3553254070124522
test_mse = mean_squared_error(Y_test,Y_pred)


#### Gráfico Predicción
plt.subplots(figsize=(12,6))
plt.plot(test['day_hour'],Y_test,label="Real")
plt.plot(test['day_hour'],Y_pred,label="Predicted")
plt.legend()
plt.xlabel('Date')
plt.ylabel('Horizontal Visibility')
plt.xticks(rotation=30)
plt.title("Trees: 100 Max Depth: 7 Random Forest \n Training MSE: {:0.3f} Test MSE: {:0.3f}".format(train_mse,test_mse))
plt.savefig("data exploration/SKBO_train_vsby.png",bbox_inches='tight',dpi=300)
plt.show()


#### Importancias Relativas
importances = {"Regressor":X_train.columns,"Importance":forest.feature_importances_}
importances = pd.DataFrame(importances)
importances = importances.sort_values(by='Importance',ascending=False)


#### Grafico de Importancias
plt.subplots(figsize=(12,5))
plt.bar(importances['Regressor'],height=importances['Importance'])
plt.title('Relative Importance of the Regressors')
plt.xticks(rotation=90,ha='right')
plt.savefig("data exploration/SKBO_importances_vsby.png",bbox_inches='tight',dpi=300)
plt.show()

#####Exportar Modelo
import pickle
# save the model to disk
filename = 'dash/SKBO_vsby.sav'
pickle.dump(forest, open(filename, 'wb'))








# #################
# ### Neural Network #####

# ##### La red neuronal hace over fit, el error de entrenamiento es bueno, el de test muy alto.

# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import auc, mean_squared_error, mean_absolute_error
# from sklearn.model_selection import RandomizedSearchCV

# #### Parametrizacion del Modelo y Grilla de Validacion Cruzada
# estim = MLPRegressor(hidden_layer_sizes=(40,20,10,5,1),random_state=1234,solver='adam',max_iter=1000,tol=1e-4,early_stopping=True)
# grid = {'activation':['identity', 'logistic', 'tanh', 'relu'],
#         'alpha':np.linspace(0.0001,0.9999,num=60)}
# nn = RandomizedSearchCV(estimator=estim,cv=5,param_distributions=grid,verbose=2,random_state=1234)

# ##### Entrenar el modelo sobre la grilla de validacion
# nn.fit(X=X_train,y=Y_train)

# ##### Mejor Parámetro
# nn.best_params_
# ### {'alpha': 0.10177457627118645, 'activation': 'tanh'}

# #### Mejor Error Cuadratico Medio
# nn.best_score_
# ## 0.03878751702896733

# #### Prediccion
# Y_pred = nn.predict(X_test)

# #### MSE de train
# train_mse = mean_squared_error(Y_train,nn.predict(X_train))

# #### MSE de test
# test_mse = mean_squared_error(Y_test,Y_pred)


# #### Gráfico Predicción
# plt.subplots(figsize=(12,6))
# plt.plot(test['day_hour'],Y_test,label="Real")
# plt.plot(test['day_hour'],Y_pred,label="Predicted")
# plt.legend()
# plt.xlabel('Date')
# plt.ylabel('Horizontal Visibility')
# plt.xticks(rotation=30)
# plt.title(" Neural Network \n Training MSE: {:0.3f} Test MSE: {:0.3f}".format(train_mse,test_mse))
# plt.show()



