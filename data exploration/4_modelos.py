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

###########################
#### MODELOS PARA LA VISIBILIDAD HORIZONTAL (vsby)
##########################

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
### Rezagar la Info una, seis y doce horas para crear un data frame 
df1 = df_num.shift(periods=1)
df1.columns = [col+'_1' for col in df1.columns] 
df2 = df_num.shift(periods=6)
df2.columns = [col+'_2' for col in df2.columns] 
df3 = df_num.shift(periods=12)
df3.columns = [col+'_3' for col in df3.columns] 


##############
### Consolidar datos
df_fin = pd.concat([Y,df1,df2,df3],axis=1)
df_fin = df_fin.sort_values(by=['day_hour'],ascending=False).reset_index(drop=True)
df_fin.dropna(inplace=True)

## Extraer Año Mes Dia Hora de la fecha
#df_fin['year']=df_fin['day_hour'].dt.year
df_fin['month']=df_fin['day_hour'].dt.month
df_fin['day']=df_fin['day_hour'].dt.day 
df_fin['hour']=df_fin['day_hour'].dt.hour



###################
#### Seleccionar Set de Entrenamiento y Test

## Set de entrenamiento (tomar todo menos el ultimo mes de datos)
df.day_hour.max()
train = df_fin[df_fin['day_hour']<='2019-10-16']
test = df_fin[df_fin['day_hour']>'2019-10-16']


#######################
### RANDOM FOREST #####

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import auc, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

### Matriz X & y de entrenamiento
X_train = train.drop(['day_hour','vsby'],axis=1)
Y_train = train['vsby']

#### Parametrizacion del Modelo y Grilla de Validacion Cruzada
estim = RandomForestRegressor(n_estimators=100,criterion='mse',n_jobs=1,random_state=123)
forest = GridSearchCV(estimator=estim,cv=5,param_grid={'max_depth':np.arange(X_train.shape[1])+1},verbose=2)

##### Entrenar el modelo sobre la grilla de validacion
forest.fit(X=X_train,y=Y_train)

##### Matriz X & Y de test
Y_test = test['vsby']
X_test = test.drop(['day_hour','vsby'],axis=1)

##### Mejor Parámetro
forest.best_params_
### {'max_depth': 8}


#### Mejor Error Cuadratico Medio
forest.best_score_
## 0.23321001075392683

#### Prediccion
Y_pred = forest.predict(X_test)

#### MSE de test
test_mse = mean_squared_error(Y_test,Y_pred)

#### Gráfico Predicción
plt.subplots(figsize=(12,6))
plt.plot(test['day_hour'],Y_test,label="Real")
plt.plot(test['day_hour'],Y_pred,label="Predicted")
plt.legend()
plt.xlabel('Date')
plt.ylabel('Horizontal Visibility')
plt.xticks(rotation=30)
plt.title("100 Trees 8 Max Depth Random Forest \n Training MSE: {:0.3f} Test MSE: {:0.3f}".format(forest.best_score_,test_mse))
plt.show()


#### Importancias Relativas
forest.feature_importances_


#################
### XGBOOST #####

from xgboost import XGBRegressor
from sklearn.metrics import auc, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


### Matriz X & y de entrenamiento
X_train = train.drop(['day_hour','vsby'],axis=1)
Y_train = train['vsby']

#### Parametrizacion del Modelo y Grilla de Validacion Cruzada
estim = XGBRegressor(booster='gbtree',verbosity=0,random_state=123)
np.random.seed(1234)
param_dist = {"max_depth": np.linspace(4,10,num=6,dtype='int'),
              "eta":np.random.uniform(0,1,size=20),
              "subsample":np.random.uniform(0,1,size=20),
              "max_features": randint(1, 11),
              "min_samples_split": randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

xgb = RandomizedSearchCV(estimator=estim,cv=5,param_distributions=param_dist,verbose=2)

##### Entrenar el modelo sobre la grilla de validacion
xgb.fit(X=X_train,y=Y_train)

##### Matriz X & Y de test
Y_test = test['vsby']
X_test = test.drop(['day_hour','vsby'],axis=1)

##### Mejor Parámetro
xgb.best_params_
### {'bootstrap': False, 'criterion': 'gini', 'eta': 0.8759326347420947, 'max_depth': 4, 'max_features': 9, 'min_samples_split': 8, 'subsample': 0.8021476420801591}

#### Mejor Error Cuadratico Medio
xgb.best_score_
## 0.2450765273697973

#### Prediccion
Y_pred = xgb.predict(X_test)

#### MSE de test
test_mse = mean_squared_error(Y_test,Y_pred)

#### Gráfico Predicción
plt.subplots(figsize=(12,6))
plt.plot(test['day_hour'],Y_test,label="Real")
plt.plot(test['day_hour'],Y_pred,label="Predicted")
plt.legend()
plt.xlabel('Date')
plt.ylabel('Horizontal Visibility')
plt.xticks(rotation=30)
plt.title("XGBoost \n Training MSE: {:0.3f} Test MSE: {:0.3f}".format(xgb.best_score_,test_mse))
plt.show()


#### Importancias Relativas
forest.feature_importances_






### Grafico
a = df[df['day_hour']>='2019-11-13']
cols = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'alti', 'vsby', 'skyl1', 'feel']
plt.subplots(figsize=(12,12),ncols=3,sharex=True)
for i,col in enumerate(cols):
    plt.subplot(3,3,i+1)
    plt.plot(a['day_hour'],a[col],label=col,color='navy')
    plt.ylabel(col)
    plt.xlabel('')
    #plt.legend()
    if i in [6,7,8]:
        plt.xticks(rotation=30,ha='right')
    else:
        plt.xticks([''])
    plt.title('{}'.format(col))

plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.show()

df['day_hour'].max()