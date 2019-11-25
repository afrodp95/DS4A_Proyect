#####################
# Importar librerías
import pandas as pd 
import numpy as np
import datetime
import time
from sqlalchemy import create_engine
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import auc, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

#####################
### Fecth Clean data from AWS DB

stations = ['SKAR', 'SKBG', 'SKBO', 'SKCC', 'SKCG', 'SKCL', 'SKMR', 'SKPE', 'SKSM', 'SKSP']
engine = create_engine('postgresql://ds4a_18:ds4a2019@ds4a18.cmlpaj0d1yqv.us-east-2.rds.amazonaws.com:5432/Airports_ds4a')

#############################################
####  HORIZONTAL VISBILITY MODELS (vsby) ####
#############################################

print("Model Training for Horizontal Visibiliy Initiated",end="\n\n")

for station in stations:

    print("Training Models of {} Station".format(station),end="\n\n")
    
    ## Get Data
    print("Fetching {} Clean Data".format(station))
    query = "SELECT * FROM dataclean WHERE station = {}".format(station)
    df = pd.read_sql(query, engine.connect(), parse_dates=('valid',))
    df = df.sort_values(by=['day_hour'],ascending=True).reset_index(drop=True)
    print('Clean Data of {} Succesfully Fetched From AWS RDS'.format(station),end="\n\n")

    ## Save vsby and date_hour
    print("Preparing Data For Model Training")
    Y = df[['day_hour','vsby']]

    ## Select only numeric variables
    numeric_cols = ['tmpf','dwpf','relh','drct','sknt','alti','skyl1']
    df_num = df[numeric_cols]

    ## Lag Data
    lagged_lists = []

    for i in [1,2,4,6,8,10,12]:
        lag = df_num.shift(periods=1)
        lag.columns = [col+'_{}'.format(i) for col in lag.columns] 
        lagged_lists.append(lag)

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
    print("Data For Model Training Ready",end="\n\n")

    print("Selecting Train and Test Data",end="\n\n")
    train = df_fin[df_fin['day_hour']<='2019-10-23']
    test = df_fin[df_fin['day_hour']>'2019-10-23']

    ### Matriz X & y de entrenamiento
    X_train = train.drop(['day_hour','vsby'],axis=1)
    Y_train = train['vsby']

    ##### Matriz X & Y de test
    Y_test = test['vsby']
    X_test = test.drop(['day_hour','vsby'],axis=1)

    print("Starting Model Training",end="\n\n")

    #### Parametrizacion del Modelo y Grilla de Validacion Cruzada
    estim = RandomForestRegressor(n_estimators=100,criterion='mse',n_jobs=1,random_state=123)
    forest = GridSearchCV(estimator=estim,cv=5,param_grid={'max_depth':np.arange(11)+1},verbose=2)

    ##### Entrenar el modelo sobre la grilla de validacion
    forest.fit(X=X_train,y=Y_train)

    ##### Re calibración del modelo con los mejores parámetros
    m_depth = forest.best_params['max_depth']
    forest2 = RandomForestRegressor(n_estimators=100,criterion='mse',
                                    n_jobs=1,random_state=123,
                                    max_depth=m_depth)
    forest2.fit(X=X_train,y=Y_train)
    
    #### Prediccion 
    Y_pred = forest2.predict(X_test)

    #### MSE de train 0.32294936345991776
    train_mse = mean_squared_error(Y_train,forest2.predict(X_train))

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
    plt.title("Trees: 100 Max Depth: {} Random Forest \n Training MSE: {:0.3f} Test MSE: {:0.3f}".format(m_depth,train_mse,test_mse))
    plt.savefig("data exploration/{}_train_vsby.png".format(station),bbox_inches='tight',dpi=300)


    #### Importancias Relativas
    importances = {"Regressor":X_train.columns,"Importance":forest2.feature_importances_}
    importances = pd.DataFrame(importances)
    importances = importances.sort_values(by='Importance',ascending=False)


    #### Grafico de Importancias
    plt.subplots(figsize=(12,5))
    plt.bar(importances['Regressor'],height=importances['Importance'])
    plt.title('Horizontal Visibility of {} Model \n Relative Importance of the Regressors'.format(station))
    plt.xticks(rotation=90,ha='right')
    plt.savefig("data exploration/{}_importances_vsby.png".format(station),bbox_inches='tight',dpi=300)

    #####Exportar Modelo
    filename = 'dash/{}_vsby.sav'.format(station)
    pickle.dump(forest, open(filename, 'wb'))

    del(forest,forest2)

    print("Model Training For Horizontal Visibility in {} Ended...".format(station))
    print("----------------------------------------------------------",end="\n\n\n")



#############################################
####  VERTICAL VISBILITY MODELS (skyl1) ####
#############################################

print("Model Training for Vertical Visibiliy Initiated",end="\n\n")

for station in stations:

    print("Training Models of {} Station".format(station),end="\n\n")
    
    ## Get Data
    print("Fetching {} Clean Data".format(station))
    query = "SELECT * FROM dataclean WHERE station = {}".format(station)
    df = pd.read_sql(query, engine.connect(), parse_dates=('valid',))
    df = df.sort_values(by=['day_hour'],ascending=True).reset_index(drop=True)
    print('Clean Data of {} Succesfully Fetched From AWS RDS'.format(station),end="\n\n")

    ## Save vsby and date_hour
    print("Preparing Data For Model Training")
    Y = df[['day_hour','skyl1']]

    ## Select only numeric variables
    numeric_cols = ['tmpf','dwpf','relh','drct','sknt','alti','vsby']
    df_num = df[numeric_cols]

    ## Lag Data
    lagged_lists = []

    for i in [1,2,4,6,8,10,12]:
        lag = df_num.shift(periods=1)
        lag.columns = [col+'_{}'.format(i) for col in lag.columns] 
        lagged_lists.append(lag)

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
    print("Data For Model Training Ready",end="\n\n")

    print("Selecting Train and Test Data",end="\n\n")
    train = df_fin[df_fin['day_hour']<='2019-10-23']
    test = df_fin[df_fin['day_hour']>'2019-10-23']

    ### Matriz X & y de entrenamiento
    X_train = train.drop(['day_hour','skyl1'],axis=1)
    Y_train = train['skyl1']

    ##### Matriz X & Y de test
    Y_test = test['skyl1']
    X_test = test.drop(['day_hour','skyl1'],axis=1)

    print("Starting Model Training",end="\n\n")

    #### Parametrizacion del Modelo y Grilla de Validacion Cruzada
    estim = RandomForestRegressor(n_estimators=100,criterion='mse',n_jobs=1,random_state=123)
    forest = GridSearchCV(estimator=estim,cv=5,param_grid={'max_depth':np.arange(11)+1},verbose=2)

    ##### Entrenar el modelo sobre la grilla de validacion
    forest.fit(X=X_train,y=Y_train)

    ##### Re calibración del modelo con los mejores parámetros
    m_depth = forest.best_params['max_depth']
    forest2 = RandomForestRegressor(n_estimators=100,criterion='mse',
                                    n_jobs=1,random_state=123,
                                    max_depth=m_depth)
    forest2.fit(X=X_train,y=Y_train)

    #### Prediccion 
    Y_pred = forest2.predict(X_test)

    #### MSE de train 0.32294936345991776
    train_mse = mean_squared_error(Y_train,forest2.predict(X_train))

    #### MSE de test 0.3553254070124522
    test_mse = mean_squared_error(Y_test,Y_pred)


    #### Gráfico Predicción
    plt.subplots(figsize=(12,6))
    plt.plot(test['day_hour'],Y_test,label="Real")
    plt.plot(test['day_hour'],Y_pred,label="Predicted")
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Vertical Visibility (skyl1)')
    plt.xticks(rotation=30)
    plt.title("Trees: 100 Max Depth: {} Random Forest \n Training MSE: {:0.3f} Test MSE: {:0.3f}".format(m_depth,train_mse,test_mse))
    plt.savefig("data exploration/{}_train_skyl1.png".format(station),bbox_inches='tight',dpi=300)


    #### Importancias Relativas
    importances = {"Regressor":X_train.columns,"Importance":forest2.feature_importances_}
    importances = pd.DataFrame(importances)
    importances = importances.sort_values(by='Importance',ascending=False)


    #### Grafico de Importancias
    plt.subplots(figsize=(12,5))
    plt.bar(importances['Regressor'],height=importances['Importance'])
    plt.title('Vertical Visibility of {} Model \n Relative Importance of the Regressors'.format(station))
    plt.xticks(rotation=90,ha='right')
    plt.savefig("data exploration/{}_importances_skyl1.png".format(station),bbox_inches='tight',dpi=300)

    #####Exportar Modelo
    filename = 'dash/{}_vsby.sav'.format(station)
    pickle.dump(forest, open(filename, 'wb'))

    del(forest,forest2)

    print("Model Training For Vertical Visibility in {} Ended...".format(station))
    print("----------------------------------------------------------",end="\n\n\n")

































