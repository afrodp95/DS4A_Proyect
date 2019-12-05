import pandas as pd 
import numpy as np
import os
import re
from sqlalchemy import create_engine
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import datetime
import dash_table
import pickle
from sklearn.ensemble import RandomForestRegressor

#############
### Get Avaliable Models Index

files = np.array(os.listdir("dash/"))
files = [file for file in files if '.sav' in file]

models = {'station':[x[0:4] for x in files],
          'variable':[x[5:9] for x in files],
          'model':files}

models = pd.DataFrame(models).sort_values(by='station').reset_index(drop=True)          


#def get_station_data(df,station,variable):


def prepare_pred_data(df,station,variable):
    print('Preparing Data For {} {} Prediction'.format(station,variable),end="\n\n")
    df_air = df[df['station']==station].sort_values(by=['day_hour'],ascending=True).reset_index(drop=True).drop_duplicates(subset='day_hour')
    ## Save date_hour
    print("Preparing Model Inputs")
    date = df_air[['day_hour']]
    ## Select only numeric variables
    if variable=='vsby':
        numeric_cols = ['tmpf','dwpf','relh','drct','sknt','alti','skyl1']
        df_num = df_air[numeric_cols]
    else:
        numeric_cols = ['tmpf','dwpf','relh','drct','sknt','alti','vsby']
        df_num = df_air[numeric_cols]


    print('Preparing Data For {} {} Prediction'.format(station,variable),end="\n\n")
    df_air = df[df['station']==station].sort_values(by=['day_hour'],ascending=True).reset_index(drop=True).drop_duplicates(subset='day_hour')
    
    ## Save date_hour
    print("Preparing Model Inputs")
    date = df_air[['day_hour']]
    ## Select only numeric variables
    if variable=='vsby':
        numeric_cols = ['tmpf','dwpf','relh','drct','sknt','alti','skyl1']
        df_num = df_air[numeric_cols]
    else:
        numeric_cols = ['tmpf','dwpf','relh','drct','sknt','alti','vsby']
        df_num = df_air[numeric_cols]
    
    ## Lag Data
    lagged_lists = []
    for i in [0,1,2,3,4,5,6,18,19]:
        lag = df_num.shift(periods=i)
        lag.columns = [col+'_{}'.format(i+6) for col in lag.columns] 
        lagged_lists.append(lag)

    df_fin = pd.concat([date]+lagged_lists,axis=1)
    df_fin = df_fin.sort_values(by=['day_hour'],ascending=False).reset_index(drop=True)
    df_fin.dropna(inplace=True)
    df_fin = df_fin.head(6)

    ## Crear Fechas de Prediccion
    max_date = df_air['day_hour'].max()
    start_date = max_date+datetime.timedelta(hours=1)
    end_date = max_date+datetime.timedelta(hours=6)
    predict_dates = pd.date_range(start=str(start_date),end=str(end_date),freq='H').to_list()
    df_fin['day_hour']=predict_dates
    
    ## Extraer Año Mes Dia Hora de la fecha
    df_fin['year']=df_fin['day_hour'].dt.year
    df_fin['month']=df_fin['day_hour'].dt.month
    df_fin['day']=df_fin['day_hour'].dt.day 
    df_fin['hour']=df_fin['day_hour'].dt.hour

    ## Recategorizar año
    years = np.linspace(2016,2030,num=13,dtype='int')
    years_dict = {}
    for i, year in enumerate(years):
        years_dict[year]=i+1

    df_fin['year']=df_fin['year'].map(years_dict)
    print("Data For Model Prediction Ready",end="\n\n")

    return df_fin


def get_model(station,variable):
    model_name = models.loc[(models['station']==station) & (models['variable']==variable),'model'].values[0]
    rf = pickle.load(open('dash/'+model_name, 'rb'))
    return rf


def create_plot_data(df,station,variable):
    print('Preparing Data For {} {} Plotting'.format(station,variable),end="\n\n")
    df_air = df[df['station']==station].sort_values(by=['day_hour'],ascending=True).reset_index(drop=True).drop_duplicates(subset='day_hour')
    ## Seleccionar Ultimas 48 horas de info
    end_date = df_air['day_hour'].max()
    start_date = end_date-datetime.timedelta(hours=40)
    date_range = pd.date_range(start=str(start_date),end=str(end_date),freq='H').to_list()
    df_air = df_air.loc[df_air['day_hour'].isin(date_range),['day_hour',variable]]
    df_air['type']='Current'
    ## Preparar datos de prediccion
    df_to_pred = prepare_pred_data(df=df,station=station,variable=variable)
    ## Cargar Modelo
    rf = get_model(station=station,variable=variable)
    ## Hacer Prediccion
    if variable == 'vsby':
        df_to_pred[variable] = rf.predict(X=df_to_pred.drop('day_hour',axis=1))
    else:
        df_to_pred[variable] = np.exp(rf.predict(X=df_to_pred.drop('day_hour',axis=1)))
    df_to_pred = df_to_pred[['day_hour',variable]]
    df_to_pred['type']='Prediction'
    ## Concatenar Data Sets
    df_out = pd.concat([df_air,df_to_pred])
    
    return df_out
