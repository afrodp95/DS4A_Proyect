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
#import apputils as apputils


##############
### Import Clean Data

stations = ['SKBO','SKBG','SKCL','SKCC','SKCG','SKPE','SKSP','SKSM','SKMR']
engine = create_engine('postgresql://ds4a_18:ds4a2019@ds4a18.cmlpaj0d1yqv.us-east-2.rds.amazonaws.com:5432/Airports_ds4a')
query = "SELECT * FROM dataclean"
df = pd.read_sql(query, engine.connect(), parse_dates=('day_hour',))
df = df[df['station'].isin(stations)]

#df = pd.read_csv("dash/airports_clean.csv.gz",encoding='UTF-8',parse_dates=['day_hour',],index_col=0)
# df.columns


################
#### Unificar Nombres de Aeropuertos


air_names = {#'SKAR':'Aeropuerto Internacional el Edén (Armenia - Quindío)',
             'SKBG':'Aeropuerto Internacional Palonegro (Bucaramanga - Santander)',
             'SKBO':'Aeropuerto Internacional El Dorado (Bogotá - Bogotá D.C.)', 
             'SKCC':'Aeropuerto Internacional Camilo Daza (Cúcuta - Norte de Santander)', 
             'SKCG':'Aeropuerto Internacional Rafael Núñez (Cartagena - Bolívar)', 
             'SKCL':'Aeropuerto Internacional Alfonso Bonilla Aragón (Cali - Valle)', 
             'SKMR':'Aeropuerto Internacional Los Garzones (Montería - Córdoba)', 
             'SKPE':'Aeropuerto Internacional Matecaña (Pereira - Risaralda)',
             'SKSM': 'Aeropuerto Internacional Simón Bolivar (Santa Marta - Magdalena)',
             'SKSP':'Aeropuerto Internacional Gustavo Rojas Pinilla (San Andrés - Colombia)'}

df['station_name'] = df['station'].map(air_names)


################
#### Lista de diccionarios de Nombre De Variables Para el Dropdown

vars_list = [
    {'label':'Airport','value':'station_name'},
    {'label':'Date','value':'day_hour'},
    #{'label':'Longitude','value':'lon'},
    #{'label':'Latitude','value':'lat'},
    {'label':'Air Temperature (F)','value':'tmpf'},
    {'label':'Dew Point (F)','value':'dwpf'},
    {'label':'Relative Humidity %','value':'relh'},
    {'label':'Wind Direction (Degrees From North)','value':'drct'},
    {'label':'Wind Speed (Knots)','value':'sknt'},
    {'label':'Pressure in Altimeter','value':'alti'},
    {'label':'Horizontal Visibility','value':'vsby'},
    {'label':'Vertical Visibility','value':'skyl1'},
    {'label':'Apparent Temperature (Wind Chill or Heat Index in F)','value':'feel'},
    
] 


################
#### Lista de diccionarios de Nombre De Variables Para el la Tabla 

vars_list_dt = [
    {'name':'Airport','id':'station_name'},
    {'name':'Date','id':'day_hour'},
    #{'name':'Longitude','id':'lon'},
    #{'name':'Latitude','id':'lat'},
    {'name':'Air Temperature (F)','id':'tmpf'},
    {'name':'Dew Point (F)','id':'dwpf'},
    {'name':'Relative Humidity %','id':'relh'},
    {'name':'Wind Direction (Degrees From North)','id':'drct'},
    {'name':'Wind Speed (Knots)','id':'sknt'},
    {'name':'Pressure in Altimeter','id':'alti'},
    {'name':'Horizontal Visibility','id':'vsby'},
    {'name':'Vertical Visibility','id':'skyl1'},
    {'name':'Apparent Temperature (Wind Chill or Heat Index in F)','id':'feel'},
    
] 



################
### Initialize app

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/uditagarwal/pen/oNvwKNP.css', 'https://codepen.io/uditagarwal/pen/YzKbqyV.css'])


################
### Set Up UI

app.layout = html.Div(children=[
        html.Div(
            className='study-browser-banner row',
            children=[
                html.H2(children="Airports Dash 0.1", className='h2-title'),
            ]
        ),
        html.Div(
            className='row-app-body',
            children=[
                html.Div(
                    className='twelve columns card',
                    children=[
                        html.Div(
                            className='padding row',
                            children=[
                                html.Div(
                                    className='five columns card',
                                    children=[
                                        html.H6("Select Airport"),
                                        dcc.Dropdown(id='select-airport',
                                                     options=[{'label':label,'value':val} for label, val in zip(df['station_name'].unique(),df['station'].unique())],
                                                     value='SKBO'        
                                        )
                                    ]
                                ),
                                html.Div(
                                    className='seven columns card',
                                    children=[
                                        html.H6("Select Columns"),
                                        dcc.Dropdown(id='select-columns',
                                                     options=vars_list,
                                                     multi=True,
                                                     value=df.columns
                                        )
                                    ]

                                )



                            ]
                        )
                    ]
                ),
                html.Div(
                    id = 'table-div',
                    className='twelve columns card',
                    children=[
                        dash_table.DataTable(
                            id='table',
                            columns=vars_list_dt,
                            data=df.to_dict('records'),
                            style_cell={'width': '50px'},
                            style_table={
                                    'maxHeight': '450px',
                                    'overflowY': 'scroll'
                                }
                        )
                    ]
                ),
                html.Div(
                    id = 'plots-div',
                    className='twelve columns card',
                    children=[
                        html.Div(
                            className="padding row",
                            children = [
                                html.Div(
                                    className="six columns card",
                                    children=[
                                        dcc.Graph(id="horizontal-vis-plot") 
                                    ]
                                ),
                                html.Div(
                                    className="six columns card",
                                    children=[
                                        dcc.Graph(id="vertical-vis-plot")
                                    ]
                                )
                            ] 
                        )
                    ]
                )
            ]
        )
                         
])




#########
### Call Back for the data table

#a = apputils.prepare_pred_data(df=df,station='SKBO',variable='vsby')
#b = apputils.get_model(station='SKBO',variable='vsby')
#c = apputils.create_plot_data(df=df,station='SKBG',variable='vsby')
#c


@app.callback(
    Output('table-div', 'children'),
    [ 
    Input('select-airport', 'value'),
    Input('select-columns', 'value')
    ]
)

def update_table(airport,columns):
    vars_list_dt2 = [a for a in vars_list_dt if a['id'] in columns]
    dff = df.copy()
    if airport:
        dff = dff[dff['station']==airport]
    if columns:
        dff = dff[columns]

    children=[dash_table.DataTable(
                    id='table',
                    columns=vars_list_dt2,
                    data=dff.to_dict('records'),
                    style_cell={'width': '50px'},
                    style_table={
                        'maxHeight': '450px',
                        'overflowY': 'scroll'
                    })
                ]
    return children


#############
### Get Avaliable Models Index

files = np.array(os.listdir("dash/"))
files = [file for file in files if '.sav' in file]

models = {'station':[x[0:4] for x in files],
          'variable':[x[5:9] for x in files],
          'model':files}

models = pd.DataFrame(models).sort_values(by='station').reset_index(drop=True)          

###############
### Utils For Models Prediction

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
    start_date = end_date-datetime.timedelta(hours=30)
    date_range = pd.date_range(start=str(start_date),end=str(end_date),freq='H').to_list()
    df_air = df_air.loc[df_air['day_hour'].isin(date_range),['day_hour',variable]]
    df_air['type']='Current'
    ## Preparar datos de prediccion
    df_to_pred = prepare_pred_data(df=df,station=station,variable=variable)
    ## Cargar Modelo
    rf = get_model(station=station,variable=variable[0:4])
    ## Hacer Prediccion
    if variable == 'vsby':
        df_to_pred[variable] = rf.predict(X=df_to_pred.drop('day_hour',axis=1))
    else:
        df_to_pred[variable] = np.exp(rf.predict(X=df_to_pred.drop('day_hour',axis=1)))
    df_to_pred = df_to_pred[['day_hour',variable]]
    df_to_pred['type']='Prediction'
    ## Link 
    df_link = df_to_pred.head(1)
    df_link['type']='Current'
    ## Concatenar Data Sets
    df_out = pd.concat([df_air,df_link,df_to_pred],ignore_index=True)
    
    return df_out



#########
### Call Back for the models prediction

#################
### Horizontal Visibility

@app.callback(
    Output('horizontal-vis-plot', 'figure'),
    [ 
    Input('select-airport', 'value')
    ]
)


def update_hvis_plot(station):
    dff = create_plot_data(df=df,station=station,variable='vsby')
    plot_data = []
    for key, data in dff.groupby('type'):
        plot_data.append(
            go.Scatter(x=data['day_hour'],y=data['vsby'],name=key,mode='lines+markers')
        )
    layout = go.Layout(title="Horizontal Visibility Prediction",
                       yaxis={"title":"Horizontal Visibility"},
                       xaxis={"title":"Date"})   
    return {
        "data":plot_data,
        "layout": layout
    }    

#################
### Vertical Visibility

@app.callback(
    Output('vertical-vis-plot', 'figure'),
    [ 
    Input('select-airport', 'value')
    ]
)


def update_vvis_plot(station):
    dff = create_plot_data(df=df,station=station,variable='skyl1')
    plot_data = []
    for key, data in dff.groupby('type'):
        plot_data.append(
            go.Scatter(x=data['day_hour'],y=data['skyl1'],name=key,mode='lines+markers')
        )
    layout = go.Layout(title="Vertical Visibility Prediction",
                       yaxis={"title":"Vertical Visibility"},
                       xaxis={"title":"Date"})   
    return {
        "data":plot_data,
        "layout": layout
    }    


### Horizontal Visibility Prediction



# min_date = str(max_date-datetime.timedelta(hours=6))

###########
### Run App

if __name__ == "__main__":
    #app.run_server(debug=True,host="0.0.0.0")
    app.run_server(debug=True)

