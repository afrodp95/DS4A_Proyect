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


##############
### Import Clean Data


engine = create_engine('postgresql://ds4a_18:ds4a2019@ds4a18.cmlpaj0d1yqv.us-east-2.rds.amazonaws.com:5432/Airports_ds4a')
query = "SELECT * FROM dataclean"
df = pd.read_sql(query, engine.connect(), parse_dates=('day_hour',))

# df = pd.read_csv("dash/airports_clean.csv.gz",encoding='UTF-8',parse_dates=['day_hour',],index_col=0)
# df.columns


#############
### Get Avaliable Models Index

files = np.array(os.listdir("dash/"))
files = [file for file in files if '.sav' in file]

models = {'station':[x[0:4] for x in files],
          'variable':[x[5:9] for x in files],
          'model':files}

models = pd.DataFrame(models).sort_values(by='station').reset_index(drop=True)          

################
#### Unificar Nombres de Aeropuertos


air_names = {'SKAR':'Aeropuerto Internacional el Edén (Armenia - Quindío)',
             'SKBG':'Aeropuerto Internacional Palonegro (Bucaramanga - Santander)',
             'SKBO':'Aeropuerto Internacional El Dorado (Bogotá - Bogotá D.C.)', 
             'SKCC':'Aeropuerto Internacional Camilo Daza (Cúcuta - Norte de Santander)', 
             'SKCG':'Aeropuerto Internacional Rafael Núñez (Cartagena - Bolívar)', 
             'SKCL':'Aeropuerto Internacional Alfonso Bonilla Aragón (Cali - Valle)', 
             'SKMR':'Aeropuerto Internacional Los Garzones (Montería - Córdoba)', 
             'SKPE':'Aeropuerto Internacional Matecaña (Pereira - Risaralda)',
             'SKSM': 'Aeropuerto Internacional Simón Bolivar (Santa Marta - Magdalena)'}

df['station'] = df['station'].map(air_names)


################
#### Lista de diccionarios de Nombre De Variables Para el Dropdown

vars_list = [
    {'label':'Airport','value':'station'},
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
#### Lista de diccionarios de Nombre De Variables Para el Dropdown

vars_list_dt = [
    {'name':'Airport','id':'station'},
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
                                    className='three columns card',
                                    children=[
                                        html.H6("Select Airport"),
                                        dcc.Dropdown(id='select-airport',
                                                     options=[{'label':val,'value':val} for val in df['station'].unique()],
                                                     value='Aeropuerto Internacional El Dorado (Bogotá - Bogotá D.C.)'        
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
                )
            ]
        )
                         
])


#########
### Call Back for the data table

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


#########
### Call Back for the models prediction

# #def prepare_data(station,variable):


# df['day_hour'].max().hour
# df.columns


# station = 'SKBO'
# variable = 'vsby'


# print('Preparing Data For {} {} Prediction'.format(station,variable),end="\n\n")
# df_air = df[df['station']==station]
# df_air = df_air.sort_values(by=['day_hour'],ascending=True).reset_index(drop=True)
# df_air = df_air.drop_duplicates(subset='day_hour')

# max_date = df['day_hour'].max()
# start_date = max_date+datetime.timedelta(hours=1)
# end_date = max_date+datetime.timedelta(hours=6)
# predict_dates = pd.date_range(start=str(start_date),end=str(end_date),freq='H').to_list()
# len(predict_dates)

# ## Save vsby and date_hour
# print("Preparing Model Inputs")
# Y = df_air[['day_hour']]

# ## Select only numeric variables
# if variable=='vsby':
#     numeric_cols = ['tmpf','dwpf','relh','drct','sknt','alti','skyl1']
#     df_num = df[numeric_cols]
# else:
#     numeric_cols = ['tmpf','dwpf','relh','drct','sknt','alti','vsby']
#     df_num = df[numeric_cols]

# ## Lag Data
# lagged_lists = []

# for i in [0,1,2,3,4,5,6]:
#     lag = df_num.shift(periods=i)
#     lag.columns = [col+'_{}'.format(i+6) for col in lag.columns] 
#     lagged_lists.append(lag)

# df_fin = pd.concat([Y]+lagged_lists,axis=1)
# df_fin = df_fin.sort_values(by=['day_hour'],ascending=False).reset_index(drop=True)
# df_fin.dropna(inplace=True)
# df_fin = df_fin.head(6)
# df_fin['day_hour']=predict_dates


# ## Extraer Año Mes Dia Hora de la fecha
# df_fin['year']=df_fin['day_hour'].dt.year
# df_fin['month']=df_fin['day_hour'].dt.month
# df_fin['day']=df_fin['day_hour'].dt.day 
# df_fin['hour']=df_fin['day_hour'].dt.hour

# ## Recategorizar año
# years = np.linspace(2016,2030,num=13,dtype='int')
# years_dict = {}
# for i,year in enumerate(years):
#     years_dict[year]=i+1

# df_fin['year']=df_fin['year'].map(years_dict)
# print("Data For Model Prediction Ready",end="\n\n")



# min_date = str(max_date-datetime.timedelta(hours=6))

###########
### Run App

if __name__ == "__main__":
    #app.run_server(debug=True,host="0.0.0.0")
    app.run_server(debug=True)
