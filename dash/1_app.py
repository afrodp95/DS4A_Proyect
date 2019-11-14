import pandas as pd 
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import datetime
import dash_table


##############
### Import Clean Data

df = pd.read_csv("airports_clean.csv.gz",encoding='UTF-8',parse_dates=['day_hour',],index_col=0)
df.columns

df.dtypes


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
    {'label':'Longitude','value':'lon'},
    {'label':'Latitude','value':'lat'},
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
    {'name':'Longitude','id':'lon'},
    {'name':'Latitude','id':'lat'},
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

df.head(2).to_dict()

#########
### Call Back for the 

###########
### Run App

if __name__ == "__main__":
    app.run_server(debug=True,host="0.0.0.0")
    #app.run_server(debug=True)
