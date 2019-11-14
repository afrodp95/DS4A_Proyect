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
### Import Raw Data

from sqlalchemy import create_engine

engine = create_engine('postgresql://ds4a_18:ds4a2019@ds4a18.cmlpaj0d1yqv.us-east-2.rds.amazonaws.com:5432/Airports_ds4a')
df = pd.read_sql("SELECT * from reads2_raw", engine.connect(), parse_dates=('valid',))
df = df.dropna(subset=['valid'],axis=0)
df['DateTime'] = df['valid'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
df.replace(['M',None],np.nan,inplace=True)

print('Data Succesfully Fetched From AWS RDS',end="\n\n")

#####################
#### Select Columns With Na % lower than 22% (previously selected)

print('Selecting Columns')

cols = ['id','station','DateTime', 'valid', 'lon', 'lat', 'tmpf', 'dwpf', 'relh', 'drct',
       'sknt', 'p01i', 'alti', 'vsby', 'skyc1', 'skyl1', 'feel']

df = df[cols]

###################
#### Convert to Numeric (in sql are all text)
numeric_cols = ['lon','lat','tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'alti', 'vsby', 'skyl1', 'feel']

for col in numeric_cols:
    df[col] = df[col].astype(float)


#################
### Find  Missing Date-Hour Intervals

print('Finding Missing Data Intervals')

df['day_hour'] = df['DateTime'].apply(lambda x: datetime.datetime.strftime(x, "%Y-%m-%d %H"))
df = df.groupby(['station', 'day_hour','lon','lat']).mean().reset_index() ## Se pierde p01i skyc1 porque son categóricas

#### Crear las horas wue no tienen observaciones
df['day_hour'] = pd.to_datetime(df['day_hour'])
minimo = df.day_hour.min()
maximo = df.day_hour.max()

# Acotar la longitud y latitud para tener siempre las mismas coordenadas
df['lon'] = df['lon'].round(4)
df['lat'] = df['lat'].round(4)

#aux = df_clean[df_clean['station'].isin(inter_airport)]
lista = df[['station','lon','lat']].drop_duplicates().values.tolist()

#Crear las fechas
df2 = pd.DataFrame()
for j,i in enumerate(lista):
    x = pd.DataFrame(pd.date_range(start=minimo, end=maximo, freq = 'H'), columns = ['day_hour'])
    x['station'] = i[0]
    x['lon'] = i[1]
    x['lat'] = i[2]
    if j == 0:
        df2 = x
    else:
        df2 = df2.append(x)

print(len(df2))


# Pegarle la informacin
df2 = pd.merge(df,df2, left_on =['station','day_hour','lat','lon'], right_on=['station','day_hour','lat','lon'], how = 'left')
df = df2.copy()
del(df2)

########################################
##### Usar Imputacion Seleccionada #####
######## En Toda la Base ###############
########################################

print('Dealing With Missing Data')


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#### Seleccionar solo las numericas
numeric_cols = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'alti', 'vsby', 'skyl1', 'feel']

### Iniciar el imputador de datos multi
m_imputer = IterativeImputer(random_state=1234,max_iter=10)

### Se debe realizar una imputacion por cada aeropuerto
airports = df['station'].unique()

### Iniciar imputacion iterativa
df_list = []

for airport in airports:
    df2 = df[df['station']==airport].reset_index(drop=True)
    df2_num = df[numeric_cols]
    df2_imp = m_imputer.fit_transform(df2_num)
    df2_imp = pd.DataFrame(df2_imp,columns=numeric_cols)
    df2_full = pd.concat([df[['station','day_hour','lon','lat']].copy(),df2_imp],axis=1)
    print('Data for {} imputed'.format(airport))
    df_list.append(df2_full)

df = pd.concat(df_list,axis=0,ignore_index=True)

#############################
#### Exportar Solo Bogotá V1

print('Selecting Bogotá V1')

df = df[df['station']=='SKBO']

#############################
#### Exportar Solo Ultimos 2 Meses de Info

df['day_hour'] = pd.to_datetime(df['day_hour'])

df = df[df['day_hour']>'2019-09']

########################################
##### Exportar A csv para consumirla #####

print('Writing to AWS')

df.to_csv("airports_clean.csv.gz",encoding='UTF-8')

print('End... Starting App')