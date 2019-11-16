#####################
# Importar librerías
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
import datetime
import time
from sqlalchemy import create_engine

#####################
### Fecth Raw data from AWS DB

engine = create_engine('postgresql://ds4a_18:ds4a2019@ds4a18.cmlpaj0d1yqv.us-east-2.rds.amazonaws.com:5432/Airports_ds4a')
df = pd.read_sql("SELECT * from reads2_raw", engine.connect(), parse_dates=('valid',))
df = df.dropna(subset=['valid'],axis=0)
df['DateTime'] = df['valid'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
df.replace(['M',None],np.nan,inplace=True)

print('Data Succesfully Fetched From AWS RDS')


#####################
#### Select Columns With Na % lower than 22% (previously selected)

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

#### Crear las horas que no tienen observaciones
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
df2 = pd.merge(df2,df, left_on =['station','day_hour','lat','lon'], right_on=['station','day_hour','lat','lon'], how = 'left')
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
df_empty = pd.DataFrame()
for airport in airports:
    df2 = df[df['station']==airport].reset_index(drop=True)
    hora = df2.day_hour
    lati = df2.lat
    longi = df2.lon
    df2_num = df2[numeric_cols]
    df2_imp = m_imputer.fit_transform(df2_num)
    df2_imp = pd.DataFrame(df2_imp,columns=numeric_cols)
    df2_imp['day_hour'] = hora
    df2_imp['station'] = airport
    df2_imp['lat'] = lati
    df2_imp['lon'] = longi
    df2_full = pd.concat([df_empty,df2_imp],axis=1)
    print('Data for {} imputed'.format(airport))
    df_list.append(df2_full)

df = pd.concat(df_list,axis=0,ignore_index=True)
df.to_sql(name='reads_clean', con=engine, if_exists = 'append', index=False, chunksize=10000)