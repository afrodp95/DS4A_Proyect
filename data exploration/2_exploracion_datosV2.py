import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import selenium
import os
import time
import sklearn.metrics       as Metrics
import datetime
from datetime import timedelta
from   datetime import datetime
import matplotlib.dates as mdates
from   sklearn               import preprocessing

#from   collections           import Counter
#from   math                  import exp
#from   sklearn.linear_model  import LinearRegression as LinReg
from   sklearn.metrics       import mean_absolute_error
from   sklearn.metrics       import median_absolute_error
from   sklearn.metrics       import r2_score


##########################
##### Carga de Datos #####
##########################


# from sqlalchemy import create_engine
# engine = create_engine('postgresql://ds4a_18:ds4a2019@ds4a18.cmlpaj0d1yqv.us-east-2.rds.amazonaws.com:5432/Airports_ds4a')
# #df.to_sql(name='reads_raw', con=engine, if_exists = 'replace', index=False)
# data = pd.read_sql('SELECT * FROM reads_raw', engine)

cwd = os.getcwd()
df = pd.read_csv(cwd+'/data_aeropuertos_col.csv.gz',sep=',',na_values='M',low_memory=False)
ini = time.time()
df['DateTime'] = df['valid'].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M"))
fin = time.time()
print(fin-ini) # 8.79
print('---------------')
# print(df.shape)
df.tail(3)

print('-----------------------------------------------------------------------------------')
print(df.shape)
print('-----------------------------------------------------------------------------------')
print(df.columns)


#########################################
#### Exploracion de Datos Faltantes #####
#########################################

###################
## Grafico de Vacios

fig, ax = plt.subplots(figsize=(15,6))
sns.heatmap(~df.drop('station',axis=1).isnull(), cbar=False)
plt.xticks(rotation=90)
plt.title('Missing Values Plot')
#plt.show()

## Grafico de Vacios Segun numero de Vacios por columna
### Obtener serie de numero de vacios por columna
nas = ~df.drop('station',axis=1).isnull()
nas = nas.sum().sort_values(ascending=False)
# nas

### Re ordenar el df de vacios en funcion de el total de vacios por columna
nas_df = ~df.drop('station',axis=1).isnull()
nas_df = nas_df[nas.index]
nas_df.head()

### Re ordenar el df de vacios en funcion de el total de vacios por fila
nas_df['n_null']=nas_df.sum(axis=1)
nas_df = nas_df.sort_values(by='n_null')

### Graficar
fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(nas_df.drop('n_null',axis=1),cbar=False)
plt.xticks(rotation=90)
plt.title('Missing Values Plot')
plt.show()

nas = df.isnull().sum()*100.0/df.shape[0]
nas = nas.sort_values()

###################
## Proporcion de Datos Faltantes Por Variable

nas = df.isnull().sum()*100.0/df.shape[0]
nas = nas.sort_values()

### Barras de Porcentaje de Vacios
fig, ax = plt.subplots(figsize=(15,6))
sns.barplot(x=nas,y=nas.index) # ,palette='RdBu'
plt.show()


###############################
#### Limpieza de Datos ########
###############################

### Calcular porcentaje de vacios por columna
cols = df.isnull().sum()*100.0/df.shape[0]
cols = cols[cols<=22]
cols.index

#### Seleccionar todas las columnas
df_clean = df.copy();
df_clean = df_clean[cols.index].copy();

#### La columna metar contiene el reporte sin procesar. Eliminarla.
df_clean.drop('metar',axis=1,inplace=True)
print(df_clean.columns); 
print(df_clean.shape)
print('-----------')

########
### Seleccionar solo aeropuertos internacionales

# Seleccionar solamente los aeropuertos internacionales
df_clean.shape # (479849, 13)
inter_airport=['SKAR','SKBQ','SKBO','SKBG','SKCL','SKCC','SKCG','SKRG','SKPE','SKSP','SKSM','SKMR'] #12
df_clean = df_clean[df_clean['station'].isin(inter_airport)]
print(df_clean.shape) # (254530, 13)

##########
### Encontrar los intervalos de fecha hora faltantes

df_clean['day_hour'] = df_clean['DateTime'].apply(lambda x: str(x)[:13])
df_clean = df_clean.groupby(['station', 'day_hour']).mean().reset_index()
print(df.shape)
print(df_clean.shape)     # (493596, 32)
df_clean.head(5)          # (479849, 21)
df_clean.columns

#### Crear las horas wue no tienen observaciones
df_clean.day_hour = pd.to_datetime(df_clean.day_hour)
minimo = df_clean.day_hour.min()
maximo = df_clean.day_hour.max()
print(minimo)
print(maximo)

# Acotar la longitud y latitud para tener siempre las mismas coordenadas
df_clean['lon'] = df_clean['lon'].round(4)
df_clean['lat'] = df_clean['lat'].round(4)
aux = df_clean[df_clean['station'].isin(inter_airport)]
lista = aux[['station','lon','lat']].drop_duplicates().values.tolist()

lista

#Crear las fechas
df_clean2 = pd.DataFrame()
for j,i in enumerate(lista):
    x = pd.DataFrame(pd.date_range(start=minimo, end=maximo, freq = 'H'), columns = ['day_hour'])
    x['station'] = i[0]
    x['lon'] = i[1]
    x['lat'] = i[2]
    if j == 0:
        df_clean2 = x
    else:
        df_clean2 = df_clean2.append(x)

print(len(df_clean2))

# Pegarle la informacin
df_clean2 = df_clean.merge(df_clean2, left_on =['station','day_hour','lat','lon'], right_on=['station','day_hour','lat','lon'], how = 'right')

# Ordenar los datos
df_clean2.sort_values(['station','day_hour'],inplace=True)
df_clean2.reset_index(drop=True,inplace=True)

## Grafico de Vacios con la nueva distribucion de datos
## se ve un claro patrón en la falta de información
fig, ax = plt.subplots(figsize=(15,6))
sns.heatmap(~df_clean2.drop('station',axis=1).isnull(), cbar=False)
plt.xticks(rotation=90)
plt.title('Missing Values Plot')
plt.show()


#############################
#### Comparar Metodos de imputacion

##############
### Seleccionar una MUESTRA de los datos

df_sample = df_clean2[(df_clean2['day_hour']<='2017-01-12') & (df_clean2['station']=='SKBO')].reset_index(drop=True)

###############
## Lineal

numeric_cols = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'alti', 'vsby', 'skyl1', 'feel']
df_lin = df_sample.copy()

for col in numeric_cols:
    df_lin[col].interpolate(method='linear', inplace = True)


plt.subplots(figsize=(12,12),ncols=3,sharex=True)
for i,col in enumerate(numeric_cols):
    plt.subplot(3,3,i+1)
    plt.plot(df_sample['day_hour'],df_sample[col],label='{} original'.format(col),color='navy')
    plt.plot(df_lin['day_hour'],df_lin[col],label='{} lin imp'.format(col),linestyle='dashed',color='red',alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('')
    plt.legend()
    if i in [6,7,8]:
        plt.xticks(rotation=30,ha='right')
    else:
        plt.xticks([''])    
    plt.title('{} Missing vs Lineal interpolation'.format(col))
      
plt.subplots_adjust(wspace=0.4,hspace=0.4)
plt.show()


###############
## MICE

from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#### Seleccionar solo las numericas
df_mice = df_sample.copy()
numeric_cols = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'alti', 'vsby', 'skyl1', 'feel']
df_numeric = df_mice[numeric_cols]

### Iniciar el imputador de datos multi
m_imputer = IterativeImputer(random_state=1234,max_iter=10)
df_imputed = m_imputer.fit_transform(df_numeric)

### Convertir el numpy array en df
df_imputed = pd.DataFrame(df_imputed,columns=numeric_cols)

### Pegarle la fecha, aeropuerto, longitud y latitud al df imputado
df_mice = pd.concat([df_mice[['station','day_hour','lon','lat']].copy(),df_imputed],axis=1)
df_mice


### Grafico
plt.subplots(figsize=(12,12),ncols=3,sharex=True)
for i,col in enumerate(numeric_cols):
    plt.subplot(3,3,i+1)
    plt.plot(df_sample['day_hour'],df_sample[col],label='{} original'.format(col),color='navy')
    plt.plot(df_mice['day_hour'],df_mice[col],label='{} MICE'.format(col),linestyle='dashed',color='red',alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('')
    plt.legend()
    if i in [6,7,8]:
        plt.xticks(rotation=30,ha='right')
    else:
        plt.xticks([''])
    plt.title('{} Missing vs MICE'.format(col))

plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.show()


########################################
##### Usar Imputacion Seleccionada #####
######## En Toda la Base ###############
########################################

from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


#### Seleccionar solo las numericas
numeric_cols = ['tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'alti', 'vsby', 'skyl1', 'feel']

### Iniciar el imputador de datos multi
m_imputer = IterativeImputer(random_state=1234,max_iter=10)

### Se debe realizar una imputacion por cada aeropuerto
airports = df_clean2['station'].unique()

### Iniciar imputacion iterativa
df_list = []

for airport in airports:
    df = df_clean2[df_clean2['station']==airport].reset_index(drop=True)
    df_num = df[numeric_cols]
    df_imp = m_imputer.fit_transform(df_num)
    df_imp = pd.DataFrame(df_imp,columns=numeric_cols)
    df_full = pd.concat([df[['station','day_hour','lon','lat']].copy(),df_imp],axis=1)
    print('Data for {} imputed'.format(airport))
    df_list.append(df_full)

len(df_list)

df_clean3 = pd.concat(df_list,axis=0,ignore_index=True)
    

##########################################
##### Analisis Exploratorio de Datos #####
##########################################

df_clean3.head()
df_clean3.isna().sum()
df_clean3.tail()
df_clean2.tail()


df_clean2.tail(13)

df_clean3.tail(13)


"""
Data cleaning proccess (half page)
Markup at the front end (detailed)
Infraestructure (pull something to the database)
"""

