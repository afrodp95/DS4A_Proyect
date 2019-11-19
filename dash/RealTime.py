#####################
# Importar librerías
from __future__ import print_function
import json
import time
import datetime as datetime
import pandas as pd
from datetime import timedelta
from sqlalchemy import create_engine

#en caso de utlizar python2 en lugar de python3
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

#maximo de intentos en caso de falla del internet
MAX_ATTEMPTS = 6

#Llamando al servicio de Iowa State University
SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
#La fecha iniciar es el dia de hoy a las 00:00
start=datetime.datetime.now()
#La fecha final es manana a las 00:00

end=start + timedelta(days=1)
startts = datetime.datetime(start.year,start.month,start.day)
endts = datetime.datetime(end.year,end.month,end.day)
#startts = datetime.datetime(2019,11,1)
#endts = datetime.datetime(2019,11,17)

##################################
#funcion para descargar la data
def download_data(uri):
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            data = urlopen(uri, timeout=300).read().decode("utf-8")
            if data is not None and not data.startswith("ERROR"):
                return data
        except Exception as exp:
            print("download_data(%s) failed with %s" % (uri, exp))
            time.sleep(5)
        attempt += 1
    print("Exhausted attempts to download, returning empty data")
    return ""
####################################

#creacion de los parametros para el API
service = SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=no&"
service += startts.strftime("year1=%Y&month1=%m&day1=%d&")
service += endts.strftime("year2=%Y&month2=%m&day2=%d&")
#definicion de network y estaciones de donde descargaremos
networks = ["CO__ASOS"]
#stations=["SKAR","SKQL","SKBO","SKBG","SKCL","SKCC","SKCG","SKPE","SKSP","SKSM","SKMR"]
stations=["SKAR"]

#iniciar la descarga por cada estacion
df = pd.DataFrame()
for station in stations:
    uri = "%s&station=%s" % (service, station)
    print("Downloading: %s" % (station,))
    data = download_data(uri)
    data = pd.DataFrame([x.split(',') for x in data.split('\n')]).iloc[5:].reset_index(drop=True)
    data.columns = data.iloc[0]
    data = data.drop(data.index[0])
    df = df.append(data)

#eliminacion de campos no usados
del df['metar']
del df['wxcodes']
del df['skyc2']
del df['skyc3']
del df['skyc4']
del df['skyl2']
del df['skyl3']
del df['skyl4']
del df['ice_accretion_1hr']
del df['ice_accretion_3hr']
del df['ice_accretion_5hr']
del df['peak_wind_gust']
del df['peak_wind_drct']
del df['peak_wind_time']
print(df.shape)

#creacion del motor de base de datos en postgres
engine = create_engine('postgresql://ds4a_18:ds4a2019@ds4a18.cmlpaj0d1yqv.us-east-2.rds.amazonaws.com:5432/Airports_ds4a')
#subir DataFrame a la base de datos
df.to_sql(name='DataRaw', con=engine, if_exists = 'append', index=False, chunksize=10000)
#eliminacion de duplicados con el query. Esta como una funcion en la base de datos
engine.execute('select delete_duplicates()')
#eliminacion de datos nulos (donde VALID es nulo)
engine.execute('delete from DataRaw where valid is null')
print('FINISH')

