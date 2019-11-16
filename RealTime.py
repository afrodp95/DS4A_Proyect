from __future__ import print_function
import json
import time
import datetime as datetime
import pandas as pd
from datetime import timedelta
from sqlalchemy import create_engine

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

MAX_ATTEMPTS = 6

SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
start=datetime.datetime.now()
end start + timedelta(days=1)
startts = datetime.datetime(start.year,start.month,start.day)
endts = datetime.datetime(end.year,end.month,end.day)

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


service = SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=no&"
service += startts.strftime("year1=%Y&month1=%m&day1=%d&")
service += endts.strftime("year2=%Y&month2=%m&day2=%d&")

networks = ["CO__ASOS"]

stations=["SKAR","SKQL","SKBO","SKBG","SKCL","SKCC","SKCG","SKPE","SKSP","SKSM","SKMR"]

df = pd.DataFrame()
for station in station:
    uri = "%s&station=%s" % (service, station)
    print("Downloading: %s" % (station,))
    data = download_data(uri)
    data = pd.DataFrame([x.split(',') for x in data.split('\n')]).iloc[5:].reset_index(drop=True)
    data.columns = data.iloc[0]
    data = data.drop(data.index[0])
    df = df.append(data)

del df['metar']
del df['wxcodes']
print(df.shape)

engine = create_engine('postgresql://ds4a_18:ds4a2019@ds4a18.cmlpaj0d1yqv.us-east-2.rds.amazonaws.com:5432/Airports_ds4a')
df.to_sql(name='reads2_raw', con=engine, if_exists = 'append', index=False, chunksize=10000)
engine.execute('select delete_duplicates()')
engine.execute('delete from reads2_raw where valid is null')
print('FINISH')

