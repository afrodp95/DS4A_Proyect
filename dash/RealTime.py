#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:16:04 2019

@author: mario
"""

from __future__ import print_function
import json
import time
import datetime as datetime
import pandas as pd

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

# Number of attempts to download data
MAX_ATTEMPTS = 6

# HTTPS here can be problematic for installs that don't have Lets Encrypt CA
SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
startts = datetime.datetime(2019,11,14)
endts = datetime.datetime(2019, 11, 15)

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


service = SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=yes&"
service += startts.strftime("year1=%Y&month1=%m&day1=%d&")
service += endts.strftime("year2=%Y&month2=%m&day2=%d&")

"""Build a station list by using a bunch of IEM networks."""

stations = []
networks = ["CO__ASOS"]
 
#for network in networks:
#    # Get metadata
#    uri = (
#        "https://mesonet.agron.iastate.edu/" "geojson/network/%s.geojson"
#    ) % (network,)
#    data = urlopen(uri)
#    jdict = json.load(data)
#    for site in jdict["features"]:
#        stations.append(site["properties"]["sid"])
    
estaciones=["SKAR","SKQL","SKBO","SKBG","SKCL","SKCC","SKCG","SKPE","SKSP","SKSM","SKMR"]



df = pd.DataFrame()
for station in estaciones:
#    if station in estaciones:
    uri = "%s&station=%s" % (service, station)
    print("Downloading: %s" % (station,))
    data = download_data(uri)
    data = pd.DataFrame([x.split(',') for x in data.split('\n')]).iloc[5:].reset_index(drop=True)
    data.columns = data.iloc[0]
    data = data.drop(data.index[0])
    df = df.append(data)

 
#postString = data.split("\n",6)[6].split('\n')
#listas=[row.split(',') for row in postString]
#df=pd.DataFrame.from_records(listas , columns=['station', 'valid', 'lon', 'lat', 'tmpf', 'dwpf', 'relh', 'drct', 'sknt', 'p01i', 'alti', 'mslp', 'vsby', 'gust', 'skyc1', 'skyc2', 'skyc3', 'skyc4', 'skyl1', 'skyl2', 'skyl3', 'skyl4', 'wxcodes', 'ice_accretion_1hr', 'ice_accretion_3hr', 'ice_accretion_6hr', 'peak_wind_gust', 'peak_wind_drct', 'peak_wind_time', 'feel', 'metar'])
del df['metar']
del df['wxcodes']
print({'shape':df.shape})

from sqlalchemy import create_engine
engine = create_engine('postgresql://ds4a_18:ds4a2019@ds4a18.cmlpaj0d1yqv.us-east-2.rds.amazonaws.com:5432/Airports_ds4a')
df.to_sql(name='reads2_raw', con=engine, if_exists = 'append', index=False, chunksize=10000)
engine.execute('select delete_duplicates()')
engine.execute('delete from reads2_raw where valid is null')
print('FINISH')

