### Thresold management

#####################
# Import Libraries
import pandas as pd 

dfThresold =  pd.read_csv('thresold.csv')

#Airports information helper functions
def getThresold(strCovar, strStation):
    return dfThresold[(dfThresold['ICAO'] == strStation) & (dfThresold['Covar'] == strCovar)]['value'].values[0]

def getVSBY(strStation):
    return getThresold('vsby',strStation)

def getSKYL1(strStation):
    return getThresold('skyl1',strStation)


#print(dfThresold)

#print(getSKYL1('SKBG'))

#print(getVSBY('SKBG'))

