### Model management

#####################
# Import Libraries
import pandas as pd 
import numpy as np
import datetime
import time
from sqlalchemy import create_engine
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import auc, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#Enviroment setup
strConnection = 'postgresql://ds4a_18:ds4a2019@ds4a18.cmlpaj0d1yqv.us-east-2.rds.amazonaws.com:5432/Airports_ds4a'
intCVEstimators = 100
intBestEstimators = 100
intCV = 5
intJobs = -1
intRandomState = 123
intStartYear = 2016
intEndYear = 2030

dfAirports =  pd.read_csv('airports.csv')
dfCovars = pd.DataFrame({'ID':['vsby','skyl1'],'Description':['Horizontal visibility','Vertical visibility']})

lstAirports = dfAirports['ICAO'].values.tolist()

lstCovars = dfCovars['ID'].values.tolist()
lstNumCols = ['tmpf','dwpf','relh','drct','sknt','alti','vsby','skyl1']
lstLag = [6,7,8,9,10,12,24,25]


#Covar Description helper function
def getCovarDescription(strID):
    return str(dfCovars[dfCovars['ID'] == strID]['Description'].values[0])


#Airports information helper functions
def getStationData(strField, strStation):
    return dfAirports[dfAirports['ICAO'] == strStation][strField].values[0]

def getAirport(strStation):
    return getStationData('airport',strStation)

def getCity(strStation):
    return getStationData('city',strStation)

def getCoordinates(strStation):
    return getStationData('coordinates',strStation)

def getLongitude(strStation):
    return float(getStationData('longitude',strStation))

def getLatitude(strStation):
    return float(getStationData('latitude',strStation))


### Fetch Clean data from AWS DB for a given station
def fetchStationData(strStation, connection):

    print("Training models for {} station".format(getAirport(strStation)),end="\n\n")
    
    ## Get Data
    print("Fetching {} clean data".format(strStation))
    query = "SELECT * FROM dataclean WHERE station = '{}'".format(strStation)
    dfResult = pd.read_sql(query, connection.connect(), parse_dates=('valid',))
    dfResult = dfResult.sort_values(by=['day_hour'],ascending=True).reset_index(drop=True)
    dfResult = dfResult.drop_duplicates(subset='day_hour')
    print('{} clean data succesfully fetched from AWS RDS'.format(getCity(strStation)),end="\n\n")

    print("Preparing data for model training")

    return dfResult


def createModel(strStation, dfTrain, dfTest, strCovar):
    #### Model and cross validation grid parameters
    estim = RandomForestRegressor(n_estimators=intBestEstimators,criterion='mse',n_jobs=1,random_state=intRandomState)
    forest = GridSearchCV(estimator=estim,cv=intCV,param_grid={'max_depth':np.arange(11)+1},verbose=1)

    print('Training for ' + getCovarDescription(strCovar))

    Y_train = dfTrain[strCovar]
    X_train = dfTrain.drop(['day_hour',strCovar],axis=1)

    ### Test X & y matrix
    Y_test = dfTest[strCovar]
    X_test = dfTest.drop(['day_hour',strCovar],axis=1)

    ##### Model training over the validation grid
    #print('X:' + str(X_train.shape))
    #print('Y:' + str(Y_train.shape))
    forest.fit(X=X_train,y=Y_train)

    ##### Model recalibration with the best parameters
    mDepth = forest.best_params_['max_depth']
    forest2 = RandomForestRegressor(n_estimators=intCVEstimators,criterion='mse',
                                    n_jobs=1,random_state=intRandomState,
                                    max_depth=mDepth)
    forest2.fit(X=X_train,y=Y_train)
    
    #### Prediction 
    Y_pred = forest2.predict(X_test)#TODO

    #### Train's MSE  0.32294936345991776
    train_mse = mean_squared_error(Y_train,forest2.predict(X_train))

    #### Test's MSE 0.3553254070124522
    test_mse = mean_squared_error(Y_test,Y_pred)

    #### Prediction chart
    plt.subplots(figsize=(12,6))
    plt.plot(dfTest['day_hour'],Y_test,label="Real")
    plt.plot(dfTest['day_hour'],Y_pred,label="Predicted")
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel(getCovarDescription(strCovar))
    plt.xticks(rotation=30)
    plt.title('Trees: {} Max Depth: {} Random Forest \n Training MSE: {:0.3f} Test MSE: {:0.3f}'.format(intCVEstimators,mDepth,train_mse,test_mse))
    plt.savefig('DE/{}_train_{}.png'.format(strStation,strCovar),bbox_inches='tight',dpi=300)
    plt.clf()

    #### Relative importance
    importances = {"Regressor":X_train.columns,"Importance":forest2.feature_importances_}
    importances = pd.DataFrame(importances)
    importances = importances.sort_values(by='Importance',ascending=False)

    #### Importance chart
    plt.subplots(figsize=(12,6))
    plt.bar(importances['Regressor'],height=importances['Importance'])
    plt.title('{} of {} Model \n Relative Importance of the Regressors'.format(getCovarDescription(strCovar),getCity(strStation)))
    plt.xticks(rotation=90,ha='right')
    plt.savefig('DE/{}_importances_{}.png'.format(strStation,strCovar),bbox_inches='tight',dpi=300)
    plt.clf()

    #####Export model
    fileName = 'dash/{}_{}.sav'.format(strStation,strCovar)
    pickle.dump(forest, open(fileName, 'wb'))

    del(forest,forest2)
    print("Model Training for {} {} Ended...".format(getAirport(strStation), getCovarDescription(strCovar)))
    print("----------------------------------------------------------",end="\n\n\n")



def main(lstStations=lstAirports, lstFields=lstCovars):
    engine = create_engine(strConnection)


    ## Recategorize the year
    years = np.linspace(intStartYear,intEndYear,num=13,dtype='int')
    years_dict = {}
    for i,year in enumerate(years):
        years_dict[year]=i+1

    for station in lstStations:
        dfResult = fetchStationData(station,engine)

        #Lag setup start
        
        ############### MODIFIED PROCESS ################
        # Create additional dates for prediction
        dtStart = pd.to_datetime(dfResult['day_hour'].max(), infer_datetime_format=True)
        dtEnd = dtStart + datetime.timedelta(hours=25)
        print(dtStart)
        print(dtEnd)
        
        dfDates = pd.DataFrame(pd.date_range(start=dtStart, end=dtEnd, freq = 'H'), columns = ['day_hour'])
        dfDates['station'] = station
        
        dfStation = pd.merge(dfResult,dfDates, left_on =['day_hour','station'], right_on=['day_hour','station'], how = 'outer')

        #Lag setup end

        ## Select only numeric variables
        dfNumeric = dfStation[lstNumCols]

        print("Training {} station models".format(getCity(station)),end="\n\n")

        ## Lag Data
        lstLagged = []
        for i in lstLag:
            lag = dfNumeric.shift(periods=i)
            lag.columns = [col+'_{}'.format(i) for col in lag.columns] 
            lstLagged.append(lag)
        
        #for var in lstVars:
        Y = dfStation[['day_hour','vsby','skyl1']]
        dfFinal = pd.concat([Y]+lstLagged,axis=1)

        #Lagging start
        v=25-6;v
        dfFinal=dfFinal.iloc[25:,:]
        dfFinal=dfFinal.iloc[:-v,:]
        
        dfPrediction = dfFinal[dfFinal.isnull().any(axis=1)] # Guardar las X para las predicciones
        dfFinal = dfFinal[~dfFinal.isnull().any(axis=1)] # Separar el resto para el train y test
        #Lagging end

        dfFinal = dfFinal.sort_values(by=['day_hour'],ascending=False).reset_index(drop=True)
        #dfFinal.dropna(inplace=True)

        ## Extract year,month,day and hour from the date
        dfFinal['year']=pd.DatetimeIndex(dfFinal['day_hour']).year
        dfFinal['month']=pd.DatetimeIndex(dfFinal['day_hour']).month
        dfFinal['day']=pd.DatetimeIndex(dfFinal['day_hour']).day 
        dfFinal['hour']=pd.DatetimeIndex(dfFinal['day_hour']).hour

        dfFinal['year']=dfFinal['year'].map(years_dict)
        print("Data for model training ready",end="\n\n")

        print("Selecting train and test data",end="\n\n")
        train = dfFinal[dfFinal['day_hour']<='2019-10-23']
        test = dfFinal[dfFinal['day_hour']>'2019-10-23']

        for var in lstFields:
            createModel(station, train, test, var)


## main (kind of)...
main()