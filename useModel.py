# Use model for a station and a covar
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

def getPrediction(strStation,strCovar,X_test,Y_test):
    fileName = 'dash/{}_{}.sav'.format(strStation,strCovar)

    loaded_model = pickle.load(open(filename, 'rb'))
    fResult = loaded_model.score(X_test, Y_test)
    print(fResult)
    return fResult