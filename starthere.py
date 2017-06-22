import numpy as np
import pandas as pd
import quandl, math
from sklearn import preprocessing #scaling is done on features. try to get values between -1 and 1
from sklearn import cross_validation, svm
from sklearn.linear_model import LinearRegression
quandl.ApiConfig.api_key = '4EDgpr_zPmbrRoAXj6gQ'
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')



df = quandl.get('WIKI/AMD')

#print(df.tail())

### Notice the double brackets when redefining the df.

df= df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

###defining the relationships of the above dataframe. Creates new features in dataset

df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Low']*100

df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100

#           price         x          x           x
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

#print(df.head())

forecast_col = 'Adj. Close' #can change the forecast_col to be what ever label you're looking for

df.fillna(-99999, inplace=True) #need to replace NaN data in ML. -99999 will be treated as outlier

forecast_out = int(math.ceil(0.0025*len(df))) #math.ceil gets to ceiling which means it rounds all decimals up to nearest whole number
### this equation allows us to change the number of days out we are predicting.
###In this case we are setting the number of days we are predicint out to 10% of
###the total number of data rows in our dataframe
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

###our label will be the prediction of the label column (defined by forecast_col)
###predicted out some number of days (defined by forecast_out

#print(df.head())
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X=X[:-forecast_out]


df.dropna(inplace=True)
y = np.array(df['label'])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
### Save classifier to avoid training classifier over and over again....do this using pickle. Retrain as often as needed
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)


pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,y_test)
forecast_set = clf.predict(X_lately)
#print(accuracy)

print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan


'''
clfsvm = svm.SVR()
clfsvm.fit(X_train, y_train)
accuracy1 = clfsvm.score(X_test, y_test)
print(accuracy1)
'''



last_date = df.iloc[-1].name
print(last_date)
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

#print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
