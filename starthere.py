import numpy as np
import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing #scaling is done on features. try to get values between -1 and 1
from sklearn import cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

#print(df.head())

### Notice the double brackets when redefining the df.

df= df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

###defining the relationships of the above dataframe. Creates new features in dataset

df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Low']*100

df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

#print(df.head())

forecast_col = 'Adj. Close' #can change the forecast_col to be what ever label you're looking for

df.fillna(-99999, inplace=True) #need to replace NaN data in ML. -99999 will be treated as outlier

forecast_out = int(math.ceil(0.01*len(df))) #math.ceil gets to ceiling which means it rounds all decimals up to nearest whole number
### this equation allows us to change the number of days out we are predicting.
###In this case we are setting the number of days we are predicint out to 10% of
###the total number of data rows in our dataframe

df['label'] = df[forecast_col].shift(-forecast_out)

###our label will be the prediction of the label column (defined by forecast_col)
###predicted out some number of days (defined by forecast_out
df.dropna(inplace=True)
print(df.head())
