#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:30:55 2018

@author: samas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy
df=pd.read_csv(r'/Users/samas/Desktop/MLTrainingDocs/Day3And4/international-airline-passengers.csv',skipfooter=3)


#In Time series analysis we need to make dateAndTime column as index.as below...
df['Month']=pd.to_datetime(df['Month'])
df.index=df['Month']
df.head()

df.drop(['Month'],axis=1,inplace=True)
df.head(2)



plt.figure(figsize=(12,5))
plt.plot(df)
plt.show()

rollmean=df.rolling(10).mean()
rollmean.head(20)

rollstd=df.rolling(10).std()

plt.figure(figsize=(12,5))
plt.plot(df,'r')
plt.plot(rollmean,'g')
plt.plot(rollstd,'black')
plt.show()

dflog=numpy.log(df)
plt.figure(figsize=(12,5))
plt.plot(dflog)
plt.show()

dflogdiff=dflog-dflog.shift(1)
dflogdiff.head()

rollmean1=dflog.rolling(10).mean()
plt.plot(dflog,'r')
plt.plot(rollmean1,'g')
plt.show()

rollmean2=dflogdiff.rolling(10).mean()
plt.plot(dflogdiff,'r')
plt.plot(rollmean2,'g')
plt.show()

  
from statsmodels.tsa.stattools import acf,pacf

dflogdiff.dropna(inplace=True)
ac=acf(dflogdiff,nlags=10)
plt.plot(ac)
plt.grid(True)
plt.show()

pac=pacf(dflogdiff,nlags=10)
plt.plot(pac)
plt.grid()
plt.plot()

#Using Arima Model
from statsmodels.tsa import arima_model
                              #P d Q
model=arima_model.ARIMA(dflog,(2,1,2))
out=model.fit()
out.forecast(168)
forecast,error,c_interval=out.forecast(168)
numpy.exp(forecast)


#Components in time series
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition=seasonal_decompose(dflog)
trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid

plt.figure(figsize=(12,5))
plt.subplot(411)
plt.plot(dflog,label='original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residual')
plt.legend(loc='best')


















