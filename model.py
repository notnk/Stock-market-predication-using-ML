import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
df = pd.read_csv("Dataset/apples.csv")

y=df[['Close']].copy()
n1=input('date:')
n2=df[df['Date'] ==n1].index.values
n3=n2[0]
n4=n3+2
print(n4)
n=int(len(df)*.8)
print(n)
train=y[:n4]
test=y[n4:]
from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(train,order=(6,1,3))
result=model.fit(disp=0)
print(result.summary())
step=1
fc=result.forecast(step)
print(fc)
predictions = list()
for t in range(len(test)):
		model = ARIMA(train, order=(6,1,3))
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()
		predictions.append(yhat)
	# calculate out of sample error
error = mean_squared_error(test, predictions)
print(error)
