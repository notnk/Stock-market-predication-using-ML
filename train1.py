import pickle
import datetime as dt
import pandas as pd
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
df = pd.read_csv("Dataset/data2.csv")
#df.set_index("Date", inplace=True)
#df.dropna(inplace=True)
#y = df.iloc[:, 3].values
#x = df.iloc[:, 0:5].values
X=df.drop('Date',axis=1)
x=X.drop('Close',axis=1)
print(x)
y=X['Close']

print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.26,  random_state=0)
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)
grid_rf = {
'n_estimators': [20, 50, 100, 500, 1000],  
'max_depth': np.arange(1, 15, 1),  
'min_samples_split': [2, 10, 9], 
'min_samples_leaf': np.arange(1, 15, 2, dtype=int),  
'bootstrap': [True, False], 
'random_state': [1, 2, 30, 42]
}

model = RandomForestRegressor(n_estimators=500, random_state=42, min_samples_split=2, min_samples_leaf=1, max_depth=10, bootstrap=True)
model.fit(x_train, y_train)
predict = model.predict(x_test)
print(x_test)
print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, predict), 4))
with open('scale1.pickle', 'wb') as handle:
    pickle.dump(scale, handle, protocol=pickle.HIGHEST_PROTOCOL)
f=open("rfregressn1.pickle","wb")
pickle.dump(model,f)
f.close()