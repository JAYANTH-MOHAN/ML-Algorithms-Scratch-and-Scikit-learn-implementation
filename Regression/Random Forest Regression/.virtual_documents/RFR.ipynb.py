import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv('Position_Salaries.csv')
dataset


X=dataset.iloc[:, 1:2]
Y=dataset.iloc[:,2]


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=50,random_state=0)
regressor.fit(X,Y)


out=regressor.predict(X)


plt.plot(X,out)


X_grid = np.arange(min(X.values), max(X.values), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')



