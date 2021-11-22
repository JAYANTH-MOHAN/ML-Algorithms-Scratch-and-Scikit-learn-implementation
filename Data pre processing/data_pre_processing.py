# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# importing dataset
dataset=pd.read_csv('data.csv')
x=dataset.iloc[:, : -1].values
y=dataset.iloc[:, -1].values
print(dataset.isnull().values.any()) # to check wheather the dataframe has any nan values
pd.isnull(x)  # to check wheather the object has any nan values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print(x)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [0])],remainder='passthrough')
X=np.array(ct.fit_transform(x))
print(X)
# encoding dependent variable
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
print('y')
print(y)
# splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

print('xtrain')
print(X_train)
print('xtest')
print(X_test)
print('y train')
print(y_train)
print('y test')
print(y_test)
# now we have 2 test dataset and 8 train dataset
# now we will feature scale it
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
X_test[:,3:]=sc.fit_transform(X_test[:,3:])
print('xtrain')
print(X_train)
print('xtest')
print(X_test)