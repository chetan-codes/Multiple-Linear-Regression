
# Multiple Linear Regression

#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing the dataset
dataset=pd.read_csv('C:/Downloads/50_Startups.csv')
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

#Convert the column into categorical columns
states=pd.get_dummies(X['State'],drop_first=True)

#Drop the state columns 
X=X.drop('State',axis=1)

#Concat the dummy variables
X=pd.concat([X,states],axis=1)

#Splitting the dataset into the training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred=regressor.predict(X_test)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
