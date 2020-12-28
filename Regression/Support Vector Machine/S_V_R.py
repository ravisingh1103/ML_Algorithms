# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:18:30 2020

@author: 718
"""


#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
#print(y)
#transforming y

y = y.reshape(len(y),1)
#print(y) 

#feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()  #scaler of x
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
#print(y)

#training the SVR model
from sklearn.svm import SVR
 
regressor = SVR(kernel = 'rbf') # creating svr model with radial basis function kernel
regressor.fit(X, y) #this fit method will train our SVR model

#predicting the new result
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]]))))
#print("END")
#this will return the result of the predicted salary but in the scale that was applied to y
#we need to return the result so we have to reverse the scaling


#visualizing the result of SVR model
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X)),color = 'blue')
plt.title('Truth or Bluf')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

# visualizing the resukt of SVR model for higher resolution and smoother curve

X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


















