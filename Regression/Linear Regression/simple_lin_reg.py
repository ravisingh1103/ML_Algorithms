# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 13:02:43 2020

@author: 718
"""
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing the dataset
dataset =  pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#spliting the dataset into the traing and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#print(x_train)
#print(x_test)

#training the simple linear regression model on the tarining 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


#predicting the test set results
y_pred = regressor.predict(x_test)
#print(y_pred)

#visualizing the trainig set results
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train) , color = 'blue')
plt.title('Salary vs Experience (Traing set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualizing the test set results

plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,regressor.predict(x_train) , color = 'blue')
plt.title('Salary vs Experience (Traing set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

