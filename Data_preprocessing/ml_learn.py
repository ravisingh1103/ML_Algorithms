# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:18:42 2020

@author: 718
"""

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#print(x)
#print(y)

#take care of missing data from the dataset
from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan,  strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#print(x)
#print(y)


#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
ct = ColumnTransformer( transformers= [('encoder',OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
#print(x)

le= LabelEncoder()
y = le.fit_transform(y)
#print(y)

#splitting the dataset into training set and test data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=1)
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)

#feature scaling

sc= StandardScaler()
x_train[:,3:]= sc.fit_transform(x_train[:,3:])           #fit get the mean and standard deviation of each of your features 
x_test[:,3:]= sc.transform(x_test[:,3:])                               #and transform will aapply the formula of standarisation to transform your values so that they can all be in the same scale
                                
#print(x_train)
print(x_test)

