# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:35:46 2020

@author: 718
"""


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv(r'C:\Users\718\Desktop\Final Folder\Dataset\breast_cancer.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#spliting the dataset into taring and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2 , random_state = 0)

#training the logistic regression on traing set and test set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

#predicting the test set result
y_pred = classifier.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

#computing the accuracy with k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
