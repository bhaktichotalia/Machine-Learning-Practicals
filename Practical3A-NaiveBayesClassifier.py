# -*- coding: utf-8 -*-
"""
Created on 8th December 2021 by Bhakti Chotalia, SAP ID : 53004200024, MSC IT Part 2 2020-2022 Batch.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn import metrics
from sklearn.metrics import  confusion_matrix

myData=pd.read_csv('C:\\Users\\bhakt\\Desktop\\Bhakti_53004200024_MSc_IT\\Part 2\\Sem 3\\Machine Learning\\Practicals\\Practical 3\\Practical 3a\\diabetes.csv')

#Reading the headers data
print(myData.head())

#Rows and colums of data.
print(myData.shape)

#Training set and target Seperation
myData_train= np.array(myData)[:,1]
myData_target=np.array(myData)[:,-1]

#70% data to training data and 30% to testing data
X_trainData, X_testData, y_trainData, y_testData = train_test_split(myData_train, myData_target, test_size=0.30)

#Converting 1D array to 2D
X_trainData=pd.DataFrame(X_trainData)
X_testData=pd.DataFrame(X_testData)

#Running the Gaussian Naive Bayes Classifier
model = GaussianNB().fit(X_trainData, y_trainData)

#Predicting the value of Y.
predicted_y_val = model.predict(X_testData)

#Finding out the model accuracy.
accuracy_score = accuracy_score(y_testData, predicted_y_val) 
print ("Accuracy:",accuracy_score)

#Getting Confusion Matrix.
confm=np.array(confusion_matrix(y_testData,predicted_y_val))
print("Confusion Matrix",confm)
