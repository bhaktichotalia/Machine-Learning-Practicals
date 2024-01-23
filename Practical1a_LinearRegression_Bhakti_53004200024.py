# -*- coding: utf-8 -*-
"""
Created on 17th October 2021 by Bhakti Chotalia, SAP ID : 53004200024, MSC IT Part 2 2020-2022 Batch.
 
"""
#imports libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from math import sqrt
from sklearn.metrics import mean_squared_error


#Read Data set
dataset = pd.read_csv('C:\\Users\\Bhakti\\Desktop\\MSC IT\\Machine Learning\\Practicals\\Practical 1\\Practical 1a\\archive\\insurance.csv',
                      skiprows=6, names=['Claims','Total_Payment'])
dataset.head()

#Understanding our data with descriptive statistics and visualisation
datasetDescription=dataset.describe()
print(datasetDescription)
dataset.hist()
plt.show()
plt.scatter(dataset.Claims,dataset.Total_Payment)
plt.title('Claims vs Total Payment')
plt.xlabel('Claims')
plt.ylabel('Total Payment')
plt.show()

#Training the ML Model by splitting the data into training set and test set
X = dataset[['Claims']]
y = dataset['Total_Payment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=13/69, random_state=0)
regressor = LinearRegression()
model = regressor.fit(X_train,y_train)

#Predicting and visualising Training Set Values
y_pred = model.predict(X_train)
print('Training Set',y_pred)
error = mean_squared_error(y_train,y_pred)
sqrt(error)
plt.scatter(X_train,y_train)
plt.plot(X_train,model.predict(X_train),color='red')
plt.xlabel('Claims')
plt.ylabel('Total Payment')
plt.show()

#Predicting and visualising Test Set Values
y_test_pred = model.predict(X_test)
print('Testing Set',y_test_pred)
error = mean_squared_error(y_test,y_test_pred)
sqrt(error)
plt.scatter(X_test,y_test)
plt.plot(X_test,model.predict(X_test),color='green')
plt.xlabel('Claims')
plt.ylabel('Total Payment')
plt.show()






