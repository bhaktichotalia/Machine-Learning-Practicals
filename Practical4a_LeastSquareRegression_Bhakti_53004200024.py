# -*- coding: utf-8 -*-
"""
Created on 29th October 2021 by Bhakti Chotalia, SAP ID : 53004200024, MSC IT Part 2 2020-2022 Batch.
 
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
dataset4a = pd.read_csv('C:\\Users\\bhakt\\Desktop\\Bhakti_53004200024_MSc_IT\\Part 2\\Sem 3\\Machine Learning\\Practicals\\Practical 4\\Practical 4a\\student_scores.csv')
print(dataset4a)

#statistical details of the dataset
datasetDescription=dataset4a.describe()
print(datasetDescription)

#Preparing Data
X = dataset4a.iloc[:, 0]
Y = dataset4a.iloc[:, 1]

#Current Data Scatterplot
plt.scatter(X, Y)
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

# Building the model
X_mean = np.mean(X)
Y_mean = np.mean(Y)
num = 0
den = 0
n=len(X)
for i in range(len(X)):
    num += (X[i] - X_mean)*(Y[i] - Y_mean)
    den += (X[i] - X_mean)**2
m = num / den
c = Y_mean - m*X_mean
print (m, c)

# Making predictions
Y_pred = m*X + c
plt.scatter(X, Y) # actual

# plt.scatter(X, Y_pred, color='red')
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') # predicted
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()
rmse = 0
for i in range(n):
    y_pred = c + m * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/n)
print("RMSE: ",rmse)

