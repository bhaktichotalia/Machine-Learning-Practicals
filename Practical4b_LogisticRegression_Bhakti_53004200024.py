# -*- coding: utf-8 -*-
"""
Created on 30th October 2021 by Bhakti Chotalia, SAP ID : 53004200024, MSC IT Part 2 2020-2022 Batch.
 
"""

#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pandas import Series
from sklearn.metrics import accuracy_score , confusion_matrix, roc_curve , classification_report,auc

#read data
df=pd.read_csv("C:\\Users\\bhakt\\Desktop\\Bhakti_53004200024_MSc_IT\\Part 2\\Sem 3\\Machine Learning\\Practicals\\Practical 4\\Practical 4b\\Social_Network_Ads.csv")
print(df.head())
Corrected_data=df.corr()
print(Corrected_data)

#dropping the USER ID column and Converting Gender col into int by convertong male and female values to 0 and 1 respectivly
df.drop('User ID',axis = 1, inplace = True)
label = {'Male': 0 ,"Female" : 1}
df['Gender'].replace(label, inplace= True)

#preparing Dataset for prediction 
X = df.drop('Purchased',axis = 1)
y = df['Purchased']

#preprocessing the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 42)
log = LogisticRegression()
model = log.fit(X_train,y_train)
y_pred = model.predict(X_test)
matrix = confusion_matrix(y_test,y_pred)
print(matrix)
print(classification_report(y_test,y_pred))
print('Accuracy: ', accuracy_score(y_test,y_pred))


