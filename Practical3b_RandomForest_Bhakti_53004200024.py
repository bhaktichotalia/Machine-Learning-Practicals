# -*- coding: utf-8 -*-
"""
Created on 2nd December 2021 by Bhakti Chotalia, SAP ID : 53004200024, MSC IT Part 2 2020-2022 Batch.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import  confusion_matrix
from sklearn.ensemble import RandomForestRegressor

#Loading data
my_df=pd.read_csv('C:\\Users\\bhakt\\Desktop\\Bhakti_53004200024_MSc_IT\\Part 2\\Sem 3\\Machine Learning\\Practicals\\Practical 3\\Practical 3b\\gapminder_alcohol.csv')

#Showing data headers and few values
print(my_df.head())

#Showing data dimensions
print(my_df.shape)

#Seperating X and Y values
df_X = my_df.iloc[:, 0:5].values
df_y = my_df.iloc[:, 5].values

#splitting Traing and testing data
X_traindata, X_testdata, y_traindata, y_testdata = train_test_split(df_X, df_y, test_size=0.2, random_state=0)

#Standardsing the data
sc = StandardScaler()
X_traindata = sc.fit_transform(X_traindata)
X_testdata = sc.transform(X_testdata)

#Applying random forest
randomForestRegressor = RandomForestRegressor(n_estimators=20, random_state=0)
randomForestRegressor.fit(X_traindata, y_traindata)

#Predicting the output
y_pred = randomForestRegressor.predict(X_testdata)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_testdata, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_testdata, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_testdata, y_pred)))
