# -*- coding: utf-8 -*-
"""
Created on 26th October 2021 by Bhakti Chotalia, SAP ID : 53004200024, MSC IT Part 2 2020-2022 Batch.
 
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
sns.set()
#Using iris dataset as per aim
knn_dataset=pd.read_csv("C:\\Users\\bhakt\\Desktop\\Bhakti_53004200024_MSc_IT\\Part 2\\Sem 3\\Machine Learning\\Practicals\\Practical 5\\Practical 5b\\iris.csv")
#View Data
print(knn_dataset.head())
#Dimensions
print(knn_dataset.shape)
#determining feature set
featureset=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
#determining Target
target=["Species"]
#X set
knn_feature=knn_dataset[featureset] 
#Y set
knn_target=knn_dataset[target]
#dividing traing and test dataset
X_train, X_test, y_train, y_test = train_test_split(knn_feature, knn_target, random_state=1)
#Applying k-nearest neighbor classifier
knn_neigh = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
#Applying fit
knn_neigh.fit(X_train, y_train.values.ravel())
#Predicting Y
y_pred = knn_neigh.predict(X_test)
#Plotting scatter graph
sns.scatterplot(
   
    data=X_test.join(y_test, how='outer')
)
plt.scatter(
    X_test['SepalLengthCm'],
    X_test['SepalWidthCm'],
)
plt.show()

#Confusion Matrix
print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))

