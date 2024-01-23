# -*- coding: utf-8 -*-
"""
Created on 29th November 2021 by Bhakti Chotalia, SAP ID : 53004200024, MSC IT Part 2 2020-2022 Batch.
 """

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import seaborn as sns
sns.set()
wcss=[]
kmean_dataset=pd.read_csv("C:\\Users\\bhakt\\Desktop\\Bhakti_53004200024_MSc_IT\\Part 2\\Sem 3\\Machine Learning\\Practicals\\Practical 6\\Practical 6b\\Mall_Customers.csv")
#View Data
print(kmean_dataset.head())
#Dimensions
print(kmean_dataset.shape)

#determining feature set
featureset=["Annual Income (k$)","Spending Score (1-100)"]
#determining Target
target=["pump"]
#X set
kmean_feature=kmean_dataset.iloc[:, [3,4]].values 
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(kmean_feature)
    wcss.append(kmeans.inertia_)
    
#Visualizing the ELBOW method to get the optimal value of K 
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(kmean_feature)
target=kmean_dataset.iloc[:, [6]].values


#Visualizing all the clusters 

plt.scatter(kmean_feature[y_kmeans == 0, 0], kmean_feature[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(kmean_feature[y_kmeans == 1, 0], kmean_feature[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(kmean_feature[y_kmeans == 2, 0], kmean_feature[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmean_feature[y_kmeans == 3, 0], kmean_feature[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmean_feature[y_kmeans == 4, 0], kmean_feature[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

print("Confusion Matrix",confusion_matrix(y_kmeans,target))
print("Test Score: ",f1_score(y_kmeans,target,average='micro'))

