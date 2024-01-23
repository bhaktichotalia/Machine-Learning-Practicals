# -*- coding: utf-8 -*-
"""
Created on 30th November 2021 by Bhakti Chotalia, SAP ID : 53004200024, MSC IT Part 2 2020-2022 Batch.
 """

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

hc_data=pd.read_csv("C:\\Users\\bhakt\\Desktop\\Bhakti_53004200024_MSc_IT\\Part 2\\Sem 3\\Machine Learning\\Practicals\\Practical 7\\Practical 7a\\Country-data.csv ")
print(hc_data.head())
print(hc_data.shape)
                    
#X data
x_data = hc_data.iloc[:, 3:5].values
plt.figure(figsize=(10, 7))
plt.title("Country Dendograms")
                    
#Create dendogram
dend = shc.dendrogram(shc.linkage(x_data, method='ward'))
                    
#Hierarchial clustering object, ussing algoremetaive clustering
h_cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean',linkage='ward')
h_cluster.fit_predict(x_data)

#Plotting dendogram and clusters
plt.figure(figsize=(10, 7))
plt.scatter(x_data[:,0], x_data[:,1], c=h_cluster.labels_, cmap='rainbow')
plt.show()
