# -*- coding: utf-8 -*-
"""
Created on 2nd December 2021 by Bhakti Chotalia, SAP ID : 53004200024, MSC IT Part 2 2020-2022 Batch.
"""


# Feature Extraction with PCA
import pandas as pd
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
pca_data=pd.read_csv('C:\\Users\\bhakt\\Desktop\\Bhakti_53004200024_MSc_IT\\Part 2\\Sem 3\\Machine Learning\\Practicals\\Practical 2\\Practical 2a\\employee_promotion.csv')
print(pca_data.shape)
print(pca_data.head(20))
names = ['EmployeeId', 'Client Impact Score', 'Firm Impact Score', 'Overall Rating', 'Resume', 'Business Leader Rating', 'Cumulative Grade', 'Manager Support']
target=['Promoted']
X = pca_data[names]
Y = pca_data[target]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)
