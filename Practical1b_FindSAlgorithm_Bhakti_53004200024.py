# -*- coding: utf-8 -*-
"""
Created on 18th October 2021 by Bhakti Chotalia, SAP ID : 53004200024, MSC IT Part 2 2020-2022 Batch.
"""

import pandas as pd
import numpy as np

#Import dataset for find S algorithm
dataset=pd.read_csv("C:\\Users\\bhakt\\Desktop\\Bhakti_53004200024_MSc_IT\\Part 2\\Sem 3\\Machine Learning\\Practicals\\Practical 1\\Practical 1b\\archive\\employee_promotion.csv")

#Reading the dataset
print(dataset)

#building an array of all the attributes
attributes = np.array(dataset)[:,1:-1:]
print("\n The attributes are: ",attributes)

#segragating the target that has positive and negative examples
target = np.array(dataset)[:,-1]
print("\n The target is: ",target)

#Training the model for Find-S Algorithm
def train_findSalgorithm(c,t):
    for i,j in enumerate(t):
        if j=="YES":
            H0=c[i].copy()
            H1=c[i].copy()
            break;
    for k,l in enumerate(c):
        if t[k]=="YES":
            for r in range(len(H0)):
                if(H0[r]=='?'):
                    pass
                elif(l[r])>int(H0[r]):
                    H1[r]='?'
                    break
                else:
                    pass
    return H1

Promoted=train_findSalgorithm(attributes, target)
print("\n The Final Hypothesis is",Promoted)
