# -*- coding: utf-8 -*-
"""
Created on 2nd December 2021 by Bhakti Chotalia, SAP ID : 53004200024, MSC IT Part 2 2020-2022 Batch.
"""

import pandas as pd
import numpy as np

#Import dataset for find S algorithm
cel_dataset=pd.read_csv("C:\\Users\\bhakt\\Desktop\\Bhakti_53004200024_MSc_IT\\Part 2\\Sem 3\\Machine Learning\\Practicals\\Practical 2\\Practical 2b\\employee_promotion.csv")
#Viewing dataset
print(cel_dataset)
#making an array of all the attributes
s = np.array(cel_dataset)[:,1:-1:]
print("\n The values are: ",s)
#segragating the target that has positive and negative examples
decider = np.array(cel_dataset)[:,-1]
print(decider)

def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("Initialization of specific_h and general_h")
    print("specific_h: ",specific_h)
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("general_h: ",general_h)
    print("concepts: ",concepts)
    for i, h in enumerate(concepts):
        if target[i] == "YES":
            for x in range(len(specific_h)):
                #print("h[x]",h[x])
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        if target[i] == "NO":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
    print("\nSteps of Candidate Elimination Algorithm: ",i+1)
    print("Specific_h: ",i+1)
    print(specific_h,"\n")
    print("general_h :", i+1)
    print(general_h)
    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    print("\nIndices",indices)
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    return specific_h, general_h
s_final,g_final = learn(s, decider)
print("\nFinal Specific_h:", s_final, sep="\n")
print("Final General_h:", g_final, sep="\n")
