# -*- coding: utf-8 -*-
"""
Created on 9th December 2021 by Bhakti Chotalia, SAP ID : 53004200024, MSC IT Part 2 2020-2022 Batch.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori

rule_dataset=pd.read_csv("C:\\Users\\bhakt\\Desktop\\Bhakti_53004200024_MSc_IT\\Part 2\\Sem 3\\Machine Learning\\Practicals\\Practical 7\\Practical 7b\\Market_Basket_Optimisation.csv")
print(rule_dataset)

records = []
for i in range(0, 1000):
    for j in range(1,4):
        records.append([str(rule_dataset.values[i,j])])
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2)
association_results = list(association_rules)
print(association_results)

