import numpy as np
import csv
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
#read Cleveland Heart Disease data

names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
heartDisease = pd.read_csv('C:\\Users\\Bhakti\\Desktop\\MSC IT\\Machine Learning\\Practicals\\Practical 8\\Practical 8a\\cleveland.csv',names=names)
#heartDisease = heartDisease.replace('?',np.nan)
#display the data
print('Few examples from the dataset are given below')
print(heartDisease.head())
del heartDisease['ca']
del heartDisease['slope']
del heartDisease['thal']
del heartDisease['oldpeak']

heartDisease = heartDisease.replace('?', np.nan)
#print(heartDisease.dtypes)
#print(heartDisease.columns)
#Model Bayesian Network
model = BayesianNetwork([('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('sex', 'trestbps'), 
                       ('exang', 'trestbps'),('trestbps','heartdisease'),('fbs','heartdisease'),
                      ('heartdisease','restecg'),('heartdisease','thalach'),('heartdisease','chol')])
#Learning CPDs using Maximum Likelihood Estimators
print('\\n Learning CPD using Maximum likelihood estimators')
model.fit(heartDisease,estimator=MaximumLikelihoodEstimator)
print(model.get_cpds('age'))
# Inferencing with Bayesian Network
print('\\n Inferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)
#computing the Probability of HeartDisease given Age
print('\\n 1. Probability of HeartDisease given Age=30')
q1 = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 28})
print(q1)
#computing the Probability of HeartDisease given cholesterol
print('\\n 2. Probability of HeartDisease given cholesterol=100')
q2=HeartDisease_infer.query(variables=['heartdisease'],evidence={'chol':100})
print(q2)
