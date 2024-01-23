# -*- coding: utf-8 -*-
"""
Created on 29th November 2021 by Bhakti Chotalia, SAP ID : 53004200024, MSC IT Part 2 2020-2022 Batch.
 
"""

"""# calculating hamming distance between bit strings
 
# calculate hamming distance
def hamming_distance(a, b):
	return sum(abs(e1 - e2) for e1, e2 in zip(a, b)) / len(a)
 
# define data
row1 = [0, 1, 0, 1, 0, 1]
row2 = [1, 0, 1, 0, 0, 0]
# calculate distance
dist = hamming_distance(row1, row2)
print("Hamming Distance: ",dist) """

"""# calculating euclidean distance between vectors
from math import sqrt
 
# calculate euclidean distance
def euclidean_distance(a, b):
	return sqrt(sum((e1-e2)**2 for e1, e2 in zip(a,b)))
 
# define data
row1 = [10, 15, 13, 22, 3]
row2 = [10, 4, 2, 8, 21]
# calculate distance
dist = euclidean_distance(row1, row2)
print("Euclidean Distance: ",dist)"""

"""# calculating manhattan distance between vectors
from math import sqrt
 
# calculate manhattan distance
def manhattan_distance(a, b):
	return sum(abs(e1-e2) for e1, e2 in zip(a,b))
 
# define data
row1 = [10, 15, 13, 22, 3]
row2 = [10, 4, 2, 8, 21]
# calculate distance
dist = manhattan_distance(row1, row2)
print("Manhattan Distance: ",dist)"""

# calculating minkowski distance between vectors
from math import sqrt
 
# calculate minkowski distance
def minkowski_distance(a, b, p):
	return sum(abs(e1-e2)**p for e1, e2 in zip(a,b))**(1/p)
 
# define data
row1 = [10, 15, 13, 22, 3]
row2 = [10, 4, 2, 8, 21]
# calculate distance (p=1)
dist = minkowski_distance(row1, row2, 1)
print("Minkowski Distance for p=1: ",dist)
# calculate distance (p=2)
dist = minkowski_distance(row1, row2, 2)
print("Minkowski Distance for p=2: ",dist)
