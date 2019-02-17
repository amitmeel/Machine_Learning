# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 19:31:33 2018

@author: amit
"""
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading dataset
data=pd.read_csv("Market_Basket_Optimisation.csv",header=None)
transactions=[]
for i in range(0,7501):
    transactions.append([str(data.values[i,j]) for j in range(0,20)])
    

#training apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.20, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

#results_list = []
#for i in range(0, len(results)):
#    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]) + '\nCONF:\t' + str(results[i][2]))
#    
#""" 
#
#results_list = []
#for i in range(0, len(results)):
#    results_list.append('RULE:\t' + str(results[i][0]) + 
#                        '\nSUPPORT:\t' + str(results[i][1]) +
#                        '\nCONF:\t' + str(results[i][2][0][2]) +
#                        '\nLIFT:\t' + str(results[i][2][0][3]))
#
#"""

# This function takes as argument your results list and return a tuple list with the format:
# [(rh, lh, support, confidence, lift)] 
def inspect(results):
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))
# this command creates a data frame to view
resultDataFrame=pd.DataFrame(inspect(results),
                columns=['rhs','lhs','support','confidence','lift'])