# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 23:29:04 2018

@author: amit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#reading the dataset
dataset=pd.read_csv("Ads_CTR_Optimisation.csv")

#implementing UCB
N=10000 #number of rounds
d=10 #number of version fro a aid
ad_selected = []
numbers_of_selection = [0] * d #vector of size d containg only zeroes
sums_of_reward = [0] * d
total_reward = 0

for n in range(0, N): #(loop for each version)
    max_upper_bound = 0
    ad = 0
    for i in range(0, d): #(loop for each version of aid)
        if (numbers_of_selection[i] > 0):
            average_reward = sums_of_reward[i]/numbers_of_selection[i]
            delta_i=math.sqrt(3/2 * math.log(n+1)/numbers_of_selection[i])
            upper_bound=average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
    ad_selected.append(ad)
    numbers_of_selection[ad] = numbers_of_selection[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_reward[ad] = sums_of_reward[ad] + reward
    total_reward = total_reward + reward
    
#visualizing the results    
plt.hist(ad_selected)  
plt.title('Histogram of ad selection')
plt.xlabel('Ads')
plt.ylabel('Number of times Ad selected')
plt.show()   

    
    