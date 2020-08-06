# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:41:43 2020

@author: bchan
"""


import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def rsquared(observed, predicted, n, k):
    
    zbar = np.mean(observed)
    rsquared = 1 - np.sum(np.square(observed - predicted))/np.sum(np.square(observed - zbar))
    adjr2 = 1 - ((1-rsquared)*(n-1)/(n-k-1))
    
    return rsquared, adjr2


def rmse(observed, predicted):
    
    rmse = np.sqrt(np.mean(np.square(observed - predicted)))

    return rmse    


    