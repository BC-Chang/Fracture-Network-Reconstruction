# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 19:06:47 2020

@author: bchan
"""
import numpy as np
from gof import *
from best_fitting_plane import *
from sklearn.metrics import r2_score, mean_squared_error

def min_gof(data, num_frac, X, Y):
    point = []
    normal = []
    polyVert = []
    min_gg = 1
    r2, adjr2, RMSE = [np.ones((num_frac,)) for gof in range(3)]
    for i in range(num_frac):
        verticalFracture = False
        tempdata = data[data[:, 3] == i]
        tempdata = tempdata[:, :3]
    
        temppoint, tempnormal = best_fitting_plane(tempdata, equation=False)
        # a,b,c,d = best_fitting_plane(tempdata,equation=True)
        
        # Tests if fracture is vertical
        if tempnormal[0] == 0: 
            verticalFracture = True
            tempnormal[0] = 1e-20

        d = -np.dot(temppoint, tempnormal.T)
        Z = (-tempnormal[2]*X - tempnormal[1]*Y - d)/tempnormal[0]
        ZPred = (-tempnormal[2]*tempdata[:,2] - tempnormal[1]*tempdata[:,1] - d)/tempnormal[0]
        
        # Goodness of Fit
        r2[i], adjr2[i] = rsquared(tempdata[:,0], ZPred, len(tempdata[:,0]), 3, verticalFracture)
        acc = r2_score(tempdata[:,0], ZPred)
        RMSE[i] = rmse(tempdata[:,0], ZPred, verticalFracture)

        # print("-----------------------------")
        # print("R-squared of cluster %d: %f" % (i+1, r2[i]))
        # print("Adjusted R-squared of cluster %d: %f" %(i+1, adjr2[i]))
        # print("RMSE of cluster %d: %f" %(i+1, RMSE[i]))
        # print("-----------------------------")
        
    min_gg = np.min((np.min(adjr2), min_gg))
        
    return adjr2