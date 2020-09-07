# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 19:06:47 2020

@author: bchan
"""
import numpy as np
from gof import *
from best_fitting_plane import *

def min_gof(data, num_frac, X, Y):
    point = []
    normal = []
    polyVert = []
    min_gg = 1   
    for i in range(num_frac):
        print(f'Analyzing cluster {i}')
        verticalFracture = False
        tempdata = data[data[:, 3] == i]
        tempdata = tempdata[:, :3]
    
        temppoint, tempnormal = best_fitting_plane(tempdata, equation=False)
        # a,b,c,d = best_fitting_plane(tempdata,equation=True)
        
        # Tests if fracture is vertical
        if tempnormal[0] == 0: 
            verticalFracture = True
            tempnormal[0] = 1e-20
        print('Found best fit plane, finding goodness of fit')
        d = -np.dot(temppoint, tempnormal.T)
        Z = (-tempnormal[2]*X - tempnormal[1]*Y - d)/tempnormal[0]
        ZPred = (-tempnormal[2]*tempdata[:,2] - tempnormal[1]*tempdata[:,1] - d)/tempnormal[0]
        
        # Goodness of Fit
        r2, adjr2 = rsquared(tempdata[:,0], ZPred, len(tempdata[:,0]), 3, verticalFracture)
        print('Found r2')
        RMSE = rmse(tempdata[:,0], ZPred, verticalFracture)
        print('Found RMSE')
        # print("-----------------------------")
        # print("R-squared of cluster %d: %f" % (i+1, r2[i]))
        # print("Adjusted R-squared of cluster %d: %f" %(i+1, adjr2[i]))
        # print("RMSE of cluster %d: %f" %(i+1, RMSE[i]))
        # print("-----------------------------")
        
        min_gg = np.min((adjr2, min_gg))
        
    return min_gg