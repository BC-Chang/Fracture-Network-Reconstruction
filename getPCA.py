'''
Created on May 12, 2020

@author: Bernard Chang

Applies Principal Component Analysis

Inputs:
-----------------------------------------------------------------------------
data: array 
    Array of NxM dimensions containing the data.
    Each of the N rows represents a different individual record
    Each of the M columns represents a different variable recorded (x,y,z)

correlation(optional): bool
    Set the type of the matrix to be computed:
        If True, compute the correlation matrix.
        If False(Default), compute the covariance matrix.
        
sort(optional): bool
    Set the order that the eigenvalues & eigenvectors will have
        If True(Default), sorted in descending order
        If False, unsorted

Outputs:
-----------------------------------------------------------------------------
Eigenvalues: (1,M) array
Eigenvectors: (M,M) array

'''

import numpy as np
def getPCA(data, correlation = False, sort = True):

    #data = tempdata[:,:3]
    
    mean = np.mean(data,axis=0)
    
    data_adjust = data - mean
    
    if correlation:
        matrix = np.corrcoef(data_adjust.T)
    else:
        matrix = np.cov(data_adjust.T)
    
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]
        
    return eigenvalues, eigenvectors    