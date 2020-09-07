# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 17:44:49 2020

@author: bchan
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering, MeanShift
from skimage import data, io, color
import time
from sklearn import mixture
from getPCA import *
from best_fitting_plane import *
from coordTrans import *
from scipy.spatial import ConvexHull
from collections import OrderedDict 
from xlrd.formula import num2strg
from gof import *
from mingg import min_gof

from mayavi import mlab

imgGeom = 1  # 0 for 2D, 1 for 3D
val_frac = 0 # Label value of fracture in image

# Domain size
nx = 400
ny = 100
nz = 450

im = io.imread('./input/Segmented_SubSample1_small.tif')
result = np.where(im == val_frac)
result = np.asarray(result).transpose()

x = result[:, 2]
y = result[:, 1]

if imgGeom == 1:
    z = result[:, 0]

elif imgGeom == 0:
    z = np.zeros([len(x), 1])
    result = np.append(result, z, axis=1)


num_fracs = 1 # Initial guess for number of fractures
gg = 0.80 # Convergence criterion for adjusted r-squared gof
min_gg = 0 # Initial convergence
best_min_gg = [-1e20, 0] # Initial value to find best convergence in max_num_fracs
converged = False

# Maximum number of fractures. 
# If max_n is reached, takes the number of fractures with largest minimum adjusted r2
max_num_fracs = 5

# Open template file for dfnWorks
#fin = open('./dfnRectInputTempl.dat', 'rt')  
#template = fin.read()
#fin.close()

#fin = open('user_specified_rect_coord.dat', 'wt')  # Input file for dfnWorks



# Create meshgrid for plotting
X, Y = np.meshgrid(np.linspace(0, nx, 20), np.linspace(0, ny, 20))

# Plot input image
# mlab.figure(bgcolor=(1,1,1), size=(1200,1200))
# mlab.points3d(x, y, z)

# mlab.figure(bgcolor=(1,1,1), size=(1200,1200))
# pltcolor = ((1, 0, 0), (0, 1, 0), (0, 0 ,1))
#%%
t0 = time.time()
print("========================================================")
print("Beginning clustering")
while min_gg < gg and num_fracs <= max_num_fracs:
    print("-----------------------------")
    print(f"Testing {num_fracs} fracture(s)...")
    clustering = mixture.BayesianGaussianMixture(
        n_components=num_fracs, covariance_type='full', weight_concentration_prior=1e-2,
        weight_concentration_prior_type='dirichlet_process',
        mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(3),
        init_params="random", max_iter=100, random_state=2).fit(result)
    labels = clustering.predict(result)
    
    labels = labels.reshape(len(labels), 1)
    
    data = np.append(result, labels, axis=1)
    print('Found clusters, analyzing...')

    #template = template.replace('{nEllipses_templ}', str(numfrac[0]))
    #template = template.replace('{nNodes_templ}', str(4))
   
    min_gg = min_gof(data, num_fracs, X, Y)
     

    print(f"Minimum adjusted R-squared of {num_fracs} cluster(s) = {min_gg}")
    print("-----------------------------")
    
    if min_gg < gg:
        if min_gg > best_min_gg[0]:
            best_min_gg[0] = min_gg
            best_min_gg[1] = num_fracs
        num_fracs += 1
            
        del data
    else:
        converged=True
        
t1 = time.time()

if converged:
    print(f"Converged using {num_fracs} clusters.")
else:
    print(f"Did not converge using up to {max_num_fracs} clusters. The best convergence was using {best_min_gg[1]} clusters")
    
print("Time to find number of clusters = %f sec" %(t1-t0))
print("========================================================")
