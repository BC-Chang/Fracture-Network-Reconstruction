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
from clean_data import *

from mayavi import mlab

imgGeom = 1  # 0 for 2D, 1 for 3D
val_frac = 1 # Label value of fracture in image

# Domain size
nx = 150
ny = 100
nz = 400

im = io.imread('./input/Segmented_SubSample1_small.tif')
im = im/val_frac
im, connected_labels = clean_data3D(im, max_element_size=500)
result = np.where(im == 1)
result = np.asarray(result).transpose()

x = result[:, 2]
y = result[:, 1]

if imgGeom == 1:
    z = result[:, 0]

elif imgGeom == 0:
    z = np.zeros([len(x), 1])
    result = np.append(result, z, axis=1)


num_fracs = 1 # Initial guess for number of fractures
gg = 0.1 # Convergence criterion for adjusted r-squared gof
min_gg = 0 # Initial convergence
best_min_gg = [1e20, 0] # Initial value to find best convergence in max_num_fracs
converged = False
target = 0.8 # Target adjusted-r2 value

# Maximum number of fractures. 
# If max_n is reached, takes the number of fractures with largest minimum adjusted r2
max_num_fracs = 3

# Open template file for dfnWorks
#fin = open('./dfnRectInputTempl.dat', 'rt')  
#template = fin.read()
#fin.close()

#fin = open('user_specified_rect_coord.dat', 'wt')  # Input file for dfnWorks



# Create meshgrid for plotting
X, Y = np.meshgrid(np.linspace(0, nx, 20), np.linspace(0, ny, 20))

# Plot input image
mlab.figure(bgcolor=(1,1,1), size=(1200,1200))
mlab.points3d(x, y, z)


#plt.scatter(x,y)
# mlab.figure(bgcolor=(1,1,1), size=(1200,1200))
# pltcolor = ((1, 0, 0), (0, 1, 0), (0, 0 ,1))

t0 = time.time()
print("========================================================")
print("Beginning clustering")
while converged==False and num_fracs <= max_num_fracs:
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


    #template = template.replace('{nEllipses_templ}', str(numfrac[0]))
    #template = template.replace('{nNodes_templ}', str(4))
    adjr2 = min_gof(data, num_fracs, X, Y) # Calculate adjusted r2
    l2norm = np.linalg.norm(adjr2-target) # Calculate l2 norm of adjusted r2
    print(f"L2 Norm of {num_fracs} cluster(s) = {l2norm}")
    print("-----------------------------")
    
    
    if l2norm > gg:
        if l2norm < best_min_gg[0]:
            best_min_gg[0] = l2norm
            best_min_gg[1] = num_fracs
        num_fracs += 1
            
        del data, labels, clustering
    else:
        converged=True
        
t1 = time.time()

if converged:
    print(f"Converged using {num_fracs} clusters.")
else:
    print(f"Did not converge using up to {max_num_fracs} clusters. The best convergence was using {best_min_gg[1]} clusters")
    
print("Time to find number of clusters = %f sec" %(t1-t0))
print("========================================================")


if converged==False: 
    num_fracs = best_min_gg[1]
    clustering = mixture.BayesianGaussianMixture(
        n_components=num_fracs, covariance_type='full', weight_concentration_prior=1e-2,
        weight_concentration_prior_type='dirichlet_process',
        mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(3),
        init_params="random", max_iter=100, random_state=2).fit(result)
    labels = clustering.predict(result)
    
    labels = labels.reshape(len(labels), 1)
    
    data = np.append(result, labels, axis=1)

mlab.figure(bgcolor=(1,1,1), size=(1200,1200))

pltcolor = ((1, 0, 0), (0, 1, 0), (0, 0 ,1), (0.5, 0.5, 0.5))
point = []
normal = []
print(f'Using {num_fracs} clusters')
# Plot clustered image
mlab.points3d(data[:,2], data[:,1], data[:,0], data[:,3], colormap='summer', scale_mode='none') 
for i in range(num_fracs):
    tempdata = data[data[:, 3] == i]
    tempdata = tempdata[:, :3]

    temppoint, tempnormal = best_fitting_plane(tempdata, equation=False)
    
    point = np.append(point, temppoint, axis=0)
    normal = np.append(normal, tempnormal, axis=0)

    hullData = tempdata
    if imgGeom == 1:
        # Project points of fracture to respective plane
        dshift = np.dot(tempnormal, (tempdata-temppoint).T)
        vect = [j*tempnormal for j in dshift]
        p = tempdata - vect
        # Transform coplanar data to xy-plane
        fullData = coord_trans(p.T, tempnormal, [0, 0, 1])
        fullData = fullData.T
        hullData = fullData

    # Convex hull calculation
    hull = ConvexHull(hullData[:, 0:2])
    vert = hull.vertices
    vertices = p[vert, :].flatten()
    vertices = np.abs(vertices)
    vertices = (vertices-nx/2)/nx
    vertices = np.around(vertices, decimals=1)
    
    # j = 0
    # while j < len(vertices):
    #     template = template.replace('{Coordinates_templ}', ' {'+str(vertices[j])+',' + str(
    #         vertices[j+1])+','+str(vertices[j+2])+'} {Coordinates_templ}')
    #     j += 3
    # template = template.replace('{Coordinates_templ}', "\n{Coordinates_templ}")
    #polyVert = np.append(polyVert,vertices,axis=0)
    #normal = np.array(normal).reshape(numfrac[0],3)
    
    # Plot Planes
    #mlab.points3d(p[:, 2], p[:, 1], p[:, 0], color=pltcolor[i])
    
mlab.show()
