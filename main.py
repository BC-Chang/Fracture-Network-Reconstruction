'''
Created on May 12, 2020

@author: Bernard Chang
'''

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

from mayavi import mlab

imgGeom = 1  # 0 for 2D, 1 for 3D
n = 2 # Initial guess for number of fractures
gg = 0.80 # Convergence criterion for adjusted r-squared gof
ggtest = 0 # Initial convergence
#fin = open('./dfnRectInputTempl.dat', 'rt')  # Open template file for dfnWorks
##template = fin.read()
#fin.close()

#fin = open('user_specified_rect_coord.dat', 'wt')  # Input file for dfnWorks

# Domain size
nx = 400
ny = 100
nz = 450

im = io.imread('./input/Segmented_SubSample1_small.tif')
#im = np.zeros([100, 100, 100])
#im[:, 49:52, :] = 255
#im[:, :, 49:52] = 255
result = np.where(im == 255)
result = np.asarray(result).transpose()



x = result[:, 2]
y = result[:, 1]

if imgGeom == 1:
    z = result[:, 0]

elif imgGeom == 0:
    z = np.zeros([len(x), 1])
    result = np.append(result, z, axis=1)
    
t0 = time.time()

# clustering = KMeans(n_clusters=3,n_init=10).fit(result)
# labels = clustering.labels_

# clustering = SpectralClustering(n_clusters = 3, assign_labels="discretize", n_jobs=-1).fit(result)
# labels = clustering.labels_

# clustering = DBSCAN(eps=0.5,min_samples=3).fit(result)
# labels = clustering.labels_

# clustering = AgglomerativeClustering(n_clusters=3,linkage="complete").fit(result)
# labels = clustering.labels_

# clustering = MeanShift(bandwidth=1).fit(result)
# labels = clustering.labels_

# clustering = mixture.GaussianMixture(n_components=3,covariance_type='full',max_iter=100,init_params='random').fit(result)
# labels = clustering.predict(result)

clustering = mixture.BayesianGaussianMixture(
    n_components=n, covariance_type='full', weight_concentration_prior=1e-2,
    weight_concentration_prior_type='dirichlet_process',
    mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(3),
    init_params="random", max_iter=100, random_state=2).fit(result)
labels = clustering.predict(result)
    
    

# clustering = mixture.BayesianGaussianMixture(
#   n_components=3, covariance_type='full', weight_concentration_prior=1e+2,
#    weight_concentration_prior_type='dirichlet_process',
#    mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(3),
#    init_params="kmeans", max_iter=300, random_state=2).fit(result)
# labels = clustering.predict(result)
    

t1 = time.time()
print("Time to find clusters = %f sec" %(t1-t0))

labels = labels.reshape(len(labels), 1)
data = np.append(result, labels, axis=1)

# Create meshgrid for plotting
X, Y = np.meshgrid(np.linspace(0, 100, 20), np.linspace(0, 100, 20))

numfrac = max(labels)+1
#template = template.replace('{nEllipses_templ}', str(numfrac[0]))
#template = template.replace('{nNodes_templ}', str(4))
point = []
normal = []
polyVert = []
mlab.figure(bgcolor=(1,1,1), size=(1200,1200))
mlab.points3d(x, y, z)
mlab.figure(bgcolor=(1,1,1), size=(1200,1200))
pltcolor = ((1, 0, 0), (0, 1, 0), (0, 0 ,1), (0.5, 0.5, 0.5))

for i in range(numfrac[0]):
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
    r2, adjr2 = rsquared(tempdata[:,0], ZPred, len(tempdata[:,0]), 3, verticalFracture)
    RMSE = rmse(tempdata[:,0], ZPred, verticalFracture)
    print("-----------------------------")
    print("R-squared of cluster %d: %f" % (i+1, r2))
    print("Adjusted R-squared of cluster %d: %f" %(i+1, adjr2))
    print("RMSE of cluster %d: %f" %(i+1, RMSE))
    print("-----------------------------")
    
    
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
    mlab.points3d(p[:, 2], p[:, 1], p[:, 0], color=pltcolor[i])


mlab.show()

#template = template.replace('{Coordinates_templ}', "")


#fin.write(template)
#fin.close()
