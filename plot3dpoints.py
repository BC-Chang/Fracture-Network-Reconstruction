'''
Created on May 14, 2020

@author: Bernard Chang
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN,AgglomerativeClustering, MeanShift
from skimage import data, io, color
import time
from sklearn import mixture
from getPCA import *
from best_fitting_plane import *
from coordTrans import *
from scipy.spatial import ConvexHull
from collections import OrderedDict
from xlrd.formula import num2strg

im = io.imread('45_1_Fracture_nxyz150_1.tif')

result  = np.where(im == 255)
result = np.asarray(result).transpose()


x = result[:,0]
y = result[:,1]
z = result[:,2]

fig = plt.figure()
ax = plt.axes(projection='3d')

#clustering = KMeans(n_clusters=3,n_init=10).fit(result)
#labels = clustering.labels_

#clustering = SpectralClustering(n_clusters = 3, assign_labels="discretize", n_jobs=-1).fit(result)
#labels = clustering.labels_

#clustering = DBSCAN(eps=0.5,min_samples=3).fit(result)
#labels = clustering.labels_

#clustering = AgglomerativeClustering(n_clusters=3,linkage="complete").fit(result)
#labels = clustering.labels_

#clustering = MeanShift(bandwidth=1).fit(result)
#labels = clustering.labels_

#clustering = mixture.GaussianMixture(n_components=3,covariance_type='full',max_iter=100,init_params='kmeans').fit(result)
#labels = clustering.predict(result)

clustering = mixture.BayesianGaussianMixture(
    n_components=3, covariance_type='full', weight_concentration_prior=1e-2,
    weight_concentration_prior_type='dirichlet_process',
    mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(3),
    init_params="random", max_iter=100, random_state=2).fit(result)
labels = clustering.predict(result)  

#clustering = mixture.BayesianGaussianMixture(
#   n_components=3, covariance_type='full', weight_concentration_prior=1e+2,
#    weight_concentration_prior_type='dirichlet_process',
#    mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(3),
#    init_params="kmeans", max_iter=300, random_state=2).fit(result)
#labels = clustering.predict(result)


#ax.plot(x,y,z,'o') #,c=labels.astype(float))
ax.scatter3D(x,y,z,c=labels.astype(float),alpha=0.5)
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#ax.view_init(elev=35,azim = 0)
ax.grid(False)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.axis('on')
#ax.set_zlim3d(0,100)
plt.show()
plt.show()