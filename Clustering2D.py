'''
Created on May 13, 2020

@author: Bernard Chang
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN,AgglomerativeClustering, MeanShift
from skimage import data, io, color
from sklearn import mixture

im = io.imread('3Fractures.tif')
result  = np.where(im == 255)
result = np.asarray(result).transpose()
x = result[:,0]
y = result[:,1]
fig = plt.figure()
ax = plt.axes()

#clustering = KMeans(n_clusters=3,n_init=100).fit(result)
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

#clustering = mixture.BayesianGaussianMixture(
#    n_components=3, covariance_type='full', weight_concentration_prior=1e-2,
#    weight_concentration_prior_type='dirichlet_process',
#    mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(2),
#    init_params="random", max_iter=100, random_state=2).fit(result)
#labels = clustering.predict(result)  

#clustering = mixture.BayesianGaussianMixture(
#   n_components=3, covariance_type='full', weight_concentration_prior=1e+2,
#    weight_concentration_prior_type='dirichlet_process',
#    mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(3),
#    init_params="kmeans", max_iter=300, random_state=2).fit(result)
#labels = clustering.predict(result)

plt.scatter(y, -x, c=labels.astype(float),alpha=0.5)
ax.set_aspect(aspect=1)
ax.set_xticks([])
ax.set_yticks([])
plt.show()