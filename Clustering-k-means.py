'''
Created on May 12, 2020

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


imgGeom = 1 #0 for 2D, 1 for 3D
fin = open('./dfnRectInputTempl.dat','rt') #Open template file for dfnWorks
template = fin.read()
fin.close()

fin=open('user_specified_rect_coord.dat','wt') #Input file for dfnWorks

# Domain size
nx = 100
ny = 100
nz = 100

#im = io.imread('Fractures_May12.tif')
im = np.zeros([100,100,100])
im[:,49:52,:]=255
im[:,:,49:52]=255
result  = np.where(im == 255)
result = np.asarray(result).transpose()


x = result[:,0]
y = result[:,1]

if imgGeom == 1:
    z = result[:,2]

elif imgGeom == 0:
    z = np.zeros([len(x),1])
    result = np.append(result,z,axis=1)

fig = plt.figure()
ax = plt.axes(projection='3d')
t0 = time.time()

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

#clustering = mixture.GaussianMixture(n_components=3,covariance_type='full',max_iter=100,init_params='random').fit(result)
#labels = clustering.predict(result)

clustering = mixture.BayesianGaussianMixture(
    n_components=2, covariance_type='full', weight_concentration_prior=1e-2,
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

t1 = time.time()
#ax.plot(x,y,z,c=labels.astype(float))
#ax.scatter3D(z,y,x,c=labels.astype(float),alpha=0.5)
#plt.plot()
print("Time to find clusters = " + num2strg(t1-t0))


labels = labels.reshape(len(labels),1)
data = np.append(result,labels,axis=1)

# Create meshgrid for plotting
X,Y = np.meshgrid(np.linspace(0,100,20), np.linspace(0,100,20)) 

numfrac = max(labels)+1
template = template.replace('{nEllipses_templ}',str(numfrac[0]))
template = template.replace('{nNodes_templ}',str(4))
point = []
normal = []
polyVert = []

for i in range(numfrac[0]):
    tempdata = data[data[:,3]==i]
    tempdata = tempdata[:,:3]
    
    temppoint,tempnormal = best_fitting_plane(tempdata,equation=False)
    #a,b,c,d = best_fitting_plane(tempdata,equation=True)
    if tempnormal[2] == 0:
        tempnormal[2] = 0.01
        
    d = -np.dot(temppoint,tempnormal.T)
    Z = (-tempnormal[0]*X - tempnormal[1]*Y - d)/tempnormal[2]
        
    point = np.append(point,temppoint,axis=0)
    normal = np.append(normal,tempnormal,axis=0)
    
    hullData = tempdata
    if imgGeom == 1:
        #Project points of fracture to respective plane
        dshift = np.dot(tempnormal,(tempdata-temppoint).T)
        vect = [j*tempnormal for j in dshift]
        p = tempdata - vect
        # Transform coplanar data to xy-plane
        fullData = coordTrans(p.T,tempnormal,[0,0,1])
        fullData = fullData.T
        hullData = fullData
            
    # Convex hull calculation
    hull = ConvexHull(hullData[:,0:2])
    vert = hull.vertices
    vertices = p[vert,:].flatten()
    vertices = np.abs(vertices)
    vertices = (vertices-nx/2)/nx
    vertices = np.around(vertices,decimals = 1)
    print(vertices)
    j = 0
    while j < len(vertices):
        template = template.replace('{Coordinates_templ}',' {'+str(vertices[j])+','+ str(vertices[j+1])+','+str(vertices[j+2])+'} {Coordinates_templ}')
        j += 3
    template = template.replace('{Coordinates_templ}',"\n{Coordinates_templ}")   
    #polyVert = np.append(polyVert,vertices,axis=0)
    #normal = np.array(normal).reshape(numfrac[0],3)

    #ax.plot_surface(X,Y,-Z, rstride=1, cstride=1, alpha = 0.5)
    

    #ax.scatter3D(tempdata[:,0], tempdata[:,1],tempdata[:,2],alpha=0.5)
    #ax.plot(p[:,0], p[:,2],-p[:,1] ,'o')
    #ax.plot(p[vert,0], p[vert,2], -p[vert,1], 'k--', lw=6)
    #ax.plot(p[vert,0], p[vert,2], -p[vert,1], 'ro', lw=6)
    ax.plot(p[:,2], p[:,1],p[:,0] ,'o')
    ax.plot(p[vert,2], p[vert,1], p[vert,0], 'k--', lw=6)
    ax.plot(p[vert,2], p[vert,1], p[vert,0], 'ro', lw=6)

#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
ax.view_init(elev=40,azim = 300)
ax.grid(False)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.axis('on')
#ax.set_zlim3d(0,100)
plt.show()


template = template.replace('{Coordinates_templ}',"")


fin.write(template)
fin.close()
