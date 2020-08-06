'''
Created on May 12, 2020

@author: Bernard Chang

Computes the best fit plane of the given points

Inputs:
-----------------------------------------------------------------------------
points: array
    x,y,z coordinates
equation(optional): bool
    Set the output plane format:
        If True, return the coefficients (a,b,c,d) of the plane
        If False (Default) return 1 point and 1 normal vector

Outputs:
-----------------------------------------------------------------------------
a,b,c,d: float
    Coefficients solving the plane equation

or

point, normal: array
    The plane defined by 1 point and 1 normal vector
'''

from getPCA import *
import numpy as np


def best_fitting_plane(points, equation=False):

    w, v = get_PCA(points)

    # Normal of the plane is the last eigenvector
    normal = v[:, 2]

    # get a point on the plane
    point = np.mean(points, axis=0)
    

    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d
    else:
        return point, normal
