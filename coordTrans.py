'''
Created on May 13, 2020

@author: Bernard Chang

Perform coordinate transform to arbitrary plane
v is the normal vector of the input plane
k is the normal vector of the output plane
'''
import numpy as np
import math

def coordTrans(data,v, k):
    mag1 = np.linalg.norm(v)
    mag2 = np.linalg.norm(k)
    
    theta = math.acos(np.dot(v,k)/(mag1*mag2))
    
    u = np.cross(v,k)/(mag1*mag2)
    
       
   
    # rotate about x
    x = u[0]
    y = u[1]
    z = u[2]
    rotmat = lambda l,m,n,angle: [[l*l*(1-math.cos(angle))+math.cos(angle), m*l*(1-math.cos(angle))-n*math.sin(angle), n*l*(1-math.cos(angle))+m*math.sin(angle)],
                            [l*m*(1-math.cos(angle))+n*math.sin(angle), m*m*(1-math.cos(angle))+math.cos(angle), n*m*(1-math.cos(angle))-l*math.sin(angle)],
                            [l*n*(1-math.cos(angle))-m*math.sin(angle), m*n*(1-math.cos(angle))+l*math.sin(angle), n*n*(1-math.cos(angle))+math.cos(angle)]]
     

    rotData = np.asarray(rotmat(x,y,z,theta)).dot(data)
    
    return rotData
