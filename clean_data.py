# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 16:56:01 2020

@author: bchan
"""

import numpy as np
from scipy.ndimage import label
from skimage import io
import mayavi.mlab as mlab


def clean_data3D(data, connectivity=6, max_element_size=100):
    if connectivity == 6:
        str_3d = np.array([[[0, 0, 0],[0, 1, 0], [0, 0, 0]],
                           [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=np.uint8)
    elif connectivity == 18:
        str_3d = np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                           [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                           [[0, 1, 0], [1, 1, 1], [0, 1, 0]]], dtype=np.uint8)
    elif connectivity == 27:
        str_3d = np.ones((3,3,3), dtype=np.uint8)
    else:
        raise Exception(f"Connectivity of {connectivity} is invalid. Choose a connectivity of 6, 18, or 28")
    
    labels = label(data, structure=str_3d)
    
    for label_num in range(labels[1]+1):
        num_elements = np.count_nonzero(labels[0]==label_num)
        #print(f'Size of label {label_num} = {num_elements}')
        if num_elements < max_element_size:
            # Remove elements if size is less than max_element_size
            data = np.where(labels[0]==label_num, 0, data) 
        
                
    
    return data, labels
