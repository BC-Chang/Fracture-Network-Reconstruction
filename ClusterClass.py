# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 18:53:52 2020

@author: bchan
"""

import numpy as np

class Cluster:
    def __init__(self, cluster_num, data, normal, point, adjr2):
        self.cluster_num = cluster_num
        self.data = data
        self.normal = normal
        self.point = point
        self.adjr2 = adjr2
        