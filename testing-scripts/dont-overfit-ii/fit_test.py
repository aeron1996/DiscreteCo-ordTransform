# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:04:48 2023

@author: asanche2
"""

import pandas as pd
import numpy as np
from coord_transform_delaunay import transform

train_data = pd.read_csv("D:/discrete co-ord transform/testing-scripts/dont-overfit-ii/test.csv")



n_dim = train_data.shape[1]-2
n_points = train_data.shape[0]

train_data_mat = []

for i in range(300):
    col = np.array(train_data[str(i)])
    if(i==0):
        train_data_mat = col
    else:
        train_data_mat = np.vstack([train_data_mat,col])
        
target = train_data['target']
target = np.vstack([target,np.ones([n_dim-1,n_points])])

tf = transform(n_dim,train_data_mat,target)