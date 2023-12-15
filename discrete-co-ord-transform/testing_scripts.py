# -*- coding: utf-8 -*-
"""
Created on Mon May  1 15:12:15 2023

@author: asanche2
"""
from discrete_coord_transform_2D import *

import numpy as np
import math
import matplotlib.pyplot as plt


a = np.array([[0,0],[1,0],[1,1]])
b = np.array([[1,1],[2,1],[2,2]])

a = a.transpose()
b = b.transpose()

zero = np.ones([1,3])

a_z = np.concatenate((a,zero),axis=0)
b_z = np.concatenate((b,zero),axis=0)

a_z_inv = np.linalg.inv(a_z)
u = np.matmul(a_z_inv,b_z)

check = np.matmul(a_z,u)


a_test = np.array([[0,0],[1,0],[1,1],[0,1]])
b_test = np.array([[2,2],[3,2],[4,4],[2,3]])
#b_test = np.array([[2,2],[2,2],[2,4],[2,3]])
tf = transform(a_test,b_test)
print(tf.map_point(0.8, 0.9))

for y in np.arange(0,1.01,0.1):
    for x in np.arange(0,1.01,0.1):
        color1 = [(0.1+0.8*x,0,0.1+0.8*y)]
        #color2 = [(0,0.1+0.8*y,0.1+0.8*x)]
        plt.scatter(x,y,color=color1,alpha = 0.3)
        xt,yt = tf.map_point(x, y)
        plt.scatter(xt,yt,color=color1,alpha = 0.3)
        
plt.show()


for y in np.arange(0,1.01,0.1):
    for x in np.arange(0,1.01,0.1):
        color1 = [(0.1+0.8*x,0,0.1+0.8*y)]
        #color2 = [(0,0.1+0.8*y,0.1+0.8*x)]
        plt.scatter(x,y,color=color1,alpha = 0.3)
        xt,yt = tf.map_point_ln(x, y)
        plt.scatter(xt,yt,color=color1,alpha = 0.3)
        
plt.show()

#Mapping x-y to spherical

def polar_mapping(x,y):
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    
    return (r,theta)

gal = []
pol = []
for y in np.arange(0,10.01,1):
    for x in np.arange(0,10.01,1):
        gal.append([x,y]);
        r,theta = polar_mapping(x,y)
        pol.append([r,theta])
 
gal_np = np.matrix(gal)
pol_np = np.matrix(pol)  
tf_pol = transform(gal_np, pol_np)

def polar_mapping(x,y):
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    
    return (r,theta)
        
        
gal = []
pol = []
for y in np.arange(-10,10.01,2):
    for x in np.arange(-10,10.01,2):
        gal.append([x,y]);
        r,theta = polar_mapping(x,y)
        pol.append([r,theta])
 
gal_np = np.matrix(gal)
pol_np = np.matrix(pol)  
tf_pol = transform(gal_np, pol_np)

fig, (ax1, ax2) = plt.subplots(1, 2)


for a in range(50):
    x = random.uniform(-10, 10);
    y = random.uniform(-10, 10);
    polars_tp = tf_pol.map_point(x, y)
    polars_tp_ln = tf_pol.map_point_ln(x, y)
    polars = polar_mapping(x,y)
    color1 = [(0.1+0.08*x,0,0.1+0.08*y)]
    
    ax1.scatter(x, y, color = 'b')
    ax2.scatter(polars_tp[0], polars_tp[1], color = 'r')
    ax2.scatter(polars_tp_ln[0], polars_tp_ln[1], color = 'g')
    ax2.scatter(polars[0], polars[1], color = 'k', marker = 'x')