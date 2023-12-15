# -*- coding: utf-8 -*-
"""
Created on Mon May  8 10:58:42 2023

@author: asanche2
"""

import numpy as np
import math
import scipy.spatial as spat
import matplotlib.pyplot as plt


class transform:
    def __init__(self, dimensions:int, values0, values1):
        self.dimensions = dimensions
        if(values0.shape != values1.shape):
            raise Exception("Both input grids must be of same size")
        cols0,rows0 = values0.shape
        
        if(cols0 == rows0):
            print("both rows and columns contain same number of values, make sure matrices are of correct orrientation")
        
        
        if(cols0 == dimensions):
            
            self.values0 = values0.transpose()
            self.values1 = values1.transpose()
            self.points = cols0
            
        elif(rows0 == dimensions):
            
            self.values0 = values0
            self.values1 = values1
            self.points = rows0
            
        else:
            raise Exception("array inuts must be size dimensions x N")
            
        self.delaunay = self.get_simplexes()
        
        self.mapping = self.create_mapping()
        
        
    def get_simplexes(self):
        simp = spat.Delaunay(self.values0)
        return simp
    
    def create_mapping(self):
        #choose weather to do lazy implementation
        #or calculate all transformations before hand
        return {}
        
    def get_simplex(self, loc):
        
        #loop over simplexes
        
        #find simplex where point is within max and mins
        #check to see if point is within simplex
        #if point is within simplex return simplex
        
        #throw error if simplex is never found
        a =1
        
    def map_point(self, loc):
        #find simplex point is within
        simp_ind = self.find_simp(loc)
        simp_key = tuple(simp_ind)
        #if map already contains simplex look up transformation
        
        
        
        if(simp_key in self.mapping):
            transform = self.mapping.get(simp_key)

        else: #otherwise calculate transform
            ones = np.ones([1,self.dimensions+1])
            simp0 = self.values0[simp_ind,:]
            mat0 = np.row_stack((simp0.transpose(),ones))
            simp1 = self.values1[simp_ind,:]
            mat1 = np.row_stack((simp1.transpose(),ones))
            mat0Inv = np.linalg.inv(mat0)
            
            
            transform = mat1@mat0Inv
            self.mapping[simp_key] = transform
        
        loc_v = np.append(loc,1)
        loc_t = transform@loc_v
            
        
        return loc_t[:-1]
    
    def find_simp(self,loc):
        simp_points = self.values0[self.delaunay.simplices]
        simp_ind = self.delaunay.simplices
        for i in range(simp_points.shape[2]):
            #run quick check to see fi point is within shape bounds
            #simp already contains max/min_bounds
            simp_p = simp_points[i,:,:]
            maxs = np.max(simp_p,axis = 0)
            mins = np.min(simp_p,axis = 0)
            within_bounds = np.all(np.greater_equal(loc,mins) & np.greater_equal(maxs,loc))
            if(within_bounds):
                #check if is point is within shape         
                if(is_within(loc,simp_p)):
                    a = simp_ind[i,:]
                    return simp_ind[i,:].transpose()
                
        #point not within list of simplices
        
        min_dist = math.inf
        min_simp_id = -1
        #find closest simplex
        for i in range(simp_points.shape[2]):
            simp_centre = np.sum(simp_points,axis=0)/(self.dimensions+1)
            distance = np.sqrt(np.sum((simp_centre - loc) ** 2))
            if(distance<min_dist):
                min_dist = distance
                min_simp_id = i
          
        return simp_ind[min_simp_id,:].transpose()
        
                

def get_facets(points):
    n,d = points.shape
    facets = np.zeros((n,n-1,d))
    for i in range(n):
        facets[i,:,:] = np.delete(points, (i), axis=0)
        
    return facets

def is_within(point,simplex):
    facets = get_facets(simplex)
    n_facets,dim,ign =facets.shape
    a = np.random.rand(dim,1)
    #a = np.array([[0.32],[0.64]])
    b = np.array(point)
    C = np.empty((dim,n_facets))
    C[:] = np.nan
    count = 0
    for i in range(n_facets):
        facet = facets[i,:,:]
        S = (facet[1:,:]-facet[0,:]).transpose()
        N = external_product(S)
        c = -facet[0,:]@N
        t = (-c-N.transpose()@b)/(N.transpose()@a)
        C[:,i] = (a*t + b.reshape(dim,1)).ravel()
        SI = np.concatenate((np.concatenate((np.zeros((dim,1)),np.eye(dim)),axis=1),np.ones((1,dim+1))),axis=0)
        IM = np.concatenate((np.concatenate((facet.transpose(),N.reshape(dim,1)),axis=1), np.ones((1,dim+1))), axis=0)
        if(np.linalg.det(IM) ==0):
            facet = np.flip(facet,axis=0)
            S = (facet[1:,:]-facet[0,:]).transpose()
            N = external_product(S)
            c = -facet[0,:]@N
            t = (-c-N.transpose()@b)/(N.transpose()@a)
            C[:,i] = (a*t + b.reshape(dim,1)).ravel()
            SI = np.concatenate((np.concatenate((np.zeros((dim,1)),np.eye(dim)),axis=1),np.ones((1,dim+1))),axis=0)
            IM = np.concatenate((np.concatenate((facet.transpose(),N.reshape(dim,1)),axis=1), np.ones((1,dim+1))), axis=0)
            
        M = SI@np.linalg.inv(IM)
        temp = M@np.concatenate((C[:,i],[1]))
        cp = temp[0:dim-1]
        
        #if( np.all(cp>=0) and np.sum(cp)<=1 and t>0 and not(np.any(np.sum(np.abs(C[:,0:i-1]-C[:,i]))==0))):
        if( np.all(cp>=0) and np.sum(cp)<=1 and t>0):
            count+=1
            
    return count%2==1

def external_product(A):
    n,cols = A.shape
    if(n != cols+1):
        raise Exception("external_product: A n by n-1 matrix is required.")
        
    w = np.zeros(n)
    indices = np.arange(n)
    for i in range(n):
        w[i] = math.pow(-1,i)*np.linalg.det(A[indices != i,:])
    
    return w


#a_test = np.array([[0,0],[1,0],[1,1],[0,1],[0,2],[1,2]])
#b_test = np.array([[2,2],[4,2],[3,3],[2,3],[2,4],[4,4]])
a_test = np.array([[0,0],[1,0],[1,0.5],[0,0.5]])
b_test = np.array([[2,2],[4,2],[3,3],[2,3]])
tf = transform(2,a_test,b_test)
p = tf.map_point([0.2,0.2])


a_test = np.array([[0,0],[1,0],[1,1],[0,1]])
b_test = np.array([[2,2],[3,2],[4,4],[2,3]])
#b_test = np.array([[2,2],[2,2],[2,4],[2,3]])
tf = transform(2,a_test,b_test)
print(tf.map_point([0.8, 0.9]))

for y in np.arange(0,1.01,0.1):
    for x in np.arange(0,1.01,0.1):
        color1 = [(0.1+0.8*x,0,0.1+0.8*y)]
        #color2 = [(0,0.1+0.8*y,0.1+0.8*x)]
        plt.scatter(x,y,color=color1,alpha = 0.3)
        xt,yt = tf.map_point([x, y])
        plt.scatter(xt,yt,color=color1,alpha = 0.3)
        
plt.show()