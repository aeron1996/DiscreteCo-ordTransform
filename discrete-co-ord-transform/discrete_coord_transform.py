# -*- coding: utf-8 -*-
"""
Created on Mon May  1 15:09:20 2023

@author: asanche2
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import random

class Grid:
    
    def __init__(self,dimensions,mins: np.ndarray,sizes: np.ndarray,n_cells: np.ndarray):
        self.dimensions = int(dimensions)
        self.mins = mins
        self.sizes = sizes
        self.n_cells = n_cells.astype(int)
        self.maxs = n_cells*sizes + mins
        
    def cell_id(self,loc):
        if(loc.size != self.dimensions):
            raise RuntimeError("location entered differnet number of dimensions to grid")
        
        indices = np.floor((loc-self.mins)/self.sizes)
        
        #if any of the cells are on the upper bound set equal to one cell index lower
        #can replace with parallel function later
        for i in range(self.dimensions):
            if(loc[i] == self.maxs[i]):
                indices[i] = math.floor((loc[i]-self.mins[i])/self.sizes[i])
                
        cell_id = self.cell_id_index(indices)
        return cell_id
    
    def cell_id_array(self,loc_i):
        if(loc_i.size != self.dimensions):
            raise RuntimeError("location entered differnet number of dimensions to grid")
        
        indices = np.floor((loc_i-self.mins)/self.sizes)
        
        #if any of the cells are on the upper bound set equal to one cell index lower
        #can replace with parallel function later
        for i in range(self.dimensions):
            if(loc_i[i] == self.maxs[i]):
                indices[i] = math.floor((loc_i[i]-self.mins[i])/self.sizes[i])
                
        return indices.astype(int)
    
    def cell_id_index(self,loc_i):
        return tuple(loc_i.astype(int))
    
    def point_id_index(self,loc_i):
        return tuple(loc_i.astype(int))
    
    def point_id(self,loc):
        loc_i = (loc-self.mins)/self.sizes
        return self.point_id_index(loc_i)
    
    def get_cell_edges(self,loc):
        
        if(loc.size != self.dimensions):
            raise RuntimeError("location entered differnet number of dimensions to grid")
        
        indices = np.floor((loc-self.mins)/self.sizes)
        
        #if any of the cells are on the upper bound set equal to one cell index lower
        #can replace with parallel function later
        for i in range(self.dimensions):
            if(loc[i] == self.maxs[i]):
                indices[i] = math.floor((loc[i]-self.mins[i])/self.sizes[i])
        
        cell_mins = indices*self.sizes + self.mins
        cell_maxs = (indices+1)*self.sizes + self.mins
        
        return (cell_mins,cell_maxs)
    
    def get_cell_edges_i(self,loc_i):
        
        indices = np.array(loc_i)
        
        cell_mins = indices*self.sizes + self.mins
        cell_maxs = (indices+1)*self.sizes + self.mins
        
        return (cell_mins,cell_maxs)
    
    def get_cell_points_list(self,loc_i):
        point_list = []
        
        cell_mins, cell_maxs = self.get_cell_edges(loc_i)
        
        to_be_per = np.row_stack((cell_mins,cell_maxs))
        already_per = np.array([])
        
        point_list = self.get_permutations(to_be_per, already_per)
        return point_list
    
    def get_cell_points_id_list(self,loc_i):
        point_list = []
        
        cell_mins = loc_i
        cell_maxs = loc_i+1
        
        to_be_per = np.row_stack((cell_mins,cell_maxs))
        already_per = np.array([])
        
        point_list = self.get_permutations(to_be_per, already_per)
        return point_list
        
    #recursion to get iterations of a set of mins and maxes
    def get_permutations(self,to_be_per,already_per):
        point_list = []
        if(to_be_per.size == 2):
            for d_p in to_be_per[:,0]:
                point_list.append(np.append(already_per,d_p))
            
        else:
            for d_p in to_be_per[:,0]:
                perm = self.get_permutations(to_be_per[:,1:],np.append(already_per,d_p))
                point_list = point_list+perm
                
        return point_list
    
    def get_ratios(self,loc):
        
        edge_mins, edge_maxs = self.get_cell_edges(loc)
        fractions = (edge_maxs-loc)/(edge_maxs-edge_mins)
        
        return fractions
    
class transform:
    def __init__(self,dimensions:int , values0, values1):
        
        self.dimensions = dimensions
        if(values0.shape != values1.shape):
            raise Exception("Both input grids must be of same size")
        cols0,rows0 = values0.shape
        
        if(cols0 == rows0):
            print("both rows and columns contain same number of values, make sure matrices are of correct orrientation")
        
        
        if(cols0 == dimensions):
            if(not math.pow(rows0,1/dimensions).is_integer()):
                raise Exception("number of points does not create a cube for the dimensions input")
            
            self.values0 = values0
            self.values1 = values1
            self.points = rows0
            
        elif(rows0 == dimensions):
            
            self.values0 = values0.transpose()
            self.values1 = values1.transpose()
            self.points = cols0
            
        else:
            raise Exception("array inuts must be size dimensions x N")
            
        self.grid = self.create_grid(self.values0)
        self.index_map = self.create_index_map()
        self.mapping = self.create_mapping()
        
    def create_grid(self,values0):
        mins = np.amin(values0, axis = 1)
        maxs = np.amax(values0, axis = 1)
        numbers = np.zeros([self.dimensions,]).astype(int)
        for d in range(self.dimensions):
            numbers[d] = int(np.unique(values0[d,:]).size-1)
        sizes = (maxs-mins)/numbers
        
        return Grid(self.dimensions, mins, sizes, numbers)
    
    def create_index_map(self):
        
        index_map = {}
        for i in range(self.points):
            point = self.values0[:,i]
            key = self.grid.cell_id(point)
            index_map[key] = i
            
        return index_map
    
    def create_mapping(self):
        
        transform_map = {}
        current_loci = np.array([])
        
        self.loop_dimensions(current_loci,0,self.grid.n_cells,transform_map)
        return transform_map
        
        #add recursional function to loop over each dimension cell to loop over every cell 
            
    def loop_dimensions(self,current_loci, current_dimension, remaining_dimension_n_cells, transform_map):
        
        if(remaining_dimension_n_cells.size == 1):
            for j in range(int(remaining_dimension_n_cells[0])):
                loci = np.append(current_loci,j)
                key = tuple(loci)
                transform_map[key] = self.get_transform_map(loci)
                
        else:
            for i in range(int(remaining_dimension_n_cells[0])):
                loci = np.append(current_loci,i)
                self.loop_dimensions(loci, current_dimension+1, remaining_dimension_n_cells[1:], transform_map)
                
        
                
        
    def get_transform_map(self,loc_i):
        t_map = {}
        n_points = int(math.pow(2,self.dimensions))
        points = self.grid.get_cell_points_list(loc_i)#speed up could be to get this from the already calulated index points
        points_i = self.grid.get_cell_points_id_list(loc_i)
        if(n_points != len(points) or n_points != len(points_i)):
            raise Exception("Incorrect number of points in points list")
        points = np.array(points)
        points_i = np.array(points_i)
        
        #get initial and fina points from finals list
        points_initial = []
        points_final = []
        for i in range(n_points):
            p = points_i[i,:]
            key = self.grid.point_id_index(p)
            index = self.index_map[key]
            points_initial.append(self.values0[:,index])
            points_final.append(self.values1[:,index])
            
        points_initial = np.array(points_initial)
        points_final = np.array(points_final)
            
        
        #get the points that are part of this
        for i in range(n_points):
            #find points for the simplex of a given point
            p = points_i[i,:]
            p_m = np.tile(p,(n_points,1))
            p_adjacent = np.sum(np.abs(points_i-p),axis=1)
            #selected_p = points[p_adjacent <= 1,:]
            selected_p = points_initial[p_adjacent <= 1,:]
            selected_p_final = points_final[p_adjacent <= 1,:]
            if(selected_p.shape[0] != self.dimensions+1):
                raise Exception("Points selected is of incorrect number")
            #convert into correct matrix form
            ones = np.ones([1,self.dimensions+1])
            initial_tr_mat = np.row_stack((selected_p.transpose(),ones))
            initial_tr_mat_inv = np.linalg.inv(initial_tr_mat)
            
            final_tr_mat = np.row_stack((selected_p_final.transpose(),ones))
            transformation_matrix = np.matmul(final_tr_mat,initial_tr_mat_inv)
            
            p_key = self.grid.point_id_index(p)
            t_map[p_key] = transformation_matrix
            
            
        return t_map
    
    def map_point(self,loc):
        
        loc_array = np.array(loc)
        if(np.shape(loc_array) != (self.dimensions,)):
            raise Exception("input array is not of correct number of dimensions")
        cell_id = self.grid.cell_id(loc_array)
        cell_loci = np.array(cell_id)
        mins,maxs = self.grid.get_cell_edges(loc_array)
        #get which point it is nearest
        
        transforms = self.mapping[cell_id]
        
        final_tr_mat = np.zeros((self.dimensions+1,self.dimensions+1))
        for point_id,transform in transforms.items():
            
            point_array = np.array(point_id)
            within_cell_loc = point_array-np.array(cell_id)
            point_loc = self.grid.point_id(point_array)
            
            frac = (maxs-loc_array)/(self.grid.sizes)
            
            a = np.abs(frac-within_cell_loc)
            mul = np.prod(a)
            final_tr_mat = final_tr_mat + mul*transform
         
        
        point_v = np.append(loc_array,1)
        point_v_trans = np.matmul(final_tr_mat,point_v)
        return point_v_trans[:-1]
            
        
                    
                
                
         
    def get_point_loc_from_within_cell_id(self,point_id,cell_id):
        mins,maxs = self.grid.get_cell_edges_i(cell_id)
        point_array = np.array(point_id)
        point_array_values = np.empty(point_array.size)
        for i in range(point_array.size):
            if(point_array[i] == 0):
                point_array_values[i] = mins[i]
            elif(point_array[i] == 1):
                 point_array_values[i] = maxs[i]
            else:
                raise Exception("All points should be 0 or 1")
        
        return point_array_values;
            
        
        
#2d points 
'''
a_test = np.array([[0,0],[1,0],[1,1],[0,1],[0,2],[1,2]])
b_test = np.array([[2,2],[4,2],[3,3],[2,3],[2,4],[4,4]])
tf = transform(2,a_test,b_test)

b = tf.map_point([0.5,1.5])

for y in np.arange(0,2,0.1):
    for x in np.arange(0,1,0.1):
        color1 = [(0.1+0.8*x,0,0.1+0.3*y)]
        #color2 = [(0,0.1+0.8*y,0.1+0.8*x)]
        plt.scatter(x,y,color=color1,alpha = 0.3)
        xt,yt = tf.map_point([x,y])
        plt.scatter(xt,yt,color=color1,alpha = 0.3)
        
plt.show()
'''
#3d points
a_test = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
b_test = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])+2

tf_3D = transform(3,a_test,b_test)
a = tf_3D.map_point([0.5,0.5,0.5])
