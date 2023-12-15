# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:52:26 2023

@author: asanche2
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import random


class Grid:
    
    def __init__(self,x_min,y_min,x_size,y_size,x_number,y_number):
        
        self.x_min = x_min
        self.y_min = y_min
        self.x_size = x_size
        self.y_size = y_size
        self.x_n_cells = x_number
        self.y_n_cells = y_number
        self.x_max = x_number*x_size + x_min
        self.y_max = y_number*y_size + y_min
        
    def cell_id(self,x,y):
        x_number = self.x_n_cells
       
        x_i = math.floor((x-self.x_min)/self.x_size)
        y_i = math.floor((y-self.y_min)/self.y_size)
        
        if(x == self.x_max):
            x_i = math.floor((x-self.x_min)/self.x_size)-1
            
        if(y == self.y_max):
            y_i = math.floor((y-self.y_min)/self.y_size)-1
            
        cell_id = y_i*x_number + x_i
        return cell_id
    
    def cell_id_index(self,i,j):
        cell_id = j*(self.x_n_cells) + i
        return cell_id
    
    def point_id(self,i,j):
        point_id = j*(self.x_n_cells+1) + i
        return point_id
    
    def tri_id(self,x,y):
        
        edges = self.get_cell_edges(x,y)
        m = (edges['y_max']-edges['y_min'])/(edges['x_max']-edges['x_min'])
        y1 = m*(x-edges['x_min']) + edges['y_min']
        y2 = -m*(x-edges['x_max']) + edges['y_max']
        
        if(y<=y1 and y<=y2):
            sector_id = 0
        
        elif(y>y1 and y<=y2):
            sector_id = 1
            
        elif(y>y1 and y>y2):
            sector_id = 2
            
        else:
            sector_id = 3
            
        tri_id = self.cell_id(x,y)*4 + sector_id
        
    def get_quartile(self,x,y):
        edges = self.get_cell_edges(x,y)
        m = (edges['y_max']-edges['y_min'])/(edges['x_max']-edges['x_min'])
        y1 = m*(x-edges['x_min']) + edges['y_min']
        y2 = -m*(x-edges['x_min']) + edges['y_max']
        
        if(x == edges['x_max']):
            sector_id = 3
            
        elif(y == edges['y_max']):
            sector_id = 2
        
        elif(y<=y1 and y<=y2):
            sector_id = 0
        
        elif(y>y1 and y<=y2):
            sector_id = 1
            
        elif(y>y1 and y>y2):
            sector_id = 2
            
        else:
            sector_id = 3
            
        return sector_id
            
    def get_cell_edges(self,x,y):
        
        x_i = math.floor((x-self.x_min)/self.x_size)
        y_i = math.floor((y-self.y_min)/self.y_size)
        
        if(x == self.x_max):
            x_i = math.floor((x-self.x_min)/self.x_size)-1
            
        if(y == self.y_max):
            y_i = math.floor((y-self.y_min)/self.y_size)-1
        
        cell_x_min = x_i*self.x_size + self.x_min
        cell_x_max = (x_i+1)*self.x_size + self.x_min
        
        cell_y_min = y_i*self.y_size + self.y_min
        cell_y_max = (y_i+1)*self.y_size + self.y_min
        
        return {"x_min":cell_x_min,
                "x_max":cell_x_max,
                "y_min":cell_y_min,
                "y_max":cell_y_max
                }
    
    def get_cell_points(x,y):
        return [(x,y),(x,y+1),(x+1,y+1),(x+1,y)]
    
        
class transform:
    
    def __init__(self,values0,values1):
            
        if(values0.shape != values1.shape):
            raise Exception("Both input grids must be of same size")
        cols0,rows0 = values0.shape
        
        if(cols0 == 2):
            self.values0 = values0
            self.values1 = values1
        elif(rows0 == 2):
            self.values0 = values0.transpose()
            self.values1 = values1.transpose()
        else:
            raise Exception("array inuts must be size 2xN")
            
        self.grid = self.create_grid(self.values0)
        self.index_list = self.create_index()
        self.mapping = self.create_mapping()
        
    def create_grid(self,values0):
        x_values = values0[0,:]
        y_values = values0[1,:]
        
        x_min = x_values.min()
        y_min = y_values.min()
        x_number = np.unique(x_values.tolist()).size-1
        y_number = np.unique(x_values.tolist()).size-1
        x_size = (x_values.max() - x_min)/(x_number)
        y_size = (y_values.max() - y_min)/(y_number)
        
        return Grid(x_min,y_min,x_size,y_size,x_number,y_number)
    
        
    def create_index(self):
        #mapp the values of the point grid id to the index in input values
        N_points = ((self.grid.x_n_cells+1)*(self.grid.y_n_cells+1))
        index_list = [None]*N_points
        c = 0
        for j in range(self.grid.y_n_cells+1):
            for i in range(self.grid.x_n_cells+1):
                '''
                cell_edges = grid.get_cell_edges(i, j)
                x = cell
                pairs = [
                    (cell_edges['x_min'],cell_edges['y_min']),
                    (cell_edges['x_min'],cell_edges['y_max']),
                    (cell_edges['x_max'],cell_edges['y_max']),
                    (cell_edges['x_max'],cell_edges['y_min']),
                    ]
                for x,y in pairs:
                    x_find = np.where(values0[0,:]=x)
                    y_find = np.where(values0[1,:]=y)
                    index = np.where(x_find*y_find == 1)[0]
                    if(index_list[c] == Empty):
                        index_list[c] = [index]
                    else:
                        index_list[c].append(index)
                            
                c+=1
                '''
                x = i*(self.grid.x_size) + self.grid.x_min
                y = j*(self.grid.y_size) + self.grid.y_min
                x_find = self.values0[0,:]==x
                y_find = self.values0[1,:]==y
                mul = np.logical_and(x_find,y_find).reshape((N_points,1))
                index = np.where(mul)[0]
                index_list[c] = index.item(0)
                c+=1
                
        return index_list
        

    def create_mapping(self):
        
        cell_id = 0
        
        mapping_list = [None]*((self.grid.x_n_cells)*(self.grid.y_n_cells))
        
        for j in range(self.grid.y_n_cells):
            for i in range(self.grid.x_n_cells):
                cell_id =  self.grid.cell_id_index(i, j)
                
                mapping_list[cell_id] = []
                
                points = [(1,1),(1,0),(0,0),(0,1)]
                
                c = 0
                
                edge_points = np.empty([2,4])
                transformed_edge_points = np.empty([2,4])
                zero = np.ones([1,3])
                for ip,jp in points:
                    
                    point_id = self.grid.point_id(i+ip,j+jp)
                    index_i = self.index_list[point_id]
                    edge_points[:,c] = self.values0[:,index_i].ravel()
                    transformed_edge_points[:,c] = self.values1[:,index_i].ravel()
                    c+=1
                    
                for k in range(4):
                    initial_triangle = np.delete(edge_points,k,axis=1)
                    initial_triangle = np.concatenate((initial_triangle,zero),axis=0)
                    
                    try:
                        initial_triangle_inv = np.linalg.inv(initial_triangle)
                    except np.linalg.LinAlgError as err:
                        print("Error non invertable triangle")
                        print(initial_triangle)
    
                    initial_triangle_inv = np.linalg.inv(initial_triangle)
                    
                    final_triangle = np.delete(transformed_edge_points,k,axis=1)
                    final_triangle = np.concatenate((final_triangle,zero),axis=0)
                    
                    transformation_matrix = np.matmul(final_triangle,initial_triangle_inv)
                    
                    mapping_list[cell_id].append(transformation_matrix)
                    
        return mapping_list
    
    def map_point(self,x,y):
        if(x<self.grid.x_min or x>self.grid.x_max or y<self.grid.y_min or y>self.grid.y_max):
            raise Exception("point is out of range")
            
        #get the two triangle transforms needed for the transformation of a point 
        cell_id = self.grid.cell_id(x,y)
        quartile = self.grid.get_quartile(x,y)
        point_v = np.array([x,y,1])
        transform_dict = {
            0:(3,0),
            1:(0,1),
            2:(1,2),
            3:(2,3)
            }
        
        transform_choice = transform_dict[quartile]
        #transform_choice = (quartile-1,quartile)%4
        #transform the point by the two transformation matricies and get average
        
        transform0 = self.mapping[cell_id][transform_choice[0]]
        point0_trans = np.matmul(transform0,point_v)
        
        transform1 = self.mapping[cell_id][transform_choice[1]]
        point1_trans = np.matmul(transform1,point_v)
        
        avr_point_trans = (point0_trans+point1_trans)/2
        
        return (avr_point_trans[0],avr_point_trans[1])
    
    def map_point_ln(self,x,y):
        if(x<self.grid.x_min or x>self.grid.x_max or y<self.grid.y_min or y>self.grid.y_max):
            raise Exception("point is out of range")
            
        #find percentage of each mapping to use
        cell_id = self.grid.cell_id(x,y)
        mapping_list = self.mapping[cell_id]
        point_v = np.array([x,y,1])
        #get fraction distance to x_borders
        edges = self.grid.get_cell_edges(x,y)
        x_frac = (edges["x_max"]-x)/(edges["x_max"]-edges["x_min"])
        y_frac = (edges["y_max"]-y)/(edges["y_max"]-edges["y_min"])
        
        transform = (x_frac*y_frac*mapping_list[0] +
                     x_frac*(1-y_frac)*mapping_list[1] +
                     (1-x_frac)*(1-y_frac)*mapping_list[2] +
                     (1-x_frac)*y_frac*mapping_list[3])
        
        
        
        point_trans = np.matmul(transform,point_v)
        return (point_trans[0],point_trans[1])

'''
def polar_mapping(x,y):
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    
    return (r,theta)
        
        
gal = []
pol = []
for y in np.arange(0,10.01,2):
    for x in np.arange(0,10.01,2):
        gal.append([x,y]);
        r,theta = polar_mapping(x,y)
        pol.append([r,theta])
 
gal_np = np.matrix(gal)
pol_np = np.matrix(pol)  
tf_pol = transform(gal_np, pol_np)

for a in range(50):
    x = random.uniform(0, 10);
    y = random.uniform(0, 10);
    polars_tp = tf_pol.map_point_ln(x, y)
    polars = polar_mapping(x,y)
    color1 = [(0.1+0.08*x,0,0.1+0.08*y)]
    
    plt.scatter(x, y, color = 'b')
    plt.scatter(polars[0], polars[1], color = 'r')
    plt.scatter(polars[0], polars[1], color = 'k', marker = 'x')
'''


                
                        
                    
                
            
            
            
            
 
        