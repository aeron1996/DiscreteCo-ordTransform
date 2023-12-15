# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:36:12 2023

@author: asanche2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from coord_transform_delaunay import transform


train_data = pd.read_csv("D:/discrete co-ord transform/testing-data/pistachio_class/pistachio.csv")

#train_data = train_data[['AREA','PERIMETER','MAJOR_AXIS','MINOR_AXIS','ECCENTRICITY','CONVEX_AREA','Class']]
#'CONVEX_AREA','AREA','EQDIASQ',       ,'SOLIDITY','EXTENT'
train_data = train_data[['ASPECT_RATIO','MINOR_AXIS','ECCENTRICITY','Class']]
#train_data = train_data[['SHAPEFACTOR_1','SHAPEFACTOR_4','SHAPEFACTOR_3','Class']]

types = train_data.Class.unique()

def pist_type(pist_name):
    if(pist_name == "Kirmizi_Pistachio"):
        return 0
    else:
        return 1
    
def normalize_df(df_in):
    df = df_in.copy()
    # apply normalization techniques
    for column in df.columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()   
        
    return df
    
train_data["class_int"] = train_data['Class'].apply(pist_type)

train = train_data.sample(frac=0.8,random_state=200)

#check for duplicates
if(train.duplicated().any()):
    print('duplicates')
    

#train_in = normalize_df(train[train.columns.drop(['Class','class_int'])]).to_numpy()
train_mat = train.drop(columns=['Class','class_int']).to_numpy()
train_mean = np.mean(train_mat,axis=0)
train_std = np.std(train_mat,axis=0)
train_in = (train_mat-train_mean)/train_std
if(train.duplicated().any()):
    print('duplicates')
train_target = train['class_int'].to_numpy()
train_target = train_target.reshape((train_target.size,1))
n_points,n_dim = train_in.shape
train_target_mat = np.hstack([train_target,np.ones([n_points,n_dim-1])])


test = train_data.drop(train.index)
test_mat = test.drop(columns=['Class','class_int']).to_numpy()
test_in = (test_mat-train_mean)/train_std
#test_in = normalize_df(test[test.columns.drop(['Class','class_int'])]).to_numpy()
test_target = test[['class_int']].to_numpy()

print('comupting triangulation')
tf = transform(n_dim,train_in,train_target_mat)

def progress_bar(current, total, bar_length = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces  = ' ' * (bar_length - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')
    
preds = []
diff = []
pred_values = []
n_preds = 200#test_target.size
print('Calculating transformations')
for i in range(n_preds):
    print(i,end = "\r",flush=True)
    test_i_val = test_in[i,:]
    pred_i_array = tf.map_point(test_i_val)
    pred_i_val = pred_i_array[0]
    
    pred_i = 0
    if pred_i_val>0.5:
        pred_i = 1
    
    diff.append((pred_i_val-test_target[i]))
    pred_values.append(pred_i_val)
    preds.append(pred_i)


best_thresh = 0
best_acc = 0
for p in np.arange(0.01,1,0.01):
    preds_t = np.zeros(n_preds)
    preds_t[pred_values>p] = 1
    frac_correct = np.sum(np.abs(1-preds_t-test_target[:n_preds,0]))/n_preds
    
    if(frac_correct>best_acc):
        best_acc = frac_correct
        best_thresh = p
    

preds_np = np.array(preds)
pred_values = np.array(pred_values)

frac_correct = np.sum(np.abs(1-preds_np-test_target[:n_preds,0]))/n_preds
