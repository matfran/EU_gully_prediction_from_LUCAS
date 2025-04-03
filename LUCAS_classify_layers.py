# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:13:40 2024

@author: Francis MAtthews fmatthews1381@gmail.com github: matfran
"""

#stack all arrays into 1 

from WS_preprocess_functions import read_raster, write_raster
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
from catboost import CatBoostClassifier, Pool

model_run = "FULL_FEATURES"
model_type = "RF"
type_ = "MIXED"
version = 'v5'
n_tiles = 120
tile_to_process = 93

dir_ = r"D:\ESDAC_RF_100m"
clf_dir = fr"C:\Users\fmatt\OneDrive - Universita degli Studi Roma Tre\Roma3_projects\LUCAS_gully_prediction\RESULTS\ALL_FEATURES_{type_}\{model_type}_Models"
out_dir = fr"C:\Users\fmatt\OneDrive - Universita degli Studi Roma Tre\Roma3_projects\LUCAS_gully_prediction\RESULTS\ALL_FEATURES_{type_}\Classified_tiles_{model_type}_{model_run}"


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(os.path.join(clf_dir, f'Results_{model_type}_{type_}_{version}.pkl'), 'rb') as f:
    clf_results = pickle.load(f)

if model_type == "RF":
    with open(os.path.join(clf_dir, f'{model_type}_{model_run}_{type_}_{version}.pkl'), 'rb') as f:
        clf_ = pickle.load(f)    
elif model_type == "CATBOOST":
    from_file = CatBoostClassifier()
    clf_ = from_file.load_model(os.path.join(clf_dir, f'{model_type}_{model_run}_{type_}_{version}'), format='cbm')
    

for tile in np.arange(n_tiles):
    t = str(tile)
    
    #ENSURE ALL VERSIONS CORRESPOND
    out_path = os.path.join(out_dir, f'CLASSIFIED_IMAGE_{model_run}_{version}_tile_{t}.tif')
    out_path_proba = os.path.join(out_dir, f'CLASSIFIED_IMAGE_{model_run}_PROBA_{version}_tile_{t}.tif')
    
    #tidy up the names
    l_names = clf_results[model_run]
    new_names = []
    for j in l_names:
        
        if j == 'covergence_100m.tif':
            j = 'Convergence_100m.tif'
        
        if not "100m" in j:
            j = j.replace(".", f"_100m.{t}.")
            j = j.replace(" ", "_")
        else:
            j = j.replace("_100m.", f"_100m.{t}.")
            
        if not ".tif" in j:
            j = j + f"_100m.{t}.tif" 
            
            
        new_names.append(j)
    rasters = {}
    i = 0
    
    for path, subdirs, files in os.walk(dir_):
        for name in files:
            if name in new_names and name.endswith(".tif"):
                name.replace
                rasters[name] = os.path.join(path, name)
                i = i + 1
                
    #if not tile == tile_to_process:
    #    continue

    mask_name = f"EU_CFactor_final_V7_100m.{t}.tif"
    mask_layer = read_raster(rasters[mask_name])
    mask = np.where(mask_layer["array"] >= 0, 1, 0)
    mask_layer = None
    
    array_list = []
    
    for key in new_names:
        r_data = read_raster(rasters[key])      
        shape = (r_data['all metadata']['width'], r_data['all metadata']['height'])
        array = r_data['array']
        print(key)
        print(shape)     
        array_list.append(array)
    
    
    metadata_copy = r_data["all metadata"]
    metadata_copy["dtype"] = "int16"
    metadata_copy["nodata"] = -9999
    
    
    metadata_copy2 = r_data["all metadata"]
    metadata_copy2["dtype"] = "float64"
    metadata_copy2["nodata"] = -9999.
    

    img = np.stack(array_list, axis = 2)
    array_list_ss = None
    

    new_shape = (img.shape[0] * img.shape[1], img.shape[2])

    img_as_array = img[:, :, :].reshape(new_shape)

    class_prediction = clf_.predict(img_as_array)
    prob_prediction = clf_.predict_proba(img_as_array)[:,1]

    class_prediction = class_prediction.reshape(img[:, :, 0].shape).astype(int)
    prob_prediction = prob_prediction.reshape(img[:, :, 0].shape)

    class_prediction = np.where(mask == 1, class_prediction, -9999).astype(int)
    prob_prediction = np.where(mask == 1, prob_prediction, -9999.)

    write_raster(class_prediction, metadata_copy, out_path, check_array = False)
    write_raster(prob_prediction, metadata_copy2, out_path_proba, check_array = False)

