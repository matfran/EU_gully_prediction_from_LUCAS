# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:13:24 2024

@author: fmatt
"""

from shapely.geometry import Polygon, MultiPolygon, shape, Point
import geopandas as gpd
import pandas as pd
import rasterio  as rst
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
from osgeo import gdal

"""
def extract_point_values(shp_path, raster_path):
    
    point_vals = gen_zonal_stats(shp_path, raster_path)
    
    return point_vals
"""

def EnS_downsample(lucas, class_specific = True, class_ = 0, oversample_ratio = 1):

    gullies_per_EnS = lucas[lucas["Gully"] == 1].groupby("EnS_name", as_index = False).count()[["EnS_name", "Gully"]]
    #only take regions with 2 or more observed gullies
    gullies_per_EnS = gullies_per_EnS[gullies_per_EnS["Gully"] >= 2]
    
    sample_spatial = pd.DataFrame()
    
    lucas_ng = lucas.copy()
    
    if class_specific == True:
        lucas_ng = lucas_ng[lucas_ng["Gully"] == class_]
    
    for index, row in gullies_per_EnS.iterrows():
       EnS = row["EnS_name"]
       n = row["Gully"]
       
       #if oversampling is not possible take equal sample
       try:
           ss = lucas_ng[lucas_ng["EnS_name"] == EnS].sample(int(n * oversample_ratio), random_state = 1)
       except:
           ss = lucas_ng[lucas_ng["EnS_name"] == EnS].sample(int(n), random_state = 1)
   
       sample_spatial = pd.concat([sample_spatial, ss])
       
    return sample_spatial

def random_downsample(lucas, class_ = 0, oversample_ratio = 1):
    df = lucas.copy()
    lucas_g = df[df["Gully"] == 1]
    lucas_ng = df[df["Gully"] == class_].sample(int(len(lucas_g) * oversample_ratio), random_state = 1)
    return lucas_ng

def tidy_df(df, mask_variable, downsample_method = "RANDOM DOWNSAMPLE", remove_nan_rows = False,
            replace_vals = None, oversample_ratio = 1):
    #tidy up nan values
    df = df.copy()
    for col in df.columns:
        if col == 'geometry':
            continue
        
        min_val = df[col].min()
        try:
            if min_val <= -9999:
                #optionally replace the masked areas with a value to avoid data loss
                if replace_vals is not None:
                    if col in replace_vals.keys():
                        df[col] = df[col].replace(min_val, replace_vals[col])
                    else:
                        df[col] = df[col].replace(min_val, np.nan)
                else:
                    df[col] = df[col].replace(min_val, np.nan)
        except:
            continue
    #apply the mask
    df = df[df[mask_variable] >= 0]
    #optionally remove all rows containing at least 1 nan
    if remove_nan_rows == True:
        df["nan_count"] = df.isnull().sum(axis=1)
        df = df[df["nan_count"] == 0]
    
    if downsample_method == "SPATIAL BALANCE":
        df2 = EnS_downsample(df, class_specific = True, class_ = 0, oversample_ratio = 1)
        df_gully = df[df["Gully"] == 1]
        df3 = pd.concat([df_gully, df2])
    elif downsample_method == "RANDOM DOWNSAMPLE":
        df2 = random_downsample(df, class_ = 0, oversample_ratio = oversample_ratio)
        df_gully = df[df["Gully"] == 1]
        df3 = pd.concat([df_gully, df2])        
    elif downsample_method == "MIXED":
        df_ens = EnS_downsample(df, class_specific = True, class_ = 0, oversample_ratio = 0.5)
        df_rand = random_downsample(df, class_ = 0, oversample_ratio = 0.5)
        df_gully = df[df["Gully"] == 1]
        df3 = pd.concat([df_gully, df_rand, df_ens])
    else:
        df3 = df
    return df3.copy()

def plot_correlation_matrix(df, cols):
    df = df.copy()
    sns.set_theme(style="white")
    
    df = df[cols]
    # Compute the correlation matrix
    corr = df.corr(method = 'spearman')
    
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

def reproject_raster(raster_in_path, raster_out_path, dst_crs):
    warp = gdal.Warp(raster_out_path, raster_in_path, dstSRS=dst_crs)
    warp = None
    
def raster_crs_check(raster_in_path, target_crs, dst_crs):
    with rst.open(raster_in_path) as src:
        print(src.crs)
        print(raster_in_path)
        if target_crs == src.crs:
            raster_out_path = raster_in_path.replace(".", "_reprojected.")
            reproject_raster(raster_in_path, raster_out_path, dst_crs)
            return raster_out_path
    return raster_in_path
    
def sample_rasters(points, rasters, names, check_crs = True):
    raster_vals = pd.DataFrame()
    for i in np.arange(len(rasters)):
        point_vals = raster_values_at_points(rasters[i], points, names[i], check_crs = check_crs)
        raster_vals[names[i]] = point_vals[names[i]]

    all_data = pd.concat([points, raster_vals], axis = 1)
    return all_data


def raster_values_at_points(raster_path, points_gdf, column_name, check_crs = True):

    new_gdf = points_gdf.copy()  # do not change original gdf
    #coords = points_gdf.get_coordinates().values
    coords = [(x,y) for x, y in zip(points_gdf.geometry.x, points_gdf.geometry.y)]
    print("OPENING: ", raster_path)

    with rst.open(raster_path) as src:
        meta = src.meta.copy()
        if check_crs == True:
            if str(points_gdf.crs.to_epsg()) in str(meta["crs"]):
                new_gdf[column_name] = [x[0] for x in src.sample(coords)]
            else:
                print("SKIPPED: check crs", str(meta["crs"]))
                new_gdf = new_gdf.to_crs(meta["crs"])
                new_gdf[column_name] = [x[0] for x in src.sample(coords)]
        else:
            print("CRS NOT CHECKED: layer crs is ", str(meta["crs"]))
            new_gdf[column_name] = [x[0] for x in src.sample(coords)]
            
    return new_gdf


def polygon_random_points (points_gdf, num_points):
    #https://medium.com/the-data-journal/a-quick-trick-to-create-random-lat-long-coordinates-in-python-within-a-defined-polygon-e8997f05123a
    min_x, min_y, max_x, max_y = points_gdf.geometry.total_bounds
    points = []
    while len(points) < num_points:
            random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
            points.append(random_point)
    return points



