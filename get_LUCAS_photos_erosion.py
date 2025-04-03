# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 09:51:20 2023

This code automatically downloads photos for the LUCAS 2022 survey for points
with observed gullies.The code requires the LUCAS gully dataset to be loaded and
subset based on points containing gullies. Downloading all LUCAS photos
will result in a heavy download process. 

@author: Francis MAtthews fmatthews1381@gmail.com github: matfran
"""
import shutil
import wget 
import pandas as pd
import os
import sys

# Set the working directory to the current directory
master_dir = os.getcwd()
os.chdir(master_dir)

#Load LUCAS Gully dataset, selecting only relevant columns
lucas_gully = pd.read_csv('LUCAS_Gully.csv', usecols=['FID', 'POINT_ID', 'POINT_NUTS'])

#Define the root URL for downloading LUCAS 2022 photos
root = 'https://gisco-services.ec.europa.eu/lucas/photos/2022'

# Extract necessary substrings from POINT_ID to construct the full photo URL
l1 = lucas_gully['POINT_NUTS']  # Region identifier
l2 = lucas_gully['POINT_ID'].astype(str).str.slice(0,3)  # First three digits of POINT_ID
l3 = lucas_gully['POINT_ID'].astype(str).str.slice(3,6)  # Next three digits of POINT_ID

#Construct the base photo link for each point
lucas_gully['PHOTO_LINK'] = root + '/' + l1 + '/' + l2 + '/' + l3 + '/2022' + lucas_gully['POINT_ID'].astype(str) + 'LCLU_'

#Define the output folder for storing downloaded photos
output_folder = os.path.join(master_dir, 'LUCAS_photos_gullies')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)  # Create folder if it doesn't exist

#Change directory to output folder
os.chdir(output_folder)

#Generate complete photo links for different orientations and crop images
for suffix in ['Point.jpg', 'West.jpg', 'East.jpg', 'North.jpg', 'South.jpg', 'CropLC1.jpg']:
    lucas_gully['PHOTO_LINK_' + suffix] = lucas_gully['PHOTO_LINK'] + suffix

# Optionally export CSV containing all photo links
# lucas_gully.to_csv('LUCAS_Gully_photo_links.csv')

#List to store skipped downloads
skipped = []

#Download available photos from generated links
for suffix in ['East.jpg', 'West.jpg', 'North.jpg', 'South.jpg', 'CropLC1.jpg']:
    for link in lucas_gully['PHOTO_LINK_' + suffix]:
        try:
            pic = wget.download(link)  # Attempt to download the image
        except Exception as e:
            print(f'Failed to download {link}: {e}')  # Print error message if download fails
        
        
