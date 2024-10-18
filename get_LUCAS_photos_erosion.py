# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 09:51:20 2023

@author: u0133999
"""
import shutil
import wget 
import pandas as pd
import os
import sys

master_dir = os.getcwd()
os.chdir(master_dir)


lucas_gully = pd.read_csv('LUCAS_Gully.csv', usecols=['FID', 'POINT_ID','POINT_NUTS'])
root = 'https://gisco-services.ec.europa.eu/lucas/photos/2022'
l1 = lucas_gully['POINT_NUTS']
l2 = lucas_gully['POINT_ID'].astype(str).str.slice(0,3)
l3 = lucas_gully['POINT_ID'].astype(str).str.slice(3,6)

lucas_gully['PHOTO_LINK'] = root + '/' + l1 + '/' + l2 + '/' + l3 + '/2022' + lucas_gully['POINT_ID'].astype(str) + 'LCLU_'

output_folder = os.path.join(master_dir, 'LUCAS_photos_gullies')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

skipped = []
os.chdir(output_folder)

for suffix in ['Point.jpg', 'West.jpg', 'East.jpg', 'North.jpg',  'South.jpg', 'CropLC1.jpg']:
    lucas_gully['PHOTO_LINK' + '_' + suffix] = lucas_gully['PHOTO_LINK'] + suffix


#lucas_gully.to_csv('LUCAS_Gully_photo_links.csv')
for i in ['East.jpg', 'North.jpg',  'South.jpg', 'CropLC1.jpg']:
    for link in lucas_gully['PHOTO_LINK_' + i]:
        try:
            pic = wget.download(link)
        except:
            print('failed')
    
        
        
