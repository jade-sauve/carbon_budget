# -*- coding: utf-8 -*-
# @Author: jadesauve
# @Date:   2021-01-06
"""
find the area of a grid south of 31S and the avg depth of each cells using ETOPO data

data dowload: 2021-01-03

"""

## Import modules
import pandas as pd
# pd.set_option("display.precision", 8)
import numpy as np
# import scipy.interpolate
import gsw
# import matplotlib.pyplot as plt
import xarray as xr
import datetime
from dateutil.relativedelta import relativedelta
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon
import os
import sys
pht = os.path.abspath('/Users/jadesauve/Documents/Python/scripts/python_tools/')
if pht not in sys.path:
    sys.path.append(pht)
from toolbox_float import *

#### Parameters ####

directory_relief = '/Users/jadesauve/Coding/data/'

file_relief = 'ETOPO1_Ice_g_gmt4.grd'

# to save 
directory_out =  '/Users/jadesauve/Coding/output/'
file_out_depth = 'ETOPO1_ocean_depth_26S.pkl'
file_out_area = 'grid_cell_area_26S.pkl'

######################

# open the files
ds_rel = xr.open_dataset(directory_relief + file_relief)

# convert to 0-360 from -180-180
ds_rel = ds_rel.assign_coords(x=ds_rel.x%360)
ds_rel = ds_rel.sortby('x')

# select data south of 30S
ds_rel = ds_rel.sel(y=slice(-90,-25))

## make a grid
# set a step size
dl = 0.5

# make a vector for coordinates - center of the cell
latc = np.arange(-90+(dl/2),-26,dl)
lonc = np.arange(dl/2,360-(dl/2),dl)

# Make a grid
lonc,latc = np.meshgrid(lonc,latc)

# vector for the corners of the cell
lat1 = latc-(dl/2)
lat2 = latc+(dl/2)
lon1 = lonc-(dl/2)
lon2 = lonc+(dl/2)

# find the area of each cell
area = areaquad(lat1,lon1,lat2,lon2)

# make a df
df_area = pd.DataFrame(index=latc[:,0],columns=lonc[0,:],data=area)

# find the mean depth of each cell
depth = np.empty(area.shape)
depth[:] = np.NaN

# print('Finding depth - 127')
for i in range(depth.shape[0]): # latitude - 118
	print(i)
	for j in range(depth.shape[1]): # longitude - 719
		# select the depth for the proper cells
		z = ds_rel.sel(x=slice(lon1[i,j],lon2[i,j]),y=slice(lat1[i,j],lat2[i,j]))
		# get the avg depth
		depth[i,j] = z.mean(['x','y']).to_array()

# make a df
df_depth = pd.DataFrame(index=latc[:,0],columns=lonc[0,:],data=depth)

#save
df_depth.to_pickle(directory_out+file_out_depth)
df_area.to_pickle(directory_out+file_out_area)



# plot
directory_in  = '/Users/jadesauve/Coding/output/'
file_depth = 'ETOPO1_ocean_depth_31S.pkl'
file_area ='grid_cell_area_31S.pkl'

df_ga = pd.read_pickle(directory_in + file_area)
df_depth = pd.read_pickle(directory_in + file_depth)

plt.figure()
cmap=plt.cm.seismic
cmap.set_bad('white',1.)
norm=mpl.colors.Normalize(vmin=-8000,vmax=8000)
plt.pcolormesh(df_depth.columns,df_depth.index,df_depth,cmap=cmap,norm=norm)
plt.colorbar() 
plt.show()

plt.figure()
plt.pcolormesh(df_ga.columns,df_ga.index,df_ga)
plt.colorbar() 
plt.show()








