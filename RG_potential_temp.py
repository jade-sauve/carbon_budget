# -*- coding: utf-8 -*-
# @Author: jadesauve
# @Date:   2021-01-04
"""
find potential temperature from RG argo climatology

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
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
import sys
pht = os.path.abspath('/Users/jadesauve/Documents/Python/scripts/python_tools/')
if pht not in sys.path:
    sys.path.append(pht)
from toolbox_float import *

#### Parameters ####

directory_in  = '/Users/jadesauve/Coding/output/'

daterange = '2014-2020'
file_in = 'RG_clim/RG_clim_'+daterange+'.nc'
file_in_annual = 'RG_clim/RG_annual_'+daterange+'.nc'


# choose up to which depth to compute the PT
deep = 400

# Additonal criteria to consider
# STF: T=10,11,12C at z=100m
# SAF: T=4,5C at z=400m or S=34.2 along smin at z<300m   -- what kind of S?
# PF:  T=2C along tmin at z<200m or T=2.2C along tmax at z<800m


# to save 
directory_out =  '/Users/jadesauve/Coding/output/RG_clim/'
file_out = 'RG_clim_PT_'+daterange+'_0-'+str(deep)+'.nc'
file_out_annual = 'RG_annual_PT_'+daterange+'_0-'+str(deep)+'.nc'

######################

# create a polar stereo projection #
proj = Basemap(projection='spstere', boundinglat=-20, lon_0=180, resolution='l')

# open the files
ds = xr.open_dataset(directory_in + file_in)
ds_an = xr.open_dataset(directory_in + file_in_annual)

# convert to 0-360 from 20-380
ds = ds.assign_coords(LONGITUDE=ds.LONGITUDE%360)
ds = ds.sortby('LONGITUDE')

ds_an = ds_an.assign_coords(LONGITUDE=ds_an.LONGITUDE%360)
ds_an = ds_an.sortby('LONGITUDE')

# select data south of 25S
ds = ds.sel(LATITUDE = slice(-90,-25))
ds_an = ds_an.sel(LATITUDE = slice(-90,-25))

# make a new data array
ds_pt = ds.copy(deep=True)
ds_pt = ds_pt.rename({'T':'PT','S':'SA'})
ds_pt = ds_pt.sel(PRESSURE = slice(0,deep))

ds_an_pt = ds_an.copy(deep=True)
ds_an_pt = ds_an_pt.rename({'T':'PT','S':'SA'})
ds_an_pt = ds_an_pt.sel(PRESSURE = slice(0,deep))

# fill the ds with nan
for v in ds_pt:
    ds_pt[v].loc[:] = np.NaN

for v in ds_pt:
    ds_an_pt[v].loc[:] = np.NaN


for m in range(1,13):
	print( 'm = '+str(m))
	# select data for the proper month
	dsm = ds.sel(TIME = m)

	# select T and S data at the depths of interest
	T = dsm.T.sel(PRESSURE = slice(0,deep))
	S = dsm.S.sel(PRESSURE = slice(0,deep))
	P = dsm.PRESSURE.sel(PRESSURE = slice(0,deep))

	# convert T to PT
	# SA_all = np.empty(T.shape); PT_all = np.empty(T.shape) # (P,lat,lon)
	# print('Finding Potential Temperature - 35')  
	i=0
	for lat in T.LATITUDE: # 35
		# lat = -30.5
		# j=0; 
		print(lat.values)
		for lon in T.LONGITUDE: # 360
			# k=0
			for p in P: # 40
				# convert practical salinity to absolute salinity
				# SA_all[k,i,j] = gsw.conversions.SA_from_SP(S.sel(LATITUDE=lat,LONGITUDE=lon, PRESSURE=p), p, lon, lat)


				# convert practical salinity to absolute salinity
				sa = gsw.conversions.SA_from_SP(S.sel(LATITUDE=lat,LONGITUDE=lon, PRESSURE=p), p, lon, lat)
				# convert T to potential T
				pt = gsw.conversions.pt0_from_t(sa, T.sel(LATITUDE=lat,LONGITUDE=lon,PRESSURE=p), p)

				ds_pt.SA.loc[dict(LATITUDE=lat,LONGITUDE=lon,PRESSURE=p,TIME=m)] = sa
				ds_pt.PT.loc[dict(LATITUDE=lat,LONGITUDE=lon,PRESSURE=p,TIME=m)] = pt

				if m == 1:
					sa_an = gsw.conversions.SA_from_SP(ds_an.S.sel(LATITUDE=lat,LONGITUDE=lon, PRESSURE=p), p, lon, lat)
					pt_an = gsw.conversions.pt0_from_t(sa_an, ds_an.T.sel(LATITUDE=lat,LONGITUDE=lon,PRESSURE=p), p)

					ds_an_pt.SA.loc[dict(LATITUDE=lat,LONGITUDE=lon,PRESSURE=p)] = sa_an
					ds_an_pt.PT.loc[dict(LATITUDE=lat,LONGITUDE=lon,PRESSURE=p)] = pt_an
				
				# increment pressure
				# break
				# k=k+1

			# increment longitude
			# break
			# j=j+1
			
		#increment latitude
		# break
		# i=i+1
		
	# break

# save
ds_pt.to_netcdf(path=directory_out+file_out)
ds_an_pt.to_netcdf(path=directory_out+file_out_annual)







