# -*- coding: utf-8 -*-
# @Author: jadesauve
# @Date:   2021-01-03
"""
make monthly climatology from RG argo climatology

data dowload: 2021-01-03

"""

## Import modules
import pandas as pd
# pd.set_option("display.precision", 8)
import numpy as np
# import scipy.interpolate
# import gsw
# import matplotlib.pyplot as plt
import xarray as xr
import datetime
from dateutil.relativedelta import relativedelta
import os
import sys
# pht = os.path.abspath('/Users/jadesauve/Documents/Python/scripts/python_tools/')
# if pht not in sys.path:
#     sys.path.append(pht)
# from toolbox_float import *

#### Parameters ####

directory_in  = '/Users/jadesauve/Coding/data/RG_clim/'  


# for 2004 to 2020, the complete time period
# daterange = '2004-2020'

# for 2014 to 2020, the soccom time period
daterange = '2014-2020'


# to save 
directory_out =  '/Users/jadesauve/Coding/output/RG_clim/'
file_out = 'RG_clim_'+daterange+'.nc'
file_out_annual = 'RG_annual_'+daterange+'.nc'

######################

# function to convert the time to datetime
def time_conv(time_arr):

    # convert to int
    time = time_arr.astype(int)

    # make a datetime object for 2004-01-15
    datein = datetime.datetime(2004, 1, 15)

    # make a datetime list 
    ls_datetime = []
    for t in time:
        date = datein + relativedelta(months=t)
        ls_datetime.append(date)

    return ls_datetime




# MAIN CODE

# open the files and extract
directory = os.fsencode(directory_in)

# for all files in this directory
for file in os.listdir(directory):  
    filename = os.fsdecode(file)
    #filename = ''
    print(filename)
    if filename.endswith("Temperature_2019.nc"):
        # open dataset
        ds_T = xr.open_dataset(directory_in + filename, decode_times=False)
        # produce datetime object list
        ls_datetime = time_conv(ds_T.TIME.values)
        # the temperature at each month
        T1 = ds_T.ARGO_TEMPERATURE_ANOMALY + ds_T.ARGO_TEMPERATURE_MEAN 

    elif filename.endswith("Salinity_2019.nc"):
        ds_S = xr.open_dataset(directory_in + filename, decode_times=False)
        S1 = ds_S.ARGO_SALINITY_ANOMALY + ds_S.ARGO_SALINITY_MEAN

    elif filename.endswith("_2019.nc"):
        ds_add = xr.open_dataset(directory_in + filename, decode_times=False)
        # concatenate the added files
        try:
            ds2 = xr.concat([ds2, ds_add], dim="TIME")
        except NameError:
            ds2 = ds_add

# sort with respect to time
ds2 = ds2.sortby('TIME')
# produce the datetime object list
ls_datetime2 = time_conv(ds2.TIME.values)
# monthly T and S
T2 = ds2.ARGO_TEMPERATURE_ANOMALY + ds_T.ARGO_TEMPERATURE_MEAN
S2 = ds2.ARGO_SALINITY_ANOMALY + ds_S.ARGO_SALINITY_MEAN 

# concatenate T and S
ds_concat = xr.Dataset()
ds_concat['T'] = xr.concat([T1, T2], dim="TIME")
ds_concat['S'] = xr.concat([S1, S2], dim="TIME")

# put datetime_ls in df
df = pd.DatetimeIndex(ls_datetime+ls_datetime2)


# select the proper dates
if daterange == '2004-2020':
    T = ds_concat.T
    S = ds_concat
elif daterange == '2014-2020':
    T = ds_concat.T.isel(TIME = np.where(df.year > 2013)[0]) 
    S = ds_concat.S.isel(TIME = np.where(df.year > 2013)[0]) 
    df = df[np.where(df.year > 2013)[0]]
else:
    print('Choose a valid date range')



# monhtly climatology
# make a new ds
ds_avg = ds_concat.isel(TIME=slice(0,12)).copy(deep=True)
ds_avg = ds_avg.assign_coords(TIME=range(1,13))
# fill the ds with nan
for v in ds_avg:
    ds_avg[v].loc[:] = np.NaN

for m in range(1,13):

    # select the data for one month and avg
    ds_avg.T.loc[dict(TIME=m)] = T.isel(TIME = np.where(df.month == m)[0]).mean(dim='TIME')
    # ds_avg['S'] = ds_concat.S.isel(TIME = np.where(df.month == m)[0]).mean(dim='TIME')
    ds_avg.S.loc[dict(TIME=m)] = S.isel(TIME = np.where(df.month == m)[0]).mean(dim='TIME')

# save
ds_avg.to_netcdf(path=directory_out+file_out)




# make an annual average
ds_an = ds_concat.isel(TIME=0).copy(deep=True)
ds_an = ds_an.drop('TIME')
# fill the ds with nan
for v in ds_an:
    ds_an[v].loc[:] = np.NaN

ds_an['T'] = T.mean(dim='TIME')
ds_an['S'] = S.mean(dim='TIME')

ds_an.to_netcdf(path=directory_out+file_out_annual)


