# -*- coding: utf-8 -*-
# @Author: jadesauve
# @Date:   2022-04-13
"""
find fronts to define the zones from RG argo climatology

cartopy version

data dowload: 2021-01-03

"""

## Import modules
import pandas as pd
# pd.set_option("display.precision", 8)
import numpy as np
import pickle
import copy 
# import scipy.interpolate
import gsw
import matplotlib.pyplot as plt
from matplotlib.path import Path
import xarray as xr
import datetime
from dateutil.relativedelta import relativedelta
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
# from mpl_toolkits.basemap import Basemap 
import cartopy.crs as ccrs
import cartopy.feature as cft
# import os
# import sys
# pht = os.path.abspath('/Users/jadesauve/Documents/Python/scripts/python_tools/')
# if pht not in sys.path:
#     sys.path.append(pht)
# from toolbox_float import *


#### Parameters ####

directory_in  = '/Users/jadesauve/Coding/output/'
directory_si = '/Users/jadesauve/Coding/data/era5/'
directory_ns = '/Users/jadesauve/Coding/data/NSIDC/monthly/'

daterange = '2014-2020'
file_in = 'RG_clim/RG_clim_'+daterange+'.nc'
file_depth = 'ETOPO1_ocean_depth_31S.pkl'
file_pt = 'RG_clim/RG_clim_PT_'+daterange+'_0-400.nc'
file_si1 = '2014-2018_wind_mask_seaice_monthly.nc'
file_si2 = '2019-2020_wind_mask_seaice_monthly.nc'

# select proper tag
tag = '6.0_NSIDC_Carto'
# 1.0 : first version - dic_lev = {'STF':12,'SAF':5,'PF':2}, dic_dep = {'STF':100,'SAF':400,'PF':200}
# 2.0_SAF : second version - dic_lev = {'STF':12,'SAF':4,'PF':2}, dic_dep = {'STF':100,'SAF':400,'PF':200}
# 3.0_SAF : third version - dic_lev = {'STF':12,'SAF':34.2,'PF':2}, dic_dep = {'STF':100,'SAF':300,'PF':200}
# 4.0_PF : fourth version - dic_lev = {'STF':12,'SAF':4,'PF':2.2}, dic_dep = {'STF':100,'SAF':400,'PF':800} 
# 5.0_SAF: - dic_lev = {'STF':12,'SAF':4.5,'PF':2}, dic_dep = {'STF':100,'SAF':400,'PF':200}
# 6.0: default - dic_lev = {'STF':12,'SAF':4.5,'PF':2}, dic_dep = {'STF':100,'SAF':400,'PF':200}

ls_fronts = ['STF','SAF','PF']
dic_dep = {'STF':100,'SAF':400,'PF':200} # dbar
dic_lev = {'STF':12,'SAF':4.5,'PF':2} # C
si_lev = 0.15
ice = 'NSIDC' # era5 or NSIDC

# Additonal criteria to consider
# STF: T=10,11,12C at z=100m
# SAF: T=4,5C at z=400m or S=34.2 along smin at z<300m   -- Practical S because no units are listed?
# PF:  T=2C along tmin at z<200m or T=2.2C along tmax at z<800m

# to save 
directory_out =  '/Users/jadesauve/Coding/output/'
directory_fig = '/Users/jadesauve/Coding/figures/'

file_out_coord = 'front_coord_dict_monthly_'+tag+'.pkl'
file_out_coord_plot = 'front_coord_dict_monthly_'+tag+'_plot.pkl'
file_out_poly = 'front_polygon_dict_monthly_'+tag+'.pkl'
######################


# open the files
ds = xr.open_dataset(directory_in + file_in)
ds_pt = xr.open_dataset(directory_in + file_pt)
# df_ga = pd.read_pickle(directory_in + file_area)
df_depth = pd.read_pickle(directory_in + file_depth)
ds_si1 = xr.open_dataset(directory_si+file_si1)
ds_si2 = xr.open_dataset(directory_si+file_si2)

# convert to 0-360 from 20-380
ds = ds.assign_coords(LONGITUDE=ds.LONGITUDE%360)
ds = ds.sortby('LONGITUDE')

# select data south of 25S
ds = ds.sel(LATITUDE = slice(-90,-25))

# concat ds_si1 and ds_si2.sel(expver=1) - data goes until May 2020 for download in 2020-08
ds_si = xr.concat([ds_si1,ds_si2.sel(expver=1)],dim='time')
# convert lat and lon to agree with ds
ds_si = ds_si.assign_coords(longitude=ds_si.longitude%360)
ds_si = ds_si.sortby('longitude')
ds_si = ds_si.sortby('latitude')

# make monthly clim
ds_sim = ds_si.siconc.groupby('time.month').mean(dim='time',skipna=True)


# NSIDC ice
file_ns2 = 'seaice_conc_monthly_icdr_sh_f18_202009_v01r00.nc'
ds_ns2 = xr.open_dataset(directory_ns + file_ns2)

for yr in range(2014,2020):
    file_ns1 = 'seaice_conc_monthly_sh_f17_'+str(yr)+'09_v03r01.nc'
    ds_temp = xr.open_dataset(directory_ns + file_ns1)
    try:
        ds_ns1 = xr.concat([ds_ns1, ds_temp], dim="time")
    except NameError:
        ds_ns1 = ds_temp

# convert lon to agree with ds
ds_ns1 = ds_ns1.assign_coords(longitude=ds_ns1.longitude%360)
# plt.pcolormesh(ds_ns1.goddard_merged_seaice_conc_monthly.isel(time=1))

# average 
ds_ns1_m = ds_ns1.goddard_merged_seaice_conc_monthly.mean(dim='time',skipna=True)
# plt.pcolormesh(ds_ns1.goddard_merged_seaice_conc_monthly.mean(dim='time',skipna=True))
ds_ns_m = (ds_ns1_m + ds_ns2.seaice_conc_monthly_cdr.isel(time=0).values)/2
# plt.pcolormesh(ds_ns_m)

# create projection
proj = ccrs.SouthPolarStereo()
land_50m = cft.NaturalEarthFeature('physical', 'land', '50m',
                                   edgecolor='black', facecolor='papayawhip', linewidth=0.5)

# compute a circle for map boundary
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = Path(verts * radius + center)

# create a dic to hold the lat/lon of all fronts
dic_coord = {} # format: dic[front][month][lon,lat]
for front in ls_fronts:
    # front = 'SAF'
    print()
    print(front)
    dic_coord[front] = {}
    for m in range(1,13):
        # m=1

        # select data for the proper month
        # dsm = ds.sel(TIME = m)
        dsm = ds_pt.sel(TIME = m)
        P = dsm.PRESSURE
        
        # select the depth of interest
        dep = dic_dep[front]

        # select PT at proper depth
        PT = dsm.sel(PRESSURE=dep).PT

        # for PF, choose the min T 
        if front == 'PF':
            # select the top x meters
            PT = dsm.sel(PRESSURE=slice(0,dep)).PT
            if tag == '4.0_PF':
                # select the max value in the water column for each lat/lon
                PT = np.amax(PT, axis=0)
            else:
                # select the min value in the water column for each lat/lon
                PT = np.amin(PT, axis=0)
        # for special case of SAF, select min S
        elif front == 'SAF':
            if tag == '3.0_SAF':
                PT = ds.sel(TIME = m, PRESSURE=slice(0,dep)).S
                PT = np.amin(PT, axis=0)
            
        # select the proper PT for the contour
        lev = dic_lev[front]
        # make contour
        plt.figure()
        CS = plt.contour(dsm.LONGITUDE,dsm.LATITUDE,PT,[lev])
        plt.show()

        # list of all the coordinates of the segments, one array per segment
        coord = CS.allsegs[0]
        # number of segments
        nn = len(coord)

        order = []
        fig = plt.figure(figsize=(10, 9))
        ax = plt.subplot(1, 1, 1, projection=proj)
        ax.set_extent([-280, 80, -80, -20], crs=ccrs.PlateCarree())
        ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
        ax.coastlines(resolution='50m')
        ax.set_boundary(circle, transform=ax.transAxes)
        # proj.drawmapboundary(fill_color='white')
        # proj.drawcoastlines()
        plt.title(front+' PT='+str(lev)+' m='+str(m))
        l1,l2 = np.meshgrid(df_depth.columns.values,df_depth.index.values)
        # X,Y = proj(l1,l2)
        # proj.contour(X,Y,df_depth,[-400])
        ax.contour(l1,l2,df_depth,[-400],transform=ccrs.PlateCarree())
        for n in range(nn):
            # select one segment
            seg = coord[n]
            # convert to map coords
            # x, y = proj(seg[:,0], seg[:,1])
            x, y = [seg[:,0], seg[:,1]]
            # plot
            # proj.scatter(x,y,s=1)
            ax.scatter(x,y,s=1,transform=ccrs.PlateCarree())
            plt.pause(0.05)
            # input the order of the segment
            order.append(int(input("Number or enter") or '999')) 
        plt.show()
        order = np.array(order)

        # number of good segments
        good_seg = len(order[order!=999]) 
        # select the segments to keep and put in order
        ls_seg = [0] * good_seg
        for n in range(nn):
            if order[n] != 999:
                ls_seg[order[n]-1] = coord[n]

        # # save in dic as one array, shape: (v,2)
        dic_coord[front][m] = np.vstack(ls_seg)

        # test plot
        # plt.figure()
        # ax = plt.subplot(1, 1, 1, projection=proj)
        # ax.set_extent([-280, 80, -80, -20], crs=ccrs.PlateCarree())
        # ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
        # ax.set_boundary(circle, transform=ax.transAxes)
        # ax.coastlines(resolution='50m')
        # plt.title('STF '+str(lev))
        # for n in range(4):
        #   # select one segment
        #   seg = ls_seg[n]
        #   # convert to map coords
        #   # x, y = proj(seg[:,0], seg[:,1])
        #   x, y = [seg[:,0], seg[:,1]]
        #   # plot
        #   # proj.scatter(x,y,s=1)
        #   ax.scatter(x,y,s=1,transform=ccrs.PlateCarree())


file = open(directory_out + 'front_coord_dict_monthly_workingdic_cartopy.pkl', "wb")
pickle.dump(dic_coord, file)
file.close()


# set max latitude at -30S for STF
for m in range(1,13):
    dic_coord['STF'][m][:,1][dic_coord['STF'][m][:,1]>-30]=-30  


# complete SAF contour - 0m contour - only for plotting
dic_coord_plot = copy.deepcopy(dic_coord) 
front = 'SAF'
order = []
plt.figure()
ax = plt.subplot(1, 1, 1, projection=proj)
ax.set_extent([-280, 80, -80, -20], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.set_boundary(circle, transform=ax.transAxes)
ax.coastlines(resolution='50m')
plt.title(front)#+' P='+str(lev)+' m='+str(m)

CS = plt.contour(df_depth.columns.values,df_depth.index.values,df_depth.values,[0])
coord = CS.allsegs[0]

# l1,l2 = np.meshgrid(df_depth.columns.values,df_depth.index.values)
# X,Y = proj(l1,l2)
# X,Y = l1,l2
# select the right segment of 400m depth contour
for n in range(len(coord)):
    # select one segment
    segr = coord[n]
    # convert to map coords
    # x, y = proj(segr[:,0], segr[:,1])
    x, y = [segr[:,0], segr[:,1]]
    # plot
    # proj.scatter(x,y,s=1)
    ax.scatter(x,y,s=1,transform=ccrs.PlateCarree())
    plt.pause(0.05)
    # input 1 for the right segment 
    order.append(int(input("Number or enter") or '999'))
plt.show()
order = np.array(order) 
segr = coord[np.where(order == 1)[0][0]]
# split the segment in two along min lat
ind_min = segr[:,1].argmin()
segr_e = segr[ind_min:]
segr_w = segr[:ind_min]

# complete the PT contour
for m in range(1,13):
    # m=1
    # set the range of lon/lat missing 
    lonm1  = dic_coord[front][m][-1,0]
    latm1  = dic_coord[front][m][-1,1]
    latm2  = dic_coord[front][m][0,1]
    lonm2  = dic_coord[front][m][0,0]

    # select the closest lon that's also close to the lat
    ind_lon1 = abs(segr[:,0]-lonm1)
    ind_lat1 = abs(segr[:,1]-latm1)
    ind1 = (ind_lon1 + ind_lat1).argmin()

    # ind_lon2 = abs(segr_e[:,0]-lonm2)
    # ind_lat2 = abs(segr_e[:,1]-latm2)
    # ind2 = (ind_lon2 + ind_lat2).argmin()
    ind_lat2 = abs(segr_e[:,1]-latm2).argmin()

    # seg_new = np.concatenate([segr_e[ind_lat2][np.newaxis,:],dic_coord[front][m]])  
    seg_add = segr[ind1:len(segr_w)+ind_lat2]  

    # add to the saved front coords
    # dic_coord[front][m] = np.concatenate([seg_new,segr[ind1][np.newaxis,:]]) 

    # add to the saved front coords
    dic_coord_plot[front][m] = np.concatenate([dic_coord[front][m],seg_add])   




# make a polygon for each front
# create a dic to hold the polygon of each zone
dic_poly = {}
for front in ls_fronts:
    # front = 'PF'
    print()
    print(front)
    dic_poly[front] = {}
    for m in range(1,13):
        # m=1
        # make a polygon for the zone 
        # project lat/lon of front
        xys= proj.transform_points( ccrs.PlateCarree(), dic_coord_plot[front][m][:,0], dic_coord_plot[front][m][:,1] )
        X = xys[:,0]  # all x's
        Y = xys[:,1]
        vert = np.vstack((X,Y)).transpose()
        # make the polygon
        polygon = Polygon(vert)
        # save in dic
        dic_poly[front][m] = polygon

        # test plot
        # plt.figure()
        # X,Y = polygon.exterior.xy
        # ax = plt.subplot(1, 1, 1, projection=proj)
        # ax.set_extent([-280, 80, -80, -20], crs=ccrs.PlateCarree())
        # ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
        # ax.set_boundary(circle, transform=ax.transAxes)
        # ax.coastlines(resolution='50m')
        # ax.plot(X,Y,label='STF') #,transform=ccrs.PlateCarree()
        # # proj.drawcoastlines()
        # # proj.drawparallels(np.arange(-90,-20,10.))
        # plt.show()


## make SIF ##
front='SIF'

# select september sea ice
if ice == 'era5':
    si = ds_sim.sel(month=9)
elif ice == 'NSIDC':
    si = ds_ns_m

# make contour
plt.figure()
CS = plt.contour(si.longitude,si.latitude,si,[si_lev])
plt.show()

# list of all the coordinates of the segments, one array per segment
coord = CS.allsegs[0]
# number of segments
nn = len(coord)

order = []
plt.figure()
ax = plt.subplot(1, 1, 1, projection=proj)
ax.set_extent([-280, 80, -80, -20], crs=ccrs.PlateCarree())
ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
ax.set_boundary(circle, transform=ax.transAxes)
ax.coastlines(resolution='50m')
plt.title(front+' SIC='+str(si_lev))
# l1,l2 = np.meshgrid(df_depth.columns.values,df_depth.index.values)
# X,Y = proj(l1,l2)
# proj.contour(X,Y,df_depth,[-400])
for n in range(nn):
    # select one segment
    seg = coord[n]
    # convert to map coords
    # x, y = proj(seg[:,0], seg[:,1])
    x, y = [seg[:,0], seg[:,1]]
    # plot
    # proj.scatter(x,y,s=1)
    ax.scatter(x,y,s=1,transform=ccrs.PlateCarree())
    plt.pause(0.05)
    # input the order of the segment
    order.append(int(input("Numer or enter") or '999')) 
plt.show()
order = np.array(order)

# number of good segments
good_seg = len(order[order!=999]) 
# select the segments to keep and put in order
ls_seg = [0] * good_seg
for n in range(nn):
    if order[n] != 999:
        ls_seg[order[n]-1] = coord[n]

# save in dic as one array, shape: (v,2)
dic_coord[front] = np.vstack(ls_seg)
dic_coord_plot[front] = np.vstack(ls_seg)

# remove weird lon = 112.5 point
if ice == 'NSIDC':
    dic_coord[front] = np.delete(np.vstack(ls_seg),1131,axis=0)
    dic_coord_plot[front] = np.delete(np.vstack(ls_seg),1131,axis=0)

# print to check
# for i in range(dic_coord['SIF2'].shape[0]):
#     print(dic_coord['SIF2'][i])


# Make a polygon for SIF
# X,Y = proj(dic_coord[front][:,0],dic_coord[front][:,1])
xys= proj.transform_points( ccrs.PlateCarree(), dic_coord['SIF'][:,0],dic_coord['SIF'][:,1] )
X = xys[:,0]  # all x's
Y = xys[:,1]
vert = np.vstack((X,Y)).transpose()
# make the polygon
polygon = Polygon(vert)
# save in dic
dic_poly['SIF'] = polygon


# Make a polygon for circle at 30S
lon30 = np.arange(0,360)
lat30 = np.array([-30]*len(lon30))
# X,Y = proj(lon30,lat30)
xys= proj.transform_points( ccrs.PlateCarree(),lon30,lat30 )
X = xys[:,0]  # all x's
Y = xys[:,1]
vert = np.vstack((X,Y)).transpose()
# make the polygon
polygon = Polygon(vert)
# save in dic
dic_poly['Sof30'] = polygon


# save the coord dic
file = open(directory_out + file_out_coord, "wb")
pickle.dump(dic_coord, file)
file.close()

file = open(directory_out + file_out_coord_plot, "wb")
pickle.dump(dic_coord_plot, file)
file.close()

# save the poly dic
file = open(directory_out + file_out_poly, "wb")
pickle.dump(dic_poly, file)
file.close()


# # plot the 4 contours 
for m in range(1,13):
    plt.figure()
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_extent([-280, 80, -80, -20], crs=ccrs.PlateCarree())
    ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(resolution='50m')
    plt.title('month = '+str(m))
    for front in ls_fronts:
        X,Y = dic_poly[front][m].exterior.xy
        # proj.plot(X,Y,label=front)
        ax.plot(X,Y,label=front) 
    X,Y = dic_poly['SIF'].exterior.xy
    # proj.plot(X,Y,label='SIF')
    ax.plot(X,Y,label='SIF') 
    # proj.drawparallels(np.arange(-90,-20,10.))
    # X,Y = dic_poly['SIF2'].exterior.xy
    # proj.plot(X,Y,label='SIF2')
    plt.legend()
    file_fig = 'contours_'+tag+'_'+str(m)+'.eps' 
    plt.savefig(directory_fig+file_fig,format='eps',dpi=200)
    plt.show()



# plot the integration version
for m in range(1,13):
    plt.figure()
    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_extent([-280, 80, -80, -20], crs=ccrs.PlateCarree())
    ax.add_feature(land_50m, color=[0.8, 0.8, 0.8])
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.coastlines(resolution='50m')
    plt.title('month = '+str(m))
    for front in ls_fronts:
        lon = dic_coord[front][m][:,0]
        lat = dic_coord[front][m][:,1]
        # X,Y = proj(lon,lat)
        ax.plot(lon,lat,label=front,transform=ccrs.PlateCarree())
    lon = dic_coord['SIF'][:,0]
    lat = dic_coord['SIF'][:,1]
    # X,Y = proj(lon,lat)
    ax.plot(lon,lat,label='SIF',transform=ccrs.PlateCarree())
    # proj.drawparallels(np.arange(-90,-20,10.))
    # X,Y = dic_poly['SIF2'].exterior.xy
    # proj.plot(X,Y,label='SIF2')
    plt.legend()
    file_fig = 'contours_'+tag+'_'+str(m)+'.eps' 
    # plt.savefig(directory_fig+file_fig,format='eps',dpi=200)
    plt.show()




