# -*- coding: utf-8 -*-
# @Author: jadesauve
# @Date:   2022-07-12 


## Import packages ##
import pandas as pd
import numpy as np
import xarray as xr
from shapely.geometry import Point


def zone_select_monthly(Y, X, m):
    """
    this version uses cartopy
    this new version has monthly fronts

    accepts an X,Y, already converted using
    import cartopy.crs as ccrs
    proj = ccrs.SouthPolarStereo()
    xys= proj.transform_points(ccrs.PlateCarree(),lon,lat)
    X = xys[:,0]  # all x's
    Y = xys[:,1]
    where lon is -180,180 (is that right?, is that necessary?)

    m is the month from 1 to 12
    """
    # select the right zone according to lat/lon of a point

    # polygons for each zone
    directory_in = '/Users/jadesauve/Coding/output/'
    dic_poly = pd.read_pickle(directory_in + 'front_polygon_dict_monthly_6.0_NSIDC_Carto.pkl')

    if pd.notnull(Y and X):
        point = Point(X, Y)
        if point.within(dic_poly['SIF']):
            zone = 5
        elif point.within(dic_poly['PF'][m]):
            zone = 4
        elif point.within(dic_poly['SAF'][m]):
            zone = 3
        elif point.within(dic_poly['STF'][m]):
            zone = 2
        elif point.within(dic_poly['Sof30']):
            zone = 1
        else:
            # print('There is a problem with the zone, check manually!')
            # print('5905980, 5904844, 5905981, 5905982: north of 25S put in STZ')
            zone = 1
        return zone 
    else:
        return 0

