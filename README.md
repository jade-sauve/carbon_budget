# carbon_budget
Code necessary to reproduce my work on float-based monthly mixed layer carbon budget in the High Latitude Southern Ocean

To find the location of the Southern Ocean fronts using the Roemmich-Gilson argo climatology:
1- RG_monthly_climatology.py to make a monthly climatology
2- RG_potential_temp.py to find the potential temperature 
3- ETOPO1_depth_area_for_zone_areas.py to put the ETOPO2 depth data on the same grid as the RG argo climatology
3- RG_front_contours_3.0.py to identify the fronts and make a polygon for each zone
4- RG_Front_selection.py function to identify which zone a (lat,lon) is in
