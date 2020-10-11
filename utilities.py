from geopy.distance import geodesic
from datetime import datetime,timedelta
import numpy as np
import json
import os
import csv

# -----------------------------------------------
# General
# -----------------------------------------------
def get_dir(dirname,json_file='input/dirs.json'):
    with open(json_file,'r') as f:
        all_dirs = json.load(f)
    return all_dirs[dirname]

def get_ncfiles_in_dir(input_dir):
    ncfiles = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.nc'):
            ncfiles.append(filename)
    return ncfiles

def get_daily_ncfiles_in_time_range(input_dir,start_date,end_date,timeformat='%Y%m%d'):
        all_ncfiles = get_ncfiles_in_dir(input_dir)
        ndays = (end_date-start_date).days+1
        ncfiles = []
        for n in range(ndays):
            date = start_date+timedelta(days=n)
            for ncfile in all_ncfiles:
                if ncfile.startswith(date.strftime(timeformat)):
                    ncfiles.append(ncfile)
        return ncfiles

def get_closest_index(A,target):
    # A must be sorted!
    idx = A.searchsorted(target)
    idx = np.clip(idx,1,len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target-left < right-target
    return idx

def write_data_to_csv(data,output_path):
    with open(output_path,'w') as f:
        writer = csv.writer(f,quoting=csv.QUOTE_MINIMAL)
        for row in data:
            writer.writerow(row)

def get_matrix_value_or_nan(matrix,i,j):
        if np.isnan(i):
            return np.nan
        return matrix[i.astype('int'),j.astype('int')]

# -----------------------------------------------
# Timeseries
# -----------------------------------------------
def get_time_index(time_array,time):
    '''Returns exact index of a requested time, raises
    error if this does not exist.'''
    t = np.where(time_array==time)[0]
    if len(t) > 1:
        raise ValueError('Multiple times found in time array that equal requested time.')
    elif len(t) == 0:
        raise ValueError('Requested time not found in time array.')
    else:
        return t[0]

def get_closest_time_index(time_array,time):
    '''Returns exact index of a requested time if is exists,
    otherwise returns the index of the closest time.'''
    dt = abs(time_array-time)
    i_closest = np.where(dt == dt.min())[0][0]
    return i_closest

def get_l_time_range(time,start_time,end_time):
    if type(start_time) is datetime.date:
        start_time = datetime.datetime(start_time.year,start_time.month,start_time.day)
    if type(end_time) is datetime.date:
        end_time = datetime.datetime(end_time.year,end_time.month,end_time.day)
    l_start = time >= start_time
    l_end = time <= end_time
    l_time = l_start & l_end
    return l_time

def convert_time_to_datetime(time_org,time_units):
    time = []
    if 'since' in time_units:   
        i_start_time = time_units.index('since')+len('since')+1
    elif 'after' in time_units:
        i_start_time = time_units.index('after')+len('after')+1
    else:
        raise ValueError('Unknown time units: "since" or "after" not found in units.')
    if 'T' in time_units: # YYYY-mm-ddTHH:MM format used by Parcels
        i_end_time = i_start_time+len('YYYY-mm-ddTHH:MM')
        base_time = datetime.strptime(time_units[i_start_time:i_end_time],'%Y-%m-%dT%H:%M')
    else: # YYYY-mm-dd format used by multiple numerical models
        i_end_time = i_start_time+len('YYYY-mm-dd')
        base_time = datetime.strptime(time_units[i_start_time:i_end_time],'%Y-%m-%d')
    if time_units.startswith('seconds'):
        for t in time_org:
            if not np.isnan(t):
                time.append(base_time+timedelta(seconds=t))
            else:
                time.append(np.nan)
        return np.array(time)
    elif time_units.startswith('hours'):
        for t in time_org:
            if not np.isnan(t):
                time.append(base_time+timedelta(hours=t))
            else:
                time.append(np.nan)
        return np.array(time)
    elif time_units.startswith('days'):
        for t in time_org:
            if not np.isnan(t):
                time.append(base_time+timedelta(seconds=t))
            else:
                time.append(np.nan)
        return np.array(time)
    else:
        raise ValueError('Unknown time units for time conversion to datetime.')

def convert_datetime_to_time(time_org,time_units='seconds',time_origin=datetime(1995,1,1,12,0)):
    time = []
    if time_units == 'seconds':
        conversion = 1
    elif time_units == 'hours':
        conversion = 60*60
    elif time_units == 'days':
        conversion = 24*60*60
    else:
        raise ValueError('Unknown time units requested fro time conversion from datetime.')
    for t in time_org:        
        time.append((t-time_origin).total_seconds()/conversion)
    return np.array(time)

# -----------------------------------------------
# Coordinates
# -----------------------------------------------
def get_distance_between_points(lon1,lat1,lon2,lat2):
    pos1 = (lat1,lon1)
    pos2 = (lat2,lon2)
    distance = geodesic(pos2,pos1).meters
    return distance

def convert_lon_360_to_180(lon):
    lon[lon>180] = lon[lon>180]-360
    return lon
