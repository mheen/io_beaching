from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, BrownianMotion2D, SpatiallyVaryingBrownianMotion2D, Field, Variable, ErrorCode
from pts_parcels_kernels import wrap_lon_180, delete_particle
from plastic_sources import RiverSources
from utilities import get_dir, get_daily_ncfiles_in_time_range, get_closest_index, convert_time_to_datetime
from ocean_utilities import LandMask
import log
from datetime import datetime,timedelta
import numpy as np
import time
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os
import sys

class IORiverParticles:
    def __init__(self,lon0,lat0,time0):
        self.lon0 = lon0
        self.lat0 = lat0
        self.time0 = time0

    @staticmethod
    def get_from_netcdf(start_date,constant_release=False,input_path='input/PlasticRiverSources_Lebreton2017_Hycom.nc'):
        global_sources = RiverSources.read_from_netcdf(input_path=input_path)
        io_sources = global_sources.get_riversources_from_ocean_basin('io')
        year = start_date.year
        day = start_date.day
        hour = start_date.hour
        minute = start_date.minute
        lon0 = []
        lat0 = []
        time0 = []
        for t in io_sources.time.astype('int'):
            time = datetime(year,t,day,hour,minute)
            if constant_release is False:
                lon0_temp,lat0_temp,time0_temp = io_sources.convert_for_parcels_per_time(t-1,time)
            elif constant_release is True:
                lon0_temp,lat0_temp,time0_temp = io_sources.convert_for_parcels_constant(time)
            lon0 = np.append(lon0,lon0_temp)
            lat0 = np.append(lat0,lat0_temp)
            time0 = np.append(time0,time0_temp)
        return IORiverParticles(lon0,lat0,time0)

class RestartParticles:
    def __init__(self,input_path):
        netcdf = Dataset(input_path)
        pid = netcdf['trajectory'][:,0].filled(fill_value=np.nan)
        time_all = netcdf['time'][:].filled(fill_value=np.nan)
        time_units = netcdf['time'].units
        lon_org = netcdf['lon'][:].filled(fill_value=np.nan)
        lat_org = netcdf['lat'][:].filled(fill_value=np.nan)
        netcdf.close()
        # construct time array for all particles with dt as
        # maximum output frequency. this is needed because
        # when particles are deleted from the simulation,
        # their final time is written to file. this time
        # can be smaller than the specified output frequency.
        t0 = np.nanmin(time_all)
        tend = np.nanmax(time_all)
        dt = np.nanmax(np.diff(time_all))
        time_org = np.arange(t0,tend+dt,dt)
        lon = np.empty(lon_org.shape)*np.nan
        lat = np.empty(lat_org.shape)*np.nan
        total_particles = []
        for t,time in enumerate(time_org):            
            i,j = np.where(time_all == time)            
            lon[i,np.repeat(t,len(i))] = lon_org[i,j]
            lat[i,np.repeat(t,len(i))] = lat_org[i,j]
            total_particles.append(len(i))
        time0 = convert_time_to_datetime([time_org[-1]],time_units)                     
        # get particle locations from final time to restart
        self.lon0 = lon[:,-1]
        self.lat0 = lat[:,-1]
        self.time0 = np.repeat(time0,len(self.lon0))

    def plot(self,lon_range=[0.,130.],lat_range=[-56.,40.],
                plot_mplstyle='input/plot.mplstyle'):
        plt.style.use(plot_mplstyle)
        fig = plt.figure()
        ax = plt.gca(projection=ccrs.PlateCarree())
        ax.set_extent([lon_range[0],lon_range[1],lat_range[0],lat_range[1]],ccrs.PlateCarree())
        ax.add_feature(cftr.COASTLINE,edgecolor='k',zorder=1)
        ax.scatter(self.lon0,self.lat0,marker='.',c='r',s=3,
                       transform=ccrs.PlateCarree(),zorder=2)
        plt.show()

def _get_io_indices_from_netcdf(input_path='input/hycom_landmask.nc',lon_range=[0.,130.],lat_range=[-75,40]):
    netcdf = Dataset(input_path)
    lon = netcdf['lon'][:].filled(fill_value=np.nan)
    lat = netcdf['lat'][:].filled(fill_value=np.nan)
    i_lon_start = get_closest_index(lon,lon_range[0])
    i_lon_end = get_closest_index(lon,lon_range[1])
    i_lat_start = get_closest_index(lat,lat_range[0])
    i_lat_end = get_closest_index(lat,lat_range[1])
    indices = {'lon' : range(i_lon_start,i_lon_end), 'lat': range(i_lat_start,i_lat_end)}
    return indices

def run_hycom_subset_monthly_release(input_dir,output_dir,output_name,
                                     time0,lon0,lat0,start_date,end_date,
                                     kh=10.,interp_method='linear',indices=_get_io_indices_from_netcdf()):
    # get paths
    ncfiles = get_daily_ncfiles_in_time_range(input_dir,start_date,end_date)
    output_path = output_dir+output_name
    # create fieldset
    filenames = [input_dir+ncfile for ncfile in ncfiles]
    variables = {'U':'u','V':'v'}
    dimensions = {'lat':'lat','lon':'lon','time':'time'}
    fset = FieldSet.from_netcdf(filenames,variables,dimensions,indices=indices)    
    # add constant horizontal diffusivity (zero on land)
    lm = LandMask.read_from_netcdf()    
    kh2D = kh*np.ones(lm.mask.shape)
    kh2D[lm.mask.astype('bool')] = 0.0 # diffusion zero on land    
    kh2D_subset = kh2D[indices['lat'],:][:,indices['lon']]
    fset.add_field(Field('Kh_zonal',data=kh2D_subset,lon=fset.U.grid.lon,lat=fset.U.grid.lat,mesh='spherical',interp_method=interp_method))
    fset.add_field(Field('Kh_meridional',data=kh2D_subset,lon=fset.U.grid.lon,lat=fset.U.grid.lat,mesh='spherical',interp_method=interp_method))
    # montly release    
    pset = ParticleSet(fieldset=fset,pclass=JITParticle,lon=lon0,lat=lat0,time=time0)
    # execute
    run_time = timedelta(days=(end_date-start_date).days)
    dt = timedelta(hours=1)
    output_interval = 24    
    kernel = pset.Kernel(AdvectionRK4) + SpatiallyVaryingBrownianMotion2D #+ wrap_lon_180
    output_file = pset.ParticleFile(name=output_path,outputdt=dt*output_interval)
    pset.execute(kernel,runtime=run_time,dt=dt,output_file=output_file,verbose_progress=True,
                 recovery={ErrorCode.ErrorOutOfBounds: delete_particle})

if __name__ == '__main__':
    # use "main_run_pts_parcels.sh" bash script to run simulations
    year = int(sys.argv[1])
    input_dir = get_dir('hycom_input')
    output_dir = get_dir('pts_output')
    if year == 1995:
        # --- first run ---
        start_date = datetime(1995,1,1,12,0)
        end_date = datetime(1995,12,31)
        output_name = 'constant_io_river_sources_1995'
        io_sources = IORiverParticles.get_from_netcdf(start_date)
        run_hycom_subset_monthly_release(input_dir,output_dir,output_name,
                                         io_sources.time0,io_sources.lon0,io_sources.lat0,
                                         start_date,end_date)
    else:
        # --- consecutive runs ---
        restart_path = get_dir('hycom_output')+'hycom_currents/'+'constant_io_river_sources_'+str(year-1)+'.nc'
        restart = RestartParticles(restart_path)    
        start_date = restart.time0[0]
        end_date = datetime(year,12,31)
        output_name = 'constant_io_river_sources_'+str(year)
        run_hycom_subset_monthly_release(input_dir,output_dir,output_name,
                                        restart.time0,restart.lon0,restart.lat0,
                                        start_date,end_date)
    