from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, DiffusionUniformKh, Field, ErrorCode
from pts_parcels_io_beaching import _get_io_indices_from_netcdf
from pts_parcels_kernels import delete_particle
from utilities import get_dir, get_daily_ncfiles_in_time_range
from utilities import get_distance_between_points
from ocean_utilities import LandMask
from datetime import datetime, timedelta
import numpy as np

def get_release_time_lon_lat():
    lon_org = 79.757667
    lat_org = 7.071556
    start_date = datetime(2008, 5, 22) # first explosion
    end_date = datetime(2008, 6, 2) # stern sunk
    n_particles = 1000 # per release
    release_interval_hours = 3
    n_releases = ((end_date-start_date).days+1)*int(24/release_interval_hours) # 3 hourly release
    lm = LandMask.read_from_netcdf()
    lm_halo = lm.get_landmask_with_halo()
    lon, lat = lm_halo.get_closest_ocean_point(lon_org, lat_org)
    time = np.array([start_date+timedelta(hours=i*release_interval_hours) for i in range(n_releases)])
    time0 = np.repeat(time, n_particles)
    lon0 = np.repeat(lon, len(time0))
    lat0 = np.repeat(lat, len(time0))
    return time0, lon0, lat0

def get_kh_value_from_grid_size(lat0, input_path='input/hycom_landmask.nc') -> float:
    lm = LandMask.read_from_netcdf(input_path=input_path)
    mean_dx_degree = np.nanmean([np.nanmean(np.diff(lm.lon)), np.nanmean(np.diff(lm.lat))])
    dx = get_distance_between_points(113.0, lat0, 113.0+mean_dx_degree, lat0) # m
    epsilon = 10**(-9) # m^2/s^3
    kh = epsilon**(1/3)*dx**(4/3)
    return np.round(kh, 2)

def run():
    input_dir = get_dir('hycom_input')
    output_dir = get_dir('xpresspearl_output')
    output_name = 'pts_parcels_2008-2009'
    time0, lon0, lat0 = get_release_time_lon_lat()
    start_date = datetime(2008, 5, 22)
    end_date = datetime(2010, 5, 22)
    indices=_get_io_indices_from_netcdf()
    kh = get_kh_value_from_grid_size(lat0[0])
    interp_method = 'linear'
    # get paths
    ncfiles = get_daily_ncfiles_in_time_range(input_dir, start_date, end_date)
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
    kernel = pset.Kernel(AdvectionRK4) + pset.Kernel(DiffusionUniformKh)
    output_file = pset.ParticleFile(name=output_path,outputdt=dt*output_interval)
    pset.execute(kernel,runtime=run_time,dt=dt,output_file=output_file,verbose_progress=True,
                 recovery={ErrorCode.ErrorOutOfBounds: delete_particle})

if __name__ == '__main__':
    run()
    