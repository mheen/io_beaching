from particles import BeachingParticles, Density
from ocean_utilities import get_global_grid
from utilities import get_dir, convert_time_to_datetime, convert_datetime_to_time
from datetime import datetime,timedelta
from netCDF4 import Dataset
import numpy as np
import log

# -----------------------------------------------
# Sensitivity parameters
# -----------------------------------------------
def get_dx():
    dx = [2,4,8,16]
    return dx

def get_probabilities():
    p = [0.05,0.275,0.50,0.725,0.95]
    return p

def get_defaults():
    dx = 8
    p = 0.5
    return dx,p

# -----------------------------------------------
# Write to netcdf functions
# -----------------------------------------------
def _write_density_to_netcdf(density,output_path):
    log.info(None,f'Writing density to netcdf: {output_path}')
    nc = Dataset(output_path,'w',format='NETCDF4')
    # define dimensions
    nc.createDimension('time',len(density.time))
    nc.createDimension('lat',len(density.grid.lat))
    nc.createDimension('lon',len(density.grid.lon))
    # define variables
    nc_time = nc.createVariable('time',float,'time',zlib=True)
    nc_lon = nc.createVariable('lon',float,'lon',zlib=True)
    nc_lat = nc.createVariable('lat',float,'lat',zlib=True)
    nc_density = nc.createVariable('density',float,('time','lat','lon'),zlib=True)
    nc_tp = nc.createVariable('total_particles',float,'time',zlib=True)
    # write variables
    time, time_units = convert_datetime_to_time(density.time)
    nc_time[:] = time
    nc_time.units = time_units
    nc_lon[:] = density.grid.lon
    nc_lat[:] = density.grid.lat
    nc_density[:] = density.density
    nc_tp[:] = density.total_particles
    nc.close()

def _write_to_netcdf(particles,output_path):
    log.info(None,f'Writing particle data to: {output_path}')
    nc = Dataset(output_path,'w',format='NETCDF4')
    # define dimensions
    nc.createDimension('pid',len(particles.pid))        
    nc.createDimension('time',len(particles.time))
    # define variables
    nc_pid = nc.createVariable('pid',float,'pid',zlib=True)
    nc_time = nc.createVariable('time',float,'time',zlib=True)
    nc_lon = nc.createVariable('lon',float,('pid','time'),zlib=True)
    nc_lat = nc.createVariable('lat',float,('pid','time'),zlib=True)
    nc_beached = nc.createVariable('beached',float,('pid','time'),zlib=True)
    # write variables
    nc_pid[:] = particles.pid
    time, time_units = convert_datetime_to_time(particles.time)
    nc_time[:] = time
    nc_time.units = time_units
    nc_lon[:] = particles.lon
    nc_lat[:] = particles.lat
    nc_beached[:] = particles.beached
    nc.close()

# -----------------------------------------------
# Processing
# -----------------------------------------------
def _get_parcels_particles_path(file_description,year=None,
                                dir_description='pts_output'):
    parcels_dir = get_dir(dir_description)
    if year is None:
        parcels_path = f'{parcels_dir}{file_description}.nc'
        return parcels_path
    parcels_path = f'{parcels_dir}{file_description}{year}.nc'
    return parcels_path

def _get_non_beaching_path(basin_name,description,extra_description=None,
                           dir_description='pts_processed'):
    processed_dir = get_dir(dir_description)
    if extra_description is None:
        non_beached_path = f'{processed_dir}{basin_name}_{description}.nc'
    else:
        non_beached_path = f'{processed_dir}{basin_name}_{extra_description}_{description}.nc'
    return non_beached_path

def _get_beaching_path(basin_name,dx,p,description,
                       extra_description=None,
                       dir_description='pts_processed'):
    processed_dir = get_dir(dir_description)
    p_str = str(p).replace('.','')
    if extra_description is None:
        beached_path = f'{processed_dir}{basin_name}_{description}_{dx}km_p{p_str}.nc'
    else:
        beached_path = f'{processed_dir}{basin_name}_{description}_{dx}km_p{p_str}_{extra_description}.nc'
    return beached_path

def get_particles_path(basin_name,extra_description=None,
                        dir_description='pts_processed'):
    particles_path = _get_non_beaching_path(basin_name,'particles',
                                            extra_description=extra_description,
                                            dir_description=dir_description)
    return particles_path

def get_beached_particles_path(basin_name,dx,p,
                                extra_description=None,
                                dir_description='pts_processed'):
    beached_particles_path = _get_beaching_path(basin_name,dx,p,'beached',
                                                extra_description=extra_description,
                                                dir_description=dir_description)
    return beached_particles_path

def get_density_path(basin_name,extra_description=None,
                      dir_description='pts_processed'):
    density_path = _get_non_beaching_path(basin_name,'density',
                                          extra_description=extra_description,
                                          dir_description=dir_description)
    return density_path

def get_beached_density_path(basin_name,dx,p,extra_description=None,
                              dir_description='pts_processed'):
    beached_density_path = _get_beaching_path(basin_name,dx,p,'density',
                                              extra_description=extra_description,
                                              dir_description=dir_description)
    return beached_density_path

def process_beached_density(particles,dx,p,basin_name='io_nh',dx_grid=0.5,
                            output_description=None):
    grid = get_global_grid(dx=dx_grid)
    output_path = get_beached_density_path(basin_name,dx,p,extra_description=output_description)
    density = Density.create_from_particles(grid,particles)
    _write_density_to_netcdf(density,output_path)

def process_density(particles,basin_name='io_nh',dx_grid=0.5,output_description=None):
    grid = get_global_grid(dx=dx_grid)
    output_path = get_density_path(basin_name,extra_description=output_description)
    density = Density.create_from_particles(grid,particles)
    _write_density_to_netcdf(density,output_path)

def process_beached_particles(particles,dx,p,basin_name='io_nh',output_description=None):
    beached_particles = particles.get_beachingparticles_at_distance_dx_with_probability(dx,p)
    output_path = get_beached_particles_path(basin_name,dx,p,extra_description=output_description)
    _write_to_netcdf(beached_particles,output_path)
    return beached_particles

def process_particles(input_description='io_river_sources_',
                      basin_name='io_nh',
                      years=np.arange(1995,2016,1),
                      output_description=None):
    if years is not None:
        input_path = _get_parcels_particles_path(input_description,year=years[0])
        particles = BeachingParticles.read_from_parcels_netcdf(input_path)
        for year in years[1:]:
            input_path = _get_parcels_particles_path(input_description,year=year)
            particles.add_particles_from_parcels_netcdf(input_path)
    else:
        input_path = _get_parcels_particles_path(input_description)
        particles = BeachingParticles.read_from_parcels_netcdf(input_path)
    log.info(None,f'Extracting particles from initial basin only: {basin_name}')
    basin_particles = particles.get_particles_from_initial_basin(basin_name)
    # write to netcdf
    output_path = get_particles_path(basin_name,extra_description=output_description)
    _write_to_netcdf(basin_particles,output_path)
    return basin_particles

def process_specific_file_density(input_path, output_path, dx_grid=0.5):
    particles = BeachingParticles.read_from_netcdf(input_path)
    grid = get_global_grid(dx=dx_grid)
    density = Density.create_from_particles(grid, particles)
    _write_density_to_netcdf(density, output_path)

def process_specific_file_particles_in_lon_lat_range(input_path, output_path, lon_range=None, lat_range=None, t_interval=1):
    if type(input_path) == list:
        particles = BeachingParticles.read_from_parcels_netcdf(input_path[0], t_interval=t_interval)
        for i in range(len(input_path)-1):
            particles.add_particles_from_parcels_netcdf(input_path[i+1])
    elif type(input_path) == str:
        particles = BeachingParticles.read_from_parcels_netcdf(input_path, t_interval=t_interval)
    if lon_range is not None and lat_range is not None:
        particles_in_range = particles.get_particles_from_initial_lon_lat_range(lon_range, lat_range)
        _write_to_netcdf(particles_in_range, output_path)
    else:
        _write_to_netcdf(particles, output_path)

if __name__ == '__main__':
    basin_names = ['io_nh','io_sh', 'io']
    dx,_ = get_defaults()
    ps = get_probabilities()
    for basin_name in basin_names:
        particles = process_particles(basin_name=basin_name)
        process_density(particles,basin_name=basin_name)
        for p in ps:
            beached_particles = process_beached_particles(particles,dx,p,basin_name=basin_name)
            process_beached_density(beached_particles,dx,p,basin_name=basin_name)
            beached_particles = None
        particles = None
    