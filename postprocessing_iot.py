from plastic_sources import RiverSources
from particles import BeachingParticles
import numpy as np

def get_iot_lon_lat_range():
    lon_range = [90., 128.]
    lat_range = [-20., 6.]
    return (lon_range, lat_range)

def get_iot_sources(original=False):
    iot_lon, iot_lat = get_iot_lon_lat_range()
    if not original:
        global_sources = RiverSources.read_from_netcdf()
    else:
        global_sources = RiverSources.read_from_shapefile()
    io_sources = global_sources.get_riversources_from_ocean_basin('io')
    iot_sources = io_sources.get_riversources_in_lon_lat_range(iot_lon, iot_lat)
    return iot_sources

def get_cki_box_lon_lat_range():
    lon_range = [96.6, 97.1]
    lat_range = [-12.3, -11.8]
    return (lon_range, lat_range)

def get_christmas_box_lon_lat_range():
    lon_range = [105.4, 105.9]
    lat_range = [-10.7, -10.2]
    return (lon_range, lat_range)

def get_l_particles_in_box(particles: BeachingParticles, iot_island):
    if iot_island == 'cki':
        box_lon, box_lat = get_cki_box_lon_lat_range()
    elif iot_island == 'christmas':
        box_lon, box_lat = get_christmas_box_lon_lat_range()
    else:
        raise ValueError(f'Unknown iot_island {iot_island}, valid options are: cki and christmas.')
    l_in_box = []
    for i in range(len(particles.pid)):
        l_lon = np.logical_and(box_lon[0]<=particles.lon[i, :], particles.lon[i, :]<=box_lon[1])
        l_lat = np.logical_and(box_lat[0]<=particles.lat[i, :], particles.lat[i, :]<=box_lat[1])
        l_in_box.append(np.logical_and(l_lon, l_lat).any())
    return np.array(l_in_box)

def get_particle_entry_time_box(particles: BeachingParticles, iot_island):
    l_box = get_l_particles_in_box(particles, iot_island)
    if iot_island == 'cki':
        box_lon, box_lat = get_cki_box_lon_lat_range()
    elif iot_island == 'christmas':
        box_lon, box_lat = get_christmas_box_lon_lat_range()
    else:
        raise ValueError(f'Unknown iot_island {iot_island}, valid options are: cki and christmas.')
    i_box = np.where(l_box)[0]
    time_entry = []
    for i in i_box:
        l_lon = np.logical_and(box_lon[0]<=particles.lon[i, :], particles.lon[i, :]<=box_lon[1])
        l_lat = np.logical_and(box_lat[0]<=particles.lat[i, :], particles.lat[i, :]<=box_lat[1])
        i_first = np.where(np.logical_and(l_lon, l_lat))[0][0]
        time_entry.append(particles.time[i_first])
    return np.array(time_entry)

def get_particle_release_time_box(particles: BeachingParticles, iot_island):
    l_box = get_l_particles_in_box(particles, iot_island)
    _, t_release = particles._get_release_time_and_particle_indices()
    time_release = particles.time[t_release[l_box]]
    return time_release

def get_particle_release_locations_box(particles: BeachingParticles, iot_island):
    l_box = get_l_particles_in_box(particles, iot_island)
    lon0, lat0 = particles.get_initial_particle_lon_lat()
    return (lon0[l_box], lat0[l_box])

def get_main_sources_lon_lat_n_particles(particles: BeachingParticles, iot_island):
    lon0, lat0 = get_particle_release_locations_box(particles, iot_island)
    coordinates0 = []
    for i in range(len(lon0)):
        coordinates0.append([lon0[i], lat0[i]])
    coordinates0 = np.array(coordinates0)
    coordinates0_unique, counts_unique = np.unique(coordinates0, axis=0, return_counts=True)
    lon0_unique = coordinates0_unique[:, 0]
    lat0_unique = coordinates0_unique[:, 1]
    return (lon0_unique, lat0_unique, counts_unique)

def get_original_source_based_on_lon0_lat0(lon0, lat0):
    iot_sources = get_iot_sources()
    iot_sources_org = get_iot_sources(original=True)
    i_sources = np.where(np.logical_and(iot_sources.lon == lon0, iot_sources.lat == lat0))
    lon = iot_sources_org.lon[i_sources]
    lat = iot_sources.lat[i_sources]
    yearly_waste = np.sum(iot_sources.waste[i_sources], axis=1)
    return (lon, lat, yearly_waste)

def get_n_particles_per_month_release_arrival(particles: BeachingParticles, iot_island):
    release_time = get_particle_release_time_box(particles, iot_island)
    release_months = np.array([x.month for x in release_time])
    entry_time = get_particle_entry_time_box(particles, iot_island)
    entry_months = np.array([x.month for x in entry_time])
    months = np.arange(1,13,1)
    n_release = []
    n_entry = []
    for month in months:
        n_release.append(np.sum(release_months==month))
        n_entry.append(np.sum(entry_months==month))
    n_entry_per_release_month = np.zeros((len(months), len(months)))
    for i in range(len(entry_months)):
        n_entry_per_release_month[entry_months[i]-1, release_months[i]-1] += 1
    return np.array(n_release), np.array(n_entry), n_entry_per_release_month
