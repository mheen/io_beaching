from particles import BeachingParticles
import numpy as np

def get_cki_box_lon_lat_range():
    lon_range = [96.6, 97.1]
    lat_range = [-12.3, -11.8]
    return (lon_range, lat_range)

def get_l_particles_in_box(particles: BeachingParticles):
    box_lon, box_lat = get_cki_box_lon_lat_range()
    l_in_box = []
    for i in range(len(particles.pid)):
        l_lon = np.logical_and(box_lon[0]<=particles.lon[i, :], particles.lon[i, :]<=box_lon[1])
        l_lat = np.logical_and(box_lat[0]<=particles.lat[i, :], particles.lat[i, :]<=box_lat[1])
        l_in_box.append(np.logical_and(l_lon, l_lat).any())
    return np.array(l_in_box)

def get_particle_entry_time_box(particles: BeachingParticles):
    l_box = get_l_particles_in_box(particles)
    box_lon, box_lat = get_cki_box_lon_lat_range()
    i_box = np.where(l_box)[0]
    time_entry = []
    for i in i_box:
        l_lon = np.logical_and(box_lon[0]<=particles.lon[i, :], particles.lon[i, :]<=box_lon[1])
        l_lat = np.logical_and(box_lat[0]<=particles.lat[i, :], particles.lat[i, :]<=box_lat[1])
        i_first = np.where(np.logical_and(l_lon, l_lat))[0][0]
        time_entry.append(particles.time[i_first])
    return np.array(time_entry)

def get_particle_release_time_box(particles: BeachingParticles):
    l_box = get_l_particles_in_box(particles)
    _, t_release = particles._get_release_time_and_particle_indices()
    time_release = particles.time[t_release[l_box]]
    return time_release

def get_particle_release_locations_box(particles: BeachingParticles):
    l_box = get_l_particles_in_box(particles)
    lon0, lat0 = particles.get_initial_particle_lon_lat()
    return (lon0[l_box], lat0[l_box])

def get_n_particles_per_month_release_arrival(particles: BeachingParticles):
    release_time = get_particle_release_time_box(particles)
    release_months = np.array([x.month for x in release_time])
    entry_time = get_particle_entry_time_box(particles)
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
