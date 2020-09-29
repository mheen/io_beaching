from utilities import convert_time_to_datetime, get_dir, get_ncfiles_in_dir
from utilities import get_time_index, get_closest_time_index
from ocean_utilities import Grid, get_global_grid, get_io_lon_lat_range, OceanBasinGrid, LandMask
from coast import CoastDistance
from plot_tools.map_plotter import MapPlot
from plot_tools.plot_cycler import plot_cycler
import cartopy.crs as ccrs
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import warnings
import random
import log

class Density:
    def __init__(self,grid,time,total_particles,density):
        self.grid = grid
        self.time = time
        self.total_particles = total_particles
        self.density = density

    def get_normalized_density(self):
        # density as percentage of total particles
        self.norm_density = self.density/self.total_particles[:,np.newaxis,np.newaxis]*100

    def plot(self,t_interval=1):        
        t = np.arange(0,len(self.time),t_interval)
        time = self.time[t]
        fig = plot_cycler(self._single_plot,time)
        fig.show()

    def plot_io(self,t_interval=1):
        t = np.arange(0,len(self.time),t_interval)
        time = self.time[t]
        fig = plot_cycler(self._single_plot_io,time)
        fig.show()

    def _single_plot(self,fig,req_time):
        t = get_closest_time_index(self.time,req_time)
        density = self.density[t,:,:]
        density[density == 0.] = np.nan
        title = self.time[t].strftime('%d-%m-%Y')
        ax = plt.gca(projection=ccrs.PlateCarree())
        mplot = MapPlot(ax,None,None,title=title)
        mplot.set_cbar_items(label='Particle density [# per grid cell]')
        mplot.pcolormesh(self.grid.lon,self.grid.lat,density)

    def _single_plot_io(self,fig,req_time):
        t = get_closest_time_index(self.time,req_time)
        lon_range,lat_range = get_io_lon_lat_range()        
        density = self.density[t,:,:]
        density[density == 0.] = np.nan
        ticks = np.arange(0,1.1,0.1)
        label = 'Particle density [% per '+str(self.grid.dx)+'$ \times $'+str(self.grid.dx)+'$^o$ cells]'
        title = self.time[t].strftime('%d-%m-%Y')
        ax = plt.gca(projection=ccrs.PlateCarree())
        mplot = MapPlot(ax,lon_range,lat_range,title=title)
        mplot.set_cbar_items(ticks=ticks,label=label,lim=[ticks[0],ticks[-1]])
        mplot.pcolormesh(self.grid.lon,self.grid.lat,density,cmap='rainbow')

    @staticmethod
    def create_from_particles(grid,particles):
        log.info(None,f'Calculating density from particles')
        total_particles = np.zeros((len(particles.time)))
        density = np.zeros((len(particles.time),grid.lat_size,grid.lon_size))
        lon_index,lat_index = grid.get_index(particles.lon,particles.lat)
        shape_2d_density = density[0,:,:].shape
        for t in range(len(particles.time)):
            density_1d = density[t,:,:].flatten()
            x = (lon_index[~np.isnan(lon_index[:,t]),t]).astype('int')
            y = (lat_index[~np.isnan(lat_index[:,t]),t]).astype('int')
            index_1d = np.ravel_multi_index(np.array([y,x]),shape_2d_density)
            np.add.at(density_1d,index_1d,1)
            density[t,:,:] = density_1d.reshape(shape_2d_density)
            total_particles[t] += sum((~np.isnan(particles.lon[:,:t+1])).any(axis=1))
        return Density(grid,particles.time,total_particles,density)

    @staticmethod
    def read_from_netcdf(input_path,time_start=None,time_end=None,t_index=None):
        data = Dataset(input_path)
        time_org = data['time'][:].filled(fill_value=np.nan)
        time_units = data['time'].units
        time = convert_time_to_datetime(time_org,time_units)
        if time_start is not None and time_end is not None:
            t_start = get_closest_time_index(time,time_start)
            t_end = get_closest_time_index(time,time_end)
            t = np.arange(t_start,t_end+1,1)
            density = data['density'][t,:,:].filled(fill_value=np.nan)
            total_particles = data['total_particles'][t].filled(fill_value=np.nan)
            time = time[t]
        elif t_index is not None:
            density = data['density'][t_index,:,:].filled(fill_value=np.nan)
            total_particles = data['total_particles'][t_index].filled(fill_value=np.nan)
            time = time[t_index]
        else:
            density = data['density'][:].filled(fill_value=np.nan)
            total_particles = data['total_particles'][:].filled(fill_value=np.nan)
        lon = data['lon'][:].filled(fill_value=np.nan)
        lat = data['lat'][:].filled(fill_value=np.nan)
        data.close()
        dx = np.unique(np.diff(lon))[0]
        dy = np.unique(np.diff(lat))[0]
        lon_range = [lon.min(),lon.max()]
        lat_range = [lat.min(),lat.max()]
        grid = Grid(dx,lon_range,lat_range,dy=dy)
        return Density(grid,time,total_particles,density)

class BeachingParticles():
    def __init__(self,pid,time,lon,lat,beached,t_interval):
        self.pid = pid
        self.time = time
        self.lon = lon
        self.lat = lat
        self.beached = beached # 0 (ocean), 1 (beached), 2 (stuck on land during simulation); dimensions: [pid,time]
        self.t_interval = t_interval
        self._remove_duplicate_times()

    def _remove_duplicate_times(self):
        _,i_times = np.unique(self.time,return_index=True)
        self.time = self.time[i_times]
        self.lon = self.lon[:,i_times]
        self.lat = self.lat[:,i_times]
        self.beached = self.beached[:,i_times]

    def get_beachingparticles_at_distance_dx_with_probability(self,dx,probability):
        log.info(None,f'Applying beaching at: dx = {dx}, p = {probability}')
        (lon,lat,beached) = self._beaching_probability(dx,probability)
        return BeachingParticles(self.pid,self.time,lon,lat,beached,self.t_interval)

    def get_beachingparticles_at_dx_distance_after_dt_days(self,beaching_distance,allow_beaching_after_n_days):        
        (lon,lat,beached) = self._beaching(beaching_distance,allow_beaching_after_n_days)
        (lon,lat,beached) = self.get_stuck_on_land(lon,lat,beached,allow_beaching_after_n_days)
        return BeachingParticles(self.pid,self.time,lon,lat,beached,self.t_interval)

    def get_particles_from_initial_basin(self,basin_name,dx=0.1):
        beachingparticles = self.get_particles_from_basin_at_time_index(basin_name,'initial',dx)
        return beachingparticles

    def get_particles_from_final_basin(self,basin_name,dx=0.1):
        beachingparticles = self.get_particles_from_basin_at_time_index(basin_name,-1,dx)
        return beachingparticles

    def plot(self,t_interval=1):
        t = np.arange(0,len(self.time),t_interval)
        time = self.time[t]
        fig = plot_cycler(self._single_plot,time)        
        fig.show()

    def plot_io(self,t_interval=1):
        t = np.arange(0,len(self.time),t_interval)
        time = self.time[t]
        fig = plot_cycler(self._single_plot_io,time)
        fig.show()

    def _single_plot(self,fig,req_time):        
        t = get_closest_time_index(self.time,req_time)
        title = req_time.strftime('%d-%m-%Y')
        b = self.beached[:,t] == 1 # beached
        s = self.beached[:,t] == 2 # stuck during simulation
        ax = plt.gca(projection=ccrs.PlateCarree())
        mplot = MapPlot(ax,lon_range=None,lat_range=None,title=title)
        mplot.points(self.lon[:,t],self.lat[:,t],color='#000000')        
        mplot.points(self.lon[b,t],self.lat[b,t],color='#cc0000')
        mplot.points(self.lon[s,t],self.lat[s,t],color='#0000cc')

    def _single_plot_io(self,fig,req_time):
        lon_range,lat_range = get_io_lon_lat_range()        
        t = get_closest_time_index(self.time,req_time)
        title = req_time.strftime('%d-%m-%Y')
        b = self.beached[:,t] == 1 # beached
        s = self.beached[:,t] == 2 # stuck during simulation
        ax = plt.gca(projection=ccrs.PlateCarree())
        mplot = MapPlot(ax,lon_range=lon_range,lat_range=lat_range,title=title)
        mplot.points(self.lon[:,t],self.lat[:,t],color='#000000')
        mplot.points(self.lon[b,t],self.lat[b,t],color='#cc0000')
        mplot.points(self.lon[s,t],self.lat[s,t],color='#0000cc')

    def _beaching_probability(self,dx,probability):
        distance_to_coast = self._get_distance_to_coast(0)
        beached,p_beached,t_beached = self._get_beached_probability(distance_to_coast,dx,probability)
        lon,lat = self._freeze_particles_on_beach(p_beached,t_beached)
        return (lon,lat,beached)

    def _beaching(self,beaching_distance,allow_beaching_after_n_days):
        distance_to_coast = self._get_distance_to_coast(allow_beaching_after_n_days)
        beached,p_beached,t_beached = self._get_beached(distance_to_coast,beaching_distance)
        lon,lat = self._freeze_particles_on_beach(p_beached,t_beached)
        return (lon,lat,beached)

    def _freeze_particles_on_beach(self,p_beached,t_beached):        
        lon = self.lon.copy()
        lat = self.lat.copy()
        for p in range(len(p_beached)):
            lon[p_beached[p],t_beached[p]:] = lon[p_beached[p],t_beached[p]]
            lat[p_beached[p],t_beached[p]:] = lat[p_beached[p],t_beached[p]]
        return (lon,lat)

    def _get_beached_probability(self,distance_to_coast,dx,probability):
        beached = self.beached.copy()
        close_to_coast = distance_to_coast >= -dx # distance_to_coast < 0 for ocean
        dx_to_coast = np.diff(distance_to_coast)
        moving_towards_coast = dx_to_coast > 0
        possible_beaching = np.logical_and(close_to_coast[:,1:],moving_towards_coast)
        # changing_to_coast = np.diff(close_to_coast.astype('int'),axis=1) # -1: moving away from coast, +1: moving towards coast
        # p_close,t_close = np.where(changing_to_coast==1)
        p_close,t_close = np.where(possible_beaching)
        p_first = np.unique(p_close)
        p_beached = []
        t_beached = []
        for p in p_first:            
            i_p = np.where(p_close==p)[0]
            ts = t_close[i_p]            
            for i,t in enumerate(ts):                
                l_beached = self._get_beaching_decision(probability)
                if l_beached:                    
                    beached[p,t+1:] = 1
                    p_beached.append(p)
                    t_beached.append(t+1)
                    break
        return beached,np.array(p_beached),np.array(t_beached)

    def _get_beaching_decision(self,probability):
        '''Makes a random decision (True or False)
        based on probability'''
        return random.random() < probability

    def _get_beached(self,distance_to_coast,beaching_distance):
        beached = self.beached.copy()
        close_to_coast = distance_to_coast >= -beaching_distance # distance_to_coast < 0 for ocean
        changing_to_coast = np.diff(close_to_coast.astype('int'),axis=1) # -1: moving away from coast, +1: moving towards coast
        # note: this does not catch particles that always remain within beaching_distance
        p_beached,t_beached = np.where(changing_to_coast==1)
        # get first occurrence of beaching only
        p_first_beached,i_first = np.unique(p_beached,return_index=True)
        t_first_beached = t_beached[i_first]+1
        for p in range(len(p_first_beached)):
            beached[p_first_beached[p],t_first_beached[p]:] = 1
        return beached,p_first_beached,t_first_beached

    def _get_distance_to_coast(self,allow_beaching_after_n_days):
        coast = CoastDistance.read_from_netcdf()
        distance_to_coast = np.empty(self.lon.shape)*np.nan        
        for t in range(len(self.time)):
            l_no_nan = ~np.isnan(self.lon[:,t])
            lon = self.lon[l_no_nan,t]
            lat = self.lat[l_no_nan,t]
            distance_to_coast[l_no_nan,t] = coast.get_distance(lon,lat)
        # get time indices for which beaching is allowed
        p_first,t_release = self._get_release_time_and_particle_indices()
        t_beaching_allowed = t_release+allow_beaching_after_n_days/self.t_interval
        # replace distance_to_coast values with nan for times before beaching allowed
        for p in range(len(p_first)):
            distance_to_coast[p_first[p],:int(t_beaching_allowed[p])] = np.nan
        return distance_to_coast

    def _get_release_time_and_particle_indices(self):
        # when t_interval != 1 then this is not strictly the release time (ignoring this for now)
        p_no_nan,t_no_nan = np.where(~np.isnan(self.lon))
        p_first,i_sort = np.unique(p_no_nan,return_index=True)
        t_release = t_no_nan[i_sort]
        return p_first,t_release

    def label_stuck_on_land_but_do_not_freeze(self,lon,lat,beached):
        lm = HycomLandMask.read_from_netcdf()
        mask = np.empty(lon.shape)*np.nan
        # get stuck particles
        for t in range(lon.shape[1]):
            mask[:,t] = lm.get_multiple_mask_values(lon[:,t],lat[:,t])
        # get p and t indices for stuck particles
        p_stuck,t_stuck = np.where(mask==1)
        beached[p_stuck,t_stuck] = 2
        return beached

    def get_stuck_on_land(self,lon,lat,beached,allow_beaching_after_n_days):
        '''Finds particles that are already stuck on land during simulation.
        Marked as "2" in beached matrix.'''
        lm = HycomLandMask.read_from_netcdf()
        mask = np.empty(lon.shape)*np.nan
        # get stuck particles
        for t in range(lon.shape[1]):
            mask[:,t] = lm.get_multiple_mask_values(lon[:,t],lat[:,t])        
        # get time indices for which begin stuck is allowed (needed because stuck particles can refloat)
        p_first,t_release = self._get_release_time_and_particle_indices()
        t_stuck_allowed = t_release+allow_beaching_after_n_days/self.t_interval
        # replace mask values with nan for times before being stuck allowed
        for p in range(len(p_first)):
            mask[p_first[p],:int(t_stuck_allowed[p])] = np.nan
        # get first occurrence (after t_stuck_allowed) of particles stuck on land
        p_stuck,t_stuck = np.where(mask==1)
        p_stuck_first,i_first = np.unique(p_stuck,return_index=True)
        t_stuck_first = t_stuck[i_first]
        # freeze particles and mark as "2" in beached matrix
        for p in range(len(p_stuck_first)):
            lon[p_stuck_first[p],t_stuck_first[p]:] = lon[p_stuck_first[p],t_stuck_first[p]]
            lat[p_stuck_first[p],t_stuck_first[p]:] = lat[p_stuck_first[p],t_stuck_first[p]]
            beached[p_stuck_first[p],t_stuck_first[p]:] = 2
        return (lon,lat,beached)

    def get_particles_from_basin_at_time_index(self,basin_name,t,dx):
        if t == 'initial': # need to find first time separately for each particle (because they are added during simulation)
            i_all,j_all = np.where(~np.isnan(self.lon))
            i_first,i_sort = np.unique(i_all,return_index=True)
            j_first = j_all[i_sort]
            lon = self.lon[i_first,j_first]
            lat = self.lat[i_first,j_first]
        else:
            if t == -1:
                warnings.warn('Locations of final time available in BeachingParticles being used to determine'+
                              ' final particle basin. This is not the same as the final time for each particle!')
            lon = self.lon[:,t]
            lat = self.lat[:,t]
        basin = OceanBasinGrid(basin_name,dx)
        lon_index,lat_index = basin.grid.get_index(lon,lat)
        l_basin = basin.in_basin[lat_index.astype('int'),lon_index.astype('int')]
        pid = self.pid[l_basin]
        time = self.time
        lon = self.lon[l_basin,:]
        lat = self.lat[l_basin,:]
        beached = self.beached[l_basin,:]
        return BeachingParticles(pid,time,lon,lat,beached,self.t_interval)

    def add_particles_from_parcels_netcdf(self,input_path,t_interval=5):
        log.info(None,f'Adding particles from: {input_path}')
        (_,time,lon,lat,beached) = self.get_data_from_parcels_netcdf(input_path,t_interval=t_interval)
        self.time = np.append(self.time,time)
        self.lon = np.append(self.lon,lon,axis=1)
        self.lat = np.append(self.lat,lat,axis=1)
        self.beached = np.append(self.beached,beached,axis=1)

    @staticmethod
    def get_data_from_parcels_netcdf(input_path,t_interval=5):
        data = Dataset(input_path)
        pid = data['trajectory'][:,0].filled(fill_value=np.nan)
        time_all = data['time'][:].filled(fill_value=np.nan)
        time_units = data['time'].units
        lon_org = data['lon'][:].filled(fill_value=np.nan)
        lat_org = data['lat'][:].filled(fill_value=np.nan)
        data.close()
        # construct time array for all particles with dt as
        # maximum output frequency. this is needed because
        # when particles are deleted from the simulation,
        # their final time is written to file. this time
        # can be smaller than the specified output frequency.
        t0 = np.nanmin(time_all)
        tend = np.nanmax(time_all)
        dt = np.nanmax(np.diff(time_all))
        time_org = np.arange(t0,tend+dt,dt*t_interval)
        lon = np.empty((len(pid),len(time_org)))*np.nan
        lat = np.empty((len(pid),len(time_org)))*np.nan
        for t,time in enumerate(time_org):            
            i,j = np.where(time_all == time)            
            lon[i,np.repeat(t,len(i))] = lon_org[i,j]
            lat[i,np.repeat(t,len(i))] = lat_org[i,j]
        time = convert_time_to_datetime(time_org,time_units)
        beached = np.zeros(lon.shape)
        return (pid,time,lon,lat,beached)

    @staticmethod
    def read_from_parcels_netcdf(input_path,beached0=None,lon0=None,lat0=None,t_interval=5):
        log.info(None,f'Reading particles from: {input_path}')
        (pid,time,lon,lat,beached) = BeachingParticles.get_data_from_parcels_netcdf(input_path,t_interval=t_interval)
        if beached0 is not None:
            beached[:,0] = beached0
            pid_beached = np.where(beached0.astype('bool'))
            for p in pid_beached:
                beached[p,:] = np.repeat(beached0[p],beached.shape[1])
                lon[p,:] = np.repeat(lon0[p],lon.shape[1])
                lat[p,:] = np.repeat(lat0[p],lat.shape[1])
        return BeachingParticles(pid,time,lon,lat,beached,t_interval)

    @staticmethod
    def read_from_netcdf(input_path,time_start=None,time_end=None):
        log.info(None,f'Reading particles from: {input_path}')
        data = Dataset(input_path)
        pid = data['pid'][:].filled(fill_value=np.nan)
        time_org = data['time'][:].filled(fill_value=np.nan)
        time_units = data['time'].units
        time = convert_time_to_datetime(time_org,time_units)
        if time_start is not None and time_end is not None:
            t_start = get_closest_time_index(time,time_start)
            t_end = get_closest_time_index(time,time_end)
            t = np.arange(t_start,t_end+1,1)
            lon = data['lon'][:,t].filled(fill_value=np.nan)
            lat = data['lat'][:,t].filled(fill_value=np.nan)
            beached = data['beached'][:,t].filled(fill_value=np.nan)
            time = time[t]
        else:
            lon = data['lon'][:].filled(fill_value=np.nan)
            lat = data['lat'][:].filled(fill_value=np.nan)
            beached = data['beached'][:].filled(fill_value=np.nan)
        data.close()        
        return BeachingParticles(pid,time,lon,lat,beached,t_interval=5)
