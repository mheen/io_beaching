from utilities import get_dir, get_closest_time_index, convert_time_to_datetime, convert_datetime_to_time
from utilities import write_data_to_csv, read_data_from_csv
from ocean_utilities import OceanBasinGrid
from processing import get_beached_particles_path, get_beached_density_path
from processing import get_defaults, get_probabilities, get_dx
from particles import BeachingParticles, Density
from processing import get_probabilities
from countries import get_cgrid_with_halo_and_extended_islands
import log
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import warnings

def sort_most_common_countries(countries):
    count_countries = Counter(countries)
    count_countries_sorted = count_countries.most_common()
    sorted_countries = []
    for i in range(len(count_countries_sorted)):
        sorted_countries.append(count_countries_sorted[i][0])
    sorted_countries.remove('')
    return sorted_countries

class ConnectivityMatrix:
    def __init__(self, start_countries: list, end_countries: list, blame_matrix: np.ndarray):
        self.start_countries = start_countries
        self.end_countries = end_countries
        self.blame_matrix = blame_matrix

    def return_matrix_for_requested_countries(self,requested_countries):
        requested_matrix = np.empty((len(requested_countries),len(requested_countries)))*np.nan
        for n,country in enumerate(requested_countries):
            i = self.end_countries.index(country)
            js = np.where(self.blame_matrix[:,i] > 1)[0]
            if len(js) >= 1:
                for j in js:
                    if self.start_countries[j] in requested_countries:
                        m = requested_countries.index(self.start_countries[j])
                        requested_matrix[n,m] = self.blame_matrix[j,i]
        requested_matrix[requested_matrix < 1] = np.nan
        return requested_matrix

    def write_to_csv(self,output_path):
        log.info(None,f'Writing connectivity matrix to csv file: {output_path}')
        csv_data = []
        n_rows = len(self.end_countries)+1
        for i in range(n_rows):
            row = [None]*(len(self.start_countries)+1)
            if i == 0:
                row[1:] = self.start_countries
            else:
                row[0] = self.end_countries[i-1]
                row[1:] = self.blame_matrix[i-1,:]
            csv_data.append(row)
        write_data_to_csv(csv_data,output_path)

    @staticmethod
    def get_from_particles(particles: BeachingParticles):
        log.info(None,'Creating connectivity matrix from particles...')
        cgrid = get_cgrid_with_halo_and_extended_islands()
        lon_start,lat_start = particles.get_initial_particle_lon_lat()
        lon_end,lat_end = particles.get_final_particle_lon_lat()
        _,start_countries = cgrid.get_country_code_and_name(lon_start,lat_start)
        _,end_countries = cgrid.get_country_code_and_name(lon_end,lat_end)
        sorted_start_countries = sort_most_common_countries(start_countries)
        sorted_end_countries = sort_most_common_countries(end_countries)
        # create blame matrix
        blame_matrix = np.zeros((len(sorted_end_countries),len(sorted_start_countries)))
        for p in range(len(particles.pid)):
            if end_countries[p] == '':
                continue
            if start_countries[p] == '':
                warnings.warn(f'Start position of particle not in a country: lon = {lon_start[p]}, lat = {lat_start[p]}')
                continue
            i = sorted_end_countries.index(end_countries[p])
            j = sorted_start_countries.index(start_countries[p])
            blame_matrix[i,j] += 1
        # row normalize blame matrix [percentage]
        blame_matrix_normalized = np.copy(blame_matrix)
        for i in range(blame_matrix_normalized.shape[0]):
            blame_matrix_normalized[i,:] = blame_matrix_normalized[i,:]/sum(blame_matrix_normalized[i,:])*100
        return ConnectivityMatrix(sorted_start_countries,sorted_end_countries,blame_matrix_normalized)

    @staticmethod
    def read_from_csv(input_path: str):
        log.info(None,f'Reading connectivity matrix from csv file: {input_path}')
        data = read_data_from_csv(input_path)
        start_countries = data[0][1:]
        end_countries = []
        for i in range(1,len(data)):
            end_countries.append(data[i][0])
        blame_matrix = np.zeros((len(start_countries),len(end_countries)))
        for i in range(1,len(data)):
            blame_matrix[:,i-1] = np.array(data[i][1:])
        return ConnectivityMatrix(start_countries,end_countries,blame_matrix)

class ParticlesPerCountry:
    def __init__(self,countries : np.ndarray ,particles_per_country: np.ndarray):
        self.countries = countries
        self.particles_per_country = particles_per_country

    def write_to_csv(self,output_path):
        log.info(None,f'Writing particles per country to csv file: {output_path}')
        n_rows = len(self.countries)
        csv_data = []
        for i in range(n_rows):
            row = [None]*2
            row[0] = self.countries[i]
            row[1] = self.particles_per_country[i]
            csv_data.append(row)
        write_data_to_csv(csv_data,output_path)

    @staticmethod
    def get_from_density(density: Density, t = -1):
        log.info(None,'Getting particles per country from density...')
        cgrid = get_cgrid_with_halo_and_extended_islands()
        if len(density.density.shape) == 3:
            i,j = np.where(density.density[t,:,:] > 0)
            non_zero_density = density.density[t,i,j]
        else:
            i,j = np.where(density.density > 0)
            non_zero_density = density.density[i,j]
        lon = density.grid.lon[j]
        lat = density.grid.lat[i]
        codes = cgrid.get_country_code(lon,lat)
        unique_codes = np.unique(codes[~np.isnan(codes)])
        countries = []
        particles_per_country = []
        for code in unique_codes:
            k = np.where(codes == code)
            countries.append(cgrid.get_country_name(code))
            particles_per_country.append(np.sum(non_zero_density[k]))
        countries = np.array(countries)
        particles_per_country = np.array(particles_per_country)
        i_sort_descending = particles_per_country.argsort()[::-1]
        return ParticlesPerCountry(countries[i_sort_descending],particles_per_country[i_sort_descending])

    @staticmethod
    def read_from_csv(input_path):
        log.info(None,f'Reading particles per country from csv file: {input_path}')
        data = read_data_from_csv(input_path)
        countries = []
        particles_per_country = []
        for i in range(len(data)):
            country = (data[i][0]).replace("['","").replace("']","")
            countries.append(country)
            particles_per_country.append(data[i][1])
        countries = np.array(countries)        
        particles_per_country = np.array(particles_per_country).astype(float)
        return ParticlesPerCountry(countries,particles_per_country)

class IoParticleDevelopmentTimeseries:
    def __init__(self,time,total_particles,ocean_nio,ocean_sio,beached_nio,beached_sio,escaped_io):
        self.time = time
        self.total_particles = total_particles
        self.ocean_nio = ocean_nio
        self.ocean_sio = ocean_sio
        self.beached_nio = beached_nio
        self.beached_sio = beached_sio
        self.escaped_io = escaped_io

    def plot(self,plot_style='plot_tools/plot.mplstyle'):
        plt.style.use(plot_style)
        plt.plot(self.time,self.beached_nio/self.total_particles*100,'-',color='#cc0000',linewidth=2,label='beached in NIO')
        plt.plot(self.time,self.ocean_nio/self.total_particles*100,'--',color='#cc0000',linewidth=2,label='floating in NIO')
        plt.plot(self.time,self.beached_sio/self.total_particles*100,'-',color='#0000cc',linewidth=2,label='beached in SIO')
        plt.plot(self.time,self.ocean_sio/self.total_particles*100,'--',color='#0000cc',linewidth=2,label='floating in SIO')
        plt.plot(self.time,self.escaped_io/self.total_particles*100,'-',color='#000000',linewidth=2,label='escaped IO')
        plt.ylim(0,100)
        plt.ylabel('Particles [%]')
        plt.legend()
        plt.show()

    def write_to_netcdf(self,output_path):
        log.info(None,f'Writing Indian Ocean particle development timeseries to netcdf: {output_path}')
        nc = Dataset(output_path,'w',format='NETCDF4')
        # define dimensions
        nc.createDimension('time',len(self.time))
        # define variables
        nc_time = nc.createVariable('time',float,'time',zlib=True)
        nc_total_particles = nc.createVariable('total_particles',float,'time',zlib=True)
        nc_ocean_nio = nc.createVariable('ocean_nio',float,'time',zlib=True)
        nc_ocean_sio = nc.createVariable('ocean_sio',float,'time',zlib=True)
        nc_beached_nio = nc.createVariable('beached_nio',float,'time',zlib=True)
        nc_beached_sio = nc.createVariable('beached_sio',float,'time',zlib=True)
        nc_escaped_io = nc.createVariable('escaped_io',float,'time',zlib=True)
        # write variables
        time,time_units = convert_datetime_to_time(self.time)
        nc_time[:] = time
        nc_time.units = time_units
        nc_total_particles[:] = self.total_particles
        nc_ocean_nio[:] = self.ocean_nio
        nc_ocean_sio[:] = self.ocean_sio
        nc_beached_nio[:] = self.beached_nio
        nc_beached_sio[:] = self.beached_sio
        nc_escaped_io[:] = self.escaped_io
        nc.close()

    @staticmethod
    def create_from_particles(particles,dx_oceanbasin=0.1):
        log.info(None,'Determining Indian Ocean particle development timeseries from particles...')
        io_nh_basin = OceanBasinGrid('io_nh',dx_oceanbasin)
        io_sh_basin = OceanBasinGrid('io_sh',dx_oceanbasin)
        time = particles.time
        total_particles = np.zeros((len(time)))
        beached_nio = np.zeros((len(time)))
        beached_sio = np.zeros((len(time)))
        ocean_nio = np.zeros((len(time)))
        ocean_sio = np.zeros((len(time)))
        escaped_io = np.zeros((len(time)))
        for t in range(len(time)):
            lon = particles.lon[:,t]
            lat = particles.lat[:,t]
            total_particles[t] += sum((~np.isnan(particles.lon[:,:t+1])).any(axis=1))
            lon_index,lat_index = io_nh_basin.grid.get_index(lon[~np.isnan(lon)],lat[~np.isnan(lat)])
            l_nio = io_nh_basin.in_basin[lat_index.astype('int'),lon_index.astype('int')]
            l_sio = io_sh_basin.in_basin[lat_index.astype('int'),lon_index.astype('int')]
            l_io = np.logical_or(l_nio,l_sio)
            beached = particles.beached[~np.isnan(lon),t] >= 1 # beached=1, stuck=2
            ocean = particles.beached[~np.isnan(lon),t] == 0
            beached_nio[t] = np.sum(beached[l_nio])
            beached_sio[t] = np.sum(beached[l_sio])
            ocean_nio[t] = np.sum(ocean[l_nio])
            ocean_sio[t] = np.sum(ocean[l_sio])
            escaped_io[t] = total_particles[t]-np.sum(l_io)
        return IoParticleDevelopmentTimeseries(time,total_particles,ocean_nio,ocean_sio,beached_nio,beached_sio,escaped_io)

    @staticmethod
    def read_from_netcdf(input_path):
        log.info(None,f'Reading Indian Ocean particle development timeseries from netcdf: {input_path}')
        netcdf = Dataset(input_path)
        time_org = netcdf['time'][:].filled(fill_value=np.nan)
        time = convert_time_to_datetime(time_org,netcdf['time'].units)
        total_particles = netcdf['total_particles'][:].filled(fill_value=np.nan)
        ocean_nio = netcdf['ocean_nio'][:].filled(fill_value=np.nan)
        ocean_sio = netcdf['ocean_sio'][:].filled(fill_value=np.nan)
        beached_nio = netcdf['beached_nio'][:].filled(fill_value=np.nan)
        beached_sio = netcdf['beached_sio'][:].filled(fill_value=np.nan)
        escaped_io = netcdf['escaped_io'][:].filled(fill_value=np.nan)
        return IoParticleDevelopmentTimeseries(time,total_particles,ocean_nio,ocean_sio,beached_nio,beached_sio,escaped_io)

def get_output_path(basin_name,dx,p,description,file_format,extra_description=None,dir_description='pts_postprocessed'):
    output_dir = get_dir(dir_description)
    p_str = str(p).replace('.','')
    if extra_description is None:
        output_path = f'{output_dir}{basin_name}_{description}_{dx}km_p{p_str}.{file_format}'
    else:
        output_path = f'{output_dir}{basin_name}_{description}_{dx}km_p{p_str}_{extra_description}.{file_format}'
    return output_path

def postprocess_beached(dx,p,basin_name='io_nh',extra_description=None):
    log.info(None,f'--- Postprocessing results for {basin_name} beaching particles at: dx = {dx}, p = {p} ---')
    input_path_particles = get_beached_particles_path(basin_name,dx,p,extra_description=extra_description)
    input_path_density = get_beached_density_path(basin_name,dx,p,extra_description=extra_description)
    # output paths
    output_io_dev = get_output_path(basin_name,dx,p,'dev','nc',extra_description=extra_description)
    output_particles_per_country = get_output_path(basin_name,dx,p,'ppcountry','csv',extra_description=extra_description)
    output_connectivity_matrix = get_output_path(basin_name,dx,p,'connectivity','csv',extra_description=extra_description)
    # load data
    particles = BeachingParticles.read_from_netcdf(input_path_particles)
    density = Density.read_from_netcdf(input_path_density,t_index=-1)
    # particles per country
    ppcountry = ParticlesPerCountry.get_from_density(density)
    ppcountry.write_to_csv(output_particles_per_country)
    # connectivity matrix
    cmatrix = ConnectivityMatrix.get_from_particles(particles)
    cmatrix.write_to_csv(output_connectivity_matrix)
    # IO particle development
    io_dev = IoParticleDevelopmentTimeseries.create_from_particles(particles)
    io_dev.write_to_netcdf(output_io_dev)

if __name__ == '__main__':
    basin_names = ['io_nh','io_sh']
    dx,p_default = get_defaults()
    ps = get_probabilities()
    ps.remove(p_default)
    ps_sh = np.copy(ps)
    ps_sh.remove(0.275)
    ps_sh.remove(0.725)
    for basin_name in basin_names:
        postprocess_beached(dx,p_default,basin_name=basin_name)        
        for p in ps:
            if basin_name == 'io_sh' and p == 0.275:
                continue
            if basin_name == 'io_sh' and p == 0.725:
                continue
            postprocess_beached(dx,p,basin_name=basin_name)
