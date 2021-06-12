from utilities import get_closest_index, add_month_to_timestamp, get_dir, get_distance_between_points
import numpy as np
import shapefile
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cftr
from netCDF4 import Dataset
from datetime import datetime, timedelta
import log

# -----------------------------------------------
# Classes
# -----------------------------------------------
class OceanBasins:
    def __init__(self):
        self.basin = []

    def determine_if_point_in_basin(self,basin_name,p_lon,p_lat):
        p_lon = np.array(p_lon)
        p_lat = np.array(p_lat)
        if basin_name.startswith('po'):
            basin_name = [basin_name[:2]+'_l'+basin_name[2:],basin_name[:2]+'_r'+basin_name[2:]]
        else:
            basin_name = [basin_name]
        l_in_basin = np.zeros(len(p_lon)).astype('bool')
        for i in range(len(basin_name)):
            basin = self.get_basin_polygon(basin_name[i])
            for p in range(len(p_lon)):
                point = Point(p_lon[p],p_lat[p])
                l_in_polygon = basin.polygon.contains(point)
                l_in_basin[p] = l_in_polygon or l_in_basin[p]
        return l_in_basin

    def get_basin_polygon(self,basin_name):
        for basin in self.basin:
            if basin.name == basin_name:
                return basin
        raise ValueError('Unknown ocean basin requested. Valid options are: "io","ao","po", and any of these with "_nh" or "_sh" added.')

    @staticmethod
    def read_from_shapefile(input_path='input/oceanbasins_polygons.shp'):
        ocean_basins = OceanBasins()
        sf = shapefile.Reader(input_path)        
        shape_records = sf.shapeRecords() # reads both shapes and records(->fields)
        for i in range(len(shape_records)):
            name = shape_records[i].record[1]
            points = shape_records[i].shape.points
            polygon = Polygon(points)
            ocean_basins.basin.append(OceanBasin(name,polygon))
        sf.close()
        return ocean_basins

class OceanBasin:
    def __init__(self,name,polygon):
        self.name = name
        self.polygon = polygon

class OceanBasinGrid:
    def __init__(self,basin_name,dx,lon_range=None,lat_range=None):        
        self.basin_name = basin_name
        if lon_range is None:
            lon_range = [-180,180]
        if lat_range is None:
            lat_range = [-90,90]
        self.grid = Grid(dx,lon_range,lat_range)
        lon,lat = np.meshgrid(self.grid.lon,self.grid.lat)
        self.in_basin = np.ones(lon.shape).astype('bool')
        ocean_basins = OceanBasins.read_from_shapefile()
        for i in range(lon.shape[0]):            
            self.in_basin[i,:] = ocean_basins.determine_if_point_in_basin(basin_name,lon[i,:],lat[i,:])

class Grid:
    def __init__(self,dx,lon_range,lat_range,dy=None,periodic=False):
        self.dx = dx
        if not dy:
            self.dy = dx
        else:
            self.dy = dy
        self.lon_min = lon_range[0]
        self.lon_max = lon_range[1]
        self.lat_min = lat_range[0]
        self.lat_max = lat_range[1]
        self.lon = np.arange(self.lon_min,self.lon_max+self.dx,self.dx)
        self.lat = np.arange(self.lat_min,self.lat_max+self.dy,self.dy)
        self.lon_size = len(self.lon)
        self.lat_size = len(self.lat)
        self.periodic = periodic

    def get_index(self,lon,lat):
        lon = np.array(lon)
        lat = np.array(lat)
        # get lon index
        lon_index = np.floor((lon-self.lon_min)*1/self.dx)
        lon_index = np.array(lon_index)
        l_index_lon_over = lon_index >= abs(self.lon_max-self.lon_min)*1/self.dx
        if self.periodic:
            lon_index[l_index_lon_over] = 0
        else:
            lon_index[l_index_lon_over] = np.nan
        l_index_lon_under = lon_index < 0
        if self.periodic:
            lon_index[l_index_lon_under]
        else:
            lon_index[l_index_lon_under] = np.nan
        # get lat index
        lat_index = np.floor((lat-self.lat_min)*1/self.dy)
        lat_index = np.array(lat_index)
        l_index_lat_over = lat_index >= abs(self.lat_max-self.lat_min)*1/self.dy
        lat_index[l_index_lat_over] = np.nan        
        l_index_lat_under = lat_index<0
        lat_index[l_index_lat_under] = np.nan        
        return (lon_index,lat_index)

    @staticmethod
    def get_from_lon_lat_array(lon,lat):
        dx = np.round(np.unique(np.diff(lon))[0],2)
        dy = np.round(np.unique(np.diff(lat))[0],2)
        lon_range = [np.nanmin(lon),np.nanmax(lon)]
        lat_range = [np.nanmin(lat),np.nanmax(lat)]        
        log.warning(None,f'dx ({np.unique(np.diff(lon))[0]}) to create Grid rounded to 2 decimals: dx = {dx}')
        if dy != dx:
            log.warning(None,f'dy ({np.unique(np.diff(lat))[0]}) to create Grid rounded to 2 decimals: dy = {dy}')
        return Grid(dx,lon_range,lat_range,dy=dy)

class IrregularGrid:
    def __init__(self,lon,lat):
        self.lon = lon
        self.lat = lat

    def get_index(self,lon,lat):
        lon_index = get_closest_index(self.lon,lon)
        lat_index = get_closest_index(self.lat,lat)        
        return lon_index,lat_index

class LandMask:
    def __init__(self,lon,lat,mask):
        self.lon = lon
        self.lat = lat
        self.mask = mask # 0: ocean, 1: land

    def get_landmask_with_halo(self):
        '''Increases the size of the landmask by 1 gridcell.
        This can be used to move plastic
        source locations further away from land.'''
        i,j = np.where(self.mask==1)
        ip1 = np.copy(i)
        ip1[i<len(self.lat)-1] += 1 # i+1 but preventing final point increasing out of range
        jp1 = np.copy(j)
        jp1[j<len(self.lon)-1] += 1 # j+1 but preventing final point increasing out of range
        im1 = np.copy(i)
        im1[i>0] -= 1 # i-1 but preventing first point decreasing out of range
        jm1 = np.copy(j)
        jm1[j>0] -= 1 # j-1 but preventing first point decreasing out of range
        mask = np.copy(self.mask)        
        mask[ip1,j] = 1 # extend mask up
        mask[i,jp1] = 1 # extend mask right
        mask[ip1,jp1] = 1 # extend mask upper right
        mask[im1,j] = 1 # extend mask down
        mask[i,jm1] = 1 # extend mask left
        mask[im1,jm1] = 1 # extend mask lower left
        # (note: this was corrected after creating v3 of river sources)
        mask[ip1,jm1] = 1 # extend mask upper left
        mask[im1,jp1] = 1 # extend mask lower right
        return LandMask(self.lon,self.lat,mask)

    def get_mask_value(self,p_lon,p_lat):
        j,i = self.get_index(p_lon,p_lat)        
        return self.mask[i,j]

    def get_multiple_mask_values(self,p_lon,p_lat):
        j = get_closest_index(self.lon,p_lon)
        i = get_closest_index(self.lat,p_lat)
        return self.mask[i,j]

    def get_closest_ocean_point(self,p_lon,p_lat,log_file=None):
        j,i = self.get_index(p_lon,p_lat)        
        domain_boundaries = self._get_mininum_surrounding_domain_including_ocean(i,j,log_file)
        if domain_boundaries is not None:
            lon_ocean,lat_ocean = self._get_ocean_coordinates(domain_boundaries,log_file)
            distances = np.empty((len(lon_ocean)))*np.nan
            for p in range(len(lon_ocean)):
                distances[p] = get_distance_between_points(p_lon,p_lat,lon_ocean[p],lat_ocean[p])
            p_closest = np.where(distances==np.nanmin(distances))[0][0]
            lon_closest = lon_ocean[p_closest]
            lat_closest = lat_ocean[p_closest]
            if log_file is not None:
                log.info(log_file,'Found closest ocean point: '+str(lon_closest)+', '+str(lat_closest)+
                         ' to point: '+str(p_lon)+', '+str(p_lat)+'.')
            return lon_closest,lat_closest
        return np.nan,np.nan    

    def get_index(self,p_lon,p_lat):
        dlon = abs(self.lon-p_lon)
        dlat = abs(self.lat-p_lat)
        j = np.where(dlon==np.nanmin(dlon))[0][0]
        i = np.where(dlat==np.nanmin(dlat))[0][0]        
        return j,i

    def get_edges_from_center_points(self,l_lon=None,l_lat=None):
        if l_lon is None:
            l_lon = np.ones(len(self.lon)).astype('bool')
        if l_lat is None:
            l_lat = np.ones(len(self.lat)).astype('bool')
        # convert lon and lat from center points (e.g. HYCOM) to edges (pcolormesh)
        lon_center = self.lon[l_lon]
        dlon = np.diff(lon_center)
        for i in range(len(lon_center)):
            if i == 0:
                lon_edges = lon_center[i]-0.5*dlon[i]
                lon_pcolor = np.append(lon_edges,lon_center[i]+0.5*dlon[i])
            elif i == len(lon_center)-1:
                lon_edges = np.append(lon_edges,lon_center[i]+0.5*dlon[i-1])
            else:
                lon_edges= np.append(lon_edges,lon_center[i]+0.5*dlon[i])        
        lat_center = self.lat[l_lat]
        dlat = np.diff(lat_center)
        for i in range(len(lat_center)):
            if i == 0:
                lat_edges = lat_center[i]-0.5*dlat[i]
                lat_edges= np.append(lat_edges,lat_center[i]+0.5*dlat[i])
            elif i == len(lat_center)-1:
                lat_edges = np.append(lat_edges,lat_center[i]+0.5*dlat[i-1])
            else:
                lat_edges = np.append(lat_edges,lat_center[i]+0.5*dlat[i])
        return lon_edges,lat_edges

    def _get_ocean_coordinates(self,domain_boundaries,log_file):
        i_min = domain_boundaries[0]
        i_max = domain_boundaries[1]
        j_min = domain_boundaries[2]
        j_max = domain_boundaries[3]
        lon = self.lon[j_min:j_max]
        lat = self.lat[i_min:i_max]
        dlon = np.append(np.diff(lon),np.diff(lon)[-1])
        dlat = np.append(np.diff(lat),np.diff(lat)[-1])
        ocean = self.mask[i_min:i_max,j_min:j_max] == 0
        i_ocean,j_ocean = np.where(ocean)
        # lon and lat in center of grid points:
        lon_ocean = lon[j_ocean]+dlon[j_ocean]/2
        lat_ocean = lat[i_ocean]+dlat[j_ocean]/2
        if log_file is not None:
            log.info(log_file,'Found '+str(len(lon_ocean))+' ocean points.')
        return lon_ocean,lat_ocean

    def _get_mininum_surrounding_domain_including_ocean(self,i,j,log_file):
        '''Increases number of grid cells around a specific
        point until an ocean cell is included in the domain.'''
        for n in range(50):            
            n_cells = 10+n*10
            if log_file is not None:
                log.info(log_file,'Finding domain size with ocean: n_cells='+str(n_cells))
            i_min,i_max = self._get_min_max_indices(i,n_cells,'i')
            j_min,j_max = self._get_min_max_indices(j,n_cells,'j')
            land_mask = self.mask[i_min:i_max,j_min:j_max]
            ocean = land_mask == 0
            if ocean.any():
                if log_file is not None:
                    log.info(log_file,'Success.')
                domain_boundaries = [i_min,i_max,j_min,j_max]
                return domain_boundaries
        log.info(log_file,'Did not find a boundary within n_cells='+str(n_cells)+', skipping point.')
        return None

    def _get_min_max_indices(self,i,n,i_type):
        i_min = i-n
        i_max = i+n+1
        if i_type == 'i':
            len_i = self.mask.shape[0]
        elif i_type == 'j':
            len_i = self.mask.shape[1]
        else:
            raise ValueError('Unknown i_type to get indices, should be either "i" or "j".')
        if i_min >= 0 and i_max <= len_i:
            return (i_min,i_max)
        elif i_min < 0 and i_max <= len_i:
            return (0,i_max)
        elif i_max > len_i and i_min >= 0:
            return (i_min,len_i)
        elif i_min < 0 and i_max > len_i:
            return(0,len_i)
        else:
            raise ValueError('Error getting '+i_type+' indices: '+i_type+'='+str(i)+',n='+str(n)) 

    def plot(self,plot_mplstyle='plot_tools/plot.mplstyle'):
        plt.style.use(plot_mplstyle)
        fig = plt.figure()
        ax = plt.gca(projection=ccrs.PlateCarree())
        ax.add_feature(cftr.COASTLINE,edgecolor='k',zorder=2)
        ax.set_extent([-180,180,-80,80],ccrs.PlateCarree())
        ax.pcolormesh(self.lon,self.lat,self.mask,transform=ccrs.PlateCarree())
        plt.show()

    def write_to_netcdf(self,output_path):
        nc = Dataset(output_path,'w',format='NETCDF4')
        # define dimensions
        nc.createDimension('lon',len(self.lon))
        nc.createDimension('lat',len(self.lat))
        # define variables
        nc_lon = nc.createVariable('lon',float,'lon',zlib=True)
        nc_lat = nc.createVariable('lat',float,'lat',zlib=True)
        nc_mask = nc.createVariable('mask',float,('lat','lon'),zlib=True)
        # write variables
        nc_lon[:] = self.lon
        nc_lat[:] = self.lat
        nc_mask[:] = self.mask
        nc_mask.units = '0: ocean, 1: land'
        nc.close()

    @staticmethod
    def read_from_netcdf(input_path='input/hycom_landmask.nc'):
        data = Dataset(input_path)
        lon = data['lon'][:]
        lat = data['lat'][:]
        mask = data['mask'][:]
        data.close()
        return LandMask(lon,lat,mask)

    @staticmethod
    def get_mask_from_vel(input_path):
        data = Dataset(input_path)
        lon = data['lon'][:]
        lat = data['lat'][:]
        if len(data['u'][:].shape) == 3:
            u = data['u'][0,:,:].filled()
        else:
            u = data['u'][:].filled()
        mask = np.isnan(u).astype('int')
        return LandMask(lon,lat,mask)

# -----------------------------------------------
# Functions
# -----------------------------------------------
def get_io_lon_lat_range():
    lon_range = [0.,130.]
    lat_range = [-50.,40.]
    return lon_range,lat_range

def get_global_grid(dx=1):
    return Grid(dx,[-180,180],[-90,90],periodic=True)

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

def read_mean_hycom_data(input_path):
    netcdf = Dataset(input_path)
    lon = netcdf['lon'][:].filled(fill_value=np.nan)
    lat = netcdf['lat'][:].filled(fill_value=np.nan)
    u = netcdf['u'][:].filled(fill_value=np.nan)
    v = netcdf['v'][:].filled(fill_value=np.nan)
    return lon, lat, u, v

def calculate_mean_hycom_data(months, lon_range, lat_range, input_dir=get_dir('hycom_input')):
    u_all = []
    v_all = []
    for month in months:
        start_date = datetime(2008, month, 1)
        end_date = add_month_to_timestamp(start_date, 1)
        n_days = (end_date-start_date).days
        for i in range(n_days):
            date = start_date+timedelta(days=i)
            input_path = f'{input_dir}{date.strftime("%Y%m%d")}.nc'
            log.info(None, f'Reading data from: {input_path}')
            lon, lat, u, v = _read_hycom_data(input_path, lon_range, lat_range)
            u_all.append(u)
            v_all.append(v)
    u_all = np.array(u_all)
    v_all = np.array(v_all)
    log.info(None, f'Calculating mean u and v')
    u_mean = np.nanmean(u_all, axis=0)
    v_mean = np.nanmean(v_all, axis=0)
    return lon, lat, u_mean, v_mean

def _write_mean_hycom_data_to_netcdf(lon, lat, u, v, output_path):
    log.info(None, f'Writing output to netcdf file: {output_path}')
    nc = Dataset(output_path,'w', format='NETCDF4')
    # define dimensions
    nc.createDimension('lat', len(lat))        
    nc.createDimension('lon',len(lon))
    # define variables
    nc_lon = nc.createVariable('lon', float, 'lon', zlib=True)
    nc_lat = nc.createVariable('lat', float, 'lat', zlib=True)
    nc_u = nc.createVariable('u', float, ('lat', 'lon'), zlib=True)
    nc_v = nc.createVariable('v', float, ('lat', 'lon'), zlib=True)
    # write variables
    nc_lon[:] = lon
    nc_lat[:] = lat
    nc_u[:] = u
    nc_v[:] = v
    nc.close()

def _read_hycom_data(input_path, lon_range, lat_range):
    indices = _get_io_indices_from_netcdf(lon_range=lon_range, lat_range=lat_range)
    netcdf = Dataset(input_path)
    lon = netcdf['lon'][indices['lon']].filled(fill_value=np.nan)
    lat = netcdf['lat'][indices['lat']].filled(fill_value=np.nan)
    u = netcdf['u'][0, :, :][indices['lat'], :][:, indices['lon']].filled(fill_value=np.nan)
    v = netcdf['v'][0, :, :][indices['lat'], :][:, indices['lon']].filled(fill_value=np.nan)
    return lon, lat, u, v