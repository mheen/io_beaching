from utilities import get_closest_index, get_matrix_value_or_nan
from ocean_utilities import Grid, get_io_lon_lat_range
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
from netCDF4 import Dataset
from plot_tools.map_plotter import MapPlot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import log

class CountriesGridded:
    def __init__(self,grid,codes_names,countries):
        self.grid = grid
        self.codes_names = codes_names
        self.countries = countries

    def get_countriesgridded_with_halo(self,halosize=5):
        log.info(None,f'Adding a halo to each country of size {halosize} grid cells.')
        '''Increases the size of countries by the number of gridcells
        defined by halosize (default=1). This is used to extend
        country boundaries into the ocean a bit.'''
        # get country codes in grid
        codes = np.unique(self.countries[~np.isnan(self.countries)])
        countries = np.copy(self.countries)
        # cycle through codes and add halo to country
        for code in codes:
            i,j = np.where(self.countries==code)
            for n in range(1,halosize+1):
                ip,im,jp,jm = self._get_ip_im_jp_jm(i,j,n)
                countries = self._extend_mask_up_down_right_left_diagonal(countries,code,i,j,ip,im,jp,jm)
                for m in range(1,n):
                    ipc,imc,jpc,jmc = self._get_ip_im_jp_jm(i,j,m)
                    countries = self._extend_mask_slanted(countries,code,ip,im,jp,jm,ipc,imc,jpc,jmc)
        return CountriesGridded(self.grid,self.codes_names,countries)

    def split_india_west_east(self):
        log.info(None,'Splitting Indian into an eastern (BoB) and western (AS) part.')
        india_code = 9
        india_code_as = 991
        india_code_bob = 992
        split_lon = 77.5        
        i,j = np.where(self.countries==india_code)
        split_j,_ = self.grid.get_index(split_lon,0)
        l_west = j < split_j
        i_west = i[l_west]
        j_west = j[l_west]
        i_east = i[~l_west]
        j_east = j[~l_west]
        self.codes_names[india_code_as] = 'India (AS)'
        self.codes_names[india_code_bob] = 'India (BoB)'
        self.countries[i_west,j_west] = india_code_as
        self.countries[i_east,j_east] = india_code_bob

    def extend_maldives_cocos(self):
        log.info(None,'Extending the domain of the Maldives and Cocos Keeling Islands.')
        maldives_code = list(self.codes_names.keys())[list(self.codes_names.values()).index('Maldives')]
        lon_range = [72.3,74.0]
        lat_range = [-0.1,7.2]
        i_lon = get_closest_index(self.grid.lon,lon_range)
        i_lat = get_closest_index(self.grid.lat,lat_range)
        self.countries[i_lat[0]:i_lat[1],i_lon[0]:i_lon[1]] = maldives_code
        cocos_keeling_code = list(self.codes_names.keys())[list(self.codes_names.values()).index('Indian Ocean Territories')]
        lon_range_c = [96.7,97.0]
        lat_range_c = [-12.3,-11.9]
        i_lon_c = get_closest_index(self.grid.lon,lon_range_c)
        i_lat_c = get_closest_index(self.grid.lat,lat_range_c)
        self.countries[i_lat_c[0]:i_lat_c[1],i_lon_c[0]:i_lon_c[1]] = cocos_keeling_code

    def _get_ip_im_jp_jm(self,i,j,n):
        ip = np.copy(i)
        im = np.copy(i)
        ip[i<self.countries.shape[0]-n] += n # i+n but preventing final point increasing out of range
        im[i>n-1] -= n # i-n but preventing first point decreasing out of range
        jp = np.copy(j)
        jm = np.copy(j)
        jp[j<self.countries.shape[1]-n] += n # j+n but preventing final point increasing out of range
        jm[j>n-1] -= n # j-n but preventing first point decreasing out of range
        return ip,im,jp,jm

    def _extend_mask_up_down_right_left_diagonal(self,countries,code,i,j,ip,im,jp,jm):
        countries[ip,j] = code # extend up
        countries[i,jp] = code # extend right
        countries[im,j] = code # extend below
        countries[i,jm] = code # extend left
        countries[ip,jp] = code # extend upperright diagonal
        countries[im,jm] = code # extend lowerleft diagonal
        countries[ip,jm] = code # extend upperleft diagonal
        countries[im,jp] = code # extend lowerright diagonal
        return countries

    def _extend_mask_slanted(self,countries,code,ip,im,jp,jm,ipc,imc,jpc,jmc):
        countries[ip,jmc] = code
        countries[ip,jpc] = code
        countries[im,jmc] = code
        countries[im,jpc] = code
        countries[imc,jp] = code
        countries[ipc,jp] = code
        countries[imc,jm] = code
        countries[ipc,jm] = code
        return countries

    def get_country_code_and_name(self,p_lon,p_lat):
        code = self.get_country_code(p_lon,p_lat)
        name = self.get_country_name(code)
        return code,name

    def get_country_code(self,p_lon,p_lat):
        lon_index,lat_index = self.grid.get_index(p_lon,p_lat)
        country_codes = np.empty((len(lon_index)))*np.nan
        for i in range(len(lon_index)):
            if ~np.isnan(lon_index[i]):
                country_codes[i] = get_matrix_value_or_nan(self.countries,lat_index[i],lon_index[i])
        return country_codes

    def get_country_name(self,code):
        if isinstance(code,int) or isinstance(code,float):
            code = [code]
        country_name = []
        for c in code:
            if ~np.isnan(c):
                country_name.append(self.codes_names[c.astype('int')])
            else:
                country_name.append('')
        return country_name

    def plot(self,p_lon=None,p_lat=None):
        ax = plt.gca(projection=ccrs.PlateCarree())
        mplot = MapPlot(ax,None,None)
        mplot.pcolormesh(self.grid.lon,self.grid.lat,self.countries,ranges=None,show_cbar=False)        
        if p_lon is not None:
            mplot.points(p_lon,p_lat)
        # add country names
        codes = np.unique(self.countries[~np.isnan(self.countries)])
        for code in codes:
            i,j = np.where(self.countries==code)
            # get location somewhere in middle of country block            
            m = np.floor(len(i)/2).astype('int')
            xt = self.grid.lon[j[m]]
            yt = self.grid.lat[i[m]]
            mplot.add_annotation(xt,yt,self.codes_names[code])
        plt.show()

    def write_to_netcdf(self,output_path='input/countries_gridded_dx01.nc'):
        log.info(None,f'Writing gridded countries data to netcdf: {output_path}')
        nc = Dataset(output_path,'w',format='NETCDF4')
        # define dimensions
        nc.createDimension('lon',len(self.grid.lon))
        nc.createDimension('lat',len(self.grid.lat))
        nc.createDimension('n_countries',len(list(self.codes_names.values())))
        # define variables
        nc_lon = nc.createVariable('lon',float,'lon',zlib=True)
        nc_lat = nc.createVariable('lat',float,'lat',zlib=True)
        nc_country_codes = nc.createVariable('country_codes',int,'n_countries',zlib=True)
        nc_country_names = nc.createVariable('country_names',str,('n_countries',),zlib=True)
        nc_countries = nc.createVariable('countries',float,('lat','lon'),zlib=True)
        # write variables
        nc_lon[:] = self.grid.lon
        nc_lat[:] = self.grid.lat
        codes = list(self.codes_names.keys())
        names = np.array(list(self.codes_names.values()))
        nc_country_codes[:] = codes
        nc_country_names[:] = names
        nc_countries[:] = self.countries        
        nc.close()

    @staticmethod
    def read_from_netcdf(input_path='input/countries_gridded_dx01.nc'):
        log.info(None,f'Reading gridded countries information from netcdf: {input_path}')
        netcdf = Dataset(input_path)
        lon = netcdf['lon'][:].filled(fill_value=np.nan)
        lat = netcdf['lat'][:].filled(fill_value=np.nan)
        grid = Grid.get_from_lon_lat_array(lon,lat)
        country_names = list(netcdf['country_names'][:])
        country_codes = list(netcdf['country_codes'][:])
        codes_names = dict(zip(country_codes,country_names))
        countries = netcdf['countries'][:].filled(fill_value=np.nan)
        return CountriesGridded(grid,codes_names,countries)

    @staticmethod
    def create_country_grid(dx=0.1,lon_range='io',lat_range='io'):        
        if lon_range is 'io':
            lon_range,_ = get_io_lon_lat_range()
        if lon_range is 'global':
            lon_range = [-180,180]
        if lat_range is 'io':
            _,lat_range = get_io_lon_lat_range()
        if lat_range is 'global':
            lat_range = [-90,90]
        grid = Grid(dx,lon_range,lat_range)
        x,y = np.meshgrid(grid.lon,grid.lat)
        countries = np.empty(x.shape)*np.nan
        # get polygons, codes and names of countries
        c_shp = Countries()
        codes_names = {}
        log.info(None,f'Creating gridded countries information at {dx} degrees resolution')
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for c,country in enumerate(c_shp.countries):
                    point = Point(x[i,j],y[i,j])
                    l_country = country.geometry.contains(point)
                    if l_country:
                        # (SUBUNIT defines islands belonging to mainland countries separately)
                        codes_names[c+1] = country.attributes['SUBUNIT']
                        countries[i,j] = c+1
        return CountriesGridded(grid,codes_names,countries)

class Countries:
    def __init__(self):
        '''Country information from Natural Earth data (shapefile).'''        
        shapename = 'admin_0_countries'
        log.info(None,f'Obtaining country information from Natural Earth data: {shapename}')
        shp_countries = shpreader.natural_earth(resolution='10m',category='cultural',name=shapename)
        self.countries = []
        for country in shpreader.Reader(shp_countries).records():
            self.countries.append(country)

def get_cgrid_with_halo_and_extended_islands(halosize=5):
    cgrid = CountriesGridded.read_from_netcdf()
    cgrid_halo = cgrid.get_countriesgridded_with_halo(halosize=halosize)
    cgrid_halo.extend_maldives_cocos()
    cgrid_halo.split_india_west_east()
    return cgrid_halo

if __name__ == '__main__':
    cgrid = CountriesGridded.create_country_grid()
    cgrid.write_to_netcdf()
