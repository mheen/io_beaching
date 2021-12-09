from plot_tools.map_plotter import MapPlot
from utilities import get_dir, convert_lon_360_to_180
from ocean_utilities import get_io_lon_lat_range, Grid, IrregularGrid
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from netCDF4 import Dataset
import os
import wget
import log

gshhg_distance_url = 'ftp://ftp.soest.hawaii.edu/gshhg/dist_to_GSHHG_v2.3.7_1m.nc'

class CoastDistance:
    def __init__(self,grid,distance):
        self.grid = grid
        self.distance = distance

    def get_distance(self,lon_req,lat_req):
        (i,j) = self.grid.get_index(lon_req,lat_req)
        return self.distance[j,i]

    def plot(self,lon_range=None,lat_range=None,resolution=None,output_path=None):
        ticks = np.arange(-400,500,100)
        levels = np.arange(-400,410,10)
        label = 'Distance to coast [km]'
        plt.figure(figsize=(10,7))
        ax = plt.gca(projection=ccrs.PlateCarree())
        mplot = MapPlot(ax,lon_range,lat_range,title='Distance to nearest coastline from GSHHG')
        mplot.set_cbar_items(ticks=ticks,label=label,lim=[ticks[0],ticks[-1]])
        x = self.grid.lon[0::10]
        y = self.grid.lat[0::10]
        z = self.distance[0::10,0::10]
        mplot.contourf(x,y,z,levels=levels,cmap='coolwarm')
        mplot.draw_grid()
        if output_path is not None:
            plt.savefig(output_path)
        plt.show()
        
    def plot_io(self,resolution=None,output_path=None):
        lon_range,lat_range = get_io_lon_lat_range()
        self.plot(lon_range=lon_range,lat_range=lat_range,resolution=resolution,output_path=output_path)

    @staticmethod
    def read_from_netcdf(input_path='input/dist_to_GSHHG_v2.3.7_1m.nc'):
        if not os.path.exists(input_path):
            log.info(None, f'Downloading GSHHG distance to coast file, this may take a few minutes.')
            wget.download(gshhg_distance_url,out=input_path)
        data = Dataset(input_path)
        lat = data['lat'][:].filled(fill_value=np.nan)
        lon360 = data['lon'][:].filled(fill_value=np.nan)
        lon = convert_lon_360_to_180(lon360)
        i_sort = np.argsort(lon)
        lon = lon[i_sort]
        distance = data['dist'][:].filled(fill_value=np.nan)
        distance = distance[:,i_sort]
        grid = IrregularGrid(lon,lat)
        data.close()
        return CoastDistance(grid,distance)

    @staticmethod
    def read_from_tif(input_path='input/coastline_distance.tif'):
        dataset = rasterio.open(input_path)
        # construct grid        
        width = dataset.width
        height = dataset.height
        dx = (dataset.bounds.right-dataset.bounds.left)/width
        dy = (dataset.bounds.top-dataset.bounds.bottom)/height
        min_lon = dataset.bounds.left+dx/2
        max_lon = dataset.bounds.right-dx/2
        min_lat = dataset.bounds.bottom+dy/2
        max_lat = dataset.bounds.top-dy/2
        grid = Grid(dx,[min_lon,max_lon],[min_lat,max_lat],dy=dy)
        # get distance to coast
        coast_distance = np.flipud(dataset.read(1))        
        coast_distance[coast_distance==999999.] = np.nan
        # close file
        dataset.close()
        return CoastDistance(grid,coast_distance)