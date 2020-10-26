import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.offsetbox import AnchoredText
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cartopy.feature as cftr
import numpy as np
from netCDF4 import Dataset
import shapely

def get_colormap_reds(n):
    colors = ['#fece6b','#fd8e3c','#f84627','#d00d20','#b50026','#950026','#830026']
    return colors[:n]

def get_colormap_reds_blues(n):
    n = n // 2
    reds = ['#ffebc0','#f84627','#950026']
    blues = ['#0a2a6a','#437aea','#c8d9fd']
    colors = blues[:n]
    colors.append(reds[:n])
    return colors

class MapPlot:
    def __init__(self,ax,lon_range,lat_range,
                 meridians=None,parallels=None,
                 title=[],landcolor='#cfcfcf',edgecolor='k',
                 xmarkers='bottom',ymarkers='left',
                 plot_mplstyle='plot_tools/plot.mplstyle'):
        plt.style.use(plot_mplstyle)
        self.ax = ax
        if lon_range is None:
            self.lon_range = [-180,180]
        else:
            self.lon_range = lon_range
        if lat_range is None:
            self.lat_range = [-80,80]
        else:
            self.lat_range = lat_range
        self.cbar_ticks = []
        self.cbar_label = []
        self.cbar_lim = []
        self.title = title
        self._basic_map(edgecolor=edgecolor,landcolor=landcolor)
        self.xmarkers = xmarkers
        self.ymarkers = ymarkers
        self.meridians = meridians
        self.parallels = parallels
        self._draw_grid()
        self._draw_box()
        self._set_title()

    def set_cbar_items(self,ticks=[],label=[],lim=[]):
        self.cbar_ticks = ticks
        self.cbar_label = label
        self.cbar_lim = lim

    def country(self,country,facecolor,edgecolor='k'):
        if type(country.geometry) == shapely.geometry.polygon.Polygon:
            c_geometry = [country.geometry]
        else:
            c_geometry = country.geometry
        self.ax.add_geometries(c_geometry,ccrs.PlateCarree(),zorder=3,
                               facecolor=facecolor,edgecolor=edgecolor)

    def points(self,lon,lat,facecolor='k',edgecolor='k',marker='.',markersize=3,edgewidth=None):
        p = self.ax.scatter(lon,lat,marker=marker,s=markersize,facecolor=facecolor,linewidth=edgewidth,
                            edgecolor=edgecolor,transform=ccrs.PlateCarree(),zorder=5)
        return p

    def tracks(self,lon,lat,color='#0000a0'):
        # assuming lon and lat dimensions are [pid,time]
        for p in range(lon.shape[0]):
            self.ax.plot(lon[p,:],lat[p,:],'.-',color=color,
                                 transform=ccrs.PlateCarree(),zorder=5)

    def fill(self,lon,lat,color='#3299cc'):
        self.ax.fill(lon,lat,color=color,transform=ccrs.PlateCarree(),zorder=3)

    def lines(self,lon,lat,linewidth=1.,color='#0000a0'):
        self.ax.plot(lon,lat,'-',color=color,linewidth=linewidth,
                     transform=ccrs.PlateCarree(),zorder=4)

    def pcolormesh(self,lon,lat,z,show_cbar=True,ranges=[1,10,100,10**3,10**4,10**5,10**6],cmap='Reds'):
        xx,yy = np.meshgrid(lon,lat)
        zz = np.copy(z)
        zz[zz == 0] = np.nan
        if ranges is not None:
            if cmap == 'RedBlue':
                colors = get_colormap_reds_blues(len(ranges))
            colors = get_colormap_reds(len(ranges))
            cm = LinearSegmentedColormap.from_list('cm_log_density',colors,N=len(ranges))
            norm = BoundaryNorm(ranges,ncolors=6)
            c = self.ax.pcolormesh(xx,yy,zz,cmap=cm,norm=norm,transform=ccrs.PlateCarree(),zorder=1)            
            self.set_cbar_items(ticks=ranges,lim=[ranges[0],ranges[-1]])
        else:
            c = self.ax.pcolormesh(xx,yy,zz,cmap=cmap,transform=ccrs.PlateCarree(),zorder=1)
        if len(self.cbar_lim) is not 0:
                c.set_clim(self.cbar_lim[0],self.cbar_lim[1])
        if show_cbar:
            cbar = self._add_cbar(c)            
            return c,cbar,ranges
        else:
            return c,ranges

    def contourf(self,lon,lat,z,levels=None,cmap='Reds'):
        xx,yy = np.meshgrid(lon,lat)
        if levels is not None:            
            c = self.ax.contourf(xx,yy,z,cmap=cmap,levels=levels,extend='both',
                                 transform=ccrs.PlateCarree(),zorder=2)
        else:
            c = self.ax.contourf(xx,yy,z,cmap=cmap,transform=ccrs.PlateCarree(),zorder=2)
        cbar = self._add_cbar(c)
        if len(self.cbar_lim) is not 0:
            c.set_clim(self.cbar_lim[0],self.cbar_lim[1])
        return c,cbar

    def add_subtitle(self,subtitle,location='upper left'):
        anchored_text = AnchoredText(subtitle,loc=location,borderpad=0.0)
        self.ax.add_artist(anchored_text)

    def add_annotation(self,lons,lats,texts):
        self.ax.annotate(texts,(lons,lats))

    def _draw_box(self):
        x0 = self.lon_range[0]
        x1 = self.lon_range[1]
        y0 = self.lat_range[0]
        y1 = self.lat_range[1]
        self.lines([x0,x0],[y0,y1],color='k',linewidth=0.5)
        self.lines([x0,x1],[y1,y1],color='k',linewidth=0.5)
        self.lines([x0,x1],[y0,y0],color='k',linewidth=0.5)
        self.lines([x1,x1],[y0,y1],color='k',linewidth=0.5)

    def _draw_grid(self,nlon=10,nlat=10):
        lon_formatter = cticker.LongitudeFormatter()
        lat_formatter = cticker.LatitudeFormatter()
        if self.lon_range and self.lat_range:
            dlon = (self.lon_range[-1]-self.lon_range[0])/nlon
            dlat = (self.lat_range[-1]-self.lat_range[0])/nlat
            if self.meridians is None:
                self.meridians = np.arange(self.lon_range[0],self.lon_range[-1]+dlon,dlon)
            if self.parallels is None:
                self.parallels = np.arange(self.lat_range[0],self.lat_range[-1]+dlat,dlat)
        else:
            dlon = 360./nlon
            dlat = 180./nlat
            if self.meridians is None:
                self.meridians = np.arange(-180,180+dlon,dlon)
            if self.parallels is None:
                self.parallels = np.arange(-80,80+dlat,dlat)
        self.ax.set_xticks(self.meridians,crs=ccrs.PlateCarree())
        if self.xmarkers == 'off':
            self.ax.set_xticklabels([])
            self.ax.set_axisbelow(False)
        else:
            self.ax.set_xticklabels(self.meridians)
        self.ax.xaxis.set_major_formatter(lon_formatter)
        self.ax.set_yticks(self.parallels,crs=ccrs.PlateCarree())
        if self.ymarkers == 'off':
            self.ax.set_yticklabels([])
        else:
            self.ax.set_yticklabels(self.parallels)
        self.ax.yaxis.set_major_formatter(lat_formatter)
        if self.xmarkers == 'top':
            self.ax.xaxis.tick_top()
        if self.ymarkers == 'right':
            self.ax.yaxis.tick_right()
        self.ax.grid(b=True,linewidth=0.5,color='k',linestyle=':',zorder=20)

    def _add_cbar(self,c):
        cbar = plt.colorbar(c,ticks=self.cbar_ticks)
        cbar.set_label(self.cbar_label)
        return cbar

    def _set_title(self):
        if len(self.title) is not 0:
            self.ax.set_title(self.title)

    def _basic_map(self,edgecolor,landcolor):
        self.ax.add_feature(cftr.LAND,facecolor=landcolor,edgecolor=edgecolor,zorder=2)
        self.ax.set_extent([self.lon_range[0],self.lon_range[1],
                            self.lat_range[0],self.lat_range[1]],ccrs.PlateCarree())
