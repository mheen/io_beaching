from plot_tools.map_plotter import MapPlot
from processing import get_beached_particles_path, get_particles_path
from pts_parcels_xpresspearl import get_release_time_lon_lat as get_xpresspearl_release_time_lon_lat
from utilities import get_dir
from ocean_utilities import get_io_lon_lat_range
from particles import BeachingParticles, Density
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cftr
import numpy as np
import matplotlib.animation as animation
from dateutil.relativedelta import relativedelta
from matplotlib.lines import Line2D
import matplotlib.image as image
from datetime import datetime

def create_animation_beached(time,lon,lat,beached,p,output_name=None,
                             lon_range=None,lat_range=None,title_style='time_passed',
                             plot_legend=False,plot_style='plot_tools/plot.mplstyle',
                             add_uwa_logo=False,logo_extent=None,
                             text=None,text_lon=None,text_lat=None,
                             point=False,point_lon=None,point_lat=None,
                             dpi=100,fps=25):    
    writer = animation.PillowWriter(fps=fps)
    if lon_range is None:
        lon_range, _ = get_io_lon_lat_range()
    if lat_range is None:
        _, lat_range = get_io_lon_lat_range()
    # plot map
    plt.style.use(plot_style)
    plt.rcParams.update({'font.size' : 6})
    plt.rcParams.update({'font.family': 'arial'})
    plt.rcParams.update({'figure.dpi': dpi})
    fig = plt.figure(figsize=(5,4))
    ax = plt.gca(projection=ccrs.PlateCarree())
    ax.add_feature(cftr.LAND,color='#FAF1D7')
    ax.add_feature(cftr.OCEAN,color='#67C1E7')
    ax.add_feature(cftr.COASTLINE,color='#8A8A8A')    
    ax.set_extent([lon_range[0],lon_range[1],lat_range[0],lat_range[1]],ccrs.PlateCarree())    
    # plot legend
    if plot_legend:
        legend_elements = [Line2D([0],[0],marker='o',markersize=10,color='k',markerfacecolor='#000000',label='drifting'),                        
                           Line2D([0],[0],marker='o',markersize=10,color='#EA5622',markerfacecolor='#EA5622',label='beached')]
        ax.legend(handles=legend_elements,loc='upper right')
    # add logo
    if add_uwa_logo:
        im = image.imread('input/uwa_logo.png')        
        ax.imshow(im, aspect=1, extent = logo_extent, transform=ccrs.PlateCarree(), zorder=10)
    if text is not None:
        ax.text(text_lon,text_lat,text,transform=ccrs.PlateCarree(),ha='center',va='top',
                bbox=dict(facecolor='w', alpha=0.3, edgecolor='w',pad=1),zorder=11)
    if point is True:
        ax.scatter(point_lon,point_lat,c='#C70039',s=1.8)
    # animated points
    point_o = ax.plot([],[],marker='o',color='k',markersize=0.4,linestyle=None,linewidth=0)[0]
    point_b = ax.plot([],[],'.',markersize=0.4,color='#EA5622',fillstyle='full')[0]
    # animated text
    ttl = ax.text(0.5,0.92,'',transform=ax.transAxes,ha='left',bbox=dict(facecolor='w', alpha=0.3, edgecolor='w',pad=2))
    ttl.set_animated(True)
    
    def init():
        point_o.set_data([],[])
        point_b.set_data([],[])
        ttl.set_text('')
        return point_o,point_b,ttl

    def animate(i):        
        xo,yo = (lon[:,i],lat[:,i])
        point_o.set_data(xo,yo)
        if beached is not None:
            b = beached[:,i] == 1 # beached
            xb,yb = (lon[b,i],lat[b,i])
            point_b.set_data(xb,yb)
        passed_years = str(relativedelta(time[i],time[0]).years)
        passed_months = str(relativedelta(time[i],time[0]).months)
        passed_days = str(relativedelta(time[i],time[0]).days)
        if beached is not None:
            title = f'Beaching: $\Delta$x = 8 km, p = {p}\nSimulation duration: {passed_years} years, {passed_months} months, {passed_days} days'
        elif title_style == 'time_passed':
            title = f'Simulation duration: {passed_years} years, {passed_months} months, {passed_days} days'
        elif title_style == 'time_passed_simple':
            title = f'{passed_years} years, {passed_months} months'
        elif title_style == 'month':
            title = time[i].strftime('%B')
        else:
            raise ValueError(f'''Unknown title style requested: {title_style}.
                             Valid options are: time_passed, time_passed_simple, month.''')
        ttl.set_text(title)
        return point_o,point_b,ttl

    anim = animation.FuncAnimation(plt.gcf(),animate,init_func=init,frames=len(time),blit=True)
    if output_name is not None:
        output_path = f'{get_dir("animation_output")}{output_name}_{fps}fps_{dpi}dpi.gif'
        anim.save(output_path,writer=writer)
    else:
        plt.show()

def io_beaching_animations():
    ps = [0.05,0.5,0.95]
    dx = 8
    start_time = datetime(1995,1,1,12,0)
    end_time = datetime(2005,1,1,12,0)
    for p in ps:
        output_name = f'beaching_8km_p{str(p).replace(".","")}'
        input_path_nh = get_beached_particles_path('io_nh',dx,p)
        input_path_sh = get_beached_particles_path('io_sh',dx,p)
        particles_nh = BeachingParticles.read_from_netcdf(input_path_nh,time_start=start_time,time_end=end_time)
        particles_sh = BeachingParticles.read_from_netcdf(input_path_sh,time_start=start_time,time_end=end_time)
        
        time = particles_nh.time
        lon = np.concatenate((particles_nh.lon,particles_sh.lon),axis=0)
        lat = np.concatenate((particles_nh.lat,particles_sh.lat),axis=0)
        beached = np.concatenate((particles_nh.beached,particles_sh.beached),axis=0)

        create_animation_beached(time,lon,lat,beached,p,output_name=output_name,plot_legend=True)
        del(particles_nh,particles_sh,time,lon,lat,beached)

def christmas_island_animation():
    particles = BeachingParticles.read_from_netcdf(get_dir('christmas_island_input'))

    lon_range = [100,115]
    lat_range = [-12,-5]
    output_name = 'plastic_waste_christmas_island'
    logo_extent = (lon_range[1]-2,lon_range[1]-0.1,lat_range[0]+0.2,lat_range[0]+2)    
    create_animation_beached(particles.time,particles.lon,particles.lat,None,None,
                             output_name=output_name,lon_range=lon_range,lat_range=lat_range,
                             text='Christmas Island',text_lon=105.6,text_lat=-11.,
                             dpi=300,fps=2,title_style='month',add_uwa_logo=True,logo_extent=logo_extent)
    
def cocos_keeling_islands_animation():
    particles = BeachingParticles.read_from_netcdf(get_dir('iot_input_2008'))

    lon_range = [90., 128.]
    lat_range = [-20., 6.]
    output_name = 'plastic_waste_cki_2008'
    create_animation_beached(particles.time, particles.lon, particles.lat, None, None,
                             output_name= output_name, lon_range=lon_range, lat_range=lat_range,
                             text='Cocos Keeling\nIslands', text_lon=97., text_lat=-13.,
                             point=True, point_lon=96.86, point_lat=-12.14,
                             dpi=150, fps=4, title_style='month')

def xpresspearl_animation():
    particles = BeachingParticles.read_from_netcdf(get_dir('xpresspearl_output')+'particles_2008-2009.nc')
    output_name = 'xpresspearl'
    lon_range = [40, 120]
    lat_range = [-20, 40]
    _, lon0, lat0 = get_xpresspearl_release_time_lon_lat()
    create_animation_beached(particles.time, particles.lon, particles.lat, None, None,
                             lon_range=lon_range, lat_range=lat_range,
                             output_name=output_name, point=True, point_lon=lon0, point_lat=lat0,
                             dpi=300, fps=5, title_style='time_passed_simple')
