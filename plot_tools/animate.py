from plot_tools.map_plotter import MapPlot
from processing import get_beached_particles_path, get_particles_path
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
                             dpi=100,fps=25):    
    writer = 'imagemagick'
    writer = animation.writers['ffmpeg'](fps=fps,codec='libx264',bitrate=-1,extra_args=['-pix_fmt', 'yuv420p'])
    if lon_range is None:
        lon_range, _ = get_io_lon_lat_range()
    if lat_range is None:
        _, lat_range = get_io_lon_lat_range()
    # plot map
    plt.style.use(plot_style)
    plt.rcParams.update({'font.size' : 6})
    plt.rcParams.update({'font.family': 'arial'})
    fig = plt.figure(figsize=(5,4))
    ax = plt.gca(projection=ccrs.PlateCarree())
    ax.add_feature(cftr.LAND,color='#FAF1D7')
    ax.add_feature(cftr.OCEAN,color='#67C1E7')
    ax.add_feature(cftr.COASTLINE,color='#8A8A8A')    
    ax.set_extent([lon_range[0],lon_range[1],lat_range[0],lat_range[1]],ccrs.PlateCarree())    
    # plot legend
    if plot_legend:
        legend_elements = [Line2D([0],[0],marker='o',markersize=10,color='w',markerfacecolor='#000000',label='drifting'),                        
                           Line2D([0],[0],marker='o',markersize=10,color='w',markerfacecolor='#EA5622',label='beached')]
        ax.legend(handles=legend_elements,loc='upper right')
    # add logo
    if add_uwa_logo:
        im = image.imread('input/uwa_logo.png')        
        ax.imshow(im, aspect=1, extent = logo_extent, transform=ccrs.PlateCarree(), zorder=10)
    # animated points
    point_o = ax.plot([],[],'.',markersize=1.,color='#000000')[0]
    point_b = ax.plot([],[],'.',markersize=1.,color='#EA5622')[0]
    # animated text
    ttl = ax.text(0.83,0.95,'',transform=ax.transAxes,ha='left',bbox=dict(facecolor='w', alpha=0.3, edgecolor='w',pad=3))
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
        else:
            title = time[i].strftime('%B')
        ttl.set_text(title)
        return point_o,point_b,ttl

    anim = animation.FuncAnimation(plt.gcf(),animate,init_func=init,frames=len(time),blit=True)
    if output_name is not None:
        output_path = get_dir('animation_output')+output_name+'.mpeg'
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
    start_time = datetime(1995,1,1,12,0)
    end_time = datetime(1996,1,1,12,0)    
    particles = BeachingParticles.read_from_netcdf(get_particles_path('io_sh'),time_start=start_time,time_end=end_time)    

    lon_range = [105,110]
    lat_range = [-11,-6]
    output_name = 'plastic_waste_christmas_island'
    logo_extent = (lon_range[1]-0.9,lon_range[1]-0.1,lat_range[0]+0.2,lat_range[0]+0.8)
    # ax.text(105.6,-10.8,'Christmas Island',transform=ccrs.PlateCarree(),ha='center',
    #         bbox=dict(facecolor='w', alpha=0.3, edgecolor='w',pad=1),zorder=11)
    create_animation_beached(particles.time,particles.lon,particles.lat,None,None,
                             output_name=output_name,lon_range=lon_range,lat_range=lat_range,
                             dpi=300,fps=5,title_style='month',add_uwa_logo=True,logo_extent=logo_extent)
    
