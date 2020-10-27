from ocean_utilities import get_io_lon_lat_range
from plastic_sources import RiverSources
from particles import Density
from countries import Countries
from processing import get_density_path, get_beached_density_path
from postprocessing import get_output_path as get_postprocessing_path
from postprocessing import ParticlesPerCountry, ConnectivityMatrix, IoParticleDevelopmentTimeseries
from plot_tools.map_plotter import MapPlot, get_colormap_reds
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import numpy as np
from datetime import datetime

small_islands = ['Maldives','Mauritius','France','Indian Ocean Territories','Indian Ocean Territories',
                 'Comoros','Seychelles','British Indian Ocean Territory']
small_islands_lon = [73.10,57.59,55.63,105.62,96.81,43.84,55.59,72.65]
small_islands_lat = [3.66,-20.43,-21.24,-10.75,-11.69,-11.90,-4.50,-7.32]

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
def _get_density_and_density_per_country_data(p=0.5,basin_name='io_nh',dx=8,extra_description=None):
    input_path_density = get_beached_density_path(basin_name,dx,p,extra_description=extra_description)
    density = Density.read_from_netcdf(input_path_density,t_index=-1)
    input_path_ppcountry = get_postprocessing_path(basin_name,dx,p,'ppcountry','csv',extra_description=extra_description)
    ppcountry = ParticlesPerCountry.read_from_csv(input_path_ppcountry)
    return density.density,ppcountry.particles_per_country,ppcountry.countries,density.grid.lon,density.grid.lat

# ---------------------------------------------------------
# Plot set-up
# ---------------------------------------------------------
def _logarithmic_colormap():
    colors = get_colormap_reds(6)
    ranges = [1,10,100,10**3,10**4,10**5,10**6]
    cm = LinearSegmentedColormap.from_list('cm_log_density',colors,N=6)
    norm = BoundaryNorm(ranges,ncolors=6)
    return colors,ranges,cm,norm

def _get_color_per_country(density_per_country,ranges=[1,10,100,10**3,10**4,10**5,10**6]):
    colors = get_colormap_reds(len(ranges))
    color_per_country = []
    for d in density_per_country:
        l_range = []
        for i in range(len(colors)):
            l_range.append(ranges[i] <= d < ranges[i+1])
        i_color = np.where(l_range)[0][0]
        color_per_country.append(colors[i_color])
    return color_per_country

def _source_info():
    ranges = np.array([350000.,20000.,2000.,200.,20.,2.,0.])
    labels = ['> 20,000','2,000 - 20,000','200 - 2,000','20 - 200','2 - 20','< 2']
    colors = get_colormap_reds(len(ranges)-1)[::-1]
    edge_widths = [0.7,0.7,0.7,0.5,0.5,0.5]
    sizes = [6,5,4,3,2,1]*6
    legend_sizes = [6,5,4,3,2,1]
    return (ranges,labels,colors,edge_widths,sizes,legend_sizes)

def _get_marker_colors_sizes_edgewidths_for_sources(waste_input):
    (ranges,_,colors,edge_widths,sizes,_) = _source_info()
    source_colors = []
    source_sizes = []    
    source_edge_widths = []
    for waste in waste_input:
        l_range = []
        for i in range(len(colors)):
            l_range.append(ranges[i] >= waste >= ranges[i+1])
        i_range = np.where(l_range)[0][0]
        source_colors.append(colors[i_range])
        source_sizes.append(sizes[i_range])
        source_edge_widths.append(edge_widths[i_range])
    return source_colors,source_sizes,source_edge_widths

def _get_legend_entries_for_sources():
    (_,labels,colors,edge_widths,_,legend_sizes) = _source_info()
    legend_entries = []
    for i in range(len(colors)):
        legend_entries.append(Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i],                              
                              markersize=legend_sizes[i],label=labels[i],markeredgewidth=edge_widths[i]))
    return legend_entries

def _timeseries_particle_development(ax,io_dev):
    ax.plot(io_dev.time,io_dev.beached_nio/io_dev.total_particles*100,
             '-',color='#1F1F1F',linewidth=2,label='beached in NIO')
    ax.plot(io_dev.time,io_dev.ocean_nio/io_dev.total_particles*100,
             '--',color='#1F1F1F',linewidth=2,label='afloat in NIO')
    ax.plot(io_dev.time,io_dev.beached_sio/io_dev.total_particles*100,
             '-',color='#3B7DDF',linewidth=2,label='beached in SIO')
    ax.plot(io_dev.time,io_dev.ocean_sio/io_dev.total_particles*100,
             '--',color='#3B7DDF',linewidth=2,label='afloat in SIO')
    ax.plot(io_dev.time,io_dev.escaped_io/io_dev.total_particles*100,
             '-',color='#808080',linewidth=2,label='escaped IO')
    n_years = 10
    xticks = []
    for n in range(n_years+1):
        xticks.append(datetime(io_dev.time[0].year+n,1,1,12,0))
    xticks = np.array(xticks)
    time0 = io_dev.time[0]    
    xticklabels = np.arange(0,11,1)
    xlim = [datetime(time0.year+1,time0.month,time0.day,time0.hour),datetime(time0.year+10,time0.month,time0.day,time0.hour)]
    time0 = io_dev.time[0]
    ax.set_xticks(xticks)
    ax.set_xlim(xlim)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Simulation time [years]')
    ax.set_ylabel('Particles [% of total]')
    ax.set_ylim(0,110)
    ax.set_yticks(np.arange(0,110,10))
    ax.set_yticklabels(['0','','20','','40','','60','','80','','100'])

def _io_basic_map(ax,xmarkers='bottom',ymarkers='left',lon_range=None,lat_range=None):
    if lon_range is None:
        lon_range, _ = get_io_lon_lat_range()
        meridians = [0,40,80,120]
    else:
        meridians = np.arange(lon_range[0],lon_range[1]+40,40)
    if lat_range is None:
        _, lat_range = get_io_lon_lat_range()
        parallels = [-40,-20,0,20,40]
    else:
        parallels = np.arange(lat_range[0],lat_range[1]+20,20)    
    mplot = MapPlot(ax,lon_range,lat_range,meridians=meridians,parallels=parallels,
                    xmarkers=xmarkers,ymarkers=ymarkers)
    return mplot

def _io_density_map(mplot,lon,lat,density,ranges=None,cmap='Reds'):
    z = np.copy(density)
    z[z == 0] = np.nan
    if ranges is None:
        c, ranges = mplot.pcolormesh(lon,lat,z,show_cbar=False,cmap=cmap)
    else:
        c, ranges = mplot.pcolormesh(lon,lat,z,show_cbar=False,ranges=ranges,cmap=cmap)
    return c, ranges

def _io_countries(mplot,density_per_country,country):
    color_per_country = _get_color_per_country(density_per_country)
    all_countries = Countries()
    for c in all_countries.countries:
        i_plastic = [i for i,s in enumerate(country) if s == c.attributes['SUBUNIT']]
        if len(i_plastic) is not 0:
            c_color = color_per_country[i_plastic[0]]
            mplot.country(c,c_color)
            for i, small_island in enumerate(small_islands):
                if small_island == c.attributes['SUBUNIT']:
                    mplot.points(small_islands_lon[i],small_islands_lat[i],marker='o',
                                 facecolor=c_color,markersize=10)
        else:
            mplot.country(c,'w')

def _add_horizontal_colorbar(fig,ax,c,ranges,scale_width=1,
                             ticklabels=['1','10','10$^2$','10$^3$','10$^4$','10$^5$','10$^6$'],
                             cbarlabel = 'Particle density [# per grid cell]'):
    l,b,w,h = ax.get_position().bounds
    # cbax = fig.add_axes([l+0.11,b-0.06,0.65*w,0.02])
    cbax = fig.add_axes([l,b-0.1,scale_width*w,0.02])
    cbar = plt.colorbar(c,ticks=ranges,orientation='horizontal',cax=cbax)
    cbar.ax.set_xticklabels(ticklabels)
    cbar.set_label(cbarlabel)
    return cbar

def _add_vertical_colorbar(fig,ax,c,ranges,height_factor=2,
                           ticklabels=['1','10','10$^2$','10$^3$','10$^4$','10$^5$','10$^6$'],
                           cbarlabel = 'Particle density [# per grid cell]'):
    l,b,w,h = ax.get_position().bounds
    cbax = fig.add_axes([l+w,b,0.02,h*height_factor+0.02])
    cbar = plt.colorbar(c,ticks=ranges,orientation='vertical',cax=cbax)
    cbar.ax.set_yticklabels(ticklabels)
    cbar.set_label(cbarlabel)
    return cbar

def _connectivity_matrix(ax,matrix,countries,label_top=False):
    colors_matrix = get_colormap_reds(5)
    ranges_matrix = [0,20,40,60,80,100]
    cm_matrix = LinearSegmentedColormap.from_list('cm_log_density',colors_matrix,N=5)
    norm_matrix = BoundaryNorm(ranges_matrix,ncolors=5)    
    im = ax.imshow(matrix,cmap=cm_matrix,norm=norm_matrix)
    texts = _annotate_heatmap(im)
    # grid
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.tick_params(length=0)
    ax.set_xticklabels(countries)
    ax.set_yticklabels(countries)
    ax.tick_params(top=label_top,bottom=False,labeltop=label_top,labelbottom=False)
    plt.setp(ax.get_xticklabels(),rotation='vertical',ha='center')
    ax.grid(False)
    ax.set_xticks(np.arange(matrix.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0]+1)-.5, minor=True)    
    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.25)
    ax.tick_params(which='minor', bottom=False, left=False)

def _annotate_heatmap(im, data=None, valfmt="{x:.0f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    Copied from: https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html
    Last accessed: 27-10-2020
    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] > 1.:
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

    return texts

# ---------------------------------------------------------
# Plots for main article
# van der Mheen et al. (2020)
# https://doi.org/10.5194/os-2020-50
# ---------------------------------------------------------
def figure2_plastic_sources(output_path=None,plot_style='plot_tools/plot.mplstyle'):
    global_sources = RiverSources.read_from_netcdf()
    nio_sources = global_sources.get_riversources_from_ocean_basin('io_nh')
    nio_total_waste = np.sum(nio_sources.waste,axis=0)
    nio_yearly_waste = np.sum(nio_sources.waste,axis=1)
    i_use = np.where(nio_yearly_waste >= 1)
    nio_yearly_waste = nio_yearly_waste[i_use]
    lon_sources = nio_sources.lon[i_use]
    lat_sources = nio_sources.lat[i_use]
    (source_colors,
    source_sizes,
    source_edgewidths) = _get_marker_colors_sizes_edgewidths_for_sources(nio_yearly_waste)
    # plot
    plt.style.use(plot_style)
    fig = plt.figure(figsize=(4,5))
    # map of NIO sources
    ax1 = plt.subplot(2,1,1,projection=ccrs.PlateCarree())
    mplot1 = _io_basic_map(ax1)
    mplot1.points(lon_sources,lat_sources,marker='o',facecolor=source_colors,
                  markersize=source_sizes,edgewidth=source_edgewidths)
    mplot1.add_subtitle('(a) NIO river plastic source locations')
    legend_entries = _get_legend_entries_for_sources()
    ax1.set_anchor('W')
    ax1.legend(handles=legend_entries,title='[tonnes year$^{-1}$]',loc='upper right',
               bbox_to_anchor=(1.6,0.95))
    # timeseries of NIO sources
    ax2 = plt.subplot(2,1,2)
    ax2.plot(nio_sources.time,nio_total_waste/1000,'-',color='k',linewidth=2)
    ax2.set_ylim(0,70)
    ax2.set_ylabel('Plastic input [10$^3$ tonnes]')
    ax2.set_xlim(1,12)
    ax2.set_xticks(np.arange(1,13,1))
    ax2.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    anchored_text2 = AnchoredText('(b) NIO monthly river plastic input',loc='upper left',borderpad=0.0)
    ax2.add_artist(anchored_text2)
    if output_path:
        plt.savefig(output_path,bbox_inches='tight',dpi=300)
    plt.show()

def figure3_particles_monsoon(output_path=None,plot_style='plot_tools/plot.mplstyle'):
    ne_date = datetime(2009,2,28) 
    inter1_date = datetime(2009,5,29)
    sw_date = datetime(2009,8,27)
    inter2_date = datetime(2009,11,30)
    input_path = get_density_path('io_nh',extra_description='neutral_iod')
    ne_density = Density.read_from_netcdf(input_path,time_start=ne_date,time_end=ne_date)
    inter1_density = Density.read_from_netcdf(input_path,time_start=inter1_date,time_end=inter1_date)
    sw_density = Density.read_from_netcdf(input_path,time_start=sw_date,time_end=sw_date)
    inter2_density = Density.read_from_netcdf(input_path,time_start=inter2_date,time_end=inter2_date)
    lon = ne_density.grid.lon
    lat = ne_density.grid.lat
    lon_range = [40,120]
    lat_range = [-20,40]
    # plot
    plt.style.use(plot_style)
    fig = plt.figure(figsize=(5,4))
    # NE monsoon
    ax1 = plt.subplot(2,2,1,projection=ccrs.PlateCarree())
    mplot1 = _io_basic_map(ax1,lon_range=lon_range,lat_range=lat_range,xmarkers='off')    
    mplot1.pcolormesh(lon,lat,ne_density.density[0,:,:],show_cbar=False)
    mplot1.add_subtitle('(a) NE monsoon (Feb)')
    ax1.set_xticklabels([])
    # intermonsoon period 1
    ax2 = plt.subplot(2,2,2,projection=ccrs.PlateCarree())
    mplot2 = _io_basic_map(ax2,lon_range=lon_range,lat_range=lat_range,xmarkers='off',ymarkers='off')    
    mplot2.pcolormesh(lon,lat,inter1_density.density[0,:,:],show_cbar=False)
    mplot2.add_subtitle('(b) Intermonsoon (May)')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    # SW monsoon
    ax3 = plt.subplot(2,2,3,projection=ccrs.PlateCarree())
    mplot3 = _io_basic_map(ax3,lon_range=lon_range,lat_range=lat_range)
    c, ranges = mplot3.pcolormesh(lon,lat,sw_density.density[0,:,:],show_cbar=False)
    mplot3.add_subtitle('(c) SW monsoon (Aug)')
    _add_horizontal_colorbar(fig,ax3,c,ranges,scale_width=2.2)
    # intermonsoon period 2
    ax4 = plt.subplot(2,2,4,projection=ccrs.PlateCarree())
    mplot4 = _io_basic_map(ax4,lon_range=lon_range,lat_range=lat_range,ymarkers='off')
    mplot4.pcolormesh(lon,lat,inter2_density.density[0,:,:],show_cbar=False)
    mplot4.add_subtitle('(d) Intermonsoon (Nov)')
    ax4.set_yticklabels([])
    # save
    if output_path:
        plt.savefig(output_path,bbox_inches='tight',dpi=300)
    plt.show()

def figure4_timeseries_particles(output_path=None,plot_style='plot_tools/plot.mplstyle'):
    input_path_p005 = get_postprocessing_path('io_nh',8,0.05,'dev','nc')
    input_path_p05 = get_postprocessing_path('io_nh',8,0.5,'dev','nc')
    input_path_p095 = get_postprocessing_path('io_nh',8,0.95,'dev','nc')
    io_dev_p005 = IoParticleDevelopmentTimeseries.read_from_netcdf(input_path_p005)
    io_dev_p05 = IoParticleDevelopmentTimeseries.read_from_netcdf(input_path_p05)
    io_dev_p095 = IoParticleDevelopmentTimeseries.read_from_netcdf(input_path_p095)
    # plot
    plt.style.use(plot_style)    
    plt.rcParams.update({'figure.subplot.hspace' : 0.3})
    fig = plt.figure(figsize=(3,5))
    # p = 0.95
    ax1 = plt.subplot(3,1,1)   
    # --- particle development currents p = 0.95 ---
    ax1 = plt.subplot(3,1,1)
    _timeseries_particle_development(ax1,io_dev_p095)
    # --- particle development currents p = 0.50 ---
    ax2 = plt.subplot(3,1,2)
    _timeseries_particle_development(ax2,io_dev_p05)
    # --- particle development stokes p = 0.05 ---
    ax3 = plt.subplot(3,1,3)
    _timeseries_particle_development(ax3,io_dev_p005)
    # subtitles
    anchored_text1 = AnchoredText('(a) High beaching probability (p = 0.95)',loc='upper right',borderpad=0.0)
    ax1.add_artist(anchored_text1)
    anchored_text2 = AnchoredText('(b) Beaching probability p = 0.50',loc='upper right',borderpad=0.0)
    ax2.add_artist(anchored_text2)
    anchored_text3 = AnchoredText('(c) Low beaching probability (p = 0.05)',loc='upper right',borderpad=0.0)
    ax3.add_artist(anchored_text3)
    # legend
    ax1.legend(loc='lower right')
    if output_path:
        plt.savefig(output_path,bbox_inches='tight',dpi=300)
    plt.show()

def figure5_nio_beaching(output_path=None,plot_style='plot_tools/plot.mplstyle'):
    relevant_countries = ['Bangladesh','Myanmar','Malaysia','India (AS)','India (BoB)',
                          'Indonesia','Pakistan','Sri Lanka','Thailand','Somalia',
                          'Maldives','Madagascar','Mozambique']
    # load density and particles per country data
    density_p095,ppcountry_p095,countries_p095,lon,lat = _get_density_and_density_per_country_data(p=0.95)
    density_p05,ppcountry_p05,countries_p05,_,_ = _get_density_and_density_per_country_data(p=0.5)
    density_p005,ppcountry_p005,countries_p005,_,_ = _get_density_and_density_per_country_data(p=0.05)
    density = Density.read_from_netcdf(get_density_path('io_nh'),t_index=-1)
    # load connectivity matrix data
    connectivitymatrix_p095 = ConnectivityMatrix.read_from_csv(get_postprocessing_path('io_nh',8,0.95,'connectivity','csv'))
    connectivitymatrix_p05 = ConnectivityMatrix.read_from_csv(get_postprocessing_path('io_nh',8,0.5,'connectivity','csv'))
    connectivitymatrix_p005 = ConnectivityMatrix.read_from_csv(get_postprocessing_path('io_nh',8,0.05,'connectivity','csv'))
    matrix_p095 = connectivitymatrix_p095.return_matrix_for_requested_countries(relevant_countries)
    matrix_p05 = connectivitymatrix_p05.return_matrix_for_requested_countries(relevant_countries)
    matrix_p005 = connectivitymatrix_p005.return_matrix_for_requested_countries(relevant_countries)
    # plot
    plt.style.use(plot_style)
    plt.rcParams.update({'figure.subplot.hspace' : 0.3})
    fig = plt.figure(figsize=(4,5))
    # p = 0.95 - map
    ax1 = plt.subplot(4,2,1,projection=ccrs.PlateCarree())
    mplot1 = _io_basic_map(ax1)
    c, ranges = _io_density_map(mplot1,lon,lat,density_p095)
    _io_countries(mplot1,ppcountry_p095,countries_p095)
    mplot1.add_subtitle('(a) High beaching probability (p = 0.95)')
    ax1.set_xticklabels([])
    # p = 0.95 - matrix
    ax2 = plt.subplot(4,2,2)
    _connectivity_matrix(ax2,matrix_p095,relevant_countries,label_top=True)
    anchored_text2 = AnchoredText('(e) Connectivity matrix p = 0.95',loc='lower center',borderpad=-2.5)
    ax2.add_artist(anchored_text2)
    # p = 0.5 - map
    ax3 = plt.subplot(4,2,3,projection=ccrs.PlateCarree())
    mplot3 = _io_basic_map(ax3)
    _io_density_map(mplot3,lon,lat,density_p05)
    _io_countries(mplot3,ppcountry_p05,countries_p05)
    mplot3.add_subtitle('(b) Beaching probability p = 0.50')
    ax3.set_xticklabels([])
    # p = 0.5 - matrix
    ax4 = plt.subplot(4,2,4)
    _connectivity_matrix(ax4,matrix_p05,relevant_countries)
    anchored_text4 = AnchoredText('(f) Connectivity matrix p = 0.50',loc='lower center',borderpad=-2.5)
    ax4.add_artist(anchored_text4)
    # p = 0.05 - map
    ax5 = plt.subplot(4,2,5,projection=ccrs.PlateCarree())
    mplot5 = _io_basic_map(ax5)
    _io_density_map(mplot5,lon,lat,density_p005)
    _io_countries(mplot5,ppcountry_p005,countries_p005)
    mplot5.add_subtitle('(c) Low beaching probability p = 0.05')
    ax5.set_xticklabels([])
    # p = 0.05 - matrix
    ax6 = plt.subplot(4,2,6)
    _connectivity_matrix(ax6,matrix_p005,relevant_countries)
    anchored_text6 = AnchoredText('(g) Connectivity matrix p = 0.05',loc='lower center',borderpad=-2.5)
    ax6.add_artist(anchored_text6)
    # no beaching - map
    ax7 = plt.subplot(4,2,7,projection=ccrs.PlateCarree())
    mplot7 = _io_basic_map(ax7)
    _io_density_map(mplot7,lon,lat,density.density)
    _io_countries(mplot7,[],[])
    mplot7.add_subtitle('(d) No beaching')
    # colorbar
    _add_horizontal_colorbar(fig,ax7,c,ranges)
    if output_path:
        plt.savefig(output_path,bbox_inches='tight',dpi=300)
    plt.show()

def figure6_sio_beaching(output_path=None,plot_style='plot_tools/plot.mplstyle'):
    # load source data
    global_sources = RiverSources.read_from_netcdf()
    sio_sources = global_sources.get_riversources_from_ocean_basin('io_sh')    
    sio_yearly_waste = np.sum(sio_sources.waste,axis=1)
    i_use = np.where(sio_yearly_waste >= 1)
    sio_yearly_waste = sio_yearly_waste[i_use]
    lon_sources = sio_sources.lon[i_use]
    lat_sources = sio_sources.lat[i_use]
    (source_colors,
    source_sizes,
    source_edgewidths) = _get_marker_colors_sizes_edgewidths_for_sources(sio_yearly_waste)
    # load density and particles per country data
    density_p095,ppcountry_p095,countries_p095,lon,lat = _get_density_and_density_per_country_data(basin_name='io_sh',p=0.95)
    density_p05,ppcountry_p05,countries_p05,_,_ = _get_density_and_density_per_country_data(basin_name='io_sh',p=0.5)
    density_p005,ppcountry_p005,countries_p005,_,_ = _get_density_and_density_per_country_data(basin_name='io_sh',p=0.05)
    density = Density.read_from_netcdf(get_density_path('io_sh'),t_index=-1)
    # load connectivity matrix data
    relevant_countries = ['Indonesia','East Timor','Madagascar','South Africa',
                          'Mozambique','Tanzania','Kenya','Comoros','Somalia',
                          'Mauritius','France','Indian Ocean Territories','Australia']
    connectivitymatrix_p095 = ConnectivityMatrix.read_from_csv(get_postprocessing_path('io_sh',8,0.95,'connectivity','csv'))
    connectivitymatrix_p05 = ConnectivityMatrix.read_from_csv(get_postprocessing_path('io_sh',8,0.5,'connectivity','csv'))
    connectivitymatrix_p005 = ConnectivityMatrix.read_from_csv(get_postprocessing_path('io_sh',8,0.05,'connectivity','csv'))
    matrix_p095 = connectivitymatrix_p095.return_matrix_for_requested_countries(relevant_countries)
    matrix_p05 = connectivitymatrix_p05.return_matrix_for_requested_countries(relevant_countries)
    matrix_p005 = connectivitymatrix_p005.return_matrix_for_requested_countries(relevant_countries)
    # plot
    plt.style.use(plot_style)
    plt.rcParams.update({'figure.subplot.hspace' : 0.3})
    fig = plt.figure(figsize=(4,5))
    # map of SIO sources
    ax0= plt.subplot(4,2,1,projection=ccrs.PlateCarree())
    mplot0 = _io_basic_map(ax0)
    mplot0.points(lon_sources,lat_sources,marker='o',facecolor=source_colors,
                  markersize=source_sizes,edgewidth=source_edgewidths)
    mplot0.add_subtitle('(a) SIO river plastic source locations')
    legend_entries = _get_legend_entries_for_sources()
    ax0.set_anchor('W')
    ax0.legend(handles=legend_entries,title='[tonnes year$^{-1}$]',loc='upper right',
               bbox_to_anchor=(1.7,1.))
    # p = 0.95 - map
    ax1 = plt.subplot(4,2,3,projection=ccrs.PlateCarree())
    mplot1 = _io_basic_map(ax1)
    c, ranges = _io_density_map(mplot1,lon,lat,density_p095)
    _io_countries(mplot1,ppcountry_p095,countries_p095)
    mplot1.add_subtitle('(a) High beaching probability (p = 0.95)')
    ax1.set_xticklabels([])
    # p = 0.95 - matrix
    ax2 = plt.subplot(4,2,4)
    _connectivity_matrix(ax2,matrix_p095,relevant_countries,label_top=True)
    anchored_text2 = AnchoredText('(e) Connectivity matrix p = 0.95',loc='lower center',borderpad=-2.5)
    ax2.add_artist(anchored_text2)
    # p = 0.5 - map
    ax3 = plt.subplot(4,2,5,projection=ccrs.PlateCarree())
    mplot3 = _io_basic_map(ax3)
    _io_density_map(mplot3,lon,lat,density_p05)
    _io_countries(mplot3,ppcountry_p05,countries_p05)
    mplot3.add_subtitle('(b) Beaching probability p = 0.50')
    ax3.set_xticklabels([])
    # p = 0.5 - matrix
    ax4 = plt.subplot(4,2,6)
    _connectivity_matrix(ax4,matrix_p05,relevant_countries)
    anchored_text4 = AnchoredText('(f) Connectivity matrix p = 0.50',loc='lower center',borderpad=-2.5)
    ax4.add_artist(anchored_text4)
    # p = 0.05 - map
    ax5 = plt.subplot(4,2,7,projection=ccrs.PlateCarree())
    mplot5 = _io_basic_map(ax5)
    _io_density_map(mplot5,lon,lat,density_p005)
    _io_countries(mplot5,ppcountry_p005,countries_p005)
    mplot5.add_subtitle('(c) Low beaching probability p = 0.05')
    ax5.set_xticklabels([])
    # p = 0.05 - matrix
    ax6 = plt.subplot(4,2,8)
    _connectivity_matrix(ax6,matrix_p005,relevant_countries)
    anchored_text6 = AnchoredText('(g) Connectivity matrix p = 0.05',loc='lower center',borderpad=-2.5)
    ax6.add_artist(anchored_text6)
    # colorbar
    _add_horizontal_colorbar(fig,ax5,c,ranges)
    if output_path:
        plt.savefig(output_path,bbox_inches='tight',dpi=300)
    plt.show()
