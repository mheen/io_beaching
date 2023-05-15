from plastic_sources import RiverSources
from plot_tools.map_plotter import MapPlot, get_colormap_reds
from particles import BeachingParticles, Density
from utilities import get_dir, add_month_to_timestamp
from ocean_utilities import read_mean_hycom_data
from postprocessing_iot import get_l_particles_in_box, get_cki_box_lon_lat_range, get_christmas_box_lon_lat_range
from postprocessing_iot import get_iot_lon_lat_range, get_iot_sources, get_main_sources_lon_lat_n_particles
from postprocessing_iot import get_original_source_based_on_lon0_lat0, get_n_particles_per_month_release_arrival
from postprocessing_iot import get_cki_plastic_measurements, get_christmas_plastic_measurements, get_cki_box_lon_lat_range, get_christmas_box_lon_lat_range
from countries import CountriesGridded
from geojson import Feature, FeatureCollection, LineString, dump
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cftr
import cartopy.io.shapereader as shpreader
import numpy as np
from datetime import datetime
import string
import time
import shapefile
from shapely.geometry import Polygon, Point

def get_months_colors():
    colors = ['#ffedbc', '#fece6b', '#fdc374', '#fb9d59', '#f57547', '#d00d20',
    '#c9e7f1', '#90c3dd', '#4576b4', '#000086', '#4d00aa', '#30006a']
    return colors

def _add_horizontal_colorbar(fig,ax,c,ranges,scale_width=1,
                             ticklabels=['1','10','10$^2$','10$^3$','10$^4$'],
                             cbarlabel = 'Mean particle density [# per grid cell]'):
    l,b,w,h = ax.get_position().bounds
    cbax = fig.add_axes([l,b-0.07,scale_width*w,0.02])
    cbar = plt.colorbar(c,ticks=ranges,orientation='horizontal',cax=cbax)
    cbar.ax.set_xticklabels(ticklabels)
    cbar.set_label(cbarlabel)
    return cbar

def _iot_basic_map(ax, xmarkers='bottom', ymarkers='left', lon_range=None, lat_range=None, dlon=4, dlat=2):
    if lon_range is None:
        lon_range, _ = get_iot_lon_lat_range()
        meridians = [90, 100, 110, 120, 128]        
    else:
        meridians = np.arange(lon_range[0], lon_range[1], dlon)
    if lat_range is None:
        _, lat_range = get_iot_lon_lat_range()
        parallels = [-20, -15, -10, -5, 0, 5]
    else:
        parallels = np.arange(lat_range[0], lat_range[1], dlat)    
    mplot = MapPlot(ax, lon_range, lat_range, meridians=meridians, parallels=parallels,
                    xmarkers=xmarkers, ymarkers=ymarkers)
    return mplot

def _main_source_info():
    ranges = np.array([400., 100., 50., 25., 10., 5., 1.])
    labels = ['> 100','50 - 100','25 - 50','10 - 25', '5 - 10','1 - 5']
    colors = get_colormap_reds(len(ranges)-1)[::-1]
    edge_widths = [0.7,0.7,0.7,0.5,0.5,0.5]
    sizes = [6,5,4,3,2,1]
    legend_sizes = [6,5,4,3,2,1]
    return (ranges,labels,colors,edge_widths,sizes,legend_sizes)

def _samples_info(plastic_type):
    if plastic_type == 'count':
        ranges = np.array([66000, 50000, 5000, 500, 50, 1])
        labels = ['>50000', '5000 - 50000', '500 - 5000', '50 - 500', '1 - 50']
        colors = get_colormap_reds(len(ranges)-1)[::-1]
        edge_widths = [0.7, 0.7, 0.7, 0.5, 0.5]
        sizes = [5, 4, 3, 2, 1]
        legend_sizes = [5, 4, 3, 2, 1]
    elif plastic_type == 'mass':
        ranges = np.array([2000, 1500, 1000, 500, 100, 50, 1])
        labels = ['>1500', '1000 - 1500', '500 - 1000', '100 - 500', '50 - 100', '1 - 50']
        colors = get_colormap_reds(len(ranges)-1)[::-1]
        edge_widths = [0.7, 0.7, 0.7, 0.7, 0.5, 0.5]
        sizes = [6, 5, 4, 3, 2, 1]
        legend_sizes = [6, 5, 4, 3, 2, 1]
    else:
        raise ValueError(f'Unknown plastic type requested: {plastic_type}. Valid options are: count and mass.')
    return (ranges, labels, colors, edge_widths, sizes, legend_sizes)

def _get_marker_colors_sizes_edgewidths_for_main_sources(waste_input):
    (ranges,_,colors,edge_widths,sizes,_) = _main_source_info()
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

def _get_marker_colors_sizes_for_samples(samples, plastic_type):
    ranges, _, colors, edge_widths, sizes, _ = _samples_info(plastic_type)
    sample_colors = []
    sample_sizes = []
    sample_edge_widths = []
    for sample in samples:
        if np.isnan(sample):
            continue
        l_range = []
        for i in range(len(colors)):
            l_range.append(ranges[i]>=sample>=ranges[i+1])
        i_range = np.where(l_range)[0][0]
        sample_colors.append(colors[i_range])
        sample_sizes.append(sizes[i_range])
        sample_edge_widths.append(edge_widths[i_range])
    return sample_colors, sample_sizes, sample_edge_widths

def _get_legend_entries_for_main_sources():
    (_,labels,colors,edge_widths,_,legend_sizes) = _main_source_info()
    legend_entries = []
    for i in range(len(colors)):
        legend_entries.append(Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i],                              
                              markersize=legend_sizes[i],label=labels[i],markeredgewidth=edge_widths[i]))
    return legend_entries

def _get_legend_entries_samples(plastic_type):
    _, labels, colors, edge_widths, _, legend_sizes = _samples_info(plastic_type)
    legend_entries = []
    for i in range(len(colors)):
        legend_entries.append(Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i],                              
                              markersize=legend_sizes[i],label=labels[i],markeredgewidth=edge_widths[i]))
    return legend_entries

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

def _get_sorted_river_contribution_info(lon_main_sources,
                                        lat_main_sources,
                                        waste_main_sources,
                                        main_source_colors,
                                        river_names,
                                        iot_name):
    i_sort = np.argsort(waste_main_sources)[::-1]
    i_5ormore = np.where(waste_main_sources[i_sort] >= 5)
    waste_big_sources = waste_main_sources[i_sort][i_5ormore]
    lon_big_sources = lon_main_sources[i_sort][i_5ormore]
    lat_big_sources = lat_main_sources[i_sort][i_5ormore]
    colors_big_sources = np.array(main_source_colors)[i_sort][i_5ormore]
    i_biggest_sources = waste_big_sources >= 50
    lon_biggest_sources = lon_big_sources[i_biggest_sources]
    lat_biggest_sources = lat_big_sources[i_biggest_sources]
    waste_biggest_sources = waste_big_sources[i_biggest_sources]
    percentage_waste_big_sources = waste_big_sources/np.sum(waste_main_sources)*100
    x = np.arange(0, len(waste_big_sources))
    if river_names == []:
        # write lon and lat of rivers contributing >50 particles (to find names and add to labels)
        with open('iot_main_sources.txt', 'a') as f:
            f.write(f'\n{iot_name}\n')
            f.write('Locations of largest sources (> 50 particles):\n')
            for i in range(len(lon_biggest_sources)):
                f.write(f'\nSource {i+1}: {lon_biggest_sources[i]}, {lat_biggest_sources[i]}, {waste_biggest_sources[i]}\n')
                lon_org, lat_org, waste_org = get_original_source_based_on_lon0_lat0(lon_biggest_sources[i], lat_biggest_sources[i])
                f.write('Original source locations:\n')
                for j in range(len(lon_org)):
                    f.write(f'{lon_org[j]}, {lat_org[j]}, {waste_org[j]}\n')
    return x, percentage_waste_big_sources, colors_big_sources

def _histogram_release_arrival(ax, n_release, n_entry, ylim=[0, 500]):
    p_release = n_release#/np.sum(n_release)*100
    p_entry = n_entry#/np.sum(n_entry)*100
    months = np.arange(1,13,1)
    colors = get_months_colors()
    ax.bar(months-0.2, p_release, width=0.4, label='Release', color=colors,
           hatch='////', edgecolor='k', zorder=5)
    n_entry_cumulative = p_entry.cumsum(axis=1)
    ax.bar(months+0.2, p_entry[:, 0], width=0.4, color=colors[0],
           edgecolor='k', zorder=5)
    for i in range(1, 11):
        heights = n_entry_cumulative[:, i]        
        starts = n_entry_cumulative[:, i-1]
        ax.bar(months+0.2, p_entry[:, i], bottom=n_entry_cumulative[:, i-1],
                width=0.4, color=colors[i], edgecolor='k', zorder=5)
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_ylabel('Particles [#/month]')
    ax.set_ylim(ylim)

def _get_plastic_samples(plastic_type='count') -> tuple:
    if not np.logical_or(plastic_type=='count', plastic_type=='mass'):
        raise ValueError(f'Unknown plastic type requested: {plastic_type}. Valid values are: count and mass.')

    months = np.arange(1,13,1)
    # plastic measurements
    time_cki, _, _, n_plastic_cki, kg_plastic_cki = get_cki_plastic_measurements()
    time_ci, _, _, n_plastic_ci, kg_plastic_ci = get_christmas_plastic_measurements()
    n_plastic_month_cki = np.zeros(12)
    n_months_cki = np.zeros(12)
    n_plastic_month_ci = np.zeros(12)
    n_months_ci = np.zeros(12)
    # cki
    for i, t in enumerate(time_cki):
        n_months_cki[t.month-1] += 1
        if plastic_type == 'count':
            n_plastic_month_cki[t.month-1] = np.nansum([n_plastic_month_cki[t.month-1], n_plastic_cki[i]])
        elif plastic_type == 'mass':
            n_plastic_month_cki[t.month-1] = np.nansum([n_plastic_month_cki[t.month-1], kg_plastic_cki[i]])
    # ci
    for i, t in enumerate(time_ci):
        n_months_ci[t.month-1] += 1
        if plastic_type == 'count':
            n_plastic_month_ci[t.month-1] = np.nansum([n_plastic_month_ci[t.month-1], n_plastic_ci[i]])
        elif plastic_type == 'mass':
            n_plastic_month_ci[t.month-1] = np.nansum([n_plastic_month_ci[t.month-1], kg_plastic_ci[i]])
    
    return (months, n_plastic_month_cki, n_plastic_month_ci, n_months_cki, n_months_ci)

def _histogram_samples(ax, months, n_plastic, n_months, plastic_type='count', show_legend=True,
                       yticks_left=True, yticks_right=True) -> plt.axes:
    colors = get_months_colors()
    color = '#adadad'

    if plastic_type == 'count':
        ylabel = 'Plastic items [#/month]'
        ylim = [0, 140000]
        yticks = np.arange(0, 140000, 20000)
        ylim_studies = [0, 14]
        yticks_studies = np.arange(0, 14, 2)
    elif plastic_type == 'mass':
        ylabel = 'Plastic weight [kg/month]'
        ylim = [0, 4000]
        yticks = np.arange(0, 4000, 500)
        ylim_studies = [0, 16]
        yticks_studies = np.arange(0, 16, 2)
    else:
        raise ValueError(f'Unknown plastic type requested: {plastic_type}. Valid values are: count and mass.')
    
    # number of plastic items
    ax.bar(months-0.2, n_plastic, width=0.4, color=colors[4], edgecolor='k', zorder=5)
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    if yticks_left == False:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel(ylabel)
    # number of sampling studies per month
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.bar(months+0.2, n_months, width=0.4, color=color, edgecolor='k', zorder=6)
    ax2.set_ylim(ylim_studies)
    ax2.set_yticks(yticks_studies)
    if yticks_right == False:
        ax2.set_yticklabels([])
    else:
        ax2.set_ylabel('Beach clean-ups [#/month]')
    ax2.spines['right'].set_color(color)
    ax2.tick_params(axis='y', colors=color)
    ax2.yaxis.label.set_color(color)

    # legend
    if show_legend:
        legend_elements = [Patch(facecolor=colors[3], edgecolor='k', label='Plastic items'),
                        Patch(facecolor=color, edgecolor='k', label='Beach clean-ups')]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.08, 1.0))
    return ax

def figure1_overview(output_path=None,
                     river_dir = get_dir('indonesia_rivers'),
                     river_filenames=['progo.shp', 'bogowonto.shp', 'serayu.shp', 'tanduy.shp', 'wulan.shp'],
                     river_color = '#002eb5',
                     linewidth=1.5,
                     plot_style='plot_tools/plot.mplstyle'):
    # boxes IOT
    lon_range_cki, lat_range_cki = get_cki_box_lon_lat_range()
    lon_range_ci, lat_range_ci = get_christmas_box_lon_lat_range()
    # box Java:
    lon_range_java = [105., 115.6]
    lat_range_java = [-9., -5.]
    meridians_java = [105, 110, 115]
    parallels_java = np.arange(lat_range_java[0], lat_range_java[1]+1, 1)
    # box zoom Java:
    lon_range = [107.5, 111.]
    lat_range = [-8.2, -6.]
    meridians = [108, 109, 110]
    parallels = [-8, -7, -6]
    # cities
    city_names = ['Tasikmalaya', 'Purwokerto', 'Wonosobo', 'Purworejo', 'Magelang', 'Yogyakarta']
    city_lons = [108.22, 109.25, 109.90, 110.01, 110.22, 110.37]
    city_lats = [-7.35, -7.42, -7.37, -7.71, -7.49, -7.80]

    plt.style.use(plot_style)
    fig = plt.figure(figsize=(8,6))
    plt.subplots_adjust(hspace=0.05)
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 6
    # (a) Overview NE monsoon
    ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    mplot1 = _iot_basic_map(ax1)
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 6
    mplot1.box(lon_range_cki, lat_range_cki, linewidth=0.5)
    mplot1.box(lon_range_ci, lat_range_ci, linewidth=0.5)
    mplot1.box(lon_range, lat_range, linewidth=1, color='#d00d20')
    mplot1.add_subtitle('(a) Main NE monsoon (DJF) ocean currents')
    # (b) Overview SW monsoon
    ax3 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    mplot3 = _iot_basic_map(ax3)
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 6
    mplot3.box(lon_range_cki, lat_range_cki, linewidth=0.5)
    mplot3.box(lon_range_ci, lat_range_ci, linewidth=0.5)
    mplot3.box(lon_range, lat_range, linewidth=1, color='#d00d20')
    mplot3.add_subtitle('(b) Main SW monsoon (JJA) ocean currents')
    # (c) Indonesian river sources
    ax4 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    mplot4 = _iot_basic_map(ax4)
    mplot4.box(lon_range_cki, lat_range_cki, linewidth=0.5)
    mplot4.box(lon_range_ci, lat_range_ci, linewidth=0.5)
    mplot4.box(lon_range, lat_range, linewidth=1, color='k')
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 6

    global_sources = RiverSources.read_from_netcdf()
    iot_lon_range, iot_lat_range = get_iot_lon_lat_range()
    io_sources = global_sources.get_riversources_from_ocean_basin('io')
    iot_sources = io_sources.get_riversources_in_lon_lat_range(iot_lon_range, iot_lat_range)
    iot_waste = np.sum(iot_sources.waste, axis=1)
    i_use = np.where(iot_waste >= 1)
    iot_waste = iot_waste[i_use]
    lon_sources = iot_sources.lon[i_use]
    lat_sources = iot_sources.lat[i_use]
    (iot_colors,
    iot_sizes,
    iot_edge_widths) = _get_marker_colors_sizes_edgewidths_for_sources(iot_waste)
    
    mplot4.points(lon_sources, lat_sources, marker='o', facecolor=iot_colors,
                  markersize=np.array(iot_sizes)*6, edgewidth=iot_edge_widths)
    mplot4.add_subtitle('(c) Indonesian river plastic sources')
    legend_entries = _get_legend_entries_for_sources()
    ax4.set_anchor('E')
    ax4.legend(handles=legend_entries, title='[tonnes year$^{-1}$]', loc='upper right',
                bbox_to_anchor=(-0.1, 1.0))
    # (d) zoom Java
    ax2 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    mplot2 = MapPlot(ax2, lon_range, lat_range, meridians=meridians, parallels=parallels)
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 6
    for river_file in river_filenames:
        reader = shpreader.Reader(river_dir+river_file)
        rivers = reader.records()
        for river in rivers:
            ax2.add_geometries([river.geometry], ccrs.PlateCarree(), edgecolor=river_color,
                               facecolor='None', zorder=5, linewidth=linewidth)
    mplot2.points(city_lons, city_lats, marker='o', edgecolor='k', facecolor='#d00d20', markersize=5)
    mplot2.add_subtitle('(d) Main Javanese rivers contributing to IOT plastic waste')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

def figure2_main_sources(input_path=get_dir('iot_input'),
                         river_names_cki=[], river_names_ci=[],
                         ylim_cki=[0, 45], ylim_ci=[0, 45],
                         output_path=None, plot_style='plot_tools/plot.mplstyle'):
    particles = BeachingParticles.read_from_netcdf(input_path)
    box_lon_cki, box_lat_cki = get_cki_box_lon_lat_range()
    box_lon_ci, box_lat_ci = get_christmas_box_lon_lat_range()
    l_box_cki = get_l_particles_in_box(particles, 'cki')
    l_box_ci = get_l_particles_in_box(particles, 'christmas')
    lon_main_cki, lat_main_cki, waste_main_cki = get_main_sources_lon_lat_n_particles(particles, 'cki')
    lon_main_ci, lat_main_ci, waste_main_ci = get_main_sources_lon_lat_n_particles(particles, 'christmas')
    (main_colors_cki,
    main_sizes_cki,
    main_edgewidths_cki) = _get_marker_colors_sizes_edgewidths_for_main_sources(waste_main_cki)
    (main_colors_ci,
    main_sizes_ci,
    main_edgewidths_ci) = _get_marker_colors_sizes_edgewidths_for_main_sources(waste_main_ci)
    x_cki, waste_big_cki, colors_big_cki = _get_sorted_river_contribution_info(lon_main_cki,
                                                                              lat_main_cki,
                                                                              waste_main_cki,
                                                                              main_colors_cki,
                                                                              river_names_cki,
                                                                              'CKI')
    x_ci, waste_big_ci, colors_big_ci = _get_sorted_river_contribution_info(lon_main_ci,
                                                                           lat_main_ci,
                                                                           waste_main_ci,
                                                                           main_colors_ci,
                                                                           river_names_ci,
                                                                           'CI')

    plt.style.use(plot_style)
    fig = plt.figure(figsize=(6, 4))
    plt.rcParams['font.size'] = 5
    plt.rcParams['axes.labelsize'] = 5
    plt.subplots_adjust(wspace=0.2)
    plt.subplots_adjust(hspace=0.35)
    land_color = '#cfcfcf'
    in_box_color = '#2e4999'
    # (c) main sources CKI
    ax1 = plt.subplot(2, 3, (4, 5), projection=ccrs.PlateCarree())
    mplot1 = _iot_basic_map(ax1)
    plt.rcParams['font.size'] = 5
    mplot1.tracks(particles.lon[l_box_cki, :], particles.lat[l_box_cki, :], color=in_box_color, linewidth=0.2)
    mplot1.box(box_lon_cki, box_lat_cki, linewidth=0.8, color='w')
    mplot1.box(box_lon_cki, box_lat_cki, linewidth=0.5)
    ax1.add_feature(cftr.LAND,facecolor=land_color,edgecolor='k',zorder=5)
    mplot1.points(lon_main_cki, lat_main_cki, marker='o', facecolor=main_colors_cki,
                 markersize=np.array(main_sizes_cki)*5, edgewidth=main_edgewidths_cki)
    mplot1.add_subtitle(f'(c) Source locations and tracks of particles reaching\n     Cocos Keeling Islands (CKI)')
    ax1.set_anchor('E')
    # (a) main sources CI
    ax2 = plt.subplot(2, 3, (1, 2), projection=ccrs.PlateCarree())
    mplot2 = _iot_basic_map(ax2)
    plt.rcParams['font.size'] = 5
    mplot2.tracks(particles.lon[l_box_ci, :], particles.lat[l_box_ci, :], color=in_box_color, linewidth=0.2)
    mplot2.box(box_lon_ci, box_lat_ci, linewidth=0.8, color='w')
    mplot2.box(box_lon_ci, box_lat_ci, linewidth=0.5)
    ax2.add_feature(cftr.LAND,facecolor=land_color,edgecolor='k',zorder=5)
    mplot2.points(lon_main_ci, lat_main_ci, marker='o', facecolor=main_colors_ci,
                 markersize=np.array(main_sizes_ci)*5, edgewidth=main_edgewidths_ci)
    mplot2.add_subtitle(f'(a) Source locations and tracks of particles reaching\n     Christmas Island (CI)')
    # sources legend
    legend_entries = _get_legend_entries_for_main_sources()
    ax2.set_anchor('E')
    ax2.legend(handles=legend_entries, title='[# particles]', loc='upper right',
                bbox_to_anchor=(-0.1, 1.0))
    # (d) river contributions CKI
    ax3 = plt.subplot(2, 3, 6)
    ax3.bar(x_cki, waste_big_cki, color=colors_big_cki, zorder=5)
    ax3.set_ylabel('[% particles arriving]', fontsize=5)
    ax3.set_ylim(ylim_cki)
    yticks = np.arange(0, ylim_cki[1], 5)
    yticks[0] = ylim_cki[0]
    ax3.set_yticks(yticks)
    xticks = np.arange(0, len(river_names_cki))
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(river_names_cki, rotation='vertical')
    ax3.grid(False, axis='x')
    anchored_text1 = AnchoredText(f'(d) Contributions of rivers to particles\n     reaching the CKI', loc='upper left', borderpad=0.0)
    ax3.add_artist(anchored_text1)
    ax3.set_anchor('W')
    # (b) river contributions CI
    ax4 = plt.subplot(2, 3, 3)
    ax4.bar(x_ci, waste_big_ci, color=colors_big_ci, zorder=5)
    ax4.set_ylim(ylim_ci)
    yticks2 = np.arange(0, ylim_ci[1], 5)
    yticks2[0] = ylim_ci[0]
    ax4.set_yticks(yticks2)
    ax4.set_ylabel('[% particles arriving]', fontsize=5)
    xticks = np.arange(0, len(river_names_ci))
    ax4.set_xticks(xticks)
    ax4.set_xticklabels(river_names_ci, rotation='vertical')
    ax4.grid(False, axis='x')
    anchored_text2 = AnchoredText(f'(b) Contributions of rivers to particles\n     reaching CI', loc='upper left', borderpad=0.0)
    ax4.add_artist(anchored_text2)
    ax4.set_anchor('W')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

def figure3_release_arrival_histograms(input_path=get_dir('iot_input'),
                                       output_path=None, plot_style='plot_tools/plot.mplstyle',
                                       river_names = ['Serayu', 'Progo', 'Tanduy', 'Wulan', 'Bogowonto'],
                                       river_lons = [109.1125, 110.2125, 108.7958333, 108.1458333, 110.0291667],
                                       river_lats = [-7.679166667, -7.979166667, -7.670833333, -7.779166667, -7.895833333],
                                       river_styles=['-', '-.', ':', '--', '-'],
                                       river_colors=['k', 'k', 'k', 'k', '#bfbfbf'],
                                       show=True):
    particles = BeachingParticles.read_from_netcdf(input_path)
    cki_n_release, _, cki_n_entry = get_n_particles_per_month_release_arrival(particles, 'cki')
    ci_n_release, _, ci_n_entry = get_n_particles_per_month_release_arrival(particles, 'christmas')
    all_sources = RiverSources.read_from_shapefile()
    river_waste = []
    for i in range(len(river_names)):
        i_river = np.where(np.logical_and(all_sources.lon==river_lons[i], all_sources.lat==river_lats[i]))
        river_waste.append(np.squeeze(all_sources.waste[i_river]))
    # plastic samples
    (months, n_plastic_cki, n_plastic_ci, n_months_cki, n_months_ci) = _get_plastic_samples()

    colors = get_months_colors()

    plt.style.use(plot_style)
    fig = plt.figure(figsize=(5, 6))
    plt.rcParams['font.size'] = 5
    plt.rcParams['axes.labelsize'] = 5

    # (a) seasonal waste input
    ax1 = plt.subplot(3, 2, (1, 2))
    for i in range(len(river_names)):
        ax1.plot(all_sources.time, river_waste[i], label=river_names[i], color=river_colors[i], linestyle=river_styles[i])
    ax1.set_xticks(all_sources.time)
    ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax1.set_xlim(1, 12)
    ax1.set_yticks(np.arange(0, 3000, 500))
    ax1.set_ylim(0, 3000)
    ax1.set_ylabel('Released particles [#/month]')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.18, 1.0))
    anchored_text1 = AnchoredText(f'(a) Seasonal input of plastic waste from {int(len(river_names))} main polluting rivers', loc='upper left', borderpad=0.0)
    ax1.add_artist(anchored_text1)

    # (c) seasonal particles arriving CKI
    ax2 = plt.subplot(3, 2, 4)
    _histogram_release_arrival(ax2, cki_n_release, cki_n_entry)
    ax2.set_yticklabels([])
    ax2.set_ylabel('')
    anchored_text2 = AnchoredText(f'(c) Particles reaching Cocos Keeling Islands (CKI) per month', loc='upper left', borderpad=0.0)
    ax2.add_artist(anchored_text2)
    # legend
    legend_elements = [Patch(facecolor=colors[3], edgecolor='k', hatch='//////', label='Release'),
                       Patch(facecolor=colors[3], edgecolor='k', label='Arrival')]
    ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.08, 1.0))

    # (b) seasonal particles arriving CI
    ax3 = plt.subplot(3, 2, 3)
    _histogram_release_arrival(ax3, ci_n_release, ci_n_entry)
    anchored_text3 = AnchoredText(f'(b) Particles reaching Christmas Island (CI) per month', loc='upper left', borderpad=0.0)
    ax3.add_artist(anchored_text3)
    
    # (e) seasonal measured plastic CKI
    ax4 = plt.subplot(3, 2, 6)
    ax4 = _histogram_samples(ax4, months, n_plastic_cki, n_months_cki, yticks_left=False)
    anchored_text4 = AnchoredText('(e) Plastic collected on CKI beaches per month',
                                  loc='upper left', borderpad=0.0)
    ax4.add_artist(anchored_text4)
    
    # (d) seasonal distribution measured plastic CI
    ax5 = plt.subplot(3, 2, 5)
    ax5 = _histogram_samples(ax5, months, n_plastic_ci, n_months_ci, yticks_right=False, show_legend=False)
    anchored_text5 = AnchoredText(f'(d) Plastic collected on CI beaches per month', loc='upper left', borderpad=0.0)
    ax5.add_artist(anchored_text5)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

def figure4_seasonal_density(input_path=get_dir('iot_input_density'),
                             input_dir_hycom_means=get_dir('hycom_means'),
                             output_path=None,
                             plot_style='plot_tools/plot.mplstyle'):
    lon_range_cki, lat_range_cki = get_cki_box_lon_lat_range()
    lon_range_ci, lat_range_ci = get_christmas_box_lon_lat_range()
    plt.style.use(plot_style)
    fig = plt.figure(figsize=(5, 6))
    months = np.arange(3, 8, 1)
    n_rows = 3
    n_cols = 2
    for i, month in enumerate(months):
        start_date = datetime(2008, month, 1)
        end_date = add_month_to_timestamp(start_date, 1)
        density = Density.read_from_netcdf(input_path, time_start=start_date, time_end=end_date)
        z = np.mean(density.density, axis=0)
        z[z==0] = np.nan
        lon = density.grid.lon
        lat = density.grid.lat
        input_path_hycom = f'{input_dir_hycom_means}iot_{start_date.strftime("%b")}.nc'
        lon_hycom, lat_hycom, u_hycom, v_hycom = read_mean_hycom_data(input_path_hycom)

        ax = plt.subplot(n_rows, n_cols, i+1, projection=ccrs.PlateCarree())
        mplot = _iot_basic_map(ax)
        ax.tick_params(axis='both', which='both', length=0)
        if np.remainder(i+1, n_cols) != 1: # not first column
            ax.set_yticklabels([])
        if i+1 <= n_rows*n_cols-n_cols-(n_rows*n_cols-len(months)): # not last row
            ax.set_xticklabels([])
        c, ranges = mplot.pcolormesh(lon, lat, z, ranges=[1,10,100,10**3,10**4], show_cbar=False)
        q = mplot.quiver(lon_hycom, lat_hycom, u_hycom, v_hycom, thin=12, scale=8)
        mplot.box(lon_range_cki, lat_range_cki, linewidth=1.0, color='#747474')
        mplot.box(lon_range_ci, lat_range_ci, linewidth=1.0, color='#747474')
        if i+1 == len(months):
            _add_horizontal_colorbar(fig, ax, c, ranges, scale_width=2.2)
            ax.quiverkey(q, X=1.3, Y=1.0, U=1, label='Mean surface currents [1 m/s]', labelpos='E')
        mplot.add_subtitle(f'({string.ascii_lowercase[i]}) {start_date.strftime("%B")}')
    # save
    if output_path:
        plt.savefig(output_path,bbox_inches='tight',dpi=300)
    plt.show()

def get_particles_in_indonesian_islands(particles:BeachingParticles, t_start:int, cgrid:CountriesGridded,
                                        input_file='input/major_indonesian_islands.shp'):

    shp_islands = shapefile.Reader(input_file)
    indo_code = cgrid.get_country_code_from_name('Indonesia')

    islands = []
    for p in range(len(particles.pid)):
        lon = particles.lon[p, t_start:]
        lat = particles.lat[p, t_start:]
        l_nonans = np.logical_and(~np.isnan(lon), ~np.isnan(lat))
        lon = lon[l_nonans]
        lat = lat[l_nonans]

        i, j = cgrid.grid.get_index(lon, lat)
        if np.any(cgrid.countries[j.astype(int), i.astype(int)] == indo_code):
            for shape_record in shp_islands.shapeRecords():
                name = shape_record.record[1]
                polygon = Polygon(shape_record.shape.points)
                l_island = [polygon.contains(Point(lon[k], lat[k])) for k in range(len(lon))]
                if np.any(l_island):
                    islands.append(name)
                else:
                    continue
        else:
            continue

    islands = np.array(islands)
    unique_islands = np.unique(islands)
    
    n_particles_per_island = []
    for island in unique_islands:
        n_particles_per_island.append(np.sum(islands==island))
    n_particles_per_island = np.array(n_particles_per_island)

    i_sort = np.argsort(n_particles_per_island)[::-1] # sort descending
    i_sort = i_sort[n_particles_per_island[i_sort]>10] # ignore less than 10 particles

    return unique_islands[i_sort], n_particles_per_island[i_sort]

def figure5_other_countries_affected(input_path=get_dir('iot_input'),
                                     river_lons_pts = [109.08001708984375, 110.27996826171875, 108.67999267578125, 108.1199951171875, 110.1199951171875],
                                     river_lats_pts = [-7.800000190734863, -8.119999885559082, -7.800000190734863, -7.880000114440918, -8.039999961853027],
                                     unique_islands = ['Java', 'Lombok', 'Bali', 'Sumbawa', 'Sumba', 'Sumatra', 'Flores', 'West Timor', 'Sulawesi'],
                                     n_particles_per_island = np.array([43654, 535, 432, 408, 171, 170, 72, 18, 12]),
                                     output_path=None, plot_style='plot_tools/plot.mplstyle'):

    cgrid_org = CountriesGridded.read_from_netcdf()
    cgrid = cgrid_org.get_countriesgridded_with_halo(halosize=3)
    cgrid.extend_iot()

    particles = BeachingParticles.read_from_netcdf(input_path)
    lon0, lat0 = particles.get_initial_particle_lon_lat()

    l_rivers = np.zeros(len(lon0)).astype(bool)
    for i in range(len(river_lons_pts)):
        l_rivers = np.logical_or(l_rivers, np.logical_and(lon0==river_lons_pts[i], lat0==river_lats_pts[i]))

    pid = particles.pid[l_rivers]
    lon = particles.lon[l_rivers, :]
    lat = particles.lat[l_rivers, :]
    beached = particles.beached[l_rivers, :]
    river_particles = BeachingParticles(pid, particles.time, lon, lat, beached, particles.t_interval)
    
    i, j = cgrid.grid.get_index(river_particles.lon, river_particles.lat)
    t_start = 2 # doing this to allow particles to move away from source country in first 2 days
    countries = np.empty(0)
    for p in range(len(river_particles.pid)):
        i_p = i[p, t_start:]
        j_p = j[p, t_start:]
        l_nonans = np.logical_and(~np.isnan(i_p), ~np.isnan(j_p))
        p_countries = cgrid.countries[j_p[l_nonans].astype(int), i_p[l_nonans].astype(int)]
        countries = np.concatenate([countries, np.unique(p_countries[~np.isnan(p_countries)])]) # only count particle passing by country once

    unique_countries = np.unique(countries)
    
    n_particles_per_country = []
    for c in unique_countries:
        n_particles_per_country.append(np.sum(countries==c))
    n_particles_per_country = np.array(n_particles_per_country)

    i_sort = np.argsort(n_particles_per_country)[::-1] # sort descending
    i_sort = i_sort[n_particles_per_country[i_sort]>10] # ignore less than 10 particles

    # calculation below takes a very long time, so don't calculate if values are given
    if unique_islands is None:
        unique_islands, n_particles_per_island = get_particles_in_indonesian_islands(river_particles, t_start, cgrid)
        print(unique_islands)
        print(n_particles_per_island)
    
    plt.style.use(plot_style)
    plt.rcParams['font.size'] = 5
    plt.rcParams['axes.labelsize'] = 5

    ranges = [50000, 4000, 2000, 1000, 500, 0]
    colors = get_colormap_reds(len(ranges)-1)[::-1]
    bar_colors = []
    for n in n_particles_per_country[i_sort]:
        bar_colors.append(colors[np.where(n<ranges)[0][-1]])

    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(2, 3, (1, 2))
    ax2 = fig.add_subplot(2, 3, (4, 5), sharex=ax1)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 6, sharex=ax3)
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.2)

    # (a) Plastics arriving in different countries
    tick_labels0 = cgrid.get_country_name(unique_countries[i_sort])
    tick_labels1 = [t.replace('France', 'Reunion') for t in tick_labels0]
    tick_labels2 = [t.replace('British Indian Ocean Territory', 'Chagos Archipelago') for t in tick_labels1]
    tick_labels = [t.replace('Indian Ocean Territories', 'Christmas & Cocos Keeling Islands') for t in tick_labels2]

    ax1.bar(np.arange(0, len(unique_countries[i_sort])), n_particles_per_country[i_sort], color=bar_colors,
           tick_label=tick_labels, zorder=3)
    ax2.bar(np.arange(0, len(unique_countries[i_sort])), n_particles_per_country[i_sort], color=bar_colors,
           tick_label=tick_labels, zorder=3)
    ax1.set_ylim(29000, 45500)  # outliers only
    ax2.set_ylim(0, 3300) # most of the data

    ax1.spines['bottom'].set_visible(False) # hide spines between axes
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False) # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # slanted lines to indicate broken y-axis
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    ax2.set_ylabel('Particles passing', loc='top')
    ax1.set_ylabel('close to countries [#]', loc='bottom')
    ax1.grid(False)
    ax2.grid(False)
    ax1.grid(True, axis='y', zorder=0)
    ax2.grid(True, axis='y', zorder=0)
    
    anchored_text1 = AnchoredText(f'(a) Countries affected by plastics from 5 main rivers', loc='upper left', borderpad=0.0)
    ax1.add_artist(anchored_text1)

    # (b) Plastics arriving at Indonesian islands
    ranges2 = [50000, 500, 250, 100, 50, 10, 0]
    colors2 = get_colormap_reds(len(ranges)-1)[::-1]
    bar_colors2 = []
    for n in n_particles_per_island:
        bar_colors2.append(colors2[np.where(n<ranges2)[0][-1]])

    ax3.bar(np.arange(0, len(unique_islands)), n_particles_per_island, color=bar_colors2, tick_label=unique_islands, zorder=3)
    ax4.bar(np.arange(0, len(unique_islands)), n_particles_per_island, color=bar_colors2, tick_label=unique_islands, zorder=3)
    ax3.set_ylim(29000, 45500)
    ax4.set_ylim(0, 560)

    ax3.spines['bottom'].set_visible(False) # hide spines between axes
    ax4.spines['top'].set_visible(False)
    ax3.xaxis.tick_top()
    ax3.tick_params(labeltop=False) # don't put tick labels at the top
    ax4.xaxis.tick_bottom()

    # slanted lines to indicate broken y-axis
    ax3.plot([0, 1], [0, 0], transform=ax3.transAxes, **kwargs)
    ax4.plot([0, 1], [1, 1], transform=ax4.transAxes, **kwargs)

    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=90)
    # ax4.set_ylabel('Particles passing', loc='top')
    # ax3.set_ylabel('close to islands [#]', loc='bottom')
    ax3.grid(False)
    ax4.grid(False)
    ax3.grid(True, axis='y', zorder=0)
    ax4.grid(True, axis='y', zorder=0)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position('right')
    
    anchored_text2 = AnchoredText(f'(b) Indonesian islands affected', loc='upper left', borderpad=0.0)
    ax3.add_artist(anchored_text2)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

def plastic_measurements(plastic_type='count', output_path=None, plot_style='plot_tools/plot.mplstyle'):
    time_cki, lon_cki, lat_cki, n_plastic_cki, kg_plastic_cki = get_cki_plastic_measurements()
    time_ci, lon_ci, lat_ci, n_plastic_ci, kg_plastic_ci = get_christmas_plastic_measurements()
    n_plastic_month_cki = np.zeros(12)
    n_months_cki = np.zeros(12)
    n_plastic_month_ci = np.zeros(12)
    n_months_ci = np.zeros(12)
    if plastic_type == 'count':
        samples_cki = n_plastic_cki
        samples_ci = n_plastic_ci
        ylabel = 'Plastic items [#]'
        ylim = [0, 140000]
        yticks = np.arange(0, 140000, 20000)
        ylim_studies = [0, 14]
        yticks_studies = np.arange(0, 14, 2)
    elif plastic_type == 'mass':
        samples_cki = kg_plastic_cki
        samples_ci = kg_plastic_ci
        ylabel = 'Plastic weight [kg]'
        ylim = [0, 4000]
        yticks = np.arange(0, 4000, 500)
        ylim_studies = [0, 16]
        yticks_studies = np.arange(0, 16, 2)
    else:
        raise ValueError(f'Unknown plastic type requested: {plastic_type}. Valid values are: count and mass.')
    for i, t in enumerate(time_cki):
        n_months_cki[t.month-1] += 1
        if plastic_type == 'count':
            n_plastic_month_cki[t.month-1] = np.nansum([n_plastic_month_cki[t.month-1], n_plastic_cki[i]])
        elif plastic_type == 'mass':
            n_plastic_month_cki[t.month-1] = np.nansum([n_plastic_month_cki[t.month-1], kg_plastic_cki[i]])
    for i, t in enumerate(time_ci):
        n_months_ci[t.month-1] += 1
        if plastic_type == 'count':
            n_plastic_month_ci[t.month-1] = np.nansum([n_plastic_month_ci[t.month-1], n_plastic_ci[i]])
        elif plastic_type == 'mass':
            n_plastic_month_ci[t.month-1] = np.nansum([n_plastic_month_ci[t.month-1], kg_plastic_ci[i]])
    (main_colors_cki,
    main_sizes_cki,
    main_edgewidths_cki) = _get_marker_colors_sizes_for_samples(samples_cki, plastic_type)
    (main_colors_ci,
    main_sizes_ci,
    main_edgewidths_ci) = _get_marker_colors_sizes_for_samples(samples_ci, plastic_type)
    
    plt.style.use(plot_style)
    fig = plt.figure(figsize=(8, 6))

    months = np.arange(1,13,1)
    colors = get_months_colors()
    color = '#adadad'
    land_color = '#cfcfcf'
    # (a) seasonal distribution measured plastic CKI
    ax1 = plt.subplot(2, 2, 1)
    ax1.bar(months-0.2, n_plastic_month_cki, width=0.4, color=colors, edgecolor='k', zorder=5)
    ax1.set_xticks(months)
    ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax1.set_ylim(ylim)
    ax1.set_yticks(yticks)
    ax1.set_ylabel(ylabel)
    anchored_text1 = AnchoredText(f'(a) Plastic samples from Cocos Keeling Islands (CKI)', loc='upper left', borderpad=0.0)
    ax1.add_artist(anchored_text1)
    # number of sampling studies per month
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.bar(months+0.2, n_months_cki, width=0.4, color=color, edgecolor='k', zorder=6)
    ax2.set_ylim(ylim_studies)
    ax2.set_yticks(yticks_studies)
    ax2.set_ylabel('Sampling studies [#]')
    ax2.spines['right'].set_color(color)
    ax2.tick_params(axis='y', colors=color)
    ax2.yaxis.label.set_color(color)
    # (b) seasonal distribution measured plastic CI
    ax3 = plt.subplot(2, 2, 3)
    ax3.bar(months-0.2, n_plastic_month_ci, width=0.4, color=colors, edgecolor='k', zorder=5)
    ax3.set_xticks(months)
    ax3.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax3.set_ylim(ylim)
    ax3.set_yticks(yticks)
    ax3.set_ylabel(ylabel)
    anchored_text2 = AnchoredText(f'(b) Plastic samples from Christmas Island (CI)', loc='upper left', borderpad=0.0)
    ax3.add_artist(anchored_text2)
    # number of sampling studies per month
    ax4 = ax3.twinx()
    ax4.grid(False)
    ax4.bar(months+0.2, n_months_ci, width=0.4, color=color, edgecolor='k', zorder=6)
    ax4.set_ylim(ylim_studies)
    ax4.set_yticks(yticks_studies)
    ax4.set_ylabel('Sampling studies [#]')
    ax4.spines['right'].set_color(color)
    ax4.tick_params(axis='y', colors=color)
    ax4.yaxis.label.set_color(color)
    # (c) map of measured plastic CKI
    ax5 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    lon_range_cki, lat_range_cki = get_cki_box_lon_lat_range()
    mplot1 = _iot_basic_map(ax5, lon_range=lon_range_cki, lat_range=lat_range_cki, dlon=0.2, dlat=0.1)
    l_nonan_cki = ~np.isnan(samples_cki)
    mplot1.points(lon_cki[l_nonan_cki], lat_cki[l_nonan_cki], facecolor=main_colors_cki, edgecolor='k', marker='o',
                  markersize=np.array(main_sizes_cki)*5, edgewidth=main_edgewidths_cki)
    anchored_text3 = AnchoredText(f'(c) Sampled plastics on CKI', loc='upper left', borderpad=0.0)
    anchored_text3.zorder = 8
    ax5.add_artist(anchored_text3)
    # (d) map of measured plastic CI
    ax6 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree())
    lon_range_ci, lat_range_ci = get_christmas_box_lon_lat_range()
    mplot2 = _iot_basic_map(ax6, lon_range=lon_range_ci, lat_range=lat_range_ci, dlon=0.2, dlat=0.1)
    l_nonan_ci = ~np.isnan(samples_ci)
    mplot2.points(lon_ci[l_nonan_ci], lat_ci[l_nonan_ci], facecolor=main_colors_ci, edgecolor='k', marker='o',
                  markersize=np.array(main_sizes_ci)*5, edgewidth=main_edgewidths_ci)
    anchored_text4 = AnchoredText(f'(d) Sampled plastics on CI', loc='upper left', borderpad=0.0)
    ax6.add_artist(anchored_text4)
    # legend
    legend_entries = _get_legend_entries_samples(plastic_type)
    ax5.legend(handles=legend_entries, loc='upper left', title=ylabel,
                bbox_to_anchor=(1.1, 1.0))
    # save
    if output_path:
        plt.savefig(output_path,bbox_inches='tight',dpi=300)
    plt.show()

def hycom_velocities(input_path, thin=6, scale=10,
                     output_path=None,
                     plot_style='plot_tools/plot.mplstyle'):
    lon, lat, u, v = read_mean_hycom_data(input_path)
    # boxes IOT
    lon_range_cki, lat_range_cki = get_cki_box_lon_lat_range()
    lon_range_ci, lat_range_ci = get_christmas_box_lon_lat_range()

    plt.style.use(plot_style)
    fig = plt.figure()
    ax = plt.gca(projection=ccrs.PlateCarree())
    mplot = _iot_basic_map(ax)
    mplot.quiver(lon, lat, u, v, thin=thin, scale=scale)
    mplot.box(lon_range_cki, lat_range_cki, linewidth=0.5, color='r')
    mplot.box(lon_range_ci, lat_range_ci, linewidth=0.5, color='r')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

def get_kepler_geojson(output_path, input_path=get_dir('iot_input_2008_only')):
    particles = BeachingParticles.read_from_netcdf(input_path)
    l_box_cki = get_l_particles_in_box(particles, 'cki')
    l_box_ci = get_l_particles_in_box(particles, 'christmas')
    l_box_iot = np.logical_or(l_box_cki, l_box_ci)
    lon = particles.lon[l_box_iot, :]
    lat = particles.lat[l_box_iot, :]
    pid = particles.pid[l_box_iot]
    
    features = []
    for p in range(len(pid)):
        coordinates = []
        for t in range(len(particles.time)):
            unixtime = int(time.mktime(particles.time[t].timetuple()))
            if np.isnan(lon[p, t]) or np.isnan(lat[p, t]):
                continue
            coordinates.append((lon[p, t], lat[p, t], 0, unixtime))
        linestring = LineString(coordinates)
        features.append(Feature(geometry=linestring, properties={"pid":str(p)}))
    feature_collection = FeatureCollection(features)
    
    with open(output_path, 'w') as f:
        dump(feature_collection, f)

if __name__ == '__main__':
    # figure1_overview(output_path=get_dir('iot_plots')+'fig1.jpg')
    
    # river_names_cki = ['Serayu', 'Bogowonto', 'Tanduy', 'Progo']
    # river_names_ci = ['Tanduy', 'Serayu', 'Wulan', 'Bogowonto']
    # figure2_main_sources(river_names_cki=river_names_cki, river_names_ci=river_names_ci, output_path=get_dir('iot_plots')+'fig2.jpg')
    
    # figure3_release_arrival_histograms(output_path=get_dir('iot_plots')+'fig3.jpg')
    
    # figure4_seasonal_density(output_path=get_dir('iot_plots')+'fig4.jpg')

    figure5_other_countries_affected(output_path=get_dir('iot_plots')+'fig5.jpg')
    
    # output_path = f'{get_dir("animation_output")}iot.geojson'
    # get_kepler_geojson(output_path)
    
    # --- plots with 3% wind added ---
    # river_names_cki_wind = ['Tanduy', 'Bogowonto']
    # river_names_ci_wind = ['Bogowonto', 'Progo', 'Tanduy']
    # figure2_main_sources(river_names_cki=river_names_cki_wind, river_names_ci=river_names_ci_wind,
    #                      input_path=get_dir('iot_input_wind'), output_path=get_dir('iot_plots')+'fig2_wind.jpg')
    # river_names_wind = ['Progo', 'Tanduy', 'Bogowonto']
    # river_lons_wind = [110.2125, 108.7958333, 110.0291667]
    # river_lats_wind = [-7.979166667, -7.670833333, -7.895833333]
    # figure3_release_arrival_histograms(river_names=river_names_wind, river_lons=river_lons_wind, river_lats=river_lats_wind,
    #                                    input_path=get_dir('iot_input_wind'), output_path=get_dir('iot_plots')+'fig3_wind.jpg')
    # figure4_seasonal_density(input_path=get_dir('iot_input_density_wind'), output_path=get_dir('iot_plots')+'fig4_wind.jpg')
