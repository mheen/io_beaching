from plot_tools.map_plotter import MapPlot, get_colormap_reds
from plots_vanderMheen_et_al_2020 import _get_marker_colors_sizes_edgewidths_for_sources, _get_legend_entries_for_sources
from particles import BeachingParticles
from utilities import get_dir
from postprocessing_iot import get_l_particles_in_box, get_cki_box_lon_lat_range, get_christmas_box_lon_lat_range
from postprocessing_iot import get_iot_lon_lat_range, get_iot_sources, get_main_sources_lon_lat_n_particles
from postprocessing_iot import get_original_source_based_on_lon0_lat0, get_n_particles_per_month_release_arrival
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cftr
import numpy as np
from datetime import datetime

def get_months_colors():
    colors = ['#ffedbc', '#fece6b', '#fdc374', '#fb9d59', '#f57547', '#d00d20',
    '#c9e7f1', '#90c3dd', '#4576b4', '#000086', '#4d00aa', '#30006a']
    return colors

def _logarithmic_colormap():
    colors = get_colormap_reds(6)
    ranges = [1,10,100,10**3,10**4,10**5,10**6]
    cm = LinearSegmentedColormap.from_list('cm_log_density',colors,N=6)
    norm = BoundaryNorm(ranges,ncolors=6)
    return colors,ranges,cm,norm

def _iot_basic_map(ax, xmarkers='bottom', ymarkers='left', lon_range=None, lat_range=None):
    if lon_range is None:
        lon_range, _ = get_iot_lon_lat_range()
        meridians = [90, 100, 110, 120, 128]        
    else:
        dlon = 4
        meridians = np.arange(lon_range[0], lon_range[1]+dlon, dlon)
    if lat_range is None:
        _, lat_range = get_iot_lon_lat_range()
        parallels = [-20, -15, -10, -5, 0, 5]
    else:
        dlat = 2
        parallels = np.arange(lat_range[0], lat_range[1]+dlat, dlat)    
    mplot = MapPlot(ax, lon_range, lat_range, meridians=meridians, parallels=parallels,
                    xmarkers=xmarkers, ymarkers=ymarkers)
    return mplot

def _main_source_info():
    ranges = np.array([300., 100., 50., 25., 10., 5., 1.])
    labels = ['> 100','50 - 100','25 - 50','10 - 25', '5 - 10','1 - 5']
    colors = get_colormap_reds(len(ranges)-1)[::-1]
    edge_widths = [0.7,0.7,0.7,0.5,0.5,0.5]
    sizes = [6,5,4,3,2,1]
    legend_sizes = [6,5,4,3,2,1]
    return (ranges,labels,colors,edge_widths,sizes,legend_sizes)

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

def _get_legend_entries_for_main_sources():
    (_,labels,colors,edge_widths,_,legend_sizes) = _main_source_info()
    legend_entries = []
    for i in range(len(colors)):
        legend_entries.append(Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[i],                              
                              markersize=legend_sizes[i],label=labels[i],markeredgewidth=edge_widths[i]))
    return legend_entries

def particle_tracks_and_main_sources(iot_island='cki', river_names = [], ylim_rivers = [5, 210],
                                     output_path=None, plot_style='plot_tools/plot.mplstyle'):
    particles = BeachingParticles.read_from_netcdf(get_dir('iot_input'))
    if iot_island == 'cki':
        box_lon, box_lat = get_cki_box_lon_lat_range()
        iot_short_name = 'CKI'
        iot_long_name = 'Cocos Keeling Islands'
    elif iot_island == 'christmas':
        box_lon, box_lat = get_christmas_box_lon_lat_range()
        iot_short_name = 'CI'
        iot_long_name = 'Christmas Island'
    else:
        raise ValueError(f'Unknown iot_island {iot_island}, valid options are: cki and christmas.')
    l_box = get_l_particles_in_box(particles, iot_island)
    plt.style.use(plot_style)
    fig = plt.figure(figsize=(6,4))
    plt.subplots_adjust(hspace=0.1)
    land_color = '#cfcfcf'
    # --- (a) tracks and all sources ---
    ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    # tracks
    not_in_box_color = '#626262'
    in_box_color = '#A21E1E'
    mplot1 = _iot_basic_map(ax1, xmarkers='off')
    mplot1.ax.set_xticklabels([])
    mplot1.tracks(particles.lon[~l_box, :], particles.lat[~l_box, :], color=not_in_box_color, linewidth=0.2)
    mplot1.tracks(particles.lon[l_box, :], particles.lat[l_box, :], color=in_box_color, linewidth=0.2)
    mplot1.box(box_lon, box_lat, linewidth=0.5)
    ax1.add_feature(cftr.LAND,facecolor=land_color,edgecolor='k',zorder=5)
    # sources
    iot_sources = get_iot_sources()
    iot_yearly_waste = np.sum(iot_sources.waste,axis=1)
    i_use = np.where(iot_yearly_waste >= 1)
    iot_yearly_waste = iot_yearly_waste[i_use]
    lon_sources = iot_sources.lon[i_use]
    lat_sources = iot_sources.lat[i_use]
    (source_colors,
    source_sizes,
    source_edgewidths) = _get_marker_colors_sizes_edgewidths_for_sources(iot_yearly_waste)
    mplot1.points(lon_sources,lat_sources,marker='o',facecolor=source_colors,
                  markersize=np.array(source_sizes),edgewidth=source_edgewidths)    
    # legends
    legend_entries = _get_legend_entries_for_sources()
    ax1.set_anchor('W')
    legend1 = plt.legend(handles=legend_entries, title='Plastic sources\n[tonnes/year]', loc='upper right',
                         bbox_to_anchor=(1.0, 1.0))
    ax1.add_artist(legend1)
    legend_entries_tracks = [Line2D([0], [0], color=in_box_color, label=f'Reach {iot_short_name}'),
                             Line2D([0], [0], color=not_in_box_color, label=f'Do not reach {iot_short_name}')]
    ax1.legend(handles=legend_entries_tracks, loc='lower right', bbox_to_anchor=(1.0, 0.0))
    # title
    mplot1.add_subtitle(f'(a) Particle tracks around {iot_long_name} ({iot_short_name})')
    # --- (b) main sources ---
    ax2 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    mplot2 = _iot_basic_map(ax2)
    lon_main_sources, lat_main_sources, waste_main_sources = get_main_sources_lon_lat_n_particles(particles, iot_island)
    (main_source_colors,
    main_source_sizes,
    main_source_edgewidths) = _get_marker_colors_sizes_edgewidths_for_main_sources(waste_main_sources)
    mplot2.points(lon_main_sources, lat_main_sources, marker='o', facecolor=main_source_colors,
                 markersize=np.array(main_source_sizes)*5, edgewidth=main_source_edgewidths)
    mplot2.box(box_lon, box_lat, linewidth=0.5)
    ax2.add_feature(cftr.LAND,facecolor=land_color,edgecolor='k',zorder=5)
    # legend
    legend_entries_main = _get_legend_entries_for_main_sources()
    ax2.set_anchor('W')
    ax2.legend(handles=legend_entries_main, title='Main sources\n[# particles]', loc='upper right',
                bbox_to_anchor=(1.0, 1.0))
    # title
    mplot2.add_subtitle(f'(b) Source locations of particles reaching {iot_short_name}')
    # --- (c) histogram rivers ---
    ax3 = plt.subplot(2, 2, (2,4))
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
    if river_names == []:
    # print lon and lat of rivers contributing >50 particles (to find names and add to labels)
        print('Locations of largest sources (> 50 particles):')
        for i in range(len(lon_biggest_sources)):
            print(f'{lon_biggest_sources[i]}, {lat_biggest_sources[i]}, {waste_biggest_sources[i]}')
    # plot
    ax3.bar(np.arange(0, len(waste_big_sources)), waste_big_sources, color=colors_big_sources, zorder=5)
    ax3.set_ylabel('Particles [#]')
    ax3.set_ylim(ylim_rivers)
    yticks = np.arange(0, ylim_rivers[1], 20)
    yticks[0] = ylim_rivers[0]
    ax3.set_yticks(yticks)
    # river names
    xticks = np.arange(0, len(i_biggest_sources))
    ax3.set_xticks(xticks)
    xlabels = river_names
    for i in range(len(xticks)-len(river_names)):
        xlabels.append('')
    ax3.set_xticklabels(xlabels, rotation='vertical')
    # title
    anchored_text = AnchoredText(f'(c) Contributions of rivers to particles reaching {iot_short_name}', loc='upper left', borderpad=0.0)
    ax3.add_artist(anchored_text)
    # make ax3 a bit shorter
    l3, b3, w3, h3 = ax3.get_position().bounds
    ax3.set_position([l3, b3+(0.25*h3)/2, w3, 0.75*h3])
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

def release_arrival_histogram(iot_island='cki', ylim=[0, 350], output_path=None, plot_style='plot_tools/plot.mplstyle'):
    particles = BeachingParticles.read_from_netcdf(get_dir('iot_input'))
    months = np.arange(1,13,1)
    n_release, _, n_entry = get_n_particles_per_month_release_arrival(particles, iot_island)
    colors = get_months_colors()
    plt.style.use(plot_style)
    fig = plt.figure(figsize=(3,4))
    ax = plt.gca()
    ax.bar(months-0.2, n_release, width=0.4, label='Release', color=colors,
           hatch='////', edgecolor='k', zorder=5)
    n_entry_cumulative = n_entry.cumsum(axis=1)
    ax.bar(months+0.2, n_entry[:, 0], width=0.4, color=colors[0],
           edgecolor='k', zorder=5)
    for i in range(1, 11):
        heights = n_entry_cumulative[:, i]        
        starts = n_entry_cumulative[:, i-1]
        ax.bar(months+0.2, n_entry[:, i], bottom=n_entry_cumulative[:, i-1],
                width=0.4, color=colors[i], edgecolor='k', zorder=5)
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_ylabel('Particles [#]')
    ax.set_ylim(ylim)
    # legend
    legend_elements = [Patch(facecolor='w', edgecolor='k', hatch='//////', label='Release'),
                       Patch(facecolor='w', edgecolor='k', label='CKI arrival')]
    ax.legend(handles=legend_elements, loc='upper right')
    # title
    anchored_text = AnchoredText('Seasonality of particles reaching Cocos Keeling Islands', loc='upper left', borderpad=0.0)
    ax.add_artist(anchored_text)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    river_names_cki = ['Serayu', 'Progo', 'Tanduy', 'Opak']
    particle_tracks_and_main_sources(iot_island='cki', river_names=river_names_cki, output_path=get_dir('iot_plots')+'cki_main_sources.jpg')
    release_arrival_histogram(iot_island='cki', output_path=get_dir('iot_plots')+'cki_release_arrival_times.jpg')
    river_names_ci = ['Tanduy', 'Wulan']
    particle_tracks_and_main_sources(iot_island='christmas', river_names=river_names_ci, ylim_rivers=[5, 300],
                                     output_path=get_dir('iot_plots')+'christmas_main_sources.jpg')
    release_arrival_histogram(iot_island='christmas', ylim=[0, 420],
                              output_path=get_dir('iot_plots')+'christmas_release_arrival_times.jpg')
