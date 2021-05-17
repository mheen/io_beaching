from plot_tools.map_plotter import MapPlot, get_colormap_reds
from plots_vanderMheen_et_al_2020 import _get_marker_colors_sizes_edgewidths_for_sources, _get_legend_entries_for_sources
from particles import BeachingParticles
from utilities import get_dir
from ocean_utilities import read_mean_hycom_data
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
import cartopy.io.shapereader as shpreader
import numpy as np
from datetime import datetime
from netCDF4 import Dataset

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
    ranges = np.array([400., 100., 50., 25., 10., 5., 1.])
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
    mplot1.add_subtitle('(a) Region overview and main NE monsoon ocean currents')
    # (b) Overview SW monsoon
    ax3 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree())
    mplot3 = _iot_basic_map(ax3)
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 6
    mplot3.box(lon_range_cki, lat_range_cki, linewidth=0.5)
    mplot3.box(lon_range_ci, lat_range_ci, linewidth=0.5)
    mplot3.box(lon_range, lat_range, linewidth=1, color='#d00d20')
    mplot3.add_subtitle('(b) Region overview and main SW monsoon ocean currents')
    # (c) zoom Java
    ax2 = plt.subplot(2, 2, (2, 4), projection=ccrs.PlateCarree())
    mplot2 = MapPlot(ax2, lon_range, lat_range, meridians=meridians, parallels=parallels,
                     ymarkers='right')
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 6
    for river_file in river_filenames:
        reader = shpreader.Reader(river_dir+river_file)
        rivers = reader.records()
        for river in rivers:
            ax2.add_geometries([river.geometry], ccrs.PlateCarree(), edgecolor=river_color,
                               facecolor='None', zorder=5, linewidth=linewidth)
    mplot2.points(city_lons, city_lats, marker='o', edgecolor='k', facecolor='#d00d20', markersize=5)
    mplot2.add_subtitle('(c) Main Javanese rivers contributing to IOT plastic waste')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

def figure2_main_sources(river_names_cki=[], river_names_ci=[],
                         ylim_cki=[0, 45], ylim_ci=[0, 45],
                         output_path=None, plot_style='plot_tools/plot.mplstyle'):
    particles = BeachingParticles.read_from_netcdf(get_dir('iot_input'))
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
    plt.subplots_adjust(wspace=0.0)
    plt.subplots_adjust(hspace=0.35)
    land_color = '#cfcfcf'
    in_box_color = '#2e4999'
    # (a) main sources CKI
    ax1 = plt.subplot(2, 3, (1, 2), projection=ccrs.PlateCarree())
    mplot1 = _iot_basic_map(ax1)
    plt.rcParams['font.size'] = 5
    mplot1.ax.set_xticklabels([])
    mplot1.tracks(particles.lon[l_box_cki, :], particles.lat[l_box_cki, :], color=in_box_color, linewidth=0.2)
    mplot1.box(box_lon_cki, box_lat_cki, linewidth=0.8, color='w')
    mplot1.box(box_lon_cki, box_lat_cki, linewidth=0.5)
    ax1.add_feature(cftr.LAND,facecolor=land_color,edgecolor='k',zorder=5)
    mplot1.points(lon_main_cki, lat_main_cki, marker='o', facecolor=main_colors_cki,
                 markersize=np.array(main_sizes_cki)*5, edgewidth=main_edgewidths_cki)
    mplot1.add_subtitle(f'(a) Source locations and tracks of particles reaching\n   Cocos Keeling Islands (CKI)')
    ax1.set_anchor('W')
    # (c) main sources CI
    ax2 = plt.subplot(2, 3, (4, 5), projection=ccrs.PlateCarree())
    mplot2 = _iot_basic_map(ax2)
    plt.rcParams['font.size'] = 5
    mplot2.tracks(particles.lon[l_box_ci, :], particles.lat[l_box_ci, :], color=in_box_color, linewidth=0.2)
    mplot2.box(box_lon_ci, box_lat_ci, linewidth=0.8, color='w')
    mplot2.box(box_lon_ci, box_lat_ci, linewidth=0.5)
    ax2.add_feature(cftr.LAND,facecolor=land_color,edgecolor='k',zorder=5)
    mplot2.points(lon_main_ci, lat_main_ci, marker='o', facecolor=main_colors_ci,
                 markersize=np.array(main_sizes_ci)*5, edgewidth=main_edgewidths_ci)
    mplot2.add_subtitle(f'(c) Source locations and tracks of particles reaching\n   Christmas Island (CI)')
    # sources legend
    legend_entries = _get_legend_entries_for_main_sources()
    ax2.set_anchor('W')
    ax2.legend(handles=legend_entries, title='[# particles]', loc='upper right',
                bbox_to_anchor=(1.3, 1.0))
    # (b) river contributions CKI
    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(x_cki, waste_big_cki, color=colors_big_cki, zorder=5)
    ax3.set_ylabel('[% particles arriving]')
    ax3.set_ylim(ylim_cki)
    yticks = np.arange(0, ylim_cki[1], 5)
    yticks[0] = ylim_cki[0]
    ax3.set_yticks(yticks)
    xticks = np.arange(0, len(river_names_cki))
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(river_names_cki, rotation='vertical')
    ax3.grid(False, axis='x')
    anchored_text1 = AnchoredText(f'(b) Contributions of rivers to particles\n   reaching the CKI', loc='upper left', borderpad=0.0)
    ax3.add_artist(anchored_text1)
    ax3.set_anchor('W')
    # (d) river contributions CI
    ax4 = plt.subplot(2, 3, 6)
    ax4.bar(x_ci, waste_big_ci, color=colors_big_ci, zorder=5)
    ax4.set_ylim(ylim_ci)
    yticks2 = np.arange(0, ylim_ci[1], 5)
    yticks2[0] = ylim_ci[0]
    ax4.set_yticks(yticks2)
    ax4.set_ylabel('[% particles arriving]')
    xticks = np.arange(0, len(river_names_ci))
    ax4.set_xticks(xticks)
    ax4.set_xticklabels(river_names_ci, rotation='vertical')
    ax4.grid(False, axis='x')
    anchored_text2 = AnchoredText(f'(d) Contributions of rivers to particles\n   reaching CI', loc='upper left', borderpad=0.0)
    ax4.add_artist(anchored_text2)
    ax4.set_anchor('W')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

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
    if iot_island == 'cki':
        iot_short_name = 'CKI'
        iot_long_name = 'Cocos Keeling Islands'
    elif iot_island == 'christmas':
        iot_short_name = 'CI'
        iot_long_name = 'Christmas Island'
    else:
        raise ValueError(f'Unknown iot_island {iot_island}, valid options are: cki and christmas.')
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
                       Patch(facecolor='w', edgecolor='k', label=f'{iot_short_name} arrival')]
    ax.legend(handles=legend_elements, loc='upper right')
    # title
    anchored_text = AnchoredText(f'Seasonality of particles reaching {iot_long_name} ({iot_short_name})', loc='upper left', borderpad=0.0)
    ax.add_artist(anchored_text)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
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

if __name__ == '__main__':
    # figure1_overview(output_path=get_dir('iot_plots')+'fig1.jpg')
    river_names_cki = ['Serayu', 'Brogowonto', 'Tanduy', 'Progo']
    river_names_ci = ['Tanduy', 'Serayu', 'Wulan', 'Bogowonto']
    figure2_main_sources(river_names_cki=river_names_cki, river_names_ci=river_names_ci, output_path=get_dir('iot_plots')+'fig2.jpg')
