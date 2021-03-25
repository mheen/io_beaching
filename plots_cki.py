from plot_tools.map_plotter import MapPlot, get_colormap_reds
from plots_vanderMheen_et_al_2020 import _get_marker_colors_sizes_edgewidths_for_sources, _get_legend_entries_for_sources
from plastic_sources import RiverSources
from particles import BeachingParticles
from utilities import get_dir
from postprocessing_cki import get_l_particles_in_box, get_cki_box_lon_lat_range, get_n_particles_per_month_release_arrival
from postprocessing_cki import get_particle_release_locations_box
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cftr
import numpy as np
from datetime import datetime

def get_cki_lon_lat_range():
    lon_range = [90., 128.]
    lat_range = [-20., 6.]
    return (lon_range, lat_range)

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

def _cki_basic_map(ax, xmarkers='bottom', ymarkers='left', lon_range=None, lat_range=None):
    if lon_range is None:
        lon_range, _ = get_cki_lon_lat_range()
        meridians = [90, 100, 110, 120, 128]        
    else:
        dlon = 4
        meridians = np.arange(lon_range[0], lon_range[1]+dlon, dlon)
    if lat_range is None:
        _, lat_range = get_cki_lon_lat_range()
        parallels = [-20, -15, -10, -5, 0, 5]
    else:
        dlat = 2
        parallels = np.arange(lat_range[0], lat_range[1]+dlat, dlat)    
    mplot = MapPlot(ax, lon_range, lat_range, meridians=meridians, parallels=parallels,
                    xmarkers=xmarkers, ymarkers=ymarkers)
    return mplot

def particle_tracks(output_path=None, plot_style='plot_tools/plot.mplstyle'):
    particles = BeachingParticles.read_from_netcdf(get_dir('cki_input'))
    box_lon, box_lat = get_cki_box_lon_lat_range()
    cki_lon, cki_lat = get_cki_lon_lat_range()
    l_box = get_l_particles_in_box(particles)
    plt.style.use(plot_style)
    fig = plt.figure(figsize=(4,5))
    ax = plt.gca(projection=ccrs.PlateCarree())
    # tracks
    mplot = _cki_basic_map(ax)
    mplot.tracks(particles.lon[~l_box, :], particles.lat[~l_box, :], color='#626262', linewidth=0.2)
    mplot.tracks(particles.lon[l_box, :], particles.lat[l_box, :], color='#A21E1E', linewidth=0.2)
    mplot.box(box_lon, box_lat, linewidth=0.5)
    ax.add_feature(cftr.LAND,facecolor='#cfcfcf',edgecolor='k',zorder=5)
    # sources
    global_sources = RiverSources.read_from_netcdf()
    io_sources = global_sources.get_riversources_from_ocean_basin('io')
    cki_sources = io_sources.get_riversources_in_lon_lat_range(cki_lon, cki_lat)    
    cki_yearly_waste = np.sum(cki_sources.waste,axis=1)
    i_use = np.where(cki_yearly_waste >= 1)
    cki_yearly_waste = cki_yearly_waste[i_use]
    lon_sources = cki_sources.lon[i_use]
    lat_sources = cki_sources.lat[i_use]
    (source_colors,
    source_sizes,
    source_edgewidths) = _get_marker_colors_sizes_edgewidths_for_sources(cki_yearly_waste)    
    mplot.points(lon_sources,lat_sources,marker='o',facecolor=source_colors,
                  markersize=source_sizes,edgewidth=source_edgewidths)    
    # sources legend
    legend_entries = _get_legend_entries_for_sources()
    ax.set_anchor('W')
    ax.legend(handles=legend_entries, title='Plastic sources\n[tonnes year$^{-1}$]', loc='upper right',
              bbox_to_anchor=(1.0, 1.0))
    # title
    mplot.add_subtitle('Particle tracks around Cocos Keeling Islands (2008)')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

def particle_timeseries(ylim=[0, 350], output_path=None, plot_style='plot_tools/plot.mplstyle'):
    particles = BeachingParticles.read_from_netcdf(get_dir('cki_input'))
    months = np.arange(1,13,1)
    n_release, _, n_entry = get_n_particles_per_month_release_arrival(particles)
    colors = get_months_colors()
    plt.style.use(plot_style)
    fig = plt.figure(figsize=(4,5))
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
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == '__main__':
    particle_timeseries(output_path=get_dir('cki_plots')+'release_arrival_times.jpg')
