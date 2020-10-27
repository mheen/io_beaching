# io_beaching
Beaching of plastics in the Indian Ocean

## Getting started
1. Create a conda environment using the io_beaching.yml file
2. Create a file "input/dirs.json" containing paths to relevant directories; see the "input/example_dirs.json" file. If you rename this file, make sure to change its default in the `get_dir()` function in the utilities.py script. All references to the file should then work correctly.
3. All input to apply beaching conditions and to postprocess results is contained in the "input/" folder, with the exception of a netcdf file containing the distance to the nearest coastline. This file is available from https://www.soest.hawaii.edu/pwessel/gshhg/; if it is not already available in the "input/" directory, `CoastDistance.read_from_netcdf()` will download this for you. Hycom (or other) data to force particle tracking simulations with needs to be downloaded separately.

## Particle tracking simulations with OceanParcels
Indian Ocean particle tracking simulations with plastic waste input from river sources

Use the "main_run_pts_parcels.sh" bash script to run particle tracking simulations for the entire Indian Ocean from 1995-2015. This script calls "pts_parcels_io_beaching.py" to run yearly simulations.

Run the "pts_parcels_neutral_iod.py" script with Python to run a single simulation during neutral IOD conditions from 2008-2009.

## Applying beaching conditions
Beaching conditions are applied in the `BeachingParticles` class in the "particles.py" script. These conditions can be applied to the OceanParcels particle tracking simulation results by running the "processing.py" script.

To create connectivity matrices between different beaching countries, to determine the number of particles that beach in each country, and to determine the particle development in time (i.e. percentage of particles that have beached vs percentage of particles that are still drifting in the northern and southern hemisphere Indian Ocean), run the "postprocessing.py" script.

## Plotting
The "plots_vanderMheen_et_al_2020.py" script contains functions to create the figures in the van der Mheen et al. (2020) article (https://doi.org/10.5194/os-2020-50).
