from pts_parcels_io_beaching import IORiverParticles, run_hycom_subset_monthly_release
from pts_parcels_io_beaching import run_hycom_cfsr_subset_monthly_release
from utilities import get_dir
from datetime import datetime

def run_neutral_iod_2008_with_wind(output_name='neutral_iod_2008_with_3p-wind', constant_release=False):
    output_dir = get_dir('pts_output')
    start_date = datetime(2008, 1, 1, 12, 0)
    end_date = datetime(2008, 12, 31)
    io_sources = IORiverParticles.get_from_netcdf(start_date, constant_release=constant_release)
    run_hycom_cfsr_subset_monthly_release(output_dir, output_name,
                                          io_sources.time0, io_sources.lon0, io_sources.lat0,
                                          start_date, end_date)

def run_neutral_iod(output_name='neutral_iod_2008-2009',constant_release=False):
    input_dir = get_dir('hycom_input')
    output_dir = get_dir('pts_output')
    start_date = datetime(2008,1,1,12,0)
    end_date = datetime(2009,12,31)
    io_sources = IORiverParticles.get_from_netcdf(start_date,
                                                  constant_release=constant_release)
    run_hycom_subset_monthly_release(input_dir,output_dir,output_name,
                                     io_sources.time0,io_sources.lon0,io_sources.lat0,
                                     start_date,end_date)

if __name__ == '__main__':
    # output_name = 'neutral_iod_constant_sources_2008-2009'
    # run_neutral_iod(output_name=output_name,constant_release=True)
    run_neutral_iod_2008_with_wind()
