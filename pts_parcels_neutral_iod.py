from pts_parcels_io_beaching import IORiverParticles, run_hycom_subset_monthly_release
from utilities import get_dir
from datetime import datetime

if __name__ == '__main__':
    input_dir = get_dir('hycom_input')
    output_dir = get_dir('pts_output')
    start_date = datetime(2008,1,1,12,0)
    end_date = datetime(2009,12,31)
    output_name = 'neutral_iod_constant_sources_2008-2009'
    io_sources = IORiverParticles.get_from_netcdf(start_date,constant_release=True)
    run_hycom_subset_monthly_release(input_dir,output_dir,output_name,
                                     io_sources.time0,io_sources.lon0,io_sources.lat0,
                                     start_date,end_date)
    