def delete_particle(particle,fieldset,time):
    particle.delete()

def periodic_bc(particle,fieldset,time):
    if particle.lon < fieldset.halo_west:
        particle.lon += fieldset.halo_east-fieldset.halo_west
    elif particle.lon > fieldset.halo_east:
        particle.lon -= fieldset.halo_east-fieldset.halo_west

def wrap_lon_180(particle, fieldset, time):
    if particle.lon > 180.:
        particle.lon = particle.lon - 360.
    if particle.lon < -180.:
        particle.lon = particle.lon + 360.

def wrap_lon_360(particle, fieldset, time):
    if particle.lon > 360.:
        particle.lon = particle.lon - 360.
    if particle.lon < 0.:
        particle.lon = particle.lon + 360.
