#!/software/anaconda3/bin/python

# ---- Import standard modules to the python path.
import os
import pdb
import argparse

import numpy as np
import pandas as pd

import astropy.units as u
import astropy.constants as C

from kickIT import __version__
from kickIT import galaxy_history
from kickIT import sample
from kickIT import system

import time

def parse_commandline():
    """
    Parse the arguments given on the command-line.
    """
    parser = argparse.ArgumentParser(description=
    """
    Can we kick it? Yes we can!
    """)

    # default information
    parser.add_argument('-V', '--version', action='version', version=__version__)
    parser.add_argument('-g', '--grb', type=str, help="GRB for which we want to perform analysis.")
    parser.add_argument('-t0', '--t0', type=int, default=0, help="Timestep that the tracer particles are initiated at. Note that this is an integer timestep, which will be used to choose the physical time in the gal.times array. Default is the first timestep (0).")
    parser.add_argument('-N', '--Nsys', type=int, default=1, help="Number of systems you wish to run for this particular starting time. Default is 1.")
    args = parser.parse_args()

    return args




def main(grb, Nsys, t0):
    """
    Main function. 
    """
    start = time.time()

    popsynth_path = '/Users/michaelzevin/research/sgrb/example_bns.dat'

    # get galaxy information
    gal = galaxy_history.GalaxyHistory(\
                        obs_mass_stars = float(10**grb_props['log(M*)'] * u.Msun.to(u.g)),\
                        obs_redz = float(grb_props['z']),\
                        obs_age_stars = float(grb_props['PopAge'] * u.Gyr.to(u.s)),\
                        obs_rad_eff = float(grb_props['r_e'] * u.kpc.to(u.cm)),\
                        obs_gal_sfr = float(grb_props['SFR'] * (u.Msun.to(u.g))/u.yr.to(u.s)),\
                        times = None,\
                        name = str(grb_props['GRB']))
    print('Redshift at which particles are initiated: z={0:0.2f}\n'.format(gal.redz[t0]))

    # get sampled properties of tracer particles
    sampled_parameters = sample.sample_parameters(gal, t0=t0, Nsys=Nsys, \
                            Mcomp_method='popsynth', \
                            Mns_method='popsynth', \
                            Mhe_method='popsynth', \
                            Apre_method='popsynth', \
                            epre_method='circularized', \
                            Vkick_method='maxwellian', \
                            R_method='sfr', \
                            samples = popsynth_path)

    # sample system parameters
    systems = system.Systems(sampled_parameters)

    # implement the supernova
    systems.SN()

    # check if the systems survived the supernova, and return survival fraction
    survival_fraction = systems.check_survival()

    # calculate the pre-SN galactic velocity
    systems.galactic_velocity(gal, t0)

    # transform the systemic velocity into the galactic frame and add pre-SN velocity
    systems.galactic_frame()

    # calculate the inspiral time for systems that survived
    tH_inspiral_fraction = systems.inspiral_time()
    
    # do evolution of each tracer particle (should parallelize this)
    systems.evolve(gal, t0)

    end = time.time()
    print('{0:0.2} s'.format(end-start))


# MAIN FUNCTINON
if __name__ == '__main__':
    args = parse_commandline()

    # read sgrb hostprops table as pandas dataframe
    sgrb_host_properties = pd.read_table('/Users/michaelzevin/research/sgrb/sgrb_hostprops_offsets.txt', delim_whitespace=True, na_values='-', skiprows=12) 
    grb_props = sgrb_host_properties.loc[sgrb_host_properties['GRB'] == args.grb]
    t0 = args.t0
    Nsys = args.Nsys

    main(grb_props, Nsys, t0)

