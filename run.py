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
    parser.add_argument('-i', '--t0', type=int, default=0, help="Timestep that the tracer particles are initiated at. Note that this is an integer timestep, which will be used to choose the physical time in the gal.times array. Default is the first timestep (0).")
    parser.add_argument('-N', '--Nsys', type=int, default=1, help="Number of systems you wish to run for this particular starting time. Default is 1.")
    parser.add_argument('-mp', '--multiproc', type=int, default=None, help="If specified, will parallelize over the number of cores provided as an argument. Default is None.")

    # defining grid properties
    parser.add_argument('-T', '--Tsteps', type=int, default=100, help="Number of discrete time (redshift) bins to evolve systems in. Default is 100.")
    parser.add_argument('-rg', '--Rgrid', type=int, default=100, help="Number of gridpoints for the Z-component of the interpolation model. Default is 100.")
    parser.add_argument('-zg', '--Zgrid', type=int, default=50, help="Number of gridpoints for the Z-component of the interpolation model. Default is 50.") 

    # paths to data files
    parser.add_argument('--interp-dirpath', type=str, help="Path to the directory that holds interpolation files. Default is None.")
    parser.add_argument('--sgrb-path', type=str, default='./data/sgrb_hostprops_offsets.txt', help="Path to the table with sGRB host galaxy information. Default is './data/sgrb_hostprops_offsets.txt'.")
    parser.add_argument('--samples-path', type=str, default='./data/example_bns.dat', help="Path to the samples from population synthesis for generating the initial population of binaries. Default is './data/example_bns.dat'.")

    # galaxy arguments
    parser.add_argument('--disk-profile', type=str, default='DoubleExponential', help="Profile for the galactic disk, named according to Galpy potentials. Default is 'DoubleExponential'.")
    parser.add_argument('--bulge-profile', type=str, default=None, help="Profile for the galactic bulge, named according to Galpy potentials. Default is None.")
    parser.add_argument('--dm-profile', type=str, default='NFW', help="Profile for the DM, named according to Galpy potentials. Default is NFW.")
    parser.add_argument('--z-scale', type=float, default=0.05, help="Fraction of the galactic scale radius for the scale height above/below the disk. Default=0.05.")

    # sampling arguments
    parser.add_argument('--Mcomp-method', type=str, default='popsynth', help="Method for sampling the companion mass. Default is 'popsynth'.")
    parser.add_argument('--Mns-method', type=str, default='popsynth', help="Method for sampling the mass of the neutron star formed from the NS. Default is 'popsynth'.")
    parser.add_argument('--Mhe-method', type=str, default='popsynth', help="Method for sampling the helium star mass. Default is 'popsynth'.")
    parser.add_argument('--Apre-method', type=str, default='popsynth', help="Method for sampling the pre-SN semimajor axis. Default is 'popsynth'.")
    parser.add_argument('--epre-method', type=str, default='circularized', help="Method for sampling the pre-SN eccentricity. Default is 'circularized'.")
    parser.add_argument('--Vkick-method', type=str, default='maxwellian', help="Method for sampling the SN natal kick. Default is 'maxwellian'.")
    parser.add_argument('--R-method', type=str, default='sfr', help="Method for sampling the initial distance from the galactic center. Default is 'sfr'.")

    parser.add_argument('--Mcomp-mean', type=float, default=1.33, help="Mean mass of the companion neutron star, in Msun. Default is 1.33.")
    parser.add_argument('--Mcomp-sigma', type=float, default=0.09, help="Sigma of gaussian for drawing mass of the companion neutron star, in Msun. Default is 0.09.")
    parser.add_argument('--Mns-mean', type=float, default=1.33, help="Mean mass of the neutron star formed from the supernova, in Msun. Default is 1.33.")
    parser.add_argument('--Mns-sigma', type=float, default=0.09, help="Sigma of gaussian for drawing the mass of the neutron star formed from the supernova, in Msun. Default is 0.09.")
    parser.add_argument('--Mhe-mean', type=float, default=3.0, help="Mean mass of the helium star, in Msun. Default is 3.0.")
    parser.add_argument('--Mhe-sigma', type=float, default=0.5, help="Sigma of gaussian for drawing the mass of the helium star, in Msun. Default is 0.5.")
    parser.add_argument('--Mhe-max', type=float, default=8.0, help="Maximum mass of the helium star, in Msun. Default is 8.0.")
    parser.add_argument('--Apre-mean', type=float, default=2.0, help="Mean orbital separation prior to the SN, in Rsun. Default is 2.0.")
    parser.add_argument('--Apre-sigma', type=float, default=0.5, help="Sigma of gaussian for drawing the orbital separation prior to the SN, in Rsun. Default is 0.5.")
    parser.add_argument('--Apre-min', type=float, default=0.1, help="Minimum orbital separation prior to the SN, in Rsun. Default is 0.1.")
    parser.add_argument('--Apre-max', type=float, default=10.0, help="Maximum orbital separation prior to the SN, in Rsun. Default is 10.0.")
    parser.add_argument('--Vkick-sigma', type=float, default=265.0, help="Value for the Maxwellian dispersion of the SN natal kick, in km/s. If method=='fixed', this is the fixed value that is used. Default is 265.0.")
    parser.add_argument('--Vkick-min', type=float, default=0.0, help="Minimum velocity for the SN natal kick, in km/s. Default is 0.0.")
    parser.add_argument('--Vkick-max', type=float, default=1000.0, help="Maximum velocity for the SN natal kick, in km/s. Default is 1000.0.")
    parser.add_argument('--R-mean', type=float, default=5.0, help="Mean starting distance from the galactic center, in kpc. Default is 5.0.")

    args = parser.parse_args()

    return args




def main(args):
    """
    Main function. 
    """
    start = time.time()

    # read sgrb hostprops table as pandas dataframe
    sgrb_host_properties = pd.read_table(args.sgrb_path, delim_whitespace=True, na_values='-')
    grb_props = sgrb_host_properties.loc[sgrb_host_properties['GRB'] == args.grb]

    # get galaxy information
    gal = galaxy_history.GalaxyHistory(\
                        obs_mass_stars = float(10**grb_props['log(M*)'] * u.Msun.to(u.g)),\
                        obs_redz = float(grb_props['z']),\
                        obs_age_stars = float(grb_props['PopAge'] * u.Gyr.to(u.s)),\
                        obs_rad_eff = float(grb_props['r_e'] * u.kpc.to(u.cm)),\
                        obs_gal_sfr = float(grb_props['SFR'] * (u.Msun.to(u.g))/u.yr.to(u.s)),\
                        disk_profile = args.disk_profile,\
                        bulge_profile = args.bulge_profile,\
                        dm_profile = args.dm_profile,\
                        z_scale = args.z_scale,\
                        interp_dirpath = args.interp_dirpath,\
                        Tsteps = args.Tsteps,\
                        Rgrid = args.Rgrid,\
                        Zgrid = args.Zgrid,\
                        name = grb_props['GRB'].item())

    print('Redshift at which particles are initiated: z={0:0.2f}\n'.format(gal.redz[args.t0]))


    # construct dict of params for sampling methods
    params_dict={
        'Mcomp_mean':args.Mcomp_mean, 'Mcomp_sigma':args.Mcomp_sigma, 
        'Mns_mean':args.Mns_mean, 'Mns_sigma':args.Mns_sigma, 
        'Mhe_mean':args.Mhe_mean, 'Mhe_sigma':args.Mhe_sigma, 'Mhe_max':args.Mhe_max,
        'Apre_mean':args.Apre_mean, 'Apre_sigma':args.Apre_sigma, 'Apre_min':args.Apre_min, 'Apre_max':args.Apre_max,
        'Vkick_sigma':args.Vkick_sigma, 'Vkick_min':args.Vkick_min, 'Vkick_max':args.Vkick_max,
        'R_mean':args.R_mean}
    
    # get sampled properties of tracer particles
    sampled_parameters = sample.sample_parameters(gal, t0=args.t0, Nsys=args.Nsys, \
                            Mcomp_method=args.Mcomp_method, \
                            Mns_method=args.Mns_method, \
                            Mhe_method=args.Mhe_method, \
                            Apre_method=args.Apre_method, \
                            epre_method=args.epre_method, \
                            Vkick_method=args.Vkick_method, \
                            R_method=args.R_method, \
                            params_dict = params_dict, \
                            samples = args.samples_path)


    # sample system parameters
    systems = system.Systems(sampled_parameters)

    # implement the supernova
    systems.SN()

    # check if the systems survived the supernova, and return survival fraction
    survival_fraction = systems.check_survival()

    # calculate the pre-SN galactic velocity
    systems.galactic_velocity(gal, args.t0)

    # transform the systemic velocity into the galactic frame and add pre-SN velocity
    systems.galactic_frame()

    # calculate the inspiral time for systems that survived
    tH_inspiral_fraction = systems.inspiral_time()
    
    # do evolution of each tracer particle (should parallelize this)
    systems.evolve(gal, args.t0, int_method='odeint')

    end = time.time()
    print('{0:0.2} s'.format(end-start))


# MAIN FUNCTINON
if __name__ == '__main__':
    args = parse_commandline()

    main(args)

