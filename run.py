#!/software/anaconda3/bin/python

# ---- Import standard modules to the python path.
import os
import argparse
import pdb
import warnings
import pickle

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
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-g', '--grb', type=str, help="GRB for which we want to perform analysis.")
    parser.add_argument('-N', '--Nsys', type=int, default=1, help="Number of systems you wish to run for this particular starting time. Default is 1.")
    parser.add_argument('-mp', '--multiproc', type=str, default=None, help="If specified, will parallelize over the number of cores provided as an argument. Can also use the string 'max' to parallelize over all available cores. Default is None.")
    parser.add_argument('--fixed-birth', type=int, default=None, help="Fixes the birth time of the progenitor system by specifying a timestep (t0). Default=None.")
    parser.add_argument('--fixed-potential', type=int, default=None, help="Fixes the galactic potential to the potential of the galaxy at the timestep t0. Also samples the location of the system according to this galactic model. Default=None.")


    # paths to data files
    parser.add_argument('--output-dirpath', type=str, default='./output_files/', help="Path to the output hdf file. File has key names tracers. Default is './output_files/'.")
    parser.add_argument('--sgrb-path', type=str, help="Path to the table with sGRB host galaxy information.")
    parser.add_argument('--samples-path', type=str, default=None, help="Path to the samples from population synthesis for generating the initial population of binaries. Default is None.")
    parser.add_argument('--interp-path', type=str, default=None, help="Path to the potential interpolation file you wish to use. Default is None.")
    parser.add_argument('--gal-path', type=str, default=None, help="Sets path to read in previously constructed galaxy realizzation. Default is 'None'.")

    # galaxy arguments
    parser.add_argument('--disk-profile', type=str, default='DoubleExponential', help="Profile for the galactic disk, named according to Galpy potentials. Default is 'DoubleExponential'.")
    parser.add_argument('--bulge-profile', type=str, default=None, help="Profile for the galactic bulge, named according to Galpy potentials. Default is None.")
    parser.add_argument('--dm-profile', type=str, default='NFW', help="Profile for the DM, named according to Galpy potentials. Default is NFW.")
    parser.add_argument('--z-scale', type=float, default=0.05, help="Fraction of the galactic scale radius for the scale height above/below the disk. Default=0.05.")
    parser.add_argument('--differential-prof', action='store_true', help="Uses a differential stellar profile, creating a unique galpy potential at each timestep according to the updated scale radius and accumulated mass. Default=False.")
    parser.add_argument('--smhm-relation', type=str, default='Guo', help="Chooses a stellar mass-halo mass relation. Current options are from Guo+2010 and Moster+2012. If Moster is provided, can also supply a sigma value. Default is 'Guo'.")
    parser.add_argument('--smhm-sigma', type=float, default=0.0, help="Deviation from the stellar mass-halo mass relation in Moster+2012. Can supply either positive or negative values. Default is 0.0.")

    # sampling arguments
    parser.add_argument('--sample-progenitor-props', action='store_true',help="Indicates whether to use specific sampling for the progenitor properties defined in sample.py (i.e., fro pop synth). If not specified, a grid in *only* R (based on the SF profile) and Vsys (with random initial direction for Vsys) will be used for initializing the particles. Default=False.")
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

    # integration arguments
    parser.add_argument('--int-method', type=str, default='odeint', help="Integration method for the orbits. Possible options are 'odeint' or 'leapfrog', until we get the C implementation working. Default is 'odeint'.")
    parser.add_argument('--Tint-max', type=float, default=120.0, help="Amount of time to integrate before terminating, in seconds. Default is 120.0.")
    parser.add_argument('--resolution', type=int, default=1000, help="Resolution of integration, specified by the number of timesteps per redshift bin in the integration. Default is 1000.")
    parser.add_argument('--save-traj', action='store_true',help="Indicates whether to save the full trajectories. Default=False")
    parser.add_argument('--downsample', type=int, default=None, help="Downsamples the trajectory data by taking every Nth line in the trajectories dataframe. Default=None.")


    args = parser.parse_args()

    return args




def main(args):
    """
    Main function.
    """
    start = time.time()

    # --- construct pertinent directories
    if not os.path.exists(args.output_dirpath):
        os.makedirs(args.output_dirpath)

    # --- read sgrb hostprops table as pandas dataframe, parse observed props
    sgrb_host_properties = pd.read_csv(args.sgrb_path, delim_whitespace=True, na_values='-')
    gal_info = sgrb_host_properties.loc[sgrb_host_properties['GRB'] == args.grb].iloc[0]
    obs_props = {'name':gal_info['GRB'],\
                      'pcc':gal_info['Pcc'],\
                      'mass_stars':10**gal_info['log(M*)']*u.Msun,\
                      'redz':gal_info['z'],\
                      'age_stars':gal_info['PopAge']*u.Gyr,\
                      'rad_eff':gal_info['r_e']*u.kpc,\
                      'rad_offset':gal_info['deltaR']*u.kpc,\
                      'rad_offset_error':gal_info['deltaR_err']*u.kpc,\
                      'gal_sfr':gal_info['SFR']*u.Msun/u.yr
                      }


    # --- Read in or construct galaxy class
    if args.gal_path:
        gal = pickle.load(open(args.gal_path, 'rb'))
        print('Using galaxy realization living at {0:s}...\n'.format(args.gal_path))
    else:
        gal = galaxy_history.GalaxyHistory(\
                            obs_props = obs_props,\
                            disk_profile = args.disk_profile,\
                            dm_profile = args.dm_profile,\
                            smhm_relation = args.smhm_relation,\
                            smhm_sigma = args.smhm_sigma,\
                            bulge_profile = args.bulge_profile,\
                            z_scale = args.z_scale,\
                            differential_prof = args.differential_prof,\
                            )

    # --- Save gal class
    gal.write(args.output_dirpath)


    # --- Read in interpolants here, if specified
    interpolants = None
    if args.interp_path:
        interpolants = pickle.load(open(args.interp_path, 'rb'))
        print('Using galactic potential interpolations living at {0:s}...\n'.format(args.interp_path))

        # Check that the interpolants have the same number of timesteps
        if len(interpolants) != len(gal.times):
            raise ValueError('The interpolation file you specified does not have the same parameters as your galaxy model! It has {0:d} timesteps whereas you galaxy has {1:d}!'.format(len(interpolants), len(gal.times)))

    else:
        if args.differential_prof==True:
            warnings.warn("If you're using differential stellar profiles, you might want to be using an interpolated potential instance to speed up the integrations!!!\n")




    # --- sample progenitor parameters

    # construct dict of params for sampling methods
    params_dict={
        'Mcomp_mean':args.Mcomp_mean, 'Mcomp_sigma':args.Mcomp_sigma,
        'Mns_mean':args.Mns_mean, 'Mns_sigma':args.Mns_sigma,
        'Mhe_mean':args.Mhe_mean, 'Mhe_sigma':args.Mhe_sigma, 'Mhe_max':args.Mhe_max,
        'Apre_mean':args.Apre_mean, 'Apre_sigma':args.Apre_sigma, 'Apre_min':args.Apre_min, 'Apre_max':args.Apre_max,
        'Vkick_sigma':args.Vkick_sigma, 'Vkick_min':args.Vkick_min, 'Vkick_max':args.Vkick_max,
        'R_mean':args.R_mean}

    # FIXME: maybe should move the population sampling to another function?
    # --- if fully sampling progenitor parameters...
    if args.sample_progenitor_props:
        print('Fully sampling system parameters, determining systemic velocities and inspiral times...\n')
        sampled_parameters = sample.sample_parameters(gal, Nsys=args.Nsys, \
                                Mcomp_method=args.Mcomp_method, \
                                Mns_method=args.Mns_method, \
                                Mhe_method=args.Mhe_method, \
                                Apre_method=args.Apre_method, \
                                epre_method=args.epre_method, \
                                Vkick_method=args.Vkick_method, \
                                R_method=args.R_method, \
                                params_dict = params_dict, \
                                samples = args.samples_path, \
                                fixed_birth = args.fixed_birth, \
                                fixed_potential = args.fixed_potential)

    # --- otherwise we sample in only Vsys and Tinsp
    else:
        print('Skipping sampling of progenitor parameters, sampling only R and Vsys and feeding to the integrator...\n')

        sampled_parameters = sample.sample_Vsys_R(gal, Nsys=args.Nsys, Vsys_range=(0,1000), R_method=args.R_method, fixed_birth=args.fixed_birth, fixed_potential=args.fixed_potential)


    # --- Initialize systems class
    systems = system.Systems(sampled_parameters, sample_progenitor_props=args.sample_progenitor_props)

    # --- Calculate the instantaneous particle escape velocities and galactic velocities at birth
    systems.escape_velocity(gal, interpolants)
    systems.galactic_velocity(gal, interpolants, args.fixed_potential)

    # --- If we sampled the porgenitor properties, we need to determine the impact of the SN and the inspiral time, and bring the system into the galactic frame
    if args.sample_progenitor_props:
        # implement the supernova
        systems.SN()
        # check if the systems survived the supernova, and return survival fraction
        survival_fraction = systems.check_survival()
        # calculate the inspiral time for systems that survived
        tH_inspiral_fraction = systems.inspiral_time()
        # transform the systemic velocity into the galactic frame and add pre-SN velocity
        systems.galactic_frame()


    # --- Otherwise, we just decompose the Vsys array according to SYStheta nad SYSphi
    else:
        # project systemic velocity into galactic coordinates
        systems.decompose_Vsys()



    # --- Kinematically evolve the tracer particles
    systems.evolve(gal, multiproc=args.multiproc, \
                        int_method=args.int_method, \
                        Tint_max=args.Tint_max, \
                        resolution=args.resolution, \
                        save_traj=args.save_traj, \
                        downsample=args.downsample, \
                        outdir = args.output_dirpath, \
                        fixed_potential = args.fixed_potential, \
                        interpolants = interpolants)



    # --- write data to output file and finish
    systems.write(gal, args.output_dirpath)

    end = time.time()
    print('{0:0.2} s'.format(end-start))


# MAIN FUNCTINON
if __name__ == '__main__':
    args = parse_commandline()

    main(args)

