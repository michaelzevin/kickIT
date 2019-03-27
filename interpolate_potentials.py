#!/software/anaconda3/bin/python

# --- Import standard modules to the python path.
import os
import pdb
import argparse
import pdb
import time
import pickle

import numpy as np
import pandas as pd

import itertools
import multiprocessing
from functools import partial

import astropy.units as u
import astropy.constants as C

from galpy.potential import RazorThinExponentialDiskPotential, DoubleExponentialDiskPotential, NFWPotential
from galpy.potential import interpRZPotential

from kickIT import utils

# --- Specify arguments for the interpolation function
def parse_commandline():
    """
    Parse the arguments given on the command-line.
    """
    parser = argparse.ArgumentParser()

    # default information
    parser.add_argument('-g', '--gal-path', type=str, help="Path to pickled gal file that we want to create interpolants for.")
    parser.add_argument('-mp', '--multiproc', type=str, default=None, help="If specified, will parallelize over the number of cores provided as an argument. Can also use the string 'max' to parallelize over all available cores. Default is None.")

    # defining grid properties for interpolations
    parser.add_argument('-rg', '--Rgrid', type=int, default=500, help="Number of gridpoints for the Z-component of the interpolation model. Default is 100.")
    parser.add_argument('-zg', '--Zgrid', type=int, default=300, help="Number of gridpoints for the Z-component of the interpolation model. Default is 50.") 
    parser.add_argument('--Rgrid-max', type=float, default=1e3, help="Maximum R value for interpolated potentials. Default is 1e3.")
    parser.add_argument('--Zgrid-max', type=float, default=1e2, help="Maximum Z value for interpolated potentials. Default is 1e2.")

    # paths to data files
    parser.add_argument('--interp-path', type=str, default='./interp.pkl', help="Path to where the interpolation file will be saved. Default is '/.interp.pkl'.")

    args = parser.parse_args()

    return args




def main(args):
    """
    Main function. 
    """
    start = time.time()


    # --- read galaxy file
    gal = pickle.load(open(args.gal_path, 'rb'))

    # --- create interpolatnts for the potentials in gal class
    interps = construct_interpolants(gal, \
                    multiproc = args.multiproc, \
                    Rgrid = args.Rgrid, \
                    Zgrid = args.Zgrid, \
                    Rgrid_max = args.Rgrid_max, \
                    Zgrid_max = args.Zgrid_max)

    pickle.dump(interps, open(args.interp_path, 'wb'))



def construct_interpolants(gal, multiproc=None, Rgrid=500, Zgrid=100, Rgrid_max=1000, Zgrid_max=100, ro=8*u.kpc, vo=220*u.km/u.s):
    """Creates interpolants for combined potentials specified in gal class. 
    To implement multiprocessing, specify an int for the argument 'multiproc'.
    """
    
    print('Creating interpolation models of combined galactic potentials at each redshift...\n')

    # --- create the grid of rads and heights we will be using
    rad_range = np.asarray([1e-4, Rgrid_max])*u.kpc
    height_range = np.asarray([0, Zgrid_max])*u.kpc

    # need to convert Rs and Zs to natural units
    rads = (rad_range / ro).value
    heights = (height_range / ro).value

    rs = (*rads, Rgrid)
    logrs = (*np.log10(rads), Rgrid)
    zs = (*heights, Zgrid)
            

    # --- set up the interpolation function
    func = partial(interp_func, rgrid=logrs, zgrid=zs)

    # --- enable multiprocessing, if specified
    if multiproc:
        if multiproc=='max':
            mp = multiprocessing.cpu_count()
        else:
            mp = int(multiproc)

        pool = multiprocessing.Pool(mp)
        func = partial(interp_func, rgrid=logrs, zgrid=zs)

        start = time.time()
        print('Parallelizing interpolations over {0:d} cores...\n'.format(mp))
        interpolated_potentials = pool.map(func, gal.full_potentials_natural)
        stop = time.time()
        print('   finished! It took {0:0.2f}s\n'.format(stop-start))
        

    # otherwise, do this in serial
    else:
        print('Interpolating potentials in serial...\n')
        interpolated_potentials=[]
        for ii, data in enumerate(gal.full_potentials_natural):

            start = time.time()
            ip = func(data)
            end = time.time()
            print('   interpolated potential for step {0:d} (z={1:0.2f}) created in {2:0.2f}s...'.format(ii,gal.redz[ii],end-start))

            interpolated_potentials.append(ip)

    return interpolated_potentials


# --- define interpolating function
def interp_func(potentials, rgrid, zgrid, ro=8*u.kpc, vo=220*u.km/u.s):
    ip = interpRZPotential(potentials, rgrid=rgrid, zgrid=zgrid, logR=True, interpRforce=True, interpzforce=True, zsym=True, ro=ro, vo=vo)
    return ip






# MAIN FUNCTINON
if __name__ == '__main__':
    args = parse_commandline()

    main(args)

