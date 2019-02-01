"""
"""
import warnings
import pdb

import numpy as np
import itertools
import os
import pickle
import time
import pandas as pd
from scipy.interpolate import interp1d
import multiprocessing
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy as ap
import astropy.units as u
import astropy.constants as C

from galpy.potential import RazorThinExponentialDiskPotential, DoubleExponentialDiskPotential, NFWPotential
from galpy.potential import interpRZPotential

# Set units here (standard in cgs)
MSOL = u.M_sun.to(u.g)
PC = u.pc.to(u.cm)
YR = u.yr.to(u.s)

KPC = 1e3 * PC
GYR = 1e9 * YR
GMSOL = 1e9 * MSOL

# Set galpy internal units
_ro = 8
_vo = 220

# Import of local modules must come after constants above
from .. import utils
from . import baryons, halos, cosmology


class GalaxyHistory:
    """Class for calculating SFR and mass histories of a galaxy, based on limited observations.
    """


    def __init__(self, obs_mass_stars, obs_redz, obs_age_stars, obs_rad_eff, obs_gal_sfr, disk_profile, dm_profile, bulge_profile=None, z_scale=None, interp_dirpath=None, Tsteps=100, Rgrid=100, Zgrid=50, Rgrid_max=1e3, Zgrid_max=1e2, name=None, multiproc=None, verbose=False):
        """All parameters are converted to CGS units!
        """

        # Store if verbose
        self.VERBOSE = verbose

        # Redshift at which integration/data-arrays begin
        self.REDZ_BEG = 4.0

        # Number of time-steps between `REDZ_BEG` and redshift zero
        self.NUM_TIME_STEPS = Tsteps

        # Radial points at which to construct the interpolant, in pc
        self.NUM_RADS = Rgrid
        self.RGRID_MAX = int(Rgrid_max)
        self.RADS_RANGE = np.array([1e-4,Rgrid_max]) * KPC   # cm

        # Disk height points at which to construct the interpolant, in pc
        self.NUM_HEIGHTS = Zgrid
        self.ZGRID_MAX = int(Zgrid_max)
        self.HEIGHTS_RANGE = np.array([0,Zgrid_max]) * KPC   # cm

        # Initialize cosmology
        cosmo = cosmology.Cosmology()
        self.cosmo = cosmo

        # Read in observed parameters of the sGRB host
        self.obs_mass_stars = obs_mass_stars
        self.obs_redz = obs_redz
        self.obs_age_stars = obs_age_stars
        self.obs_rad_eff = obs_rad_eff
        self.obs_gal_sfr = obs_gal_sfr
        self.disk_profile = disk_profile
        self.bulge_profile = bulge_profile
        self.dm_profile = dm_profile
        self.z_scale = z_scale
        self.name = name

        # Construct sampling times
        self.time_beg = cosmo.age(self.REDZ_BEG).cgs.value
        # End time is the time at which the galaxy is observed (calculated from redshift)
        self.time_end = cosmo.age(self.obs_redz).cgs.value
        self.time_dur = self.time_end - self.time_beg
        times = np.logspace(*np.log10([self.time_beg, self.time_end]), self.NUM_TIME_STEPS)
        self.times = times

        # Calculate array of redshifts corresponding to each `times` value (age of the universe)
        self.redz = cosmo.tage_to_z(self.times)
        if self.VERBOSE:
            textr = np.array([times.min(), times.max()])
            print("\nTimes: {:3d} between [{:.1e}, {:.1e}] Gyr (z={:.1e}, {:.1e})".format(
                times.size, *textr/GYR, self.redz.max(), self.redz.min()))

        # Construct sampling of radii for constructing interpolants (flat in log)
        self.rads = 10**np.linspace(*np.log10(self.RADS_RANGE), self.NUM_RADS)

        # Calculate galaxy properties vs time
        self.calc_total_masses_vs_time()
        self.calc_mass_profiles_vs_time()

        # Calculate the SFR weights at each timestep
        self.sfr_weights = self.gal_sfr / np.sum(self.gal_sfr)

        # Calculate galactic potentials vs time
        self.calc_potentials_vs_time()
        self.calc_potentials_vs_time(method='natural')

        # Interpoate the potentials (need to be in natural units)
        if interp_dirpath:
            self.interp = True
            self.calc_interpolated_potentials(interp_dirpath, multiproc=multiproc)

        return


    def calc_total_masses_vs_time(self):
        """Calculate the total-mass of stars, gas and DM based on galaxy scaling relations
        """

        # Initialize mass and sfr arrays
        mass_stars = np.zeros_like(self.times)
        gal_sfr = np.zeros_like(self.times)

        # Find time and redshift of SFR peak from stellar-age of galaxy
        time_sfr_peak = self.time_end - self.obs_age_stars
        redz_sfr_peak = float(self.cosmo.tage_to_z(time_sfr_peak))
        if self.VERBOSE:
            print("Time of peak SFR: {:.2f} [Gyr] (z={:.2f})".format(time_sfr_peak/GYR, redz_sfr_peak))

        # Assume constant SFR from time of observation, back to SFR peak
        idx = (self.times >= time_sfr_peak)
        gal_sfr[idx] = self.obs_gal_sfr
        mass_stars[:] = self.obs_mass_stars
        mass_stars[idx] -= self.obs_gal_sfr * (self.time_end - self.times[idx])
        growth_quiescent = self.obs_gal_sfr * (self.time_end - time_sfr_peak)
        frac = growth_quiescent / mass_stars[-1]
        if self.VERBOSE:
            print("During quiescent state, galaxy grew by {:.2e} (dM={:.2e} [Msol])".format(
            frac, growth_quiescent/MSOL))
        # For the model to be consistent, the fractional growth during "quiescent" phase should be
        # negligible
        if frac > 0.1:
            warnings.warn("Significant galaxy growth during 'quiescent' phase! Inconsistency?")

        # Assume peak of SFR follows from star-formation main-sequence
        # integrate backwards from there
        peak_idx = np.where(~idx)[0][-1]
        ii = peak_idx
        while ii >= 0:
            mstar = mass_stars[ii+1]
            # Get the SFR by assuming galaxy lies on the SFR main-sequence
            sfr = baryons.sfr_main_seq(mstar, self.redz[ii+1], self.cosmo)
            dt = self.times[ii+1] - self.times[ii]
            # Change in mass in this time-step
            dm = sfr * dt

            # time-step at 'galaxy formation' can lead to negative values, start at mass of 0 and drop out earlier times
            if dm > mstar:
                dm = mstar
                sfr = mstar / dt
                self.formation_sfr = sfr
                self.formation_idx = ii
                self.formation_time = self.times[ii]

                # now, we make all arrays begin from the step after formation (i.e. the first step that has mass)
                self.time_beg = self.times[ii+1]
                self.time_dur = self.times[-1] - self.time_beg
                mass_stars = mass_stars[(ii+1):]
                gal_sfr = gal_sfr[(ii+1):]
                self.times = self.times[(ii+1):]
                self.redz = self.redz[(ii+1):]
                break

            gal_sfr[ii] = sfr
            mass_stars[ii] = mstar - dm
            ii -= 1

        # Initialize gas and DM arrays
        mass_gas = np.zeros_like(self.times)
        mass_dm = np.zeros_like(self.times)

        # Use scaling relations to get DM and gas masses from stellar (note that we start from step 1 since the initial mass is 0)
        mass_dm[:] = halos.stellar_mass_to_halo_mass(mass_stars[:])
        mass_gas[:] = baryons.gas_mass_from_stellar_mass(mass_stars[:])

        # Store arrays
        self.mass_stars = mass_stars
        self.mass_gas = mass_gas
        self.mass_dm = mass_dm
        self.gal_sfr = gal_sfr
        if self.VERBOSE:
            print("")
            finals = np.array([mass_stars[-1], mass_gas[-1], mass_dm[-1]])/MSOL
            print("Final total masses: stars={:.1e}, gas={:.1e}, DM={:.1e} [Msol]".format(
                *finals))
            print("Final SFR={:.1e} [Msol/yr]".format(gal_sfr[-1]*YR/MSOL))

        return

    def calc_mass_profiles_vs_time(self):
        """Calculate the radial mass profiles vs time
        The profile at each timestep will be a linear interpolant
        """
        # Initialize arrays for galaxy matter profiles over time
        prof_shape = (self.times.size, self.rads.size)

        # All of these quantities are in units of mass (grams), such that the total mass out to a given radius is simply the sum out to that index.
        # Note that the SFR, stars and gas are in a thin-disk while the DM is roughly spherical.
        mass_sfr_prof = np.zeros(prof_shape)
        mass_stars_prof = np.zeros(prof_shape)
        mass_gas_prof = np.zeros(prof_shape)
        mass_dm_prof = np.zeros(prof_shape)

        # lists for saving interpolation models
        mass_sfr_interp = []
        mass_stars_interp = []
        mass_gas_interp = []
        mass_dm_interp = []

        Rscale_baryons = np.zeros(self.times.size)
        Rscale_dm = np.zeros(self.times.size)

        # Get the predicted scale radius at the time of the observation
        # FIXME: the predicted scale radius is SMALLER than the observed effective radius...we were hoping for the opposite to be true...
        mstar_final = self.mass_stars[len(self.times)-1]
        _, R_final = baryons.sfr_rad_dist(self.rads, mstar_final)
        R_scaling = self.obs_rad_eff / R_final

        # Iterate over each time-step until when the sgrb occurred
        for ii, zz in enumerate(self.redz):

            sfr = self.gal_sfr[ii]
            mstar = self.mass_stars[ii]
            mgas = self.mass_gas[ii]
            mdm = self.mass_dm[ii]


            # Calculate exponential disk-profile (normalized to 1), used for both gas and SFR radial distributions since the gas follows the SFR
            if ii == 0:
                # for the first step, the mass only comes from the initial bout of SF...assume the scale radius is the same as the next step
                dt = self.times[ii] - self.formation_time
                disk_prof, Rscale_baryons[ii] = baryons.sfr_rad_dist(self.rads,  self.mass_stars[1], scaling=R_scaling)
                mass_stars_prof[ii, :] = self.formation_sfr * dt * disk_prof

            else:
                dt = self.times[ii] - self.times[ii-1]
                disk_prof, Rscale_baryons[ii] = baryons.sfr_rad_dist(self.rads,  mstar, scaling=R_scaling)
                # Add mass of stars and SF from previous time-steps to get the mass of stars at this timestep
                mass_stars_prof[ii, :] = mass_sfr_prof[ii-1, :] + mass_stars_prof[ii-1, :]


            # Calculate the SFR profile to be used for the next timestep
            mass_sfr_prof[ii, :] = sfr * dt * disk_prof

            # Distribute gas in disk
            mass_gas_prof[ii, :] = mgas * disk_prof

            # Distribute DM in NFW profile
            if self.dm_profile not in ['NFW']:
                raise NameError('DM profile {0:s} not recognized!'.format(self.dm_profile))
            mass_dm_prof[ii, :], Rscale_dm[ii] = halos.nfw_mass_prof(self.rads, mdm, zz, self.cosmo)

            # Create interpolations for each profile, with R=0 inserted
            mass_sfr_interp.append(interp1d(np.insert(self.rads, 0, 0), np.insert(mass_sfr_prof[ii], 0, mass_sfr_prof[ii][0])))
            mass_stars_interp.append(interp1d(np.insert(self.rads, 0, 0), np.insert(mass_stars_prof[ii], 0, mass_stars_prof[ii][0])))
            mass_gas_interp.append(interp1d(np.insert(self.rads, 0, 0), np.insert(mass_gas_prof[ii], 0, mass_gas_prof[ii][0])))
            mass_dm_interp.append(interp1d(np.insert(self.rads, 0, 0), np.insert(mass_dm_prof[ii], 0, mass_dm_prof[ii][0])))


        self.mass_sfr_prof = mass_sfr_prof
        self.mass_stars_prof = mass_stars_prof
        self.mass_gas_prof = mass_gas_prof
        self.mass_dm_prof = mass_dm_prof

        self.mass_sfr_interp = mass_sfr_interp
        self.mass_stars_interp = mass_stars_interp
        self.mass_gas_interp = mass_gas_interp
        self.mass_dm_interp = mass_dm_interp

        self.Rscale_baryons = Rscale_baryons
        self.Rscale_dm = Rscale_dm
        
        
        if self.VERBOSE:
            print("")
            names_rads = ['pc', 'kpc', 'Mpc']
            print_rads = np.array([1.0, 1e3, 1e6])*PC
            print("{:8s}  ".format("[Msol]"), end='')
            for nr in names_rads:
                print("  {:10s}".format(nr), end='')
            print("")
            vals = [mass_stars_prof[-1, :], mass_gas_prof[-1, :], mass_dm_prof[-1, :]]
            nams = ['stars', 'gas', 'dm']
            for vv, nv in zip(vals, nams):
                print("{:8s}: ".format(nv), end='')
                interp = utils.log_interp_1d(self.rads, np.cumsum(vv))
                for rr, nr in zip(print_rads, names_rads):
                    yy = interp(rr)/MSOL
                    print("  {:10s}".format("{:.1e}".format(yy)), end='')

                print("\n")
        
        return


    def calc_potentials_vs_time(self, method='astropy'):
        """Calculates the gravitational potentials of each component for all redshift steps using galpy
        The gas and stars are represented by a double exponential profile by default. 
        The DM is represented by a NFW profile. 
        Can onstruct potential in both astropy units (method=='astropy') or galpy's 'natural' units (method=='natural') for the purposes of interpolation. 
        """

        if method == 'astropy':
            print("Calculating galactic potentials at each redshift...\n")
        elif method == 'natural':
            print("Calculating galactic potentials at each redshift using galpy's natural units...\n")
        else:
            raise NameError('Method {0:s} for constructing the potential not recognized!'.format(method))

        # lists for saving combined potentials at each step
        stars_potentials = []
        gas_potentials = []
        dm_potentials = []
        full_potentials = []

        if self.disk_profile not in ['RazorThinExponential','DoubleExponential']:
            raise NameError('Disk profile {0:s} not recognized!'.format(self.disk_profile))
        

        # iterate over each time-step until when the sgrb occurred
        for ii, zz in enumerate(self.redz):


            # calculate the new amount of mass at this timestep
            if ii == 0:
                # first timestep formed from nothing
                if method=='astropy':
                    mstar = (self.mass_stars[ii]) * u.g
                    mgas = (self.mass_gas[ii]) * u.g
                    mdm = (self.mass_dm[ii]) * u.g
                elif method=='natural':
                    mstar = utils.Mcgs_to_nat(self.mass_stars[ii])
                    mgas = utils.Mcgs_to_nat(self.mass_gas[ii])
                    mdm = utils.Mcgs_to_nat(self.mass_dm[ii])

            else:
                if method=='astropy':
                    mstar = (self.mass_stars[ii] - self.mass_stars[ii-1]) * u.g
                    mgas = (self.mass_gas[ii] - self.mass_gas[ii-1]) * u.g
                    mdm = (self.mass_dm[ii] - self.mass_dm[ii-1]) * u.g
                elif method=='natural':
                    mstar = utils.Mcgs_to_nat(self.mass_stars[ii] - self.mass_stars[ii-1])
                    mgas = utils.Mcgs_to_nat(self.mass_gas[ii] - self.mass_gas[ii-1])
                    mdm = utils.Mcgs_to_nat(self.mass_dm[ii] - self.mass_dm[ii-1])

            # get the scale lengths for the baryons and halo at this redshift step
            if method=='astropy':
                rs_baryons = self.Rscale_baryons[ii] * u.cm
                rs_dm = self.Rscale_dm[ii] * u.cm
            elif method=='natural':
                rs_baryons = utils.Rcgs_to_nat(self.Rscale_baryons[ii])
                rs_dm = utils.Rcgs_to_nat(self.Rscale_dm[ii])


            if self.disk_profile=='RazorThinExponential':
                # for a razor-thin disk, the amplitude is mdisk / (2 * pi * rs**2)
                amp_stars = mstar / (4 * np.pi * rs_baryons**2)
                amp_gas = mgas / (4 * np.pi * rs_baryons**2)
                # construct the potentials at this timestep
                stars_potential = RazorThinExponentialDiskPotential(amp=amp_stars, hr=rs_baryons)
                gas_potential = RazorThinExponentialDiskPotential(amp=amp_gas, hr=rs_baryons)

            elif self.disk_profile=='DoubleExponential':
                # for a double exponential disk, the amplitude is mdisk / (2 * pi * rs**2 * rz)
                # NOTE: (2 * pi * rs**2 * 2 * zs) matches the code...
                amp_stars = mstar / (4 * np.pi * rs_baryons**2 * (self.z_scale*rs_baryons))
                amp_gas = mgas / (4 * np.pi * rs_baryons**2 * (self.z_scale*rs_baryons))
                stars_potential = DoubleExponentialDiskPotential(amp=amp_stars, hr=rs_baryons, hz=self.z_scale*rs_baryons)
                gas_potential = DoubleExponentialDiskPotential(amp=amp_gas, hr=rs_baryons, hz=self.z_scale*rs_baryons)


            # assume a nfw profile, amplitude is just the total dm mass
            amp_dm = mdm
            dm_potential = NFWPotential(amp=amp_dm, a=rs_dm)


            # add the potentials to the lists for each step
            stars_potentials.append(stars_potential)
            gas_potentials.append(gas_potential)
            dm_potentials.append(dm_potential)
            full_potentials.append([stars_potential,gas_potential,dm_potential])


        if method=='astropy':
            self.stars_potentials = stars_potentials
            self.gas_potentials = gas_potentials
            self.dm_potentials = dm_potentials
            self.full_potentials = full_potentials
        if method=='natural':
            self.stars_potentials_natural = stars_potentials
            self.gas_potentials_natural = gas_potentials
            self.dm_potentials_natural = dm_potentials
            self.full_potentials_natural = full_potentials


        return




    def calc_interpolated_potentials(self, interp_dirpath=None, ro=_ro, vo=_vo, multiproc=None):
        """Creates interpolants for combined potentials. 
        First, checks to see if interpolations already exist in directory `interp_dirpath`.
        If the interpolations do not exist in this path, they are generated and saved to this path. 
        To implement multiprocessing, specify an int for the argument 'multiproc'
        """
        
        print('Creating interpolation models of galactic potentials at each redshift...\n')

        if interp_dirpath:
            # if interpolation path is provided, see if the interpolated potentials exist
            pickle_path = interp_dirpath + '/' + self.name + '_' + str(self.NUM_TIME_STEPS) + 'T_' + str(self.NUM_RADS) + 'R_' + str(self.NUM_HEIGHTS) + 'Z_' + str(self.RGRID_MAX) + 'Rmax_' + str(self.ZGRID_MAX) + 'Zmax.pkl'
            if os.path.isfile(pickle_path):

                # read in the pickled file
                print('Pickled file with galactic interpolations found at: \n  {0:s}\n    reading in this data...\n'.format(interp_dirpath))
                interpolated_potentials = pickle.load(open(pickle_path, 'rb'))
                self.interpolated_potentials = interpolated_potentials
                return

            else:
                print('Pickled file with galactic interpolations not found at \n  {0:s}\n    constructing the interpolants...\n'.format(interp_dirpath))
                

        # convert Rs and Zs to natural units, calculate the grid
        ro_cgs = ro * u.kpc.to(u.cm)
        vo_cgs = vo * u.km.to(u.cm)
        rads = self.RADS_RANGE / ro_cgs
        heights = self.HEIGHTS_RANGE / ro_cgs
        
        rs = (*rads, self.NUM_RADS)
        logrs = (*np.log10(rads), self.NUM_RADS)
        zs = (*heights, self.NUM_HEIGHTS)
                

        # combine all the potentials
        combined_potentials=[]
        for idx, pot in enumerate(self.full_potentials_natural):
            combined_potentials.append(self.full_potentials_natural[:(idx+1)])


        # enable multiprocessing, if specified
        if multiproc:
            if multiproc=='max':
                mp = multiprocessing.cpu_count()
            else:
                mp = int(multiproc)

            pool = multiprocessing.Pool(mp)
            func = partial(interp, rgrid=logrs, zgrid=zs)

            start = time.time()
            print('Parallelizing interpolations over {0:d} cores...\n'.format(mp))
            interpolated_potentials = pool.map(func, combined_potentials)
            stop = time.time()
            print('   finished! It took {0:0.2f}s\n'.format(stop-start))
            

        # otherwise, do this in serial
        else:
            print('Interpolating potentials in serial...\n')
            interpolated_potentials=[]
            func = partial(interp, rgrid=logrs, zgrid=zs)
            for ii, data in enumerate(combined_potentials):

                start = time.time()
                ip = func(data)
                end = time.time()
                if self.VERBOSE == True:
                    print('   interpolated potential for step {0:d} (z={1:0.2f}) created in {2:0.2f}s...'.format(ii,self.redz[ii],end-start))

                interpolated_potentials.append(ip)



        self.interpolated_potentials = interpolated_potentials

        # if interp_dirpath was provided, dump the interpolations 
        if interp_dirpath:
            print('\nSaving the inteprolated potentials as pickles to the provided path...\n')
            pickle.dump(interpolated_potentials, open(pickle_path,'wb'))

        return




    def write(self, outdir):
        """Writes the galaxy data to a pickled file.
        """

        print("Writing galaxy data in directory '{0:s}'...\n".format(outdir))
        pickle.dump(self, open(outdir+'/galaxy.pkl', 'wb'))
        return
                
                
# define interpolating function
def interp(combined_potential, rgrid, zgrid, ro=_ro, vo=_vo):
    ip = interpRZPotential(combined_potential, rgrid=rgrid, zgrid=zgrid, logR=True, interpRforce=True, interpzforce=True, zsym=True, ro=ro, vo=vo)
    return ip


