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
import multiprocessing
from functools import partial

import scipy as sp

import astropy as ap
import astropy.units as u
import astropy.constants as C

from galpy.potential import RazorThinExponentialDiskPotential, DoubleExponentialDiskPotential, NFWPotential
from galpy.potential import interpRZPotential

# Import of local modules must come after constants above
from .. import utils
from . import baryons, halos, cosmology


# Specify initial redshift and number of timesteps
REDZ_BEG = 4.0
NUM_TIME_STEPS = 100

# Specify parameters for the grid, in kpc
NUM_RADS = 300
RGRID_MAX = 1e3
RADS_RANGE = np.array([1e-4,RGRID_MAX])*u.kpc

NUM_HEIGHTS = 100
ZGRID_MAX = 100
HEIGHTS_RANGE = np.array([0,ZGRID_MAX])*u.kpc

# Specify time sampling variables
MAX_DT = 0.1 * u.Gyr
MAX_REDZ = 10.0
MAX_STEPS = 300
MAX_DM_FRAC = 0.01

# Tolerance in choosing SF main sequence
STAR_AGE_RTOL = 0.02

VERBOSE = True


class GalaxyHistory:
    """Class for calculating SFR and mass histories of a galaxy, based on limited observations.
    """


    def __init__(self, obs_props, disk_profile, dm_profile, smhm_relation='Guo', smhm_sigma=0.0, bulge_profile=None, z_scale=None, differential_prof=False):
        """Units are kept as astropy quantities for clarity!
        """

        # Read in observed parameters of the sGRB host
        self.obs_props = obs_props
        self.disk_profile = disk_profile
        self.dm_profile = dm_profile
        self.smhm_relation = smhm_relation
        self.smhm_sigma = smhm_sigma
        self.bulge_profile = bulge_profile
        self.z_scale = z_scale
        self.differential_prof = differential_prof

        # Initiate cosmology
        self.cosmo = cosmology.Cosmology()

        # Store tolerance for SFR calculation
        self.STAR_AGE_RTOL = STAR_AGE_RTOL

        # Construct sampling times
        self.time_beg = self.cosmo.age(REDZ_BEG)
        # End time is the time at which the galaxy is observed (calculated from redshift)
        self.time_end = self.cosmo.age(obs_props['redz'])
        self.time_dur = self.time_end - self.time_beg

        # Construct sampling of radii for constructing interpolants (flat in log)
        self.rads = np.logspace(*np.log10(RADS_RANGE.value), NUM_RADS)*u.kpc

        # Calculate galaxy component masses vs time
        self.calc_total_masses_vs_time()

        # Calculate the component mass profiles vs time (used for checking against galpy)
        self.calc_mass_profiles_vs_time()

        # Calculate the SFR weights at each timestep
        self.calc_sfr_weights()

        # Calculate galactic potentials vs time
        self.calc_potentials_vs_time(self.differential_prof)
        self.calc_potentials_vs_time(self.differential_prof, method='natural')

        return



    def calc_total_masses_vs_time(self):
        """Calculate the total-mass of stars, gas and DM based on galaxy scaling relations
        """
        # Maximum time-step interval
        MIN_TIME = self.cosmo.age(MAX_REDZ)

        # Maximum fractional change in mass
        cosmo = self.cosmo
        time_end = self.time_end
        mstar_end = self.obs_props['mass_stars']

        # Construct instance for calculating SFR Main-sequence relation (units in cgs)
        sfr_ms = baryons.SFR_MS_Speagle_1405_2041()
        MIN_MSTAR = 2*sfr_ms._MIN_MSTAR*u.g.to(u.Msun)*u.Msun

        # Find time and redshift of SFR peak from stellar-age of galaxy
        time_sfr_peak = time_end - self.obs_props['age_stars']
        redz_sfr_peak = float(cosmo.tage_to_z(time_sfr_peak.cgs.value))
        self.redz_sfr_peak = redz_sfr_peak
        self.time_sfr_peak = time_sfr_peak

        if VERBOSE:
            print("Time of peak SFR: {0:0.2f} (z={1:0.2f})".format(time_sfr_peak, redz_sfr_peak))

        # Determine the approximate quenching time; use a grid-search to find quenching time and location in SFR--stellar-mass space to produce a galaxy with the correct properties
        time_quench, sfr_ms_sigma, sfr_age, frac_err = self.calc_sfr_params()
        time_quench *= u.s.to(u.Gyr)*u.Gyr
        sfr_age *= u.s.to(u.Gyr)*u.Gyr
        redz_quench = cosmo.tage_to_z(time_quench.cgs.value)
        print("SFR history solution: Tq = {0:.1e} (z={1:.2f}), sigma = {2:.2f}".format(time_quench, redz_quench, sfr_ms_sigma))
        print("   Mean Stellar Age = {0:.2e} vs. {1:.2e}  (error: {2:.2e})\n".format(sfr_age, self.obs_props['age_stars'], frac_err))
        self.time_quench = time_quench
        self.redz_quench = redz_quench

        # Initialize arrays
        print("Constructing quiescent phase...")
        q_times = np.arange(time_end.value, time_quench.value, -MAX_DT.value)[::-1]*u.Gyr
        q_redz = cosmo.tage_to_z(q_times.cgs.value)
        print("{0} steps between {1:0.2f} (z={2:0.2f}), {3:0.2f} (z={4:0.2f})".format(q_times.size, q_times[0], q_redz[0], q_times[-1], q_redz[-1]))
        q_gal_sfr = self.obs_props['gal_sfr'] * np.ones_like(q_times.value)
        q_mass_stars = mstar_end - q_gal_sfr * (time_end - q_times)

        growth_quiescent = q_mass_stars[-1] - q_mass_stars[0]
        frac = growth_quiescent / q_mass_stars[-1]
        print("During quiescent state, galaxy grew by {0:0.2e} (dM={1:0.2e})\n".format(frac, growth_quiescent))
        # For consistent model, the fractional growth during "quiescent" phase should be negligible
        if frac > 0.05:
            warnings.warn("Significant galaxy growth during 'quiescent' phase!")

        times = []
        sfr = []
        mass_stars = []

        prev_mstar = q_mass_stars[0]
        prev_time = q_times[0]
        prev_sfr = q_gal_sfr[0]

        temp_time = prev_time
        temp_mstar = prev_mstar
        ii = 0
        while (temp_time > MIN_TIME) and (temp_mstar > MIN_MSTAR):
            temp_dt = MAX_DT
            temp_time = prev_time - temp_dt
            if temp_time < 0.0:
                temp_time = MIN_TIME
                temp_dt = prev_time - temp_time

            temp_mstar = prev_mstar - temp_dt * prev_sfr
            if temp_mstar <= 0.0:
                temp_dt = (prev_mstar - MIN_MSTAR)/prev_sfr
                temp_time = prev_time - temp_dt
                temp_mstar = prev_mstar - temp_dt * prev_sfr

            try:
                temp_sfr = sfr_ms.sfr_from_mstar(temp_mstar.cgs.value, temp_time.cgs.value, sfr_ms_sigma)[0] * u.g.to(u.Msun)/u.s.to(u.yr) * u.Msun/u.yr
            except ValueError:
                print("temp_mstar = {0:.2e}".format(tmp_mstar))
                print("temp_time = {0:.2e}".format(temp_time))
                print("sfr_ms_sigma = {0:.2e}".format(sfr_ms_sigma))
                raise
            if not np.isfinite(temp_sfr):
                raise ValueError("Infinite `temp_sfr`!")

            temp_dm = temp_sfr * temp_dt.to(u.yr)

            if temp_dm/mstar_end > MAX_DM_FRAC:
                temp_dt = (0.98*MAX_DM_FRAC*mstar_end / temp_sfr).to(u.Gyr)
                temp_time = prev_time - temp_dt
                temp_dm = temp_sfr * temp_dt

                if temp_dm/self.obs_props['mass_stars'] > MAX_DM_FRAC:
                    raise ValueError("Mass-change STILL too large!")

            temp_mstar = prev_mstar - temp_dm
            temp_sfr = sfr_ms.sfr_from_mstar(temp_mstar.cgs.value, temp_time.cgs.value, sfr_ms_sigma)[0] * (u.g.to(u.Msun)/u.s.to(u.yr))*(u.Msun/u.yr)
            temp_sfr = 0.5 * (temp_sfr + prev_sfr)
            temp_dm = temp_sfr * temp_dt.to(u.yr)

            if temp_dm/mstar_end > MAX_DM_FRAC:
                temp_dt = (MAX_DM_FRAC*mstar_end / temp_sfr).to(u.Gyr)
                temp_time = prev_time - temp_dt
                temp_dm = temp_sfr * temp_dt.to(u.yr)

            if prev_mstar - temp_dm < MIN_MSTAR:
                temp_dm = (prev_mstar - MIN_MSTAR)
                temp_dt = (temp_dm / temp_sfr).to(u.Gyr)
                temp_time = prev_time - temp_dt

            temp_mstar = prev_mstar - temp_dm
            times.append(temp_time.value)
            sfr.append(temp_sfr.value)
            mass_stars.append(temp_mstar.value)

            prev_mstar = temp_mstar
            prev_sfr = temp_sfr
            prev_time = temp_time

            ii += 1
            if ii > MAX_STEPS:
                raise RuntimeError("Excceded maximum steps!")

        times = times[::-1]
        sfr = sfr[::-1]
        mass_stars = mass_stars[::-1]

        times = np.append(times, q_times.value)
        sfr = np.append(sfr, q_gal_sfr.value)
        mass_stars = np.append(mass_stars, q_mass_stars.value)
        if np.any(np.diff(times) < 0.0):
            raise ValueError("BAD TIMES!")
        if np.any(np.diff(mass_stars) < 0.0):
            raise ValueError("BAD MASSES!")

        # convert to astropy units
        times *= u.Gyr
        sfr *= u.Msun/u.yr
        mass_stars *= u.Msun

        # Use scaling relations to get DM and gas masses from stellar
        mass_dm = (halos.stellar_mass_to_halo_mass(mass_stars.cgs.value, relation=self.smhm_relation, sigma=self.smhm_sigma)*u.g).to(u.Msun)
        mass_gas = (baryons.gas_mass_from_stellar_mass(mass_stars.cgs.value)*u.g).to(u.Msun)

        # Store arrays
        self.mass_stars = mass_stars
        self.mass_gas = mass_gas
        self.mass_dm = mass_dm
        self.sfr = sfr
        self.times = times
        self.redz = cosmo.tage_to_z(times.cgs.value)
        if VERBOSE:
            print("Final total masses: s={0:0.1e}, g={1:0.1e}, dm={2:0.1e}".format(mass_stars[-1], mass_gas[-1], mass_dm[-1]))
            print("Final SFR={0:0.1e}".format(sfr[-1]))

            star_ages = (times[-1] - times[1:])
            star_weights = np.diff(mass_stars)
            star_weights = np.nan_to_num(star_weights)

            ave_age = np.average(star_ages, weights=star_weights)
            idx = np.argsort(star_ages)
            csum = np.cumsum(star_ages[idx] * star_weights[idx]) / np.sum(star_weights)
            med_age = np.interp(0.5, csum, star_ages[idx])
            print("Stellar Ages: average {0:0.2f}, median {1:0.2f}\n".format(ave_age, med_age))

        return





    def calc_mass_profiles_vs_time(self):
        """Calculate the radial mass profiles vs time
        The profile at each timestep will be a linear interpolant
        """
        # Initialize arrays for galaxy matter profiles over time
        prof_shape = (self.times.size, self.rads.size)

        # All of these quantities are in units of mass (Msun), such that the total mass out to a given radius is simply the sum out to that index.
        # Note that the SFR, stars and gas are in a thin-disk while the DM is roughly spherical.
        mass_sfr_prof = np.zeros(prof_shape)
        mass_stars_prof = np.zeros(prof_shape)
        mass_gas_prof = np.zeros(prof_shape)
        mass_dm_prof = np.zeros(prof_shape)

        Rscale_baryons = np.zeros(self.times.size)
        Rscale_dm = np.zeros(self.times.size)

        # --- Compare the observed scale radius with the predicted scale radius, to account for the "width" of the star-formation main sequence
        _, R_final = baryons.sfr_rad_dist(self.rads.cgs.value, self.obs_props['mass_stars'].cgs.value)
        R_final = (R_final*u.cm).to(u.kpc)
        R_scaling = (self.obs_props['rad_eff'] / R_final).value

        dt = 0 * u.yr

        # --- Iterate over each time-step until when the sgrb occurred
        for ii, zz in enumerate(self.redz):

            sfr = self.sfr[ii]
            mstar = self.mass_stars[ii]
            mgas = self.mass_gas[ii]
            mdm = self.mass_dm[ii]

            # Skips times before galaxy 'formed' (i.e. when it had negligible mass/SFR)
            if ii == 0 or sfr <= 0.0 or mstar <= 0.0:
                continue

            # --- Calculate exponential disk-profile (normalized to 1), used for both gas and SFR radial distributions since the gas follows the SFR
            dt = (self.times[ii] - self.times[ii-1]).to(u.yr)
            disk_prof, Rscale_baryons[ii] = baryons.sfr_rad_dist(self.rads.cgs.value,  mstar.cgs.value, scaling=R_scaling)
            Rscale_baryons[ii] *= u.cm.to(u.kpc)

            # --- Add the new mass of stars from the ongoing SF
            mass_stars_prof[ii, :] = mass_sfr_prof[ii-1, :] + mass_stars_prof[ii-1]

            # --- Calculate the SFR profile at this timestep, which is used for the next timestep
            mass_sfr_prof[ii, :] = sfr * dt * disk_prof

            # --- Distribute gas in disk
            mass_gas_prof[ii, :] = mgas * disk_prof

            # --- Distribute DM in NFW profile
            if self.dm_profile not in ['NFW']:
                raise NameError('DM profile {0:s} not recognized!'.format(self.dm_profile))

            mass_dm_prof[ii, :], Rscale_dm[ii] = halos.nfw_mass_prof(self.rads.cgs.value, mdm.cgs.value, zz, self.cosmo)
            mass_dm_prof[ii, :] *= u.g.to(u.Msun)
            Rscale_dm[ii] *= u.cm.to(u.kpc)


        # --- store the profiles
        self.mass_sfr_prof = mass_sfr_prof * u.Msun
        self.mass_stars_prof = mass_stars_prof * u.Msun
        self.mass_gas_prof = mass_gas_prof * u.Msun
        self.mass_dm_prof = mass_dm_prof * u.Msun

        self.Rscale_baryons = Rscale_baryons * u.kpc
        self.Rscale_dm = Rscale_dm * u.kpc

        return


    def calc_sfr_weights(self):
        """Calculate the SFR weighting used to sample t0.

        The first step with postive SFR is set to 0, since there is no mass profile at this time.
        """

        # --- calculate the dts for the SF weight calculation, setting last step to 0
        dts = (self.times[1:]-self.times[:-1]).value
        dts = np.append(dts, 0.0) * u.Gyr

        # --- teporarily set sfr to 0 where there is no mass
        nomass_idxs = np.argwhere(self.mass_stars == 0)

        # --- find the change in masses at this step
        dms = self.sfr * (dts.to(u.yr))
        dms[nomass_idxs] = 0

        # --- calculate the weights
        self.sfr_weights = (dms / np.sum(dms)).value

        return



    def calc_potentials_vs_time(self, differential=False, method='astropy'):
        """Calculates the gravitational potentials of each component for all redshift steps using galpy
        The gas and stars are represented by a double exponential profile by default. 
        The DM is represented by a NFW profile. 
        Can construct potential in both astropy units (method=='astropy') or galpy's 'natural' units (method=='natural') for the purposes of interpolation. 

        Stars can be calculated using a differential mass profile with varying scale radii, but in this case it is best to interpolate the potentials first
        """

        if method == 'astropy':
            print("Calculating galactic potentials at each redshift using astropy units...\n")
        elif method == 'natural':
            print("Calculating galactic potentials at each redshift using galpy's natural units...\n")
        else:
            raise NameError('Method {0:s} for constructing the potential not recognized!'.format(method))

        # lists for saving potentials at each step
        stars_potentials = []
        gas_potentials = []
        dm_potentials = []
        full_potentials = []

        if self.disk_profile not in ['RazorThinExponential','DoubleExponential']:
            raise NameError('Disk profile {0:s} not recognized!'.format(self.disk_profile))

        # iterate over each time-step until when the sgrb occurred
        for ii, zz in enumerate(self.redz):

            # --- get the gas and DM masses at this step
            if method=='astropy':
                mgas = self.mass_gas[ii]
                mdm = self.mass_dm[ii]
            elif method=='natural':
                mgas = utils.Mphys_to_nat(self.mass_gas[ii])
                mdm = utils.Mphys_to_nat(self.mass_dm[ii])

            # --- if differential stellar potential not being used, just take the total stellar mass at each timestep
            # --- this is also done for the first differential timestep
            if (differential==False) or (ii == 0):
                if method=='astropy':
                    mstar = self.mass_stars[ii]
                elif method=='natural':
                    mstar = utils.Mphys_to_nat(self.mass_stars[ii])
            else:
                if method=='astropy':
                    mstar = self.mass_stars[ii] - self.mass_stars[ii-1]
                elif method=='natural':
                    mstar = utils.Mphys_to_nat(self.mass_stars[ii] - self.mass_stars[ii-1])

            # --- get the scale lengths for the baryons and halo at this redshift step
            if method=='astropy':
                rs_baryons = self.Rscale_baryons[ii]
                rs_dm = self.Rscale_dm[ii]
                # if galaxy hasn't formed yet, give the potentials neglible scale sizes to avoid dividing by 0
                if rs_baryons==0:
                    rs_baryons = 1e-10 * rs_baryons.unit
                if rs_dm==0:
                    rs_dm = 1e-10 * rs_dm.unit
            elif method=='natural':
                rs_baryons = utils.Rphys_to_nat(self.Rscale_baryons[ii])
                rs_dm = utils.Rphys_to_nat(self.Rscale_dm[ii])
                # if galaxy hasn't formed yet, give the potentials neglible scale sizes to avoid dividing by 0
                if rs_baryons==0:
                    rs_baryons = 1e-10
                if rs_dm==0:
                    rs_dm = 1e-10


            # --- construct the stellar and gas potentials

            if self.disk_profile=='RazorThinExponential':
                # for a razor-thin disk, the amplitude is mdisk / (2 * pi * rs**2)
                amp_stars = mstar / (4 * np.pi * rs_baryons**2)
                amp_gas = mgas / (4 * np.pi * rs_baryons**2)
                # construct the potentials at this timestep
                stars_potential = RazorThinExponentialDiskPotential(amp=amp_stars, hr=rs_baryons)
                gas_potential = RazorThinExponentialDiskPotential(amp=amp_gas, hr=rs_baryons)

            elif self.disk_profile=='DoubleExponential':
                # for a double exponential disk, the amplitude is mdisk / (2 * pi * rs**2 * 2*rz)
                amp_stars = mstar / (4 * np.pi * rs_baryons**2 * (self.z_scale*rs_baryons))
                amp_gas = mgas / (4 * np.pi * rs_baryons**2 * (self.z_scale*rs_baryons))
                stars_potential = DoubleExponentialDiskPotential(amp=amp_stars, hr=rs_baryons, hz=self.z_scale*rs_baryons)
                gas_potential = DoubleExponentialDiskPotential(amp=amp_gas, hr=rs_baryons, hz=self.z_scale*rs_baryons)



            # --- construct the DM potentials

            amp_dm = mdm
            dm_potential = NFWPotential(amp=amp_dm, a=rs_dm)


            # --- add the potentials to the lists for each step
            stars_potentials.append(stars_potential)
            gas_potentials.append(gas_potential)
            dm_potentials.append(dm_potential)

            # --- if differential is specified, we use *all* the stellar profiles up to this point
            if differential==True:
                combined_potentials = stars_potentials[:]
                combined_potentials.extend([gas_potential, dm_potential])
            else:
                combined_potentials = [stars_potential,gas_potential,dm_potential]
            full_potentials.append(combined_potentials)


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


    def write(self, outdir, label=None):
        """Writes the galaxy data to a pickled file.
        """

        print("Writing galaxy data in directory {0:s}...\n".format(outdir))
        if label:
            savepath = outdir+'/'+label+'.pkl'
        else:
            savepath = outdir+'/galaxy.pkl'
        pickle.dump(self, open(savepath, 'wb'))
        return





    def calc_sfr_params(self):
        """
        Grid search to find quiescence time and "sigma" on SFR relation to produce galaxy props.

        We use a SFR main-sequence (MS) parametrized by a stellar-mass, age-of-the-universe, and also a "sigma" parameter quantifying the location relative to the mean relationship. This sigma parameter, in addition to the quenching time (when the galaxy leaves the MS), need to be calculated.

        Here, a grid of quenching-times and sigma-values are considered, and star-formation histories are constructed (in parallel) for each. Average stellar-ages are then calculated, and the best fitting value is chosen. This worked much better than using an actual minimization routine.

        Units for this function are in cgs.
        """
        time_end = self.cosmo.age(self.obs_props['redz']).cgs.value
        gal_age_stars = self.obs_props['age_stars'].cgs.value
        target_time = time_end - gal_age_stars
        sfr_end = self.obs_props['gal_sfr'].to(u.g/u.s).value

        # Number of values in each dimension to use
        NSIGMA = 100
        NQUENCH = 100
        NTIMES = 200

        # Construct an instance for calculating SFR values
        sfr_ms = baryons.SFR_MS_Speagle_1405_2041()

        # Lowest stellar-mass at which interpolation functions for the MS will work
        MIN_MASS = sfr_ms._MIN_MSTAR

        # Construct parameter arrays
        times = np.linspace(0.0, time_end, NTIMES)
        quench = np.linspace(target_time, time_end, NQUENCH)
        sigma = np.linspace(-2.0, 2.0, NSIGMA)

        # Arrays to store values
        sfh = np.zeros((NQUENCH, NSIGMA, NTIMES))
        mass = np.zeros_like(sfh)
        dmh = np.zeros_like(sfh)

        dt = np.append([0.0], np.diff(times))
        # initialize last time-step to be observed mass of galaxy
        mass[..., -1] = self.obs_props['mass_stars'].cgs.value

        # Iterate backwards in time to de-construct the galaxy following the SFR MS
        for ii, tt in utils.renumerate(times):
            # (Q,S,)
            mm = mass[..., ii]
            # Sources above the minimum mass (galaxies below this mass stop evolving)
            idx = (mm > MIN_MASS)

            # Find 'active' galaxies still on the main-sequence
            act = (tt < quench[:, np.newaxis]) & idx
            num_act = np.count_nonzero(act)
            if num_act > 0:
                ss = sigma[np.newaxis, :]
                args = mm, tt*np.ones_like(mm), ss*np.ones_like(mm)
                args = [aa[act].flatten() for aa in args]
                # Calculate and store star-formation-rate for all galaxies at this time
                sfh[act, ii] = sfr_ms.sfr_from_mstar(*args)

            # Find 'passive' (i.e. quiescent) galaxies, off the main-sequence
            psv = (tt >= quench[:, np.newaxis]) & idx
            num_psv = np.count_nonzero(psv)
            if num_psv > 0:
                # Set the SFR to the quiescent (observed) value
                sfh[psv, ii] = sfr_end

            # Increment the stellar-mass of the galaxy
            if ii > 0:
                dm = sfh[..., ii] * dt[ii]
                dmh[..., ii] = dm
                mass[..., ii-1] = mm - dm

        # Calculate average times (age of the universe) of star-formation
        tsf = np.sum(dmh * times[np.newaxis, np.newaxis, :], axis=-1) / np.sum(dmh, axis=-1)

        if np.all(tsf < target_time) or np.all(tsf > target_time):
            raise RuntimeError("Average times never cross target: {:.2e}".format(target_time*u.s.to(u.Gyr)))

        # Calculate the fractional error in stellar-age relative to target galaxy age
        frac_err = np.fabs(((time_end - tsf) - gal_age_stars)/gal_age_stars)
        # Find region of parameter space within target tolerance of true value
        idx = (frac_err < self.STAR_AGE_RTOL)

        # Choose 'best' prior values on parameters
        s0 = 0.0
        q0 = np.mean([np.min(quench), np.max(quench)])

        # Construct a cost-function to choose the best parameter combination
        def cost_func(vals, ref):
            cost = np.fabs(vals - ref)
            cost = cost / np.diff([np.min(cost), np.max(cost)])
            return cost

        cost_quench = cost_func(quench, q0)[:, np.newaxis]
        cost_sigma = cost_func(sigma, s0)[np.newaxis, :]
        cost = np.sqrt(cost_quench**2 + cost_sigma**2)

        # Find the "optimal" (w.r.t. the cost-function) indices within the error-tolerance for ages
        valid = np.ma.masked_array(cost, mask=(~idx))
        argmin = np.argmin(valid)
        ii, jj = np.unravel_index(argmin, valid.shape)

        sol_tsf = tsf[ii, jj]
        sol_err = frac_err[ii, jj]
        sol_age = time_end - sol_tsf

        if sol_err > self.STAR_AGE_RTOL:
            raise ValueError("Solution accuracy did not reach tolerance (shouldn't happen)!")

        # Extract values corresponding to indices
        tq = quench[ii]
        sig = sigma[jj]

        return tq, sig, sol_age, sol_err
