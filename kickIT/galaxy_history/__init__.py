"""
"""
import warnings
import pdb

import numpy as np
import itertools
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy as ap
import astropy.units as u
import astropy.constants as C  # noqa

from galpy.potential import RazorThinExponentialDiskPotential, NFWPotential

VERBOSE = True

# Set units here (standard in cgs)
MSOL = u.M_sun.to(u.g)
PC = u.pc.to(u.cm)
YR = u.yr.to(u.s)

KPC = 1e3 * PC
GYR = 1e9 * YR
GMSOL = 1e9 * MSOL

# Import of local modules must come after constants above
from . import utils, baryons, halos, cosmology


class GalaxyHistory:
    """Class for calculating SFR and mass histories of a galaxy, based on limited observations.
    """

    # Redshift at which integration/data-arrays begin
    REDZ_BEG = 4.0
    # Number of time-steps between `REDZ_BEG` and redshift zero
    NUM_TIME_STEPS = 100
    # Radial points at which to construct the interpolant
    NUM_RADS = 100
    RADS = np.array([1e0, 1e6]) * PC

    def __init__(self, obs_mass_stars, obs_redz, obs_age_stars, obs_rad_eff, obs_gal_sfr, times=None, name=None):
        """All input parameters should be in CGS units!
        """
        # Initialize cosmology
        cosmo = cosmology.Cosmology()
        self.cosmo = cosmo

        # Read in observed parameters of the sGRB host
        self.obs_mass_stars = obs_mass_stars
        self.obs_redz = obs_redz
        self.obs_age_stars = obs_age_stars
        self.obs_rad_eff = obs_rad_eff
        self.obs_gal_sfr = obs_gal_sfr
        self.name = name

        # Construct sampling times if not provided
        if times is None:
            self.time_beg = cosmo.age(self.REDZ_BEG).cgs.value
            # End time is the time at which the galaxy is observed (calculated from redshift)
            self.time_end = cosmo.age(self.obs_redz).cgs.value
            self.time_dur = self.time_end - self.time_beg
            # times = np.linspace(self.time_beg, self.time_end, self.NUM_TIME_STEPS)  # sec
            times = np.logspace(*np.log10([self.time_beg, self.time_end]), self.NUM_TIME_STEPS)
        else:
            self.time_beg = times[0]
            self.time_end = times[-1]
            self.time_dur = times[-1] - times[0]
        self.times = times

        # Calculate array of redshifts corresponding to each `times` value (age of the universe)
        self.redz = cosmo.tage_to_z(self.times)
        if VERBOSE:
            textr = np.array([times.min(), times.max()])
            print("Times: {:3d} between [{:.1e}, {:.1e}] Gyr (z={:.1e}, {:.1e})".format(
                times.size, *textr/GYR, self.redz.max(), self.redz.min()))

        # Construct sampling of radii for constructing interpolants
        self.rads = np.logspace(*np.log10(self.RADS), self.NUM_RADS)

        # Calculate galaxy properties vs time
        self.calc_total_masses_vs_time()
        self.calc_mass_profiles_vs_time()

        # calculate galactic potentials vs time
        self.calc_potentials_vs_time()

        return


    def calc_total_masses_vs_time(self):
        """Calculate the total-mass of stars, gas and DM based on galaxy scaling relations
        """

        # Initialize mass and sfr arrays
        mass_stars = np.zeros_like(self.times)
        gal_sfr = np.zeros_like(self.times)

        # Find time and redshift of SFR peak from stellar-age of galaxy
        time_sfr_peak = self.time_end - self.obs_age_stars
        redz_sfr_peak = self.cosmo.tage_to_z(time_sfr_peak)
        if VERBOSE:
            print("Time of peak SFR: {:.2f} [Gyr] (z={:.2f})".format(
                time_sfr_peak/GYR, redz_sfr_peak))

        # Assume constant SFR from time of observation, back to SFR peak
        idx = (self.times >= time_sfr_peak)
        gal_sfr[idx] = self.obs_gal_sfr
        mass_stars[:] = self.obs_mass_stars
        mass_stars[idx] -= self.obs_gal_sfr * (self.time_end - self.times[idx])
        growth_quiescent = self.obs_gal_sfr * (self.time_end - time_sfr_peak)
        frac = growth_quiescent / mass_stars[-1]
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
                # timestep at which M=0
                self.tstep_0 = ii

                formation_idx = ii+1
                gal_sfr = gal_sfr[formation_idx:]
                mass_stars = mass_stars[formation_idx:]
                # adjust times and redshifts accordingly
                self.times = self.times[formation_idx:]
                self.redz = self.redz[formation_idx:]
                self.time_beg = self.times[0]
                self.time_dur = self.times[-1] - self.times[0]
                break

            gal_sfr[ii] = sfr
            mass_stars[ii] = mstar - dm
            ii -= 1


        # Initialize gas and DM arrays
        mass_gas = np.zeros_like(self.times)
        mass_dm = np.zeros_like(self.times)

        # Use scaling relations to get DM and gas masses from stellar
        mass_dm[:] = halos.stellar_mass_to_halo_mass(mass_stars[:])
        mass_gas[:] = baryons.gas_mass_from_stellar_mass(mass_stars[:])

        # Store arrays
        self.mass_stars = mass_stars
        self.mass_gas = mass_gas
        self.mass_dm = mass_dm
        self.gal_sfr = gal_sfr
        if VERBOSE:
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

        # Iterate over each time-step until when the sgrb occurred
        for ii, zz in enumerate(self.redz):

            sfr = self.gal_sfr[ii]
            mstar = self.mass_stars[ii]
            mgas = self.mass_gas[ii]
            mdm = self.mass_dm[ii]

            if ii==0:
                dt = self.times[ii] - self.tstep_0
            else:
                dt = self.times[ii] - self.times[ii-1]

            # Calculate exponential disk-profile (normalized to 1), used for both gas and SFR radial distributions since the gas follows the SFR
            disk_prof, Rscale_baryons[ii] = baryons.sfr_rad_dist(self.rads, mstar)

            # Mass profile of stars formed in this timestep
            mass_sfr_prof[ii, :] = sfr * dt * disk_prof
            # Add mass of stars from previous time-steps to get total
            mass_stars_prof[ii, :] = mass_sfr_prof[ii, :] + mass_stars_prof[ii-1, :]

            # Distribute gas in disk
            mass_gas_prof[ii, :] = mgas * disk_prof

            # Distribute DM in NFW profile
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

        # Get rid of the first step of times, redshifts, and masses at which there is no stellar mass yet (should just be the first index)
        
        
        if VERBOSE:
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

                print("")
        
        return


    def calc_potentials_vs_time(self):
        """Calculates the gravitational potentials of each component for all redshift steps using Galpy
        The gas and stars are represented by a razor-thin disk with an exponential profile. 
        The DM is represented by a NFW profile. 
        Uses astropy units to construct the potential. 
        """

        print('\nCalculating galactic potentials at each redshift...\n')

        # lists for saving combined potentials at each step
        stars_potentials = []
        gas_potentials = []
        dm_potentials = []
        full_potentials = []

        # Iterate over each time-step until when the sgrb occurred
        for ii, zz in enumerate(self.redz):

            # Calculate the new amount of mass at this timestep
            if ii == 0:
                # First timestep formed from nothing
                mstar = (self.mass_stars[ii]) * u.g
                mgas = (self.mass_gas[ii]) * u.g
                mdm = (self.mass_dm[ii]) * u.g

            else:
                mstar = (self.mass_stars[ii] - self.mass_stars[ii-1]) * u.g
                mgas = (self.mass_gas[ii] - self.mass_gas[ii-1]) * u.g
                mdm = (self.mass_dm[ii] - self.mass_dm[ii-1]) * u.g

            # Get the scale lengths for the baryons and halo at this redshift step
            rs_baryons = self.Rscale_baryons[ii] * u.cm
            rs_dm = self.Rscale_dm[ii] * u.cm

            # Get the amplitudes of the potential
            # For a razor-thin disk, this is Mdisk / (2 * pi * Rs**2)
            # For a NFW profile, this is just the total DM mass
            amp_stars = mstar / (2 * np.pi * rs_baryons**2)
            amp_gas = mgas / (2 * np.pi * rs_baryons**2)
            amp_dm = mdm

            # Construct the potentials at this timestep
            stars_potential = RazorThinExponentialDiskPotential(amp=amp_stars, hr=rs_baryons)
            gas_potential = RazorThinExponentialDiskPotential(amp=amp_gas, hr=rs_baryons)
            dm_potential = NFWPotential(amp=amp_dm, a=rs_dm)

            # add the potentials to the lists for each step
            stars_potentials.append(stars_potential)
            gas_potentials.append(gas_potential)
            dm_potentials.append(dm_potential)
            full_potentials.append([stars_potential,gas_potential,dm_potential])


        self.stars_potentials = stars_potentials
        self.gas_potentials = gas_potentials
        self.dm_potentials = dm_potentials
        self.full_potentials = full_potentials





    ### PLOTTING METHODS ###

    def plot_gal_mass_history(self):
        """Plot the mass-history (total masses & radial profiles of stars, gas, dm) vs time.
        """

        # Colors for each species
        cols = ['b', 'r', '0.5']
        # Names of each species (also used for retrieving parameters using `getattr`)
        pars = ['stars', 'gas', 'dm']

        # Construct Figure and Axes
        # -----------------------------------
        fig = plt.figure(figsize=[8, 8])
        axes = []

        def grid(ax):
            ax.grid(True, which='major', axis='both', c='0.5', alpha=0.25)
            ax.grid(True, which='minor', axis='both', c='0.5', alpha=0.1)

        gs = mpl.gridspec.GridSpec(
            2, 3, top=0.95, bottom=0.07, left=0.1, right=0.9, hspace=0.3, wspace=0.02)

        ax = fig.add_subplot(gs[0, :])
        ax.set(xscale='linear', xlabel='Time [Gyr]',
               yscale='log', ylabel='Mass $[M_\odot]$')
        grid(ax)
        if self.name is not None:
            ax.set_title(self.name)

        axes.append(ax)
        for ii in range(3):
            ax = fig.add_subplot(gs[1, ii])
            name = pars[ii]
            ax.set(xscale='log', xlabel='Radius [pc]',
                   yscale='log', ylabel='Mass $M(<r)$ $[M_\odot]$')
            ax.set_title(name, color=cols[ii])
            grid(ax)
            if ii == 1:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            elif ii == 2:
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_label_position('right')

            axes.append(ax)

        # Plot total-masses vs time
        # --------------------------------
        times = self.times/GYR
        rads = self.rads

        # Choose and setup axes
        ax = axes[0]

        # Plot each species
        for ii, (pp, cc) in enumerate(zip(pars, cols)):
            var_name = 'mass_' + pp
            yy = getattr(self, var_name)
            ax.plot(times, yy/MSOL, color=cc, lw=2.0, alpha=0.5)

        # Plot SFR on twin axis
        tw = ax.twinx()
        tw.set_ylabel('SFR $[M_\odot/yr]$ (dashed)')
        tw.plot(times, self.gal_sfr*YR/MSOL, 'b--', alpha=0.5, lw=2.0)

        # Plot radial-profiles over time
        # --------------------------------------------

        # Load data and colors for each time-step (for radial profiles)
        cmap = mpl.cm.get_cmap('coolwarm_r')
        colors = [cmap(ii) for ii in np.linspace(0.0, 1.0, times.size)]
        data = [getattr(self, 'mass_' + pp + '_prof') for pp in pars]
        ymax = 0.0
        # Each time-step
        for ii in range(times.size):
            cc = colors[ii]
            # Each of three species
            for jj, (ax, pp) in enumerate(zip(axes[1:], pars)):
                yy = data[jj][ii]
                # No mass at early times
                if not np.any(yy > 0.0):
                    continue

                ax.plot(rads/PC, yy/MSOL, color=cc, lw=1.0, alpha=0.5)

                # Plot cumulative mass distribution also
                zz = np.cumsum(yy)
                ax.plot(rads/PC, zz/MSOL, ls='--', color=cc, lw=1.0, alpha=0.25)

                # Store ymax values for ylim later
                ymax = np.max([ymax, zz.max()/MSOL])

        # Set ylimits
        ylim = [ymax/1e10, ymax]
        for ax in axes[1:]:
            ax.set_ylim(ylim)

        fname = 'gal_mass_history.pdf'
        if self.name is not None:
            fname = '{}_{}'.format(self.name.replace(' ', ''), fname)

        return fig, fname

    def plot_gal_scaling_relations(self):
        """Plot the scaling relations between different mass components for this galaxy.

        Not interesting.  Just for debugging (make sure it looks right).
        """

        # Create figure and axes
        # -------------------------------
        fig, axes = plt.subplots(figsize=[14, 5], ncols=3)
        plt.subplots_adjust(wspace=0.3)
        for ax in axes:
            ax.set(xscale='log', yscale='log')
            ax.grid(True, which='major', axis='both', c='0.5', alpha=0.25)
            ax.grid(True, which='minor', axis='both', c='0.5', alpha=0.1)

        stars = self.mass_stars/MSOL
        gas = self.mass_gas/MSOL
        dm = self.mass_dm/MSOL

        # Stars vs Gas
        ax = axes[0]
        ax.set(xlabel='Stellar Mass [$M_\odot$]',
               ylabel='Gas Fraction $F_g \equiv M_g / M_\star$')
        xx = stars
        yy = gas/stars
        ax.plot(xx, yy, 'r-', lw=2.0)

        # Stars vs. DM
        ax = axes[1]
        ax.set(xlabel='Halo Mass [$M_\odot$]',
               ylabel='Stellar Mass [$M_\odot$]')
        xx = dm
        yy = stars
        ax.plot(xx, yy, 'b-', lw=2.0)

        # Baryons (stars+gas) vs DM
        ax = axes[2]
        ax.set(xlabel='Halo Mass [$M_\odot$]',
               ylabel='Baryon Mass $M_b \equiv M_\star + M_g$ [$M_\odot$]')
        xx = dm
        yy = (stars + gas) / dm
        ax.plot(xx, yy, color='purple', lw=2.0)

        fname = 'gal_scaling_relations.pdf'
        if self.name is not None:
            fname = '{}_{}'.format(self.name.replace(' ', ''), fname)

        return fig, fname
