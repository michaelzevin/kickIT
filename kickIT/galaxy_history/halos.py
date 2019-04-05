"""Dark Matter Halo and Subhalo related code.
"""

import numpy as np
import astropy as ap
import scipy as sp
from . import utils, baryons

MSOL = ap.constants.M_sun.cgs.value    # gram
PC = ap.units.pc.to(ap.units.cm)       # cm
YR = 365.2425*24*3600                  # sec

KPC = 1e3 * PC    # cm
GYR = 1e9 * YR    # yr

from . cosmology import Cosmology

class Moster_1205_5807():

    DATA = {
        "M1": {
            "VALS": [11.590, 1.195],
            "SIGMA": [0.236, 0.353],
            "LOG": True
        },

        "N1": {
            "VALS": [0.0351, -0.0247],
            "SIGMA": [0.0058, 0.0069],
            "LOG": False
        },

        "B1": {
            "VALS": [1.376, -0.826],
            "SIGMA": [0.153, 0.225],
            "LOG": False
        },

        "G1": {
            "VALS": [0.608, 0.329],
            "SIGMA": [0.059, 0.173],
            "LOG": False
        }
    }

    def __init__(self, redz=0.0, store=False):
        MHALO_GRID_RANGE = [10.0, 20.0]   # log10(M_h/Msol)
        MHALO_GRID_SIZE_PER_DEX = 4      # per decade
        NUM_MC = 1000

        SIGMA_GRID_RANGE = [-3.0, 3.0]   # standard-deviations
        SIGMA_GRID_SIZE_PER_STD = 4      # per std

        # Construct a grid of halo-masses
        mhalo_grid_size = np.diff(MHALO_GRID_RANGE) * MHALO_GRID_SIZE_PER_DEX
        mhalo_grid = np.logspace(*MHALO_GRID_RANGE, mhalo_grid_size) * MSOL

        # MC Calculate stellar-masses from grid of halo-masses
        # shape: (N, M) for `N` halo-masses, and `M` MC samples
        mstar_from_mhalo_grid = self.stellar_mass(mhalo_grid[:, np.newaxis], size=NUM_MC)

        # Construct a grid of standard-deviation values
        sigma_grid_size = np.diff(SIGMA_GRID_RANGE) * SIGMA_GRID_SIZE_PER_STD
        sigma_grid = np.linspace(*SIGMA_GRID_RANGE, sigma_grid_size)
        # Convert standard-deviations to percentiles
        percentiles = sp.stats.norm.cdf(sigma_grid)
        # Calculate distribution of stellar masses
        # shape: (L, N) for `L` standard-deviation values and `N` halo masses
        mstar_dist = np.percentile(mstar_from_mhalo_grid, 100*percentiles, axis=-1)

        # Find the range of valid stellar-masses
        # Minimum value is the one reached by the *highest* percentile, at the lowest halo-mass
        # Maximum value is the one reached by the *lowest* percentile, at the highest halo-mass
        mstar_range = [np.max(mstar_dist[:, 0]), np.min(mstar_dist[:, -1])]
        # Construct a grid of stellar-masses spanning this range
        mstar_grid = np.logspace(*np.log10(mstar_range), mhalo_grid.size//2)

        # Interpolate to find halo-masses corresponding to stellar-masses at each percentile
        mhalo_dist = [sp.interpolate.interp1d(np.log10(pp), np.log10(mhalo_grid))(np.log10(mstar_grid))
                      for pp in mstar_dist]
        mhalo_dist = np.power(10, mhalo_dist)

        if store:
            self._mhalo_grid = mhalo_grid
            self._sigma_grid = sigma_grid
            self._mstar_grid = mstar_grid
            self._mhalo_dist = mhalo_dist
            self._mstar_dist = mstar_dist

        # Construct 2D interpolant between stellar-mass and standard deviation, and halo-mass
        # Note: input log(stellar-mass), output log(halo-mass)
        self._log_mstar_from_log_mhalo_sigma = sp.interpolate.interp2d(
            np.log10(mhalo_grid), sigma_grid, np.log10(mstar_dist))

        # Construct 2D interpolant between stellar-mass and standard deviation, and halo-mass
        # Note: input log(stellar-mass), output log(halo-mass)
        self._log_mhalo_from_log_mstar_sigma = sp.interpolate.interp2d(
            np.log10(mstar_grid), sigma_grid[::-1], np.log10(mhalo_dist))

        return

    def mhalo_from_mstar(self, mstar, sigma=0.0):
        log_mstar = np.log10(mstar)
        log_mhalo = self._log_mhalo_from_log_mstar_sigma(log_mstar, sigma)
        mhalo = np.power(10.0, log_mhalo)
        return mhalo

    def mstar_from_mhalo(self, mhalo, sigma=0.0):
        log_mhalo = np.log10(mhalo)
        log_mstar = self._log_mstar_from_log_mhalo_sigma(log_mhalo, sigma)
        mstar = np.power(10.0, log_mstar)
        return mstar

    @classmethod
    def param(cls, name, redz=0.0, size=None):
        data = cls.DATA[name]
        vals = data['VALS']
        sigma = data['SIGMA']
        log_flag = data['LOG']

        if size is not None:
            vals = [np.random.normal(pp, ss, size=size) for pp, ss in zip(vals, sigma)]

        par = vals[0] + vals[1] * redz / (1 + redz)
        if log_flag:
            par = np.power(10.0, par) * MSOL

        return par

    @classmethod
    def stellar_mass(cls, mhalo, redz=0.0, size=None):
        m1 = cls.param("M1", redz=redz, size=size)
        norm = cls.param("N1", redz=redz, size=size)
        beta = cls.param("B1", redz=redz, size=size)
        gamma = cls.param("G1", redz=redz, size=size)

        bterm = np.power(mhalo/m1, -beta)
        gterm = np.power(mhalo/m1, gamma)
        mstar = mhalo * 2 * norm / (bterm + gterm)
        return mstar

class KLYPIN_1411_4001:
    """Interpolate between redshifts and masses to find DM halo concentrations.

    Eq. 24 & Table 2
    """
    _redz = [0.00e+00, 3.50e-01, 5.00e-01, 1.00e+00, 1.44e+00,
             2.15e+00, 2.50e+00, 2.90e+00, 4.10e+00, 5.40e+00]
    _c0 = [7.40e+00, 6.25e+00, 5.65e+00, 4.30e+00, 3.53e+00,
           2.70e+00, 2.42e+00, 2.20e+00, 1.92e+00, 1.65e+00]
    _gamma = [1.20e-01, 1.17e-01, 1.15e-01, 1.10e-01, 9.50e-02,
              8.50e-02, 8.00e-02, 8.00e-02, 8.00e-02, 8.00e-02]
    _mass0 = [5.50e+05, 1.00e+05, 2.00e+04, 9.00e+02, 3.00e+02,
              4.20e+01, 1.70e+01, 8.50e+00, 2.00e+00, 3.00e-01]

    _zz = np.log10(1 + np.array(_redz))
    _lin_interp_c0 = utils.interp_1d(_zz, np.log10(_c0))
    _lin_interp_gamma = utils.interp_1d(_zz, np.log10(_gamma))
    _lin_interp_mass0 = utils.interp_1d(_zz, np.log10(_mass0)+np.log10(1e12 * MSOL / Cosmology.h))

    @classmethod
    def c0(cls, redz):
        xx = np.log10(1 + redz)
        yy = np.power(10.0, cls._lin_interp_c0(xx))
        return yy

    @classmethod
    def gamma(cls, redz):
        xx = np.log10(1 + redz)
        yy = np.power(10.0, cls._lin_interp_gamma(xx))
        return yy

    @classmethod
    def mass0(cls, redz):
        xx = np.log10(1 + redz)
        yy = np.power(10.0, cls._lin_interp_mass0(xx))
        return yy

    @classmethod
    def concentration(cls, mass, redz):
        c0 = cls.c0(redz)
        gamma = cls.gamma(redz)
        mass0 = cls.mass0(redz)
        f1 = np.power(mass/(1e12*MSOL/Cosmology.h), -gamma)
        f2 = 1 + np.power(mass/mass0, 0.4)
        conc = c0 * f1 * f2
        return conc


def nfw_dens_prof(rads, mhalo, redz, cosmo):
    """NFW DM Density profile.

    See: astro-ph/9611107, NFW 1997
    """
    if mhalo==0:
        dens = np.zeros(rads.shape)
        rs=0
        return dens, rs

    # Get Halo concentration
    conc = KLYPIN_1411_4001.concentration(mhalo, redz)
    log_c_term = np.log(1 + conc) - conc/(1+conc)

    # Critical over-density
    delta_c = (200/3) * (conc**3) / log_c_term
    # NFW density (*not* the density at the characteristic-radius)
    rho_s = cosmo.critical_density(redz).cgs.value * delta_c
    # scale-radius
    rs = mhalo / (4*np.pi*rho_s*log_c_term)
    rs = np.power(rs, 1.0/3.0)

    # Calculate NFW profile
    dens = (rads/rs)
    dens = dens * np.square(1 + dens)
    dens = rho_s / dens
    return dens, rs


def nfw_mass_prof(rads, mhalo, redz, cosmo):
    """Convert from NFW density profile to mass-profile in radial shells
    """
    dens, rs = nfw_dens_prof(rads, mhalo, redz, cosmo)
    vols = utils.shell_volumes(rads, relative=False)
    mass = dens * vols
    return mass, rs


# Construct interpolants for inverting halo-mass--stellar-mass relation
mass_halo_grid = np.logspace(6, 15, 1000) * MSOL
mass_stel_grid = baryons.halo_mass_to_stellar_mass(mass_halo_grid)
xx = np.log10(mass_stel_grid)
yy = np.log10(mass_halo_grid)
lin_interp = utils.interp_1d(xx, yy, fill_value=-np.inf)


def stellar_mass_to_halo_mass(mstar, relation='Guo', redz=None, sigma=None):
    """Inverted Guo+2010 relation.
    Also can call Moster_1205_5807 relation.
    """
    if relation not in ['Guo', 'Moster']:
        raise NameError('Stellar mass-Halo mass relation {0:s} not recognized!'.format(relation))

    mhalo = np.zeros_like(mstar)
    pos_vals = mstar!=0

    if relation == 'Guo':
        mhalo[pos_vals] = lin_interp(np.log10(mstar[pos_vals]))
        mhalo[pos_vals] = np.power(10.0, mhalo[pos_vals])

    elif relation == 'Moster':
        mos = Moster_1205_5807(store=True)
        mhalo[pos_vals] = mos.mhalo_from_mstar(mstar[pos_vals], sigma=sigma)


    return mhalo

