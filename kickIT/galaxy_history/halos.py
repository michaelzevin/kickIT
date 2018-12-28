"""Dark Matter Halo and Subhalo related code.
"""

import numpy as np

from . import MSOL, utils, baryons
from . cosmology import Cosmology


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


def stellar_mass_to_halo_mass(mstar):
    """Inverted Guo+2010 relation.
    """
    mhalo = lin_interp(np.log10(mstar))
    mhalo = np.power(10.0, mhalo)
    return mhalo
