"""Baryonic (galaxy) scaling relations
"""

import numpy as np

from . import utils, MSOL, KPC


class GUO_0909_4305:
    """Stellar-Mass -- Halo-Mass scaling relation.

    [Guo+2010 (0909.4305)](https://arxiv.org/abs/0909.4305) Eq. 3
    """
    c = 0.129
    M0 = (10**11.4) * MSOL
    alpha = 0.926
    beta = 0.261
    gamma = 2.440


class NELSON_1507_03999:
    """Star-Formation Disk scale radius as a function of stellar mass

    Eq. 5 & Table 5
    log10(r_s) = aa + bb * (log10(Mstar/Msol) - 10.0)
    """
    aa = 0.171  # ±0.008
    bb = 0.226  # ±0.022


def sfr_disk_rad(mstars):
    """Characteristic radii of star-forming disks.

    See: Nelson+2015, 1507.03999, Eq.5
    """
    mm = mstars / (1e10*MSOL)
    rs = 10**NELSON_1507_03999.aa * KPC
    rs *= np.power(mm, NELSON_1507_03999.bb)

    return rs


def sfr_rad_dist(rads, mstar):
    """SFR radial distributions assuming exponential disk distributions.
    """
    rs = sfr_disk_rad(mstar)

    # Density of star-formation distribution
    sfr_dens = np.exp(-rads/rs)
    # Area of each disk-section
    area = utils.annulus_areas(rads)

    sfr = sfr_dens * area
    # Normalize
    sfr /= np.sum(sfr)

    return sfr, rs


def sfr_main_seq(mass, redz, cosmo):
    """Star-forming Main-Sequence

    See: 1405.2041, Speagle+2014, Eq. 28
    """
    time = cosmo.age(redz).to('Gyr').value
    sfr_amp = -(6.51 - 0.11*time)      # Msol/yr
    gamma = 0.84 - 0.026*time
    sfr = gamma * np.log10(mass) + sfr_amp
    sfr = 10**sfr
    return sfr


def gas_mass_from_stellar_mass(mstar):
    """Gas-Mass -- Stellar-Mass Relation

    See: Peeples+2014 [1310.2253], Eq.9
    """
    gas_frac = -0.48 * np.log10(mstar/MSOL) + 4.39
    mgas = mstar * np.power(10.0, gas_frac)
    return mgas


def gas_mass_prof(rr, mstar, warm_frac=0.5):
    """Gas-Mass radial profile.

    See: Peeples+2014 [1310.2253] and Oey+2007 [0703033] for warm gas frac
    NOTE: this isn't used as of now
    """
    # Use gas-fraction (gas-mass/stellar-mass) relation to get gas-mass, account for warm-gas too
    gas_mass = gas_mass_from_stellar_mass(mstar) / warm_frac
    disk_rad = sfr_disk_rad(mstar)

    areas = utils.annulus_areas(rr)
    gas_dens = np.exp(-rr / disk_rad)

    prof = gas_dens * areas
    # normalize
    prof = gas_mass * prof / prof.sum()
    return prof


def halo_mass_to_stellar_mass(mhalo):
    """Stellar-Mass -- Halo-Mass relation.

    From: Guo+2010 [0909.4305], Eq. 3
    """
    M0 = GUO_0909_4305.M0
    t1 = np.power(mhalo/M0, -GUO_0909_4305.alpha)
    t2 = np.power(mhalo/M0, +GUO_0909_4305.beta)
    mstar = mhalo * GUO_0909_4305.c * np.power(t1 + t2, -GUO_0909_4305.gamma)
    return mstar
