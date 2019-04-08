"""Baryonic (galaxy) scaling relations
"""

import numpy as np
import astropy as ap
from . import utils

MSOL = ap.constants.M_sun.cgs.value    # gram
PC = ap.units.pc.to(ap.units.cm)       # cm
YR = 365.2425*24*3600                  # sec

KPC = 1e3 * PC    # cm
GYR = 1e9 * YR    # yr


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


def sfr_disk_rad(mstars, scaling):
    """Characteristic radii of star-forming disks.

    See: Nelson+2015, 1507.03999, Eq.5
    """
    mm = mstars / (1e10*MSOL)
    rs = 10**NELSON_1507_03999.aa * KPC
    rs *= np.power(mm, NELSON_1507_03999.bb)

    # scale the scale radius by the difference between the scale radius today and the effective radius
    # FIXME: are the scale radius and effective radius a 1:1 relationship?
    rs *= scaling

    return rs


def sfr_rad_dist(rads, mstar, scaling=1.0):
    """SFR radial distributions assuming exponential disk distributions.
    Scaling is determined by the difference between the scale radius and half-light radius in the present day and accounts for 'width' of SFMS
    """
    rs = sfr_disk_rad(mstar, scaling)

    # Density of star-formation distribution
    if rs>0:
        sfr_dens = np.exp(-rads/rs)
    else:
        sfr_dens = np.zeros(rads.shape)

    # Area of each disk-section
    area = utils.annulus_areas(rads)
    sfr = sfr_dens * area

    # Normalize
    if np.sum(sfr)>0:
        sfr /= np.sum(sfr)

    return sfr, rs


'''def sfr_main_seq(mass, redz, cosmo):
    """Star-forming Main-Sequence

    See: 1405.2041, Speagle+2014, Eq. 28
    """
    time = cosmo.age(redz).to('Gyr').value
    sfr_amp = -(6.51 - 0.11*time)      # Msol/yr
    gamma = 0.84 - 0.026*time
    if mass==0:
        sfr = 0.0
    else:
        sfr = gamma * np.log10(mass) + sfr_amp
        sfr = 10**sfr
    return sfr
'''

def gas_mass_from_stellar_mass(mstar):
    """Gas-Mass -- Stellar-Mass Relation

    See: Peeples+2014 [1310.2253], Eq.9
    """
    mgas = np.zeros_like(mstar)
    gas_frac = np.zeros_like(mstar)
    pos_vals = mstar!=0

    gas_frac[pos_vals] = -0.48 * np.log10(mstar[pos_vals]/MSOL) + 4.39
    mgas[pos_vals] = mstar[pos_vals] * np.power(10.0, gas_frac[pos_vals])
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





def sfr_main_seq(cosmo, mass, redz=None, time=None):
    """Star-forming Main-Sequence

    `mass` must be in grams!

    See: 1405.2041, Speagle+2014, Eq. 28
    """
    if time is None:
        time = cosmo.age(redz).cgs.value

    tt = (time / GYR)
    mm = (mass / MSOL)
    sfr_amp = -(6.51 - 0.11*tt)      # Msol/yr
    gamma = 0.84 - 0.026*tt
    sfr = gamma * np.log10(mm) + sfr_amp
    sfr = np.power(10, sfr)
    # sfr = gamma * np.log(mass/MSOL) + sfr_amp
    # Convert from [Msol/Yr] to [g/s]
    # sfr = np.exp(sfr) * MSOL / YR
    sfr = sfr * MSOL / YR
    return sfr



def arg_nearest(edges, value):
    # This is the index of the edge to the *right* of (i.e. above) each value
    idx = np.searchsorted(edges, value, side="left").clip(max=edges.size-1)
    # Find the distances to each nearest bin edge
    dist_lo = np.fabs(value - edges[idx-1])
    dist_hi = np.fabs(value - edges[idx])
    # If left ('lo') is nearer, mask=1, and we shift from the right edge to the left edge
    mask = (idx > 0) & ((idx == edges.size) | (dist_lo < dist_hi))
    idx = idx - mask
    return idx


def quick_sfr_history(cosmo, times, tquench, mass, wind=0.0):
    """Very coarse SFR and stellar-mass history to estimate mean stellar ages.

    Times should be a fairly-fine spacing of universe ages [seconds]
    tquench is the time SFR stops [seconds]
    mass is the final stellar-mass [grams]

    """

    ii = arg_nearest(times, tquench)
    # print("tau: {}, ii = {}, times[ii] = {}".format(tau/GYR, ii, times[ii]/GYR))
    mm = mass
    sfr = np.zeros_like(times)
    age = 0.0
    cnt = 0
    while (ii >= 0) and (mm > 0.0):
        tt = times[ii]
        dt = times[ii] - times[ii-1] if ii > 0 else times[ii+1] - times[ii]
        psi = sfr_main_seq(cosmo, mm, time=tt)
        psi = (1.0 - wind) * psi
        sfr[ii] = psi
        dm = sfr[ii] * dt
        mm -= dm
        age += dm * (times[-1] - tt)
        ii -= 1
        cnt += 1

    age /= mass
    # print("Age: {:.2f} [Gyr]".format(age/GYR))

    return sfr, age
