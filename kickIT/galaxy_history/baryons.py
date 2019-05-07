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


# --- Classes for stellar mass - halo mass relation

class Moster(utils.Outlier):

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

        # Construct a grid of standard-deviation values
        sigma_grid_size = np.diff(SIGMA_GRID_RANGE) * SIGMA_GRID_SIZE_PER_STD
        sigma_grid = np.linspace(*SIGMA_GRID_RANGE, sigma_grid_size)

        xgrid = np.log10(mhalo_grid)
        super().__init__(xgrid, sgrid=sigma_grid, nmc=NUM_MC, store=store)

        return

    def mhalo_from_mstar(self, mstar, sigma=0.0):
        log_mstar = np.log10(mstar)
        log_mhalo = self.ys_to_x(log_mstar, sigma)
        mhalo = np.power(10.0, log_mhalo)
        return mhalo

    def mstar_from_mhalo(self, mhalo, sigma=0.0):
        log_mhalo = np.log10(mhalo)
        log_mstar = self.xs_to_y(log_mhalo, sigma)
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
    def function(cls, log_mhalo, redz=0.0, size=None):
        mhalo = np.power(10.0, log_mhalo)
        mstar = cls.stellar_mass(mhalo, redz=redz, size=size)
        mstar = np.log10(mstar)
        return mstar

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



# --- Class for star formation main sequence

class SFR_MS_Speagle_1405_2041(utils.OutlierND):
    '''Star Formation Rate Main-Sequence from Speagle+2014
    Eq.28
    log(psi) = (0.84±0.02 - 0.026±0.003 t) * log(Mstar) - (6.51±0.24 - 0.11±0.03 t)
    log(psi) = (A1 - A2*t)*log(Mstar) - (B1 - B2*t)
    '''

    DATA = {
        "A1": [0.84, 0.02],
        "A2": [0.026, 0.003],

        "B1": [6.51, 0.24],
        "B2": [0.11, 0.03],
    }

    def __init__(self, store=True):
        NUM_MC = 1000

        MSTAR_GRID_RANGE = [5.0, 13.0]   # log10(M_h/Msol)
        MSTAR_GRID_SIZE_PER_DEX = 10      # per decade

        TIME_GRID_RANGE = [0.0, 13.7]   # Gyr
        TIME_GRID_SIZE_PER_VAL = 6

        SIGMA_GRID_RANGE = [-3.0, 3.0]   # standard-deviations
        SIGMA_GRID_SIZE_PER_STD = 10      # per std

        # Construct a grid of halo-masses
        mstar_grid_size = np.diff(MSTAR_GRID_RANGE) * MSTAR_GRID_SIZE_PER_DEX
        mstar_grid = np.logspace(*MSTAR_GRID_RANGE, mstar_grid_size) * MSOL

        time_grid_size = np.diff(TIME_GRID_RANGE) * TIME_GRID_SIZE_PER_VAL
        # time_grid = np.logspace(*TIME_GRID_RANGE, time_grid_size) * GYR
        # time_grid = np.linspace(*TIME_GRID_RANGE, time_grid_size) * GYR
        end = TIME_GRID_RANGE[1]
        lo = [end/1000, end/10]
        hi = [end/10, end]
        time_grid_1 = np.logspace(*np.log10(lo), time_grid_size/2, endpoint=False)
        time_grid_2 = np.linspace(*hi, time_grid_size/2, endpoint=True)
        time_grid = np.append([0.0], np.append(time_grid_1, time_grid_2)) * GYR

        # Construct a grid of standard-deviation values
        sigma_grid_size = np.diff(SIGMA_GRID_RANGE) * SIGMA_GRID_SIZE_PER_STD
        sigma_grid = np.linspace(*SIGMA_GRID_RANGE, sigma_grid_size)

        if store:
            self._mstar_grid = mstar_grid
            self._time_grid = time_grid
            self._sigma_grid = sigma_grid

        self._MIN_MSTAR = np.min(mstar_grid)
        xgrids = (np.log10(mstar_grid), time_grid)

        # initialize outlier class in utils.py
        super().__init__(xgrids, sgrid=sigma_grid, nmc=NUM_MC, store=store)

        return

    def sfr_from_mstar(self, mstar, time, sigma=0.0, check=True):
        sc_mstar = np.isscalar(mstar)
        sc_time = np.isscalar(time)
        if sc_mstar and not sc_time:
            mstar = np.broadcast_to(mstar, np.shape(time))
        if sc_time and not sc_mstar:
            time = np.broadcast_to(time, np.shape(mstar))

        log_mstar = np.log10(mstar)
        # ss = np.ones_like(mstar) * sigma
        args = np.stack((log_mstar, time), axis=-1)
        args = np.atleast_2d(args)
        # args = np.stack((log_mstar, time, ss), axis=-1)
        # log_sfr = self.xs_to_y(log_mstar, time)
        log_sfr = self.xs_to_y(args, sigma)
        sfr = np.power(10.0, log_sfr)
        if check and np.any(np.isnan(sfr)):
            print("mstar = ", mstar, zmath.stats_str(self._mstar_grid))
            print("time = ", time, zmath.stats_str(self._time_grid))
            print("sigma = ", sigma)
            print("log_sfr = ", log_sfr)
            raise ValueError("`sfr` is NaN!")

        return sfr

    @classmethod
    def param(cls, name, size=None):
        vals, sigma = cls.DATA[name]

        if size is not None:
            vals = np.random.normal(vals, sigma, size=size)

        return vals

    @classmethod
    def function(cls, log_mstar, time, samples=None):
        mhalo = np.power(10.0, log_mstar)
        sfr = cls.sfr_ms(mhalo, time, samples=samples)
        sfr = np.log10(sfr)
        return sfr

    @classmethod
    def sfr_ms(cls, mstar, time, samples=None):
        a1 = cls.param("A1", size=samples)
        a2 = cls.param("A2", size=samples)
        b1 = cls.param("B1", size=samples)
        b2 = cls.param("B2", size=samples)

        shape = np.shape(mstar) + (samples,)
        a1, a2, b1, b2 = [np.broadcast_to(zz, shape) for zz in [a1, a2, b1, b2]]

        tt = time[..., np.newaxis] / GYR
        mm = mstar[..., np.newaxis] / MSOL
        log_mass = np.log10(mm)
        log_psi = (a1 - a2*tt) * log_mass - (b1 - b2*tt)
        psi = np.power(10.0, log_psi) * MSOL / YR
        return psi
