"""utils.py : general utility methods and functions for internal use.
"""

import numpy as np
import scipy as sp


def interp_1d(xx, yy, **kwargs):
    kwargs.setdefault('kind', 'linear')
    kwargs.setdefault('bounds_error', False)
    kwargs.setdefault('fill_value', 0.0)

    interp = sp.interpolate.interp1d(xx, yy, **kwargs)
    return interp


def log_interp_1d(xx, yy, **kwargs):
    xx = np.log10(xx)
    yy = np.log10(yy)
    lin_interp = interp_1d(xx, yy, **kwargs)

    def interp(zz):
        zz = np.log10(zz)
        ww = lin_interp(zz)
        return np.power(10.0, ww)

    return interp


def annulus_areas(rads, relative=True, reset_inner=True):
    rr = rads
    if relative:
        rr = rr / rads[0]

    area = np.pi * (rr**2)
    area[1:] = area[1:] - area[:-1]
    # Assume log-distributed
    if reset_inner:
        area[0] = area[1]/(area[2]/area[1])

    return area


def shell_volumes(rads, relative=True, reset_inner=True):
    rr = rads
    if relative:
        rr = rr / rads[0]

    vol = (4.0/3.0)*np.pi * (rr**3)
    vol[1:] = vol[1:] - vol[:-1]
    # Assume log-distributed
    if reset_inner:
        vol[0] = vol[1]/(vol[2]/vol[1])

    return vol
