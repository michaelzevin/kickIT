import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import maxwell
from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.integrate import trapz

import astropy.units as u
import astropy.constants as C

from . import galaxy_history



VERBOSE=True


def weight_tinsp(gal, tracers, powerlaw_idx=1.0, Tinsp_min=10*u.Myr, normalize=True):
    """
    Weights systems by their inspiral times according to the specified method

    Returns a modified 'systems' dataframe with Tinsp weights included

    Possible methods: 
        'powerlaw': weights according to p(t) \propto t^{-x} where x=powerlaw_idx
    """

    # --- get the time of the GRB
    cosmo = gal.cosmo
    Tgrb = cosmo.age(gal.obs_props['redz']).value

    # --- get the inspiral time
    Tinsp = tracers.Tinsp

    # --- get normalization factor s.t. weight goes from 0 to 1
    N = (Tinsp_min.to(u.Gyr).value)**(-powerlaw_idx)

    # --- calculate inspiral probabilities
    weights = 1./N * Tinsp**(-powerlaw_idx)

    # --- for systems with less than Tinsp_min, set to weight to 1
    weights.loc[weights>1.0] = 1.0

    # --- normalize
    if normalize==True:
        weights = normalize_weights(weights)

    # --- write weights
    tracers['Tinsp_weight'] = weights

    return tracers




def weight_Vsys(tracers, method='maxwellian', param=265.0, normalize=True):
    """
    Weights systems by their systemic velocities according to the specified method

    Returns a modified 'systems' dataframe with Vsys weights included

    Possible methods: 
        'flat_in_log': flat in log distribution, param is Vsys val below which the weight is constant (default=30 km/s)
        'maxwellian': maxwellian distribution, where param is scale parameter
        'gaussian': gaussian distribution, where param is tuple of (mean, sigma)
    """

    if method not in ['flat_in_log', 'maxwellian', 'gaussian']:
        raise NameError('Method {0:s} not an available method for weighting Vsys!'.format(method))

    Vsys = tracers['Vsys']

    if method=='flat_in_log':
        weights = np.zeros_like(Vsys)

        xmin = param
        xmax = Vsys.max()

        # --- get points sampled uniform in log
        pts = np.exp(np.linspace(np.log(xmin), np.log(xmax), 100000))

        # --- create histogram of points
        h, bins = np.histogram(pts, bins=100)

        # --- fix the bounds for interpolation
        bins[-1] = 1000
        h = np.append(h, h[-1])
        max_weight = h[0]

        # --- create interpolation model of weights
        interp = interp1d(bins, h)

        # --- get weights
        low_idxs = np.where(Vsys < xmin)
        high_idxs = np.where(Vsys >= xmin)
        weights[low_idxs] = max_weight
        weights[high_idxs] = interp(np.asarray(Vsys)[high_idxs])

    elif method=='maxwellian':

        # --- get weights
        weights = maxwell.pdf(Vsys, loc=0, scale=param)
        

    elif method=='gaussian':

        if len(param) != 2:
            raise ValueError('For gaussian distribution, must supply tuple [param=(mean, scale)] of the Gaussian distribution!')

        # --- get weights
        weights = norm.pdf(Vsys, loc=param[0], scale=param[1])
         


    # --- normalize
    if normalize==True:
        weights = normalize_weights(weights)

    # --- write weights
    tracers['Vsys_weight'] = weights

    return tracers




def combine_weights(tracers, normalize=True):
    """
    Combines Tinsp and Vsys weights to create a final weight for each system
    """

    # --- combine weights in quadrature
    weights = np.sqrt(tracers['Tinsp_weight']**2 + tracers['Vsys_weight']**2)

    # --- normalize
    if normalize==True:
        weights = normalize_weights(weights)

    # --- write weights
    tracers['weights'] = weights

    return tracers



def normalize_weights(weights):
    """
    Normalizes a set of weights, such that the maximum weight is 1.0

    Returns a modified 'systems' dataframe with normalized weights
    """
    weights = weights*(1./weights.max())

    return weights



def weight_from_pop(tracers, Vsys, Tinsp, normalize=True):
    """
    Weights systems by comparing to a generated population of systems

    Must provide a inspiral times (in Gyr) and Vsys (in km/s) from the population

    This gives a single (combined) weight, as correlation between inspiral time and systemic velocity cannot be ignored
    """

    # --- read in data
    pop_data = np.asarray([Vsys, Tinsp])
    tracers_data = np.asarray([tracers['Vsys'], tracers['Tinsp']])

    # --- generate KDE
    kde = gaussian_kde(pop_data)

    # --- get weights
    weights = kde.pdf(tracers_data)

    # --- normalize
    if normalize==True:
        weights = normalize_weights(weights)

    # --- write weights
    tracers['weights'] = weights

    return tracers


