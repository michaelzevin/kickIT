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




def normalize_weights(weights):
    """
    Normalizes a set of weights, such that the maximum weight is 1.0

    Returns a modified 'systems' dataframe with normalized weights
    """
    weights = weights*(1./weights.max())

    return weights

def normalize_data(samps, tracers):
    """
    Normalizes sample and tracer parameters for KDE construction between 0 and 1
    """
    max_val = np.max((np.max(samps), np.max(tracers)))
    min_val = np.min((np.min(samps), np.min(tracers)))

    samps_normed = (samps-min_val)/(max_val-min_val)
    tracers_normed = (tracers-min_val)/(max_val-min_val)

    return samps_normed, tracers_normed

def combine_weights(weights, combine_method='add', normalize=True):
    """
    Combines weights in quadrature

    If weighting schemes are independent, make sure to normalize first
    """

    weights = np.asarray(weights)

    if combine_method=='add':
        # --- add in quadrature
        weights = np.sqrt(np.sum(weights**2, axis=0))
    elif combine_method=='multiply':
        # --- multiply weights together
        weights = np.prod(weights, axis=0)

    if normalize==True:
        weights = normalize_weights(weights)

    return weights



# --- Weighting tracer particles using phenominological prescriptions

def weight_tracers_by_tinsp(tracers, powerlaw_idx=-1.0, min_tinsp=10*u.Myr, normalize=True):
    """
    Weights systems by their inspiral times according to the specified method

    Returns an array of weights for all the tracer particle

    Possible methods: 
        'powerlaw': weights according to p(t) \propto t^{x} where x=powerlaw_idx
    """

    # --- get the inspiral time
    Tinsp = tracers['Tinsp']

    # --- get normalization factor s.t. weight goes from 0 to 1
    N = (min_tinsp.to(u.Gyr).value)**(powerlaw_idx)

    # --- calculate inspiral probabilities
    weights = 1./N * Tinsp**(powerlaw_idx)

    # --- for systems with less than Tinsp_min, set to weight to 1
    weights.loc[weights>1.0] = 1.0

    # --- normalize
    if normalize==True:
        weights = normalize_weights(weights)

    return np.asarray(weights)


def weight_tracers_by_vsys(tracers, method='maxwellian', param=265.0, normalize=True):
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

    Vsys = tracers.Vsys

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

    return np.asarray(weights)




# --- weight tracers according to popsynth samples

def weight_tracers_from_samples(tracers, Vsys_samps, Tinsp_samps, normalize=True):
    """
    Weights systems by comparing to a generated population of systems

    Must provide a inspiral times (in Gyr) and Vsys (in km/s) from the population

    This gives a single (combined) weight, as correlation between inspiral time and systemic velocity cannot be ignored
    """

    # --- read in and normalize data
    Vsys_samps, Vsys_tracers = normalize_data(Vsys_samps, tracers['Vsys'])
    Tinsp_samps, Tinsp_tracers = normalize_data(Tinsp_samps, tracers['Tinsp'])
    pop_data = np.asarray([Vsys_samps, Tinsp_samps])
    tracers_data = np.asarray([Vsys_tracers, Tinsp_tracers])

    # --- generate KDE
    kde = gaussian_kde(pop_data)

    # --- get weights
    weights = kde.pdf(tracers_data)

    # --- normalize
    if normalize==True:
        weights = normalize_weights(weights)

    return np.asarray(weights)




# --- weight tracers based on the observed offset of the sGRB

def weight_tracers_from_observations(tracers, offset, offset_error, normalize=True):
    """
    Weights systems according to their projected offset

    Takes in offset and offset error in kpc
    """

    weights = norm.pdf(tracers['Rproj_offset'], offset, offset_error)

    # --- normalize
    if normalize==True:
        weights = normalize_weights(weights)

    return np.asarray(weights)



# --- weight popsynth according to tracers and offset constraint

def weight_samples_from_tracers(tracers, offset, offset_error, Vsys_samps, Tinsp_samps, normalize=True):
    """
    Weights popsynth samples according to the tracer particles and the offset constraint

    Must provide a inspiral times (in Gyr) and Vsys (in km/s) from the population
    """

    # --- read in and normalize data
    Vsys_samps, Vsys_tracers = normalize_data(Vsys_samps, tracers['Vsys'])
    Tinsp_samps, Tinsp_tracers = normalize_data(Tinsp_samps, tracers['Tinsp'])
    pop_data = np.asarray([Vsys_samps, Tinsp_samps])
    tracers_data = np.asarray([Vsys_tracers, Tinsp_tracers])

    # --- get weights based on observed offset and normalize
    weights = norm.pdf(tracers['Rproj_offset'], offset, offset_error)
    if normalize==True:
        weights = normalize_weights(weights)

    # --- generate KDE
    kde = gaussian_kde(tracers_data, weights=weights)

    # --- get weights for the population
    pop_weights = kde.pdf(pop_data)

    # --- normalize
    if normalize==True:
        pop_weights = normalize_weights(pop_weights)

    return np.asarray(pop_weights)

