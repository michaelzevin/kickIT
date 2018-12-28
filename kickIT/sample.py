import numpy as np
import pandas as pd
from scipy.stats import maxwell
from scipy.integrate import trapz

import astropy.units as u
import astropy.constants as C
from astropy.table import Table

from . import galaxy_history


def sample_parameters(gal, t0=0, Nsys=1, Mns_method='gaussian', Mhe_method='uniform', Apre_method='uniform', epre_method='circularized', Vkick_method='maxwellian', R_method='sfr'):
    """
    Calls all the sampling functions defined below. 
    Returns a dataframe with the sampled parameters. 
    """
    bin_params=pd.DataFrame(columns=['Mns', 'Mcomp', 'Mhe', 'Apre', 'epre', 'Vkick', 'R'])

    # FIXME: parameters of a given method should be able to be written in the executable
    bin_params['Mcomp'], bin_params['Mns'] = sample_masses(Nsys, method=Mns_method)
    bin_params['Mhe'] = sample_Mhe(Nsys, bin_params['Mns'], method=Mhe_method)
    bin_params['Apre'] = sample_Apre(Nsys, method=Apre_method)
    bin_params['epre'] = sample_epre(Nsys, method=epre_method)
    bin_params['Vkick'] = sample_Vkick(Nsys, method=Vkick_method, sigma=265)
    bin_params['R'] = sample_R(Nsys, gal, t0, method=R_method)

    return bin_params


def sample_masses(Nsys, method='gaussian', m1_params={'mean':1.33, 'sigma': 0.09}, m2_params={'mean':1.33, 'sigma': 0.09}, samples=None, gw_samples=None):
    """
    Samples NS masses (m1 and m2)
    Inputs in units Msun
    Possible methods: 
        'posterior': NS masses are sampled from a GW posterior, path to samples must be specified
        'mean': NS masses are fixed at the mean of the posterior distritbuion, path to samples must be specified
        'median': NS masses are fixed at the median of the posterior distribution, path to samples must be specified
        'gaussian': NS masses are sampled from a Gaussian distribution tuned to the observed galactic DNS population
        'fixed': NS masses are fixed to the mean specified in m1_params and m2_params
        'popsynth': NS masses are taken from popsynth model at path 'samples'
    """

    if method=='posterior':
        samples = Table.read(gw_samples, format='ascii')
        m1 = samples['m1_source'][np.random.randint(0,len(samples['m1_source']),Nsys)]
        m2 = samples['m2_source'][np.random.randint(0,len(samples['m2_source']),Nsys)]
        return m1,m2


    elif method=='mean':
        samples = Table.read(gw_samples, format='ascii')
        m1 = np.ones(Nsys)*np.mean(samples['m1_source'])*u.Msun.to(u.g)
        m2 = np.ones(Nsys)*np.mean(samples['m2_source'])*u.Msun.to(u.g)
        return m1,m2


    elif method=='median':
        samples = Table.read(gw_samples, format='ascii')
        m1 = np.ones(Nsys)*np.median(samples['m1_source'])*u.Msun.to(u.g)
        m2 = np.ones(Nsys)*np.median(samples['m2_source'])*u.Msun.to(u.g)
        return m1,m2


    elif method=='gaussian':
        m1 = np.random.normal(m1_params['mean'], m1_params['sigma'], Nsys)*u.Msun.to(u.g)
        m2 = np.random.normal(m2_params['mean'], m2_params['sigma'], Nsys)*u.Msun.to(u.g)
        return m1,m2


    elif method=='fixed':
        m1 = np.ones(Nsys) * m1_params['mean']*u.Msun.to(u.g)
        m2 = np.ones(Nsys) * m2_params['mean']*u.Msun.to(u.g)
        return m1,m2


    else:
        raise ValueError("Undefined NS mass sampling method '{0:s}'.".format(method))


def sample_Mhe(Nsys, Mns, Mmax=8.0, method='uniform', fixed_val=None, samples=None):
    """
    Samples helium star mass (Mhe)
    Inputs in units Msun
    Maximum mass is 8 Msun (BH limit), unless otherwise specified
    Possible methods: 
        'uniform': Mhe is sampled uniformly between Mns and Mmax
        'powerlaw': Mhe is sampled from a power law with power law slope of -2.35
        'fixed': Mhe is a fixed value of Mhe=m
        'popsynth': Mhe is taken from popsynth model at path 'samples'
    """

    if method=='uniform':
        Mhe = np.asarray([np.random.uniform(Mmin, Mmax*u.Msun.to(u.g)) for Mmin in Mns])
        return Mhe


    elif method=='powerlaw':
        Mhe=[]

        def pdf(m):
            return m**-2.35
        def invpdf(ii,m):
                return (1./((m**-1.35)-(ii*1.35/Anorm)))**(1./1.35)

        for Mmin in Mns:
            xx=np.linspace(Mmin,Mmax*u.Msun.to(u.g),1000)
            A1=trapz(pdf(xx),x=xx)
            Anorm=1./A1
            II=np.random.uniform(0,1)
            Mhe.append(invpdf(II,Mmin))

        return np.asarray(Mhe)


    elif method=='fixed':
        if fixed_val==None:
            raise ValueError("No fixed mass specified!")
        if (fixed_val*u.Msun.to(u.g) < Mns).any():
            raise ValueError("Fixed mass of {0:0.2f} Msun is below one of the NS masses!".format(fixed_val))
        Mhe = np.ones(Nsys)*fixed_val*u.Msun.to(u.g)

        return Mhe


    else:
        raise ValueError("Undefined Mhe sampling method '{0:s}'.".format(method))



def sample_Apre(Nsys, Amin=0.1, Amax=10, method='uniform', fixed_val=None, samples=None):
    """
    Samples the pre-SN semimajor axis
    Inputs in units Rsun
    Possible methods: 
        'uniform': samples SMA uniformly between Amin and Amax
        'log': samples SMA flat in log between Amin and Amax
        'fixed': uses a fixed value for the SMA
        'popsynth': Apre is taken from popsynth model at path 'samples'
    """

    if method=='uniform':
        Apre = np.random.uniform(Amin, Amax, Nsys)*u.Rsun.to(u.cm)
        return Apre


    elif method=='log':
        Apre = 10**np.random.uniform(np.log10(Amin), np.log10(Amax), Nsys)*u.Rsun.to(u.cm)
        return Apre


    elif method=='fixed':
        if fixed_val==None:
            raise ValueError("No fixed SMA specified!")
        Apre = np.ones(Nsys)*fixed_val*u.Rsun.to(u.cm)
        return Apre


    else:
        raise ValueError("Undefined Apre sampling method '{0:s}'.".format(method))



def sample_epre(Nsys, method='circularized', samples=None):
    """
    Samples the pre-SN eccentricity
    Possible methods: 
        'circularized': pre-SN eccentricities are all 0
        'thermal': pre-SN eccentricities are sampled from a thermal distribution
        'popsynth': epre is taken from popsynth model at path 'samples'
    """
    if method=='circularized':
        epre = np.zeros(Nsys)
        return epre


    elif method=='thermal':
        epre = np.sqrt(np.random.uniform(0, 1, Nsys))
        return epre


    else:
        raise ValueError("Undefined epre sampling method '{0:s}'.".format(method))



def sample_Vkick(Nsys, Vmin=0, Vmax=1000, method='maxwellian', sigma=265, fixed_val=None, samples=None):
    """
    Samples velocity of SN2 natal kick
    Inputs in units km/s
    Possible methods: 
        'uniform': samples Vkick flat between Vmin and Vmax
        'maxwellian': samples Vkick from a maxwellian distribution of scale parameter sigma
        'fixed': uses fixed value for Vkick
        'popsynth': Vkick is taken from popsynth model at path 'samples'
    """

    if method=='uniform':
        Vkick_samp = np.random.uniform(Vmin, Vmax, size=Nsys)*u.km.to(u.cm)
        return Vkick


    elif method=='maxwellian':
        Vkick = maxwell.rvs(loc=0, scale=sigma, size=Nsys)*u.km.to(u.cm)
        return Vkick


    elif method=='fixed':
        if fixed_val==None:
            raise ValueError("No fixed Vkick specified!")
        Vkick = np.ones(Nsys)*fixed_val*u.km.to(u.cm)
        return Vkick


    else:
        raise ValueError("Undefined Vkick sampling method '{0:s}'.".format(method))
        


def sample_R(Nsys, gal, t0, method='sfr', fixed_val=None):
    """
    Samples the galactic radius at which to initiate the tracer particles
    Inputs in units kpc
    Possible methods: 
        'sfr': samples the location of the tracer particle according to the gas density at t0
        'fixed': takes a fixed radial distance for the location of the tracer particles
    """

    if method=='galpy_test':
        R = np.linspace(0.1,30, Nsys)*u.kpc.to(u.cm)
        return R

    if method=='sfr':
        # The SFR radial distribution is given by a gamma distribution with k=3, theta=r_s
        # So, p(R) = 1/(2*r_s^3) * R^2 * np.exp(-R/r_s)
        # The CDF of this distribution is P(R) = 1/Gamma(k) * gamma(k, x/theta)
        mstar = gal.mass_stars[t0]
        rs = galaxy_history.baryons.sfr_disk_rad(mstar) * u.cm.to(u.kpc)
        R = np.random.gamma(shape=3, scale=rs, size=Nsys)*u.kpc.to(u.cm)
        return R


    if method=='fixed':
        if fixed_val==None:
            raise ValueError("No fixed R specified!")
        R = np.ones(Nsys)*fixed_val*u.kpc.to(u.cm)
        return R
