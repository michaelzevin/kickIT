import numpy as np
import pandas as pd
from scipy.stats import maxwell
from scipy.integrate import trapz

import astropy.units as u
import astropy.constants as C
from astropy.table import Table

from . import galaxy_history



VERBOSE=True


def sample_parameters(gal, Nsys=1, Mcomp_method='gaussian', Mns_method='gaussian', Mhe_method='uniform', Apre_method='uniform', epre_method='circularized', Vkick_method='maxwellian', R_method='sfr', params_dict=None, samples=None, fixed_birth=None, fixed_potential=None):
    """
    Calls all the sampling functions defined below. 
    Returns a dataframe with the sampled parameters. 

    Can input a (space-separated) table with data from a population synthesis model
    When method='popsynth', must provide the path to the sample as the 'samples' argument
    Units of data must be: Msun, Run, km/s, and column names must be same as below
    """
    bin_params=pd.DataFrame(columns=['Mns', 'Mcomp', 'Mhe', 'Apre', 'epre', 'Vkick', 't0', 'R'])

    # if popsynth samples are provided, sample systems randomly from the popsynth data
    if 'popsynth' in (Mcomp_method, Mns_method, Mhe_method, Apre_method, epre_method, Vkick_method, R_method):
        if samples is None:
            raise NameError("No popsynth samples were provided!")
        popsynth_data = pd.read_csv(samples, index_col='idx')
        popsynth_data = popsynth_data.sample(Nsys, replace=True)
    else:
        popsynth_data = None

    # call all the sampling functions
    bin_params['Mcomp'] = sample_Mcomp(Nsys, method=Mcomp_method, mean=params_dict['Mcomp_mean'], sigma=params_dict['Mcomp_sigma'], samples=popsynth_data)
    bin_params['Mns'] = sample_Mns(Nsys, method=Mns_method, mean=params_dict['Mns_mean'], sigma=params_dict['Mns_sigma'], samples=popsynth_data)
    bin_params['Mhe'] = sample_Mhe(Nsys, bin_params['Mns'], Mmax=params_dict['Mhe_max'], method=Mhe_method, mean=params_dict['Mhe_mean'], sigma=params_dict['Mhe_sigma'], samples=popsynth_data)
    bin_params['Apre'] = sample_Apre(Nsys, Amin=params_dict['Apre_min'], Amax=params_dict['Apre_max'], method=Apre_method, mean=params_dict['Apre_mean'], samples=popsynth_data)
    bin_params['epre'] = sample_epre(Nsys, method=epre_method, samples=popsynth_data)
    bin_params['Vkick'] = sample_Vkick(Nsys, Vmin=params_dict['Vkick_min'], Vmax=params_dict['Vkick_max'], method=Vkick_method, sigma=params_dict['Vkick_sigma'], samples=popsynth_data)
    bin_params['t0'] = sample_t0(Nsys, gal, fixed_birth=fixed_birth)
    bin_params['R'] = sample_R(Nsys, gal, bin_params['t0'], method=R_method, mean=params_dict['R_mean'], samples=popsynth_data, fixed_potential=fixed_potential)

    # --- write the birth time and birth redshift
    bin_params['tbirth'] = gal.times[bin_params['t0']]
    bin_params['zbirth'] = gal.redz[bin_params['t0']]

    if VERBOSE:
        print('Parameters sampled according to the following methods:')
        print('  Mcomp: {0:s}\n  Mns: {1:s}\n  Mhe: {2:s}\n  Apre: {3:s}\n  epre: {4:s}\n  Vkick: {5:s}\n  R: {6:s}\n'.format(Mcomp_method,Mns_method,Mhe_method,Apre_method,epre_method,Vkick_method,R_method))


    return bin_params



def sample_Mcomp(Nsys, method='gaussian', mean=1.33, sigma=0.09, samples=None, gw_samples=None):
    """
    Samples companion NS mass (m1)
    Inputs in units Msun
    Possible methods: 
        'posterior': Mass is sampled from a GW posterior, path to samples must be specified
        'mean': Mass is fixed at the mean of the posterior distritbuion, path to samples must be specified
        'median': Mass is fixed at the median of the posterior distribution, path to samples must be specified
        'gaussian': Mass is sampled from a Gaussian distribution tuned to the observed galactic DNS population
        'fixed': Mass is fixed
        'popsynth': Mass is taken from popsynth model at path 'samples'
    """

    if method=='posterior':
        samples = Table.read(gw_samples, format='ascii')
        Mcomp = samples['m1_source'][np.random.randint(0,len(samples['m1_source']),Nsys)]


    elif method=='mean':
        samples = Table.read(gw_samples, format='ascii')
        Mcomp = np.ones(Nsys)*np.mean(samples['m1_source'])


    elif method=='median':
        samples = Table.read(gw_samples, format='ascii')
        Mcomp = np.ones(Nsys)*np.median(samples['m1_source'])


    elif method=='gaussian':
        Mcomp = np.random.normal(mean, sigma, Nsys)


    elif method=='fixed':
        Mcomp = np.ones(Nsys) * mean
        
    elif method=='popsynth':
        if 'Mcomp' not in samples.columns:
            raise NameError("Series '{0:s}' not in popsynth table".format('Mcomp'))
        Mcomp = np.asarray(samples['Mcomp'])


    else:
        raise ValueError("Undefined companion mass sampling method '{0:s}'.".format(method))

    return Mcomp*u.Msun




def sample_Mns(Nsys, method='gaussian', mean=1.33, sigma=0.09, samples=None, gw_samples=None):
    """
    Samples remnant NS mass (m2)
    Inputs in units Msun
    Possible methods: 
        'posterior': Mass is sampled from a GW posterior, path to samples must be specified
        'mean': Mass is fixed at the mean of the posterior distritbuion, path to samples must be specified
        'median': Mass is fixed at the median of the posterior distribution, path to samples must be specified
        'gaussian': Mass is sampled from a Gaussian distribution tuned to the observed galactic DNS population
        'fixed': Mass is fixed
        'popsynth': Mass is taken from popsynth model at path 'samples'
    """

    if method=='posterior':
        samples = Table.read(gw_samples, format='ascii')
        Mns = samples['m2_source'][np.random.randint(0,len(samples['m2_source']),Nsys)]


    elif method=='mean':
        samples = Table.read(gw_samples, format='ascii')
        Mns = np.ones(Nsys)*np.mean(samples['m2_source'])


    elif method=='median':
        samples = Table.read(gw_samples, format='ascii')
        Mns = np.ones(Nsys)*np.median(samples['m2_source'])


    elif method=='gaussian':
        Mns = np.random.normal(mean, sigma, Nsys)


    elif method=='fixed':
        Mns = np.ones(Nsys) * mean
        
    elif method=='popsynth':
        if 'Mns' not in samples.columns:
            raise NameError("Series '{0:s}' not in popsynth table".format('Mns'))
        Mns = np.asarray(samples['Mns'])


    else:
        raise ValueError("Undefined remnant mass sampling method '{0:s}'.".format(method))

    return Mns*u.Msun




def sample_Mhe(Nsys, Mns, Mmax=8.0, method='uniform', mean=3, sigma=0.5, samples=None):
    """
    Samples helium star mass (Mhe)
    Inputs in units Msun
    Maximum mass is 8 Msun (BH limit), unless otherwise specified
    Possible methods: 
        'uniform': Mhe is sampled uniformly between Mns and Mmax
        'powerlaw': Mhe is sampled from a power law with power law slope of -2.35
        'fixed': Mhe is a fixed value of Mhe=M
        'gaussian': Mhe is drawn from a gaussian
        'popsynth': Mhe is taken from popsynth model at path 'samples'
    """

    if method=='uniform':
        Mhe = np.asarray([np.random.uniform(Mmin.value, Mmax) for Mmin in Mns])


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
        Mhe = np.asarray(Mhe)


    elif method=='gaussian':
        Mhe = np.random.normal(mean, sigma, Nsys)
        # if any of the values for Mhe drawn from the gaussian are less massive than Mns, set their value to Mne
        neg_vals = np.argwhere((Mhe-Mns) < 0)
        Mhe[neg_vals] = Mns[neg_vals]


    elif method=='fixed':
        if mean==None:
            raise ValueError("No fixed mass specified!")
        if (mean*u.Msun.to(u.g) < Mns).any():
            raise ValueError("Fixed mass of {0:0.2f} Msun is below one of the NS masses!".format(mean))
        Mhe = np.ones(Nsys)*mean

    elif method=='popsynth':
        if 'Mhe' not in samples.columns:
            raise NameError("Series '{0:s}' not in popsynth table".format('Mhe'))
        Mhe = np.asarray(samples['Mhe'])


    else:
        raise ValueError("Undefined Mhe sampling method '{0:s}'.".format(method))

    return Mhe*u.Msun


def sample_Apre(Nsys, Amin=0.1, Amax=10, method='uniform', mean=None, samples=None):
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
        Apre = np.random.uniform(Amin, Amax, Nsys)


    elif method=='log':
        Apre = 10**np.random.uniform(np.log10(Amin), np.log10(Amax), Nsys)


    elif method=='gaussian':
        Apre = np.random.normal(mean, sigma, Nsys)
        # if any of the values for Apre drawn from the gaussian are below the specified lower limit, set their value to Amin
        neg_vals = np.argwhere((Apre-Amin) < 0)
        Apre[neg_vals] = Amin
        Apre = np.asarray(Apre)


    elif method=='fixed':
        if mean==None:
            raise ValueError("No fixed SMA specified!")
        Apre = np.ones(Nsys)

    elif method=='popsynth':
        if 'Apre' not in samples.columns:
            raise NameError("Series '{0:s}' not in popsynth table".format('Apre'))
        Apre = np.asarray(samples['Apre'])


    else:
        raise ValueError("Undefined Apre sampling method '{0:s}'.".format(method))

    return Apre*u.Rsun


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


    elif method=='thermal':
        epre = np.sqrt(np.random.uniform(0, 1, Nsys))

    elif method=='popsynth':
        if 'epre' not in samples.columns:
            raise NameError("Series '{0:s}' not in popsynth table".format('epre'))
        epre = np.asarray(samples['epre'])


    else:
        raise ValueError("Undefined epre sampling method '{0:s}'.".format(method))

    return epre


def sample_Vkick(Nsys, Vmin=0, Vmax=1000, method='maxwellian', sigma=265, samples=None):
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
        Vkick_samp = np.random.uniform(Vmin, Vmax, size=Nsys)


    elif method=='maxwellian':
        Vkick = maxwell.rvs(loc=0, scale=sigma, size=Nsys)


    elif method=='fixed':
        if sigma==None:
            raise ValueError("No fixed Vkick specified!")
        Vkick = np.ones(Nsys)*sigma

    elif method=='popsynth':
        if 'Vkick' not in samples.columns:
            raise NameError("Series '{0:s}' not in popsynth table".format('Vkick'))
        Vkick = np.asarray(samples['Vkick'])


    else:
        raise ValueError("Undefined Vkick sampling method '{0:s}'.".format(method))
        
    return Vkick*u.km/u.s


def sample_R(Nsys, gal, t0, method='sfr', mean=3, samples=None, fixed_potential=False):
    """
    Samples the galactic radius at which to initiate the tracer particles
    Inputs in units kpc
    Possible methods: 
        'sfr': samples the location of the tracer particle according to the gas density at t0
        'fixed': takes a fixed radial distance for the location of the tracer particles
        'popsynth': R is taken from popsynth model at path 'samples'
    """

    if method=='sfr':
        # The SFR radial distribution is given by a gamma distribution with k=3, theta=r_s
        # So, p(R) = 1/(2*r_s^3) * R^2 * np.exp(-R/r_s)
        # The CDF of this distribution is P(R) = 1/Gamma(k) * gamma(k, x/theta)

        # First, get the galaxy scaling from the observed effective radius
        _, R_final = galaxy_history.baryons.sfr_rad_dist(gal.rads.cgs.value, gal.obs_props['mass_stars'].cgs.value)
        R_final = (R_final*u.cm).to(u.kpc)
        R_scaling = (gal.obs_props['rad_eff'] / R_final).value

        # if fixed_potential, set t0 to the last step
        if fixed_potential:
            t0 = (np.ones(Nsys)*fixed_potential).astype(int)

        mstar = gal.mass_stars[t0]
        rs = (galaxy_history.baryons.sfr_disk_rad(mstar.cgs.value, R_scaling) * u.cm.to(u.kpc))
        R = np.random.gamma(shape=3, scale=rs, size=Nsys)


    elif method=='fixed':
        if mean==None:
            raise ValueError("No fixed R specified!")
        R = np.ones(Nsys)*mean

    elif method=='popsynth':
        if 'R' not in samples.columns:
            raise NameError("Series '{0:s}' not in popsynth table".format('R'))
        R = np.asarray(samples['R'])


    else:
        raise ValueError("Undefined R sampling method '{0:s}'.".format(method))

    return R*u.kpc




def sample_t0(Nsys, gal, fixed_birth=False):
    """
    Samples the initial timestep at which particles are initiated, according to the sfr of the galaxy.
    If fixed_birth is specified, then will initiate all particles at the specified t0.
    """
    if fixed_birth:
        t0 = (fixed_birth * np.ones(Nsys)).astype(int)

    else:
        t0 = np.random.choice(np.arange(0,len(gal.times)), Nsys, p=gal.sfr_weights)

    return t0


### Option B: separate out the progenitor sampling and the evolution ###

def sample_Vsys_R(gal, Nsys=1, Vsys_range=(0,1000), R_method='sfr', fixed_birth=False, fixed_potential=False):
    """
    Samples only radii and systemic velocity. 

    This allows the progenitor population sampling and trajectories to effectively run independently, allowing for the convolution of distribution as a post-processing step.

    This function supplies *everything* necessary in the galactic frame, such that it can be fed directly into the systems.evolve method.
    """

    bin_params=pd.DataFrame(columns=['Vsys', 'R', 'Tinsp', 'SNsurvive'])

    # sample Vsys
    bin_params['Vsys'] = np.random.uniform(Vsys_range[0],Vsys_range[1], size=Nsys) * u.km/u.s
    # sample t0
    bin_params['t0'] = sample_t0(Nsys, gal, fixed_birth=fixed_birth)
    # sample R
    bin_params['R'] = sample_R(Nsys, gal, bin_params['t0'], method=R_method, fixed_potential=fixed_potential)
    # fix SNsurvive and Tinsp (we'll change Tinsp after the integration)
    bin_params['SNsurvive'] = True
    bin_params['Tinsp'] = 14*u.Gyr

    # --- write the birth time and birth redshift
    bin_params['tbirth'] = gal.times[bin_params['t0']]
    bin_params['zbirth'] = gal.redz[bin_params['t0']]

    return bin_params


