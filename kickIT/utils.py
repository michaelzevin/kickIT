import numpy as np
import pandas as pd
import scipy as sp
from scipy import integrate

import astropy.units as u
import astropy.constants as C

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

def euler_rot(vectors, angles, axis):
    """
    Performed Euler angle transformation on vector. 

    Takes in vectors as (Nsamples x Ndim)
    """
    if axis=='X':
        transformations = np.asarray([[[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]] for angle in angles])
    elif axis=='Y':
        transformations = np.asarray([[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]] for angle in angles])
    elif axis=='Z':
        transformations = np.asarray([[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]] for angle in angles])
    else:
        raise ValueError("Unknown axis '{0:s}' specified in Euler transformation)".format(axis))

    rot_vectors = np.asarray([np.dot(trans,vector).T for (trans,vector) in zip(transformations, vectors)])
    return rot_vectors



def inspiral_time_peters(a0,e0,m1,m2,af=0):
    """
    Computes the inspiral time, in Gyr, for a binary
    a0 in Au, and masses in solar masses

    if different af is given, computes the time from a0,e0
    to that final semi-major axis

    for af=0, just returns inspiral time
    for af!=0, returns (t_insp,af,ef)
    """

    def deda_peters(a,e):
        num = 12*a*(1+(73./24)*e**2 + (37./96)*e**4)
        denom = 19*e*(1-e**2)*(1+(121./304)*e**2)
        return denom/num

    coef = 6.086768e-11 #G^3 / c^5 in au, gigayear, solar mass units
    beta = (64./5.) * coef * m1 * m2 * (m1+m2)

    if e0 == 0:
        if not af == 0:
            print("ERROR: doesn't work for circular binaries")
            return 0
        return a0**4 / (4*beta)

    c0 = a0 * (1.-e0**2.) * e0**(-12./19.) * (1.+(121./304.)*e0**2.)**(-870./2299.)

    if af == 0:
        eFinal = 0.
    else:
        r = ode(deda_peters)
        r.set_integrator('lsoda')
        r.set_initial_value(e0,a0)
        r.integrate(af)
        if not r.successful():
            print("ERROR, Integrator failed!")
        else:
            eFinal = r.y[0]

    time_integrand = lambda e: e**(29./19.)*(1.+(121./304.)*e**2.)**(1181./2299.) / (1.-e**2.)**1.5
    integral,abserr = integrate.quad(time_integrand,eFinal,e0)

    if af==0:
        return integral * (12./19.) * c0**4. / beta
    else:
        return (integral * (12./19.) * c0**4. / beta,af,eFinal)




def cartesian_to_cylindrical(x,y,z,vx,vy,vz):
    """
    Transforms positions and velocities from cartesian to cylindrical coordinates
    """

    R = np.sqrt(x**2 + y**2)
    vR = (x*vx + y*vy)/((x**2 + y**2)**(1./2))

    Phi = np.arctan(y/x)
    vPhi = (x*vy - y*vx)/(x**2 + y**2)

    Z = z
    vZ = vz

    return R,Phi,Z,vR,vPhi,vZ


def cylindrical_to_cartesian(R,Phi,Z,vR,vPhi,vZ):
    """
    Transforms positions and velocities from cylindrical to cartesian coordinates
    """

    x = R*np.cos(Phi)
    vx = vR*np.cos(Phi) - vR*np.sin(Phi)*vPhi

    y = R*np.sin(Phi)
    vy = vR*np.sin(Phi) + vR*np.cos(Phi)*vPhi

    z = Z
    vz = vZ

    return x,y,z,vx,vy,vz
    


def Mcgs_to_nat(M, ro=8, vo=220):
    """Converts cgs masses to galpy natural units
    """
    ro *= u.kpc.to(u.cm)
    vo *= u.km.to(u.cm)
    G = C.G.cgs.value

    Mo = vo**2 * ro / G

    return M/Mo


def Mnat_to_cgs(M, ro=8, vo=220):
    """Converts galpy natural units to cgs masses
    """
    ro *= u.kpc.to(u.cm)
    vo *= u.km.to(u.cm)
    G = C.G.cgs.value

    Mo = vo**2 * ro / G

    return M*Mo


def Rcgs_to_nat(r, ro=8, vo=220):
    """Converts cgs distance to galpy natural units
    """
    ro *= u.kpc.to(u.cm)

    return r/ro


def Rnat_to_cgs(r, ro=8, vo=220):
    """Converts galpy natural units to cgs distance
    """
    ro *= u.kpc.to(u.cm)

    return r*ro


def Tcgs_to_nat(t, ro=8, vo=220):
    """Converts cgs time to galpy natural units
    """
    ro *= u.kpc.to(u.cm)
    vo *= u.km.to(u.cm)

    to = ro/vo

    return t/to


def Tnat_to_cgs(t, ro=8, vo=220):
    """Converts galpy natural units to cgs time
    """
    ro *= u.kpc.to(u.cm)
    vo *= u.km.to(u.cm)

    to = ro/vo

    return t*to


def orbit_cgs_to_nat(R, vR, vT, Z, vZ, Phi, ro=8, vo=220):
    """Converts orbital parameters from cgs to natural units
    """
    ro *= u.kpc.to(u.cm)
    vo *= u.km.to(u.cm)

    R /= ro
    vR /= vo
    vT /= vo
    Z /= ro
    vZ /= vo
    
    return R, vR, vT, Z, vZ, Phi


def orbit_nat_to_cgs(R, vR, vT, Z, vZ, Phi, ro=8, vo=220):
    """Converts orbital parameters from natural units to cgs
    """
    ro *= u.kpc.to(u.cm)
    vo *= u.km.to(u.cm)

    R *= ro
    vR *= vo
    vT *= vo
    Z *= ro
    vZ *= vo
    
    return R, vR, vT, Z, vZ, Phi




