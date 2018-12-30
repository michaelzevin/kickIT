import numpy as np
import pandas as pd

from scipy import integrate


def euler_rot(vectors, angles, axis):
    """
    Performed Euler angle transformation on vector. 
    """
    if axis=='X':
        transformations = np.asarray([[[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]] for angle in angles])
    elif axis=='Y':
        transformations = np.asarray([[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]] for angle in angles])
    elif axis=='Z':
        transformations = np.asarray([[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]] for angle in angles])
    else:
        raise ValueError("Unknown axis '{0:s}' specified in Euler transformation)".format(axis))


    rot_vectors = np.asarray([np.dot(transform, vector) for (transform,vector) in zip (transformations, vectors)])

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
    vR = (x*vx + y*vy)/np.sqrt(x**2 + y**2)

    T = np.arctan(y/x)
    #FIXME check that this is right?
    vT = (x*vy - y*vx)/np.sqrt(x**2 + y**2)

    Z = z
    vZ = vz

    return R,T,Z,vR,vT,vZ


def cylindrical_to_cartesian(R,T,Z,vR,vT,vZ):
    """
    Transforms positions and velocities from cylindrical to cartesian coordinates
    """

    x = R*np.cos(T)
    vx = vR*np.cos(T) - vR*np.sin(T)*vT

    y = R*np.sin(T)
    vy = vR*np.sin(T) + vR*np.cos(T)*vT

    z = Z
    vz = vZ

    return x,y,z,vx,vy,vz
    






