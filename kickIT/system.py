import numpy as np
import pandas as pd

import astropy.units as u
import astropy.constants as C

from scipy.integrate import ode
from scipy.integrate import quad
from scipy.interpolate import interp1d

import galpy as gp

from . import utils


class Systems:
    """
    Places system in orbit in the galaxy model. 
    Applies the SN kick and mass loss to find post-SN trajectory.
    Calculates merger time for the binary. 
    Follows the evolution of the binary through the time-dependent galactic potential until the system merges. 

    Galactic units: r, theta (polar angle), phi (azimuthal angle). 
    System starts on a circular orbit in the r-phi (x-y) plane, on the x-axis (phi=0) and moving in the positive y direction. 
    Galaxy projection taken account when determining radial offset at merger. 
    """
    def __init__(self, sampled_parameters, Nsys, SNphi=None, SNtheta=None, SYSphi=None, SYStheta=None):

        # read in the sampled parameters
        self.Mns = np.asarray(sampled_parameters['Mns'])
        self.Mcomp = np.asarray(sampled_parameters['Mcomp'])
        self.Mhe = np.asarray(sampled_parameters['Mhe'])
        self.Apre = np.asarray(sampled_parameters['Apre'])
        self.epre = np.asarray(sampled_parameters['epre'])
        self.Vkick = np.asarray(sampled_parameters['Vkick'])
        self.R = np.asarray(sampled_parameters['R'])

        # initialize random angles
        if SNphi: self.SNphi = SNphi
        else: self.SNphi = np.random.uniform(0,2*np.pi, Nsys)

        if SNtheta: self.SNtheta = SNtheta
        else: self.SNtheta = np.arccos(np.random.uniform(0,1, Nsys))

        if SYSphi: self.SYSphi = SYSphi
        else: self.SYSphi = np.random.uniform(0,2*np.pi, Nsys)

        if SYStheta: self.SYStheta = SYStheta
        else: self.SYStheta = np.arccos(np.random.uniform(0,1, Nsys))


    def SN(self):
        """
        Implements the SN explosion, including the effect of the natal kick and the mass loss. 

        SN Coordinate System: 
        Mhe lies on origin moving in direction of positive y axis, Mcomp is on the negative X axis, Z completes right-handed coordinate system

        Variables: 
        SNtheta: angle between preSN He core velocity relative to Mcomp (i.e. the positive y axis) and the kick velocity
        SNphi: angle between Z axis and projection of kick onto X-Z plane

        Vr is velocity of preSN He core relative to Mcomp, directed along the positive y axis

        Vkick is kick velocity with components Vkx, Vky, Vkz in the above coordinate system
        V_sys is the resulting center of mass velocity of the system IN THE TRANSLATED COM FRAME, imparted by the SN


        Paper reference:

        Kalogera 1996: http://iopscience.iop.org/article/10.1086/177974/meta
            We use Eq 1, 3, 4, and 34: giving Vr, Apost, epost, and (Vsx,Vsy,Vsz) respectively
            Also see Fig 1 in that paper for coordinate system
        """
        G = C.G.cgs.value

        # Decompose the kick into its cartesian coordinates
        self.Vkx = self.Vkick*np.sin(self.SNtheta)*np.sin(self.SNphi)
        self.Vky = self.Vkick*np.cos(self.SNtheta)
        self.Vkz = self.Vkick*np.sin(self.SNtheta)*np.cos(self.SNphi)

        # Calculate the relative velocity according to Kepler's Law
        self.Vr = np.sqrt(G * (self.Mhe+self.Mcomp) / self.Apre)

        # Calculate the post-SN orbital properties (Eqs 3 and 4 from Kalogera 1996)
        Mtot_post = self.Mns+self.Mcomp

        self.Apost = G*(Mtot_post) * ((2*G*Mtot_post/self.Apre) - (self.Vkick**2) - (self.Vr**2) - 2*self.Vky*self.Vr)**(-1.0)
        x = ((self.Vkz**2 + self.Vky**2 + self.Vr**2 + 2*self.Vky*self.Vr)*self.Apre**2) / (G * Mtot_post * self.Apost)
        self.epost = np.sqrt(1-x)

        # Calculate the post-SN systemic velocity (Eq 34 from Kalogera 1996)
        self.Vsx = self.Mns*self.Vkx / Mtot_post
        self.Vsy = (self.Mns*self.Vky - ((self.Mhe-self.Mns)*self.Mcomp / (self.Mhe+self.Mcomp) * self.Vr)) / Mtot_post
        self.Vsz = self.Mns*self.Vkz / Mtot_post
        self.Vsys = np.sqrt(self.Vsx**2 + self.Vsy**2 + self.Vsz**2)

        # Calculate the tile of the orbital plane from the SN (Eq 5 from Kalogera 1996)
        self.tilt = np.arccos((self.Vky+self.Vr) / np.sqrt((self.Vky+self.Vr)**2 + self.Vkz**2))



    def check_survival(self):
        """
        Checks to see if the systems survived the supernova explosion. "True" if the system passes the check, "False" if system does not pass the check. 

        References: 
        Willems et al 2002: http://iopscience.iop.org/article/10.1086/429557/meta
            We use eq 21, 22, 23, 24, 25, 26 for checks of SN survival

        Kalogera and Lorimer 2000: http://iopscience.iop.org/article/10.1086/308417/meta

        Note: V_He;preSN is the same variable as V_r from Kalogera 1996
        """

        G = C.G.cgs.value
        Mtot_pre = self.Mhe+self.Mcomp
        Mtot_post = self.Mns+self.Mcomp

        # Check 1: Continuity demands that post-SN orbits must pass through the pre-SN positions (Eq 21 from Flannery & Van Heuvel 1975)
        self.SNcheck1 = (1-self.epost <= self.Apre/self.Apost) & (self.Apre/self.Apost <= 1+self.epost)

        # Check 2: Lower and upper limites on amount of orbital contraction or expansion that can take place for a given amount of mass loss and a given natal kick velocity (Kalogera & Lorimer 2000)
        self.SNcheck2 = (self.Apre/self.Apost < 2-((Mtot_pre/Mtot_post)*((self.Vkick/self.Vr)-1)**2)) & (self.Apre/self.Apost > 2-((Mtot_pre/Mtot_post)*((self.Vkick/self.Vr)+1)**2))

        # Check 3: The magnitude of the kick velocity imparted to the compact object at birth is restricted to a certain range (Brandy & Podsiadlowski 1995; Kalogera & Lorimer 2000)
        # The first inequality expresses the requirement that the bianry must remain bound after the SN explosion
        # The second inequality yields the minium kick velocity required to keep the system bound if more than half of the total system mass is lost in the explosion
        self.SNcheck3 = (self.Vkick/self.Vr < 1 + np.sqrt(2*Mtot_post/Mtot_pre)) & ((Mtot_post/Mtot_pre > 0.5) | (self.Vkick/self.Vr > 1 - np.sqrt(2*Mtot_post/Mtot_pre)))
        
        # Check 4: An upper limit on the mass of the compact object progenitor can be derived from the condition that the azimuthal direction of the kick is real (Eq. 26, Fryer & Kalogera 1997)

        # first need to make sure that e_post <= 1, otherwise we'll get error
        self.SNcheck4 = (self.epost <= 1)

        idxs = np.where(self.SNcheck4==True)[0]
        Mtot_post_temp = self.Mns[idxs]+self.Mcomp[idxs]

        kvar = 2*(self.Apost[idxs]/self.Apre[idxs])-(((self.Vkick[idxs]**2)*self.Apost[idxs] / (G*Mtot_post_temp))+1)
        term1 = kvar**2 * Mtot_post_temp * (self.Apre[idxs]/self.Apost[idxs])
        term2 = 2 * (self.Apost[idxs]/self.Apre[idxs])**2 * (1-self.epost[idxs]**2) - kvar
        term3 = -2 * (self.Apost[idxs]/self.Apre[idxs]) * np.sqrt(1-self.epost[idxs]**2) * np.sqrt((self.Apost[idxs]/self.Apre[idxs])**2 * (1-self.epost[idxs]**2) - kvar)
        max_val = -self.Mcomp[idxs] + term1 / (term2 + term3)

        self.SNcheck4[idxs] = (self.Mhe[idxs] <= max_val)


        # Now, create series to see if the system passes all the checks
        self.SNsurvive = ((self.SNcheck1==True) & (self.SNcheck2==True) & (self.SNcheck3==True) & (self.SNcheck4==True))

        # Also, return the survival fraction
        survival_fraction = float(np.sum(self.SNsurvive))/float(len(self.SNsurvive))
        return survival_fraction



    def galactic_velocity(self, gal, t0):
        """
        Calculates the pre-SN galactic velocity for the tracer particles at their initial radius R. 
        """

        # Using galpy's vcirc method, we can easily calculate the rotation velocity at any R
        # Note we use the combination of *all* potentials up to the timestep t0
        Vcirc = gp.potential.vcirc(gal.full_potentials[:(t0+1)], self.R*u.cm)
        self.Vcirc = Vcirc.to(u.cm/u.s).value

        # Just to have them, calculate the circular velocity of each component as well
        Vcirc_stars = gp.potential.vcirc(gal.stars_potentials[:(t0+1)], self.R*u.cm)
        self.Vcirc_stars = Vcirc_stars.to(u.cm/u.s).value
        Vcirc_gas = gp.potential.vcirc(gal.gas_potentials[:(t0+1)], self.R*u.cm)
        self.Vcirc_gas = Vcirc_gas.to(u.cm/u.s).value
        Vcirc_dm = gp.potential.vcirc(gal.dm_potentials[:(t0+1)], self.R*u.cm)
        self.Vcirc_dm = Vcirc_dm.to(u.cm/u.s).value
        

        # Also, get mass enclosed at each rad by taking cumulative sum of mass profiles
        mass_stars_enclosed = np.cumsum(gal.mass_stars_prof[t0])
        mass_gas_enclosed = np.cumsum(gal.mass_gas_prof[t0])
        mass_dm_enclosed = np.cumsum(gal.mass_dm_prof[t0])

        # Create interpolation function for enclosed masses
        mass_stars_enclosed_interp = interp1d(gal.rads, mass_stars_enclosed)
        mass_gas_enclosed_interp = interp1d(gal.rads, mass_gas_enclosed)
        mass_dm_enclosed_interp = interp1d(gal.rads, mass_dm_enclosed)
        
        # Calculate the enclosed mass for each of our tracer systems
        mass_stars_enclosed_systems = mass_stars_enclosed_interp(self.R)
        mass_gas_enclosed_systems = mass_gas_enclosed_interp(self.R)
        mass_dm_enclosed_systems = mass_dm_enclosed_interp(self.R)

        self.Menc = mass_stars_enclosed_systems+mass_gas_enclosed_systems+mass_dm_enclosed_systems



    def galactic_frame(self):
        """
        Transforms the velocity vectors of the system following the SN to the galactic frame, where the galaxy disk is in the x-y plane and the system is moving in the positive y direction prior to the SN. 

        Assume that the systemic velocity post-SN is in the same direction of the pre-SN galactic velocity (+y direction). Then perform Z-axis Euler rotation of SYSphi and Y-axis Euler rotation of SYStheta. 
        """

        # create Vsys array (Nsamples x Ndim)
        Vsys_vec = np.transpose([self.Vsx,self.Vsy,self.Vsz])

        # Rotate Vsys about the Z-axis by SYSphi
        Vsys_vec = utils.euler_rot(Vsys_vec, np.asarray(self.SYSphi), axis='Z')

        # Rotate Vsys about the Y-axis by SYStheta
        Vsys_vec = utils.euler_rot(Vsys_vec, np.asarray(self.SYStheta), axis='Y')

        # Save the velocity of the system immediately following the SN
        # Don't forget to add the pre-SN galactic velocity to the y-component!
        self.Vpx = Vsys_vec[:,0]
        self.Vpy = Vsys_vec[:,1] + self.Vcirc
        self.Vpz = Vsys_vec[:,2]
        self.Vpost = np.linalg.norm(np.asarray([self.Vpx,self.Vpy,self.Vpz]), axis=0)

        # NaN out the post-SN velocity for systems that were disrupted, as this is ambiguous
        disrupt_idx = np.argwhere(self.SNsurvive == False)
        self.Vpx[disrupt_idx] = np.nan
        self.Vpy[disrupt_idx] = np.nan
        self.Vpz[disrupt_idx] = np.nan
        self.Vpost[disrupt_idx] = np.nan
        
        


    def inspiral_time(self, Nsys):
        """
        Calculates the GW inspiral time (in seconds) for the systems given their post-SN orbital properties
        """

        self.Tinsp = np.nan * np.ones(Nsys)

        lt_tH_insp = 0
        for idx in np.arange(Nsys):

            # for systems that were disrupted, continue
            if self.SNsurvive[idx] == False:
                continue

            # if system is still bound, calculate the inspiral time using Peters 1964
            # FIXME: should have some way to pick out systems with inspiral times greater than a Hubble time so we don't have to integrate them
            else:
                m1 = self.Mcomp[idx] * u.g.to(u.Msun)
                m2 = self.Mns[idx] * u.g.to(u.Msun)
                a0 = self.Apost[idx] * u.cm.to(u.AU)
                e0 = self.epost[idx]

                self.Tinsp[idx] = utils.inspiral_time_peters(a0, e0, m1, m2) * u.Gyr.to(u.s)

                # count the number of systems that merge in more/less than a Hubble time
                if self.Tinsp[idx] < (14 * u.Gyr.to(u.s)):
                    lt_tH_insp += 1

        # return the fraction that merge within a Hubble time
        return float(lt_tH_insp)/np.sum(self.SNsurvive)


