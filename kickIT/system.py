import numpy as np
import pandas as pd

import time
import multiprocessing
from functools import partial
import copy
import os

import astropy.units as u
import astropy.constants as C

from scipy.integrate import ode
from scipy.integrate import quad
from scipy.interpolate import interp1d

from galpy.potential import vcirc
from galpy.orbit import Orbit
from galpy.potential import evaluatePotentials

from kickIT.galaxy_history import cosmology
from . import utils


VERBOSE=True

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
    def __init__(self, sampled_parameters, SNphi=None, SNtheta=None, SYSphi=None, SYStheta=None, sample_progenitor_props=False):

        self.R = np.asarray(sampled_parameters['R'])*u.kpc
        self.t0 = np.asarray(sampled_parameters['t0'])
        self.tbirth = np.asarray(sampled_parameters['tbirth'])*u.Gyr
        self.zbirth = np.asarray(sampled_parameters['zbirth'])*u.dimensionless_unscaled
        self.Nsys = len(self.R)

        # --- read in the sampled progenitor parameters specific to both sampling methods
        if sample_progenitor_props:
            self.Mns = np.asarray(sampled_parameters['Mns'])*u.Msun
            self.Mcomp = np.asarray(sampled_parameters['Mcomp'])*u.Msun
            self.Mhe = np.asarray(sampled_parameters['Mhe'])*u.Msun
            self.Apre = np.asarray(sampled_parameters['Apre'])*u.Rsun
            self.epre = np.asarray(sampled_parameters['epre'])*u.dimensionless_unscaled
            self.Vkick = np.asarray(sampled_parameters['Vkick'])*u.km/u.s
        else:
            self.Vsys = np.asarray(sampled_parameters['Vsys'])*u.km/u.s
            self.Tinsp = np.asarray(sampled_parameters['Tinsp'])*u.Gyr
            self.SNsurvive = np.asarray(sampled_parameters['SNsurvive'])
        

        #  --- initialize random angles (only need SN angles if implementing the SN)
        if sample_progenitor_props:
            if SNphi: self.SNphi = SNphi*u.rad
            else: self.SNphi = 2*np.pi*np.random.random(self.Nsys)*u.rad

            if SNtheta: self.SNtheta = SNtheta*u.rad
            else: self.SNtheta = np.arccos(2*np.random.random(self.Nsys)-1)*u.rad

        if SYSphi: self.SYSphi = SYSphi*u.rad
        else: self.SYSphi = 2*np.pi*np.random.random(self.Nsys)*u.rad

        if SYStheta: self.SYStheta = SYStheta*u.rad
        else: self.SYStheta = np.arccos(2*np.random.random(self.Nsys)-1)*u.rad


    def escape_velocity(self, gal, interpolants):
        """
        Calculates the escape velocity for each particle at their respective radius.
        """

        print('Calculating particle escape velocities...\n')

        # Read in Rvals, assuming particles start in the plane
        R_vals = self.R
        z_vals = np.zeros_like(R_vals)

        # specify radius and height at "infinity" in kpc
        R_inf = 1000*u.kpc
        z_inf = 1000*u.kpc

        Vescs = []
        for idx, (t0,R,z) in enumerate(zip(self.t0, R_vals, z_vals)):
            if interpolants:
                potential = interpolants[t0]
            else:
                potential = gal.full_potentials[t0]

            pot_at_inf = evaluatePotentials(potential, R_inf, z_inf).value
            Vescs.append(np.sqrt(2*(pot_at_inf - evaluatePotentials(potential, R, z).value)))

        self.Vesc = np.asarray(Vescs)*u.km/u.s
        return



    def galactic_velocity(self, gal, interpolants, fixed_potential=None):
        """
        Calculates the pre-SN galactic velocity for the tracer particles at their initial radius R. 
        """

        print('Calculating the pre-SN galactic velocity...\n')

        # --- Using galpy's vcirc method, we can easily calculate the rotation velocity at any R

        R_vals = self.R

        Vcircs = []
        for idx, (t0,R) in enumerate(zip(self.t0, R_vals)):
            # if fixed_potential, we take the potential at the specified timestep only
            if fixed_potential:
                t0_pot = fixed_potential
            else:
                t0_pot = t0

            # if interpolants are provided, use these for the calculation
            if interpolants:
                potential = interpolants[t0_pot]
            else:
                potential = gal.full_potentials[t0_pot]

            Vcircs.append(vcirc(potential, R).value)

        self.Vcirc = np.asarray(Vcircs)*u.km/u.s

        

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

        print('Implementing the supernova physics...\n')

        # get G in Msun/km/s units
        G = C.G.to(u.km**3 / u.Msun / u.s**2)
        # NOTE: Need to make sure Apre is converted to km!
        Apre = self.Apre.to(u.km)

        # Decompose the kick into its cartesian coordinates
        self.Vkx = self.Vkick*np.sin(self.SNtheta)*np.sin(self.SNphi)
        self.Vky = self.Vkick*np.cos(self.SNtheta)
        self.Vkz = self.Vkick*np.sin(self.SNtheta)*np.cos(self.SNphi)

        # Calculate the relative velocity according to Kepler's Law
        self.Vr = np.sqrt(G * (self.Mhe+self.Mcomp) / Apre)

        # Calculate the post-SN orbital properties (Eqs 3 and 4 from Kalogera 1996)
        Mtot_post = self.Mns+self.Mcomp

        Apost = G*(Mtot_post) * ((2*G*Mtot_post/Apre) - (self.Vkick**2) - (self.Vr**2) - 2*self.Vky*self.Vr)**(-1.0)
        x = ((self.Vkz**2 + self.Vky**2 + self.Vr**2 + 2*self.Vky*self.Vr)*Apre**2) / (G * Mtot_post * Apost)
        self.epost = np.sqrt(1-x)

        # Calculate the post-SN systemic velocity (Eq 34 from Kalogera 1996)
        self.Vsx = self.Mns*self.Vkx / Mtot_post
        self.Vsy = (self.Mns*self.Vky - ((self.Mhe-self.Mns)*self.Mcomp / (self.Mhe+self.Mcomp) * self.Vr)) / Mtot_post
        self.Vsz = self.Mns*self.Vkz / Mtot_post
        self.Vsys = np.sqrt(self.Vsx**2 + self.Vsy**2 + self.Vsz**2)

        # Calculate the tile of the orbital plane from the SN (Eq 5 from Kalogera 1996)
        self.tilt = np.arccos((self.Vky+self.Vr) / np.sqrt((self.Vky+self.Vr)**2 + self.Vkz**2))

        # Now, convert Apost to Rsun
        self.Apost = Apost.to(u.Rsun)



    def check_survival(self):
        """
        Checks to see if the systems survived the supernova explosion. "True" if the system passes the check, "False" if system does not pass the check. 

        References: 
        Willems et al 2002: http://iopscience.iop.org/article/10.1086/429557/meta
            We use eq 21, 22, 23, 24, 25, 26 for checks of SN survival

        Kalogera and Lorimer 2000: http://iopscience.iop.org/article/10.1086/308417/meta

        Note: V_He;preSN is the same variable as V_r from Kalogera 1996
        """

        print('Checking if the systems survived the supernovae...\n')

        # get G in Msun/km/s units
        G = C.G.to(u.km**3 / u.Msun / u.s**2)
        # NOTE: Need to make sure Apre and Apost is converted to km!
        Apre = self.Apre.to(u.km)
        Apost = self.Apost.to(u.km)

        Mtot_pre = self.Mhe+self.Mcomp
        Mtot_post = self.Mns+self.Mcomp

        # Check 1: Continuity demands that post-SN orbits must pass through the pre-SN positions (Eq 21 from Flannery & Van Heuvel 1975)
        self.SNcheck1 = (1-self.epost <= Apre/Apost) & (Apre/Apost <= 1+self.epost)

        # Check 2: Lower and upper limites on amount of orbital contraction or expansion that can take place for a given amount of mass loss and a given natal kick velocity (Kalogera & Lorimer 2000)
        self.SNcheck2 = (Apre/Apost < 2-((Mtot_pre/Mtot_post)*((self.Vkick/self.Vr)-1)**2)) & (Apre/Apost > 2-((Mtot_pre/Mtot_post)*((self.Vkick/self.Vr)+1)**2))

        # Check 3: The magnitude of the kick velocity imparted to the compact object at birth is restricted to a certain range (Brandy & Podsiadlowski 1995; Kalogera & Lorimer 2000)
        # The first inequality expresses the requirement that the bianry must remain bound after the SN explosion
        # The second inequality yields the minium kick velocity required to keep the system bound if more than half of the total system mass is lost in the explosion
        self.SNcheck3 = (self.Vkick/self.Vr < 1 + np.sqrt(2*Mtot_post/Mtot_pre)) & ((Mtot_post/Mtot_pre > 0.5) | (self.Vkick/self.Vr > 1 - np.sqrt(2*Mtot_post/Mtot_pre)))
        
        # Check 4: An upper limit on the mass of the compact object progenitor can be derived from the condition that the azimuthal direction of the kick is real (Eq. 26, Fryer & Kalogera 1997)

        # first need to make sure that e_post <= 1, otherwise we'll get error
        self.SNcheck4 = (self.epost <= 1)

        idxs = np.where(self.SNcheck4==True)[0]
        Mtot_post_temp = self.Mns[idxs]+self.Mcomp[idxs]

        kvar = 2*(Apost[idxs]/Apre[idxs])-(((self.Vkick[idxs]**2)*Apost[idxs] / (G*Mtot_post_temp))+1)
        term1 = kvar**2 * Mtot_post_temp * (Apre[idxs]/Apost[idxs])
        term2 = 2 * (Apost[idxs]/Apre[idxs])**2 * (1-self.epost[idxs]**2) - kvar
        term3 = -2 * (Apost[idxs]/Apre[idxs]) * np.sqrt(1-self.epost[idxs]**2) * np.sqrt((Apost[idxs]/Apre[idxs])**2 * (1-self.epost[idxs]**2) - kvar)
        max_val = -self.Mcomp[idxs] + term1 / (term2 + term3)

        self.SNcheck4[idxs] = (self.Mhe[idxs] <= max_val)


        # Now, create series to see if the system passes all the checks
        self.SNsurvive = ((self.SNcheck1==True) & (self.SNcheck2==True) & (self.SNcheck3==True) & (self.SNcheck4==True))

        # Also, return the survival fraction
        survival_fraction = float(np.sum(self.SNsurvive))/float(len(self.SNsurvive))
        return survival_fraction



    def inspiral_time(self, Tinsp_max=14):
        """
        Calculates the GW inspiral time (in seconds) for the systems given their post-SN orbital properties
        """

        print('Calculating inspiral times...\n')

        Tinsps = []
        lessthan_tH = 0

        for idx in np.arange(self.Nsys):

            # for systems that were disrupted, continue
            if self.SNsurvive[idx] == False:
                Tinsps.append(np.nan)
                continue

            # if system is still bound, calculate the inspiral time using Peters 1964 (note that it takes in A in AU, returns Tinsp in Gyr)
            else:
                m1 = self.Mcomp[idx].value
                m2 = self.Mns[idx].value
                a0 = self.Apost[idx].to(u.AU).value
                e0 = self.epost[idx].value

                Tinsp = utils.inspiral_time_peters(a0, e0, m1, m2)

                # count the number of systems that merge in more/less than a Hubble time
                if (Tinsp < Tinsp_max):
                    lessthan_tH += 1

                Tinsps.append(Tinsp)

        self.Tinsp = np.asarray(Tinsps)*u.Gyr

        # return the fraction that merge within a Hubble time
        if np.sum(self.SNsurvive) == 0:
            return 0.0
        else:
            return float(lessthan_tH)/np.sum(self.SNsurvive)



    
    def galactic_frame(self):
        """
        Transforms the velocity vectors of the system following the SN to the galactic frame, where the galaxy disk is in the x-y plane and the system is moving in the positive y direction prior to the SN. 

        Assume that the systemic velocity post-SN is in the same direction of the pre-SN galactic velocity (+y direction). Then perform Z-axis Euler rotation of SYSphi and Y-axis Euler rotation of SYStheta. 
        """

        print('Transforming systems into the galactic frame of reference...\n')

        # create Vsys array (Nsamples x Ndim)
        Vsys_vec = np.transpose([self.Vsx,self.Vsy,self.Vsz])

        # Rotate Vsys about the Z-axis by SYSphi
        Vsys_vec = utils.euler_rot(Vsys_vec, np.asarray(self.SYSphi), axis='Z')

        # Rotate Vsys about the Y-axis by SYStheta
        Vsys_vec = utils.euler_rot(Vsys_vec, np.asarray(self.SYStheta), axis='Y')

        # Save the velocity of the system immediately following the SN
        # Don't forget to add the pre-SN galactic velocity to the y-component!
        self.Vpx = Vsys_vec[:,0]*u.km/u.s
        self.Vpy = Vsys_vec[:,1]*u.km/u.s + self.Vcirc
        self.Vpz = Vsys_vec[:,2]*u.km/u.s
        self.Vpost = np.linalg.norm(np.asarray([self.Vpx,self.Vpy,self.Vpz]), axis=0)*u.km/u.s

        # NaN out the post-SN velocity for systems that were disrupted, as this is ambiguous
        disrupt_idx = np.argwhere(self.SNsurvive == False)
        self.Vpx[disrupt_idx] = np.nan
        self.Vpy[disrupt_idx] = np.nan
        self.Vpz[disrupt_idx] = np.nan
        self.Vpost[disrupt_idx] = np.nan
        
        
    def decompose_Vsys(self):
        """
        Decomposes systemic velocity (magnitude) into galactic frame using SYStheta and SYSphi, for when sampling only Vsys and R 
        """

        print('Decomposing systemic velocities into galactic frame of reference...\n')

        # Save the velocity of the system immediately following the SN
        # Don't forget to add the pre-SN galactic velocity to the y-component!
        self.Vpx = self.Vsys*np.sin(self.SYStheta)*np.cos(self.SYSphi)
        self.Vpy = self.Vsys*np.sin(self.SYStheta)*np.sin(self.SYSphi) + self.Vcirc
        self.Vpz = self.Vsys*np.cos(self.SYStheta)
        self.Vpost = np.linalg.norm(np.asarray([self.Vpx,self.Vpy,self.Vpz]), axis=0)



    def evolve(self, gal, ro=8, vo=220, multiproc=None, int_method='odeint', Tinsp_lim=False, Tint_max=60, Nsteps_per_bin=1000, save_traj=False, downsample=None, outdir=None, fixed_potential=False):
        """
        Evolves the tracer particles using galpy's 'Evolve' method
        Does for each bound systems until one of two conditions are met:
            1. The system evolves until the time of the sGRB
            2. The system merges due to GW emission (if Tinsp_lim=True, otherwise will evolve until the time of the sgrb)

        Each system will evolve through a series of galactic potentials specified in distinct redshift bins in the 'gal' class

        Note that all units are cgs unless otherwise specified, and galpy is initialized to take in astropy units
        """
        print('Evolving orbits of the tracer particles...\n')

        # set up arrays/lists of arrays for storing output
        merger_redzs = []
        R_offsets = []
        Rproj_offsets = []
        Xs = []
        Ys = []
        Zs = []
        vXs = []
        vYs = []
        vZs = []


        # get the pertinent data for the evolution function
        systems_info = []
        for idx in np.arange(self.Nsys):
            systems_info.append([idx,self.SNsurvive[idx],self.Tinsp[idx],self.R[idx],self.Vpx[idx],self.Vpy[idx],self.Vpz[idx]])

        # enable multiprocessing, if specifed
        if multiproc:
            if multiproc=='max':
                mp = multiprocessing.cpu_count()
            else:
                mp = int(multiproc)

            # initialize the parallelization, and specify function arguments
            pool = multiprocessing.Pool(mp)
            func = partial(integrate_orbits, t0=t0, gal=gal, int_method=int_method, Tinsp_lim=Tinsp_lim, Tint_max=Tint_max, Nsteps_per_bin=Nsteps_per_bin, save_traj=save_traj, downsample=downsample, outdir=outdir, fixed_potential=fixed_potential, VERBOSE=self.VERBOSE)

            start = time.time()
            print('Parallelizing integration of the orbits over {0:d} cores...'.format(mp))
            results = pool.map(func, systems_info)
            pool.close()
            pool.join()

            results = np.transpose(results)
            Xs,Ys,Zs,vXs,vYs,vZs,R_offsets,Rproj_offsets,merger_redzs = results[0],results[1],results[2],results[3],results[4],results[5],results[6],results[7],results[8]
            stop = time.time()
            print('Finished! It took {0:0.2f}s\n'.format(stop-start))



        # otherwise, loop over all tracers in serial
        else:
            start = time.time()
            print('Performing the integrations in serial...')

            for system in systems_info:

                func = partial(integrate_orbits, t0=t0, gal=gal, int_method=int_method, Tinsp_lim=Tinsp_lim, Tint_max=Tint_max, Nsteps_per_bin=Nsteps_per_bin, save_traj=save_traj, downsample=downsample, outdir=outdir, fixed_potential=fixed_potential,VERBOSE=self.VERBOSE)
                X,Y,Z,vX,vY,vZ,R_offset,Rproj_offset,merger_redz = func(system)

                Xs.append(X)
                Ys.append(Y)
                Zs.append(Z)
                vXs.append(vX)
                vYs.append(vY)
                vZs.append(vZ)
                R_offsets.append(R_offset)
                Rproj_offsets.append(Rproj_offset)
                merger_redzs.append(merger_redz)

            stop = time.time()
            print('Finished! It took {0:0.2f}s\n'.format(stop-start))


        # store everything in systems class
        self.merger_redz = np.asarray(merger_redzs)
        self.R_offset = np.asarray(R_offsets)
        self.Rproj_offset = np.asarray(Rproj_offsets)
        self.X = np.asarray(Xs)
        self.Y = np.asarray(Ys)
        self.Z = np.asarray(Zs)
        self.vX = np.asarray(vXs)
        self.vY = np.asarray(vYs)
        self.vZ = np.asarray(vZs)

        # combine the trajectories into a single hdf5 file, if save_traj==True
        print('Combining trajectory files into single hdf5 file...\n')
        if save_traj==True:
            for f in os.listdir(outdir):
                if 'trajectories_' in f:
                    temp = pd.read_hdf(outdir+'/'+f, key='system')
                    _, idx = f.split('_')
                    idx, _ = idx.split('.')
                    temp.to_hdf(outdir+'/trajectories.hdf', key='system_'+str(idx))
                    os.remove(outdir+'/'+f)

        return

            


    def write(self, outdir, t0):
        """Write data as hdf file to specified outpath.
        Each key represents a timestep at which the particles are initialed in the galaxy model.
        """

        print("Writing systems data in directory '{0:s}'...\n".format(outdir))

        tracers = pd.DataFrame()
        for attr, values in self.__dict__.items():
            if attr not in ['VERBOSE','Nsys']:
                tracers[attr] = values

        step_key = 'step_{:03d}'.format(t0)

        tracers.to_hdf(outdir+'/tracers.hdf', key=step_key, mode='a')
            
        return




def integrate_orbits(system, t0, gal, int_method='odeint', Tinsp_lim=False, Tint_max=60, Nsteps_per_bin=1000, save_traj=False, downsample=None, outdir=None, fixed_potential=False, VERBOSE=False):
    """Function for integrating orbits. 

    If Tinsp_lim==False, will integrate ALL systems until the time of the sgrb, regardless of Tinsp.
    
    Tint_max will end integration if t_int > Tint_max.

    If save_traj == True, will save the full trajectory information rather than just the last step. If downsample is also specified, will save only every Nth line in the trajectories dataframe.
    """

    start_time = time.time()

    # NaN values to initialize positions
    X,Y,Z,vX,vY,vZ,R_offset,Rproj_offset = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan

    # Lists for storing trajectories, if save_traj==True
    if save_traj:
        X_traj,Y_traj,Z_traj,vX_traj,vY_traj,vZ_traj,R_offset_traj,Rproj_offset_traj,time_traj = [],[],[],[],[],[],[],[],[]

    # system info
    idx = system[0]
    SNsurvive = system[1]
    Tinsp = system[2]
    R = system[3]
    Vpx = system[4]
    Vpy = system[5]
    Vpz = system[6]

    # gal info
    interp = gal.interp
    times = gal.times
    redz = gal.redz
    full_potentials = gal.full_potentials
    if interp:
        interpolated_potentials = gal.interpolated_potentials
    
    # initialize cosmology
    cosmo = cosmology.Cosmology()

    INSP_FLAG=False
    MERGED=False
    FINISHED_EVOLVING=False

    while FINISHED_EVOLVING==False:

        # first, check that the system survived ther supernova
        if SNsurvive == False:
            # write in NaNs here for merger_redz
            merger_redz = np.nan

            FINISHED_EVOLVING=True
            return X,Y,Z,vX,vY,vZ,R_offset,Rproj_offset,merger_redz


        # if the system survived the supernova, jot down its inspiral time
        if interp:
            Tinsp = utils.Tcgs_to_nat(Tinsp)

        # keep track of the time that has elapsed
        T_elapsed = 0


        ### MAIN LOOP ### 
        # through all redshifts and evolving potential starting at timestep t0
        tt=t0

        while tt < (len(times)-1):

            # if potential is held fixed, 'tt_pot' is the last timestep
            if fixed_potential:
                tt_pot = len(times)-2
            else:
                tt_pot = tt

            # first, transform the post-SN systemic velocity into cylindrical coordinates
            if tt==t0:
                # by construction, the systems start in the galactic plane, at x=R, y=0, and therefore phi=0
                # also, galpy's orbit integrator takes in vT = R*vPhi
                R,Phi,Z,vR,vPhi,vZ = utils.cartesian_to_cylindrical(R,0.0,0.0,Vpx,Vpy,Vpz)
                vT = R*vPhi
                if interp:
                    R,vR,vT,Z,vZ,Phi = utils.orbit_cgs_to_nat(R,vR,vT,Z,vZ,Phi)
                    
            else:
                # extract the orbital properties at the end of the previous integration (note that galpy output is [r,vR,vT,Z,vZ,T] and already in natural units)
                R,vR,vT,Z,vZ,Phi = orb.getOrbit()[-1]
                Phi = Phi % (2*np.pi)
                if not interp:
                    # convert from galpy's 'natural' units to cgs
                    R,vR,vT,Z,vZ,Phi = utils.orbit_nat_to_cgs(R,vR,vT,Z,vZ,Phi)
    
            # record the amount of time that passes in this step
            # compare the total time passed to the inspiral time of the system
            dt = times[tt+1]-times[tt]
            if interp:
                dt = utils.Tcgs_to_nat(dt)


            # --- See if the merger occurred during this step
            # note that if Tinsp_lim=False, it will continue the integration until the time of the sGRB
            if (((T_elapsed+dt) > Tinsp) and INSP_FLAG==False):

                # see where it merged
                dt_merge = (Tinsp - T_elapsed)

                # get timesteps for this integration (set to 1000 steps for now)
                if interp:
                    ts = np.linspace(0,dt_merge,Nsteps_per_bin)
                else:
                    ts = np.linspace(0*u.s,dt_merge*u.s,Nsteps_per_bin)

                # initialize the orbit and integrate, store redshift of merger
                if interp:
                    orb = gp_orbit(vxvv=[R, vR, vT, Z, vZ, Phi])
                    orb.integrate(ts, interpolated_potentials[tt_pot], method=int_method)
                    age = utils.Tcgs_to_nat(times[t0])+Tinsp
                    merger_redz = float(cosmo.tage_to_z(utils.Tnat_to_cgs(age)))
                else:
                    orb = gp_orbit(vxvv=[R*u.cm, vR*(u.cm/u.s), vT*(u.cm/u.s), Z*u.cm, vZ*(u.cm/u.s), Phi*u.rad])
                    orb.integrate(ts, full_potentials[(tt_pot+1)], method=int_method)
                    age = times[t0]+Tinsp
                    merger_redz = float(cosmo.tage_to_z(age))

                INSP_FLAG=True
                MERGED=True

                # If Tinsp_lim==True, stop the integration here
                if Tinsp_lim==True:
                    FINISHED_EVOLVING = True
                    stop_time = time.time()

                    if VERBOSE:
                        print('  Tracer {0:d}:\n    merger occurred at z={1:0.2f}...integration took {2:0.2f}s'.format(idx, merger_redz, (stop_time-start_time)))
                    break


            # --- If it didn't merge, evolve until the next timestep
            T_elapsed += dt

            # get timesteps for this integration (set to 1000 steps for now)
            if interp:
                ts = np.linspace(0,dt,Nsteps_per_bin)
            else:
                ts = np.linspace(0*u.s,dt*u.s,Nsteps_per_bin)

            # initialize the orbit and integrate
            if interp:
                orb = gp_orbit(vxvv=[R, vR, vT, Z, vZ, Phi])
                orb.integrate(ts, interpolated_potentials[tt_pot], method=int_method)
            else:
                orb = gp_orbit(vxvv=[R*u.cm, vR*(u.cm/u.s), vT*(u.cm/u.s), Z*u.cm, vZ*(u.cm/u.s), Phi*u.rad])
                orb.integrate(ts, full_potentials[(tt_pot+1)], method=int_method)


            # if it evolved until the time of the sGRB, end the integration
            if tt == (len(times)-2):
                time_evolved = times[(tt+1)]-times[t0]

                stop_time = time.time()

                if MERGED:
                    if VERBOSE:
                        print('  Tracer {0:d}:\n    merger occurred at z={1:0.2f}...integration took {2:0.2f}s'.format(idx, merger_redz, (stop_time-start_time)))

                else:
                    # set merger_redz to 0
                    merger_redz = 0
                    if VERBOSE:
                        print('  Tracer {0:d}:\n    system evolved for {1:0.2e} Myr and did not merge prior to the sGRB...integration took {2:0.2f}s'.format(idx, time_evolved*u.s.to(u.Myr), (stop_time-start_time)))

                FINISHED_EVOLVING = True
                break


            # if integration time surpasses Tint_max, end
            if (time.time()-start_time) > Tint_max:
                time_evolved = times[(tt+1)]-times[t0]

                # set merger_redz to -1 to keep track of these
                merger_redz = -1

                print('  Tracer {0:d}:\n    system evolved for {1:0.2e} Myr and longer than maximum integration time ({2:0.1f}s), integration terminated'.format(idx, time_evolved*u.s.to(u.Myr), Tint_max))

                FINISHED_EVOLVING=True
                break


            # append orbit information at this step, if save_traj==True
            X,Y,Z,vX,vY,vZ,R_offset,Rproj_offset = transform_orbits(copy.deepcopy(orb))

            if save_traj:
                X_traj.append(X)
                Y_traj.append(Y)
                Z_traj.append(Z)
                vX_traj.append(vX)
                vY_traj.append(vY)
                vZ_traj.append(vZ)
                R_offset_traj.append(R_offset)
                Rproj_offset_traj.append(Rproj_offset)
                if interp:
                    int_times = times[t0]+utils.Tnat_to_cgs(T_elapsed+ts)
                else:
                    int_times = times[t0]+ts.value+T_elapsed
                time_traj.append(int_times)

            tt += 1


    # once the system has either merged or integrated until the time of the sGRB, save trajectory information
    X,Y,Z,vX,vY,vZ,R_offset,Rproj_offset = transform_orbits(copy.deepcopy(orb))

    if VERBOSE:
        print('    final offset: {0:0.2f} kpc (proj: {1:0.2f} kpc)\n'.format(R_offset[-1]*u.cm.to(u.kpc), Rproj_offset[-1]*u.cm.to(u.kpc)))

    # append orbit information at this step, if save_traj==True
    if save_traj:
        X_traj.append(X)
        Y_traj.append(Y)
        Z_traj.append(Z)
        vX_traj.append(vX)
        vY_traj.append(vY)
        vZ_traj.append(vZ)
        R_offset_traj.append(R_offset)
        Rproj_offset_traj.append(Rproj_offset)
        if interp:
            int_times = times[t0]+utils.Tnat_to_cgs(T_elapsed+ts)
        else:
            int_times = times[t0]+ts.value+T_elapsed
        time_traj.append(int_times)

        # flatten and save trajectories
        trajectories = pd.DataFrame()
        trajectories['X'] = [item for sublist in X_traj for item in sublist]
        trajectories['Y'] = [item for sublist in Y_traj for item in sublist]
        trajectories['Z'] = [item for sublist in Z_traj for item in sublist]
        trajectories['vX'] = [item for sublist in vX_traj for item in sublist]
        trajectories['vY'] = [item for sublist in vY_traj for item in sublist]
        trajectories['vZ'] = [item for sublist in vZ_traj for item in sublist]
        trajectories['R_offset'] = [item for sublist in R_offset_traj for item in sublist]
        trajectories['Rproj_offset'] = [item for sublist in Rproj_offset_traj for item in sublist]
        trajectories['time'] = [item for sublist in time_traj for item in sublist]
        # if downsample is specified, apply here
        if downsample:
            trajectories = trajectories.iloc[::downsample, :]

        # save each trajectory separately, then combine at the end
        trajectories.to_hdf(outdir+'/trajectories_'+str(idx)+'.hdf', key='system', mode='a')


    # return the final values
    return X[-1],Y[-1],Z[-1],vX[-1],vY[-1],vZ[-1],R_offset[-1],Rproj_offset[-1],merger_redz




def transform_orbits(orb):
    """Takes in orbit, transforms to cartesian (cgs units), and calculates offsets/projected offsets.

    By default, just returns the final values in cartesian coordinates, as well as the offset and projected offset. 
    """
    # NOTE: galpy's code spits things out in natural units no matter what you input!!!

    Rs = orb.getOrbit()[:,0]
    vRs = orb.getOrbit()[:,1]
    vTs = orb.getOrbit()[:,2]
    Zs = orb.getOrbit()[:,3]
    vZs = orb.getOrbit()[:,4]
    Phis = orb.getOrbit()[:,5] % (2*np.pi)

    # convert from natural to cgs units
    # NOTE: THIS CHANGES THE INSTANCE ORBIT TO CGS TOO!!!
    Rs, vRs, vTs, Zs, vZs, Phis = utils.orbit_nat_to_cgs(Rs, vRs, vTs, Zs, vZs, Phis)
    vPhis = vTs / Rs
    R_offsets = np.sqrt(Rs**2 + Zs**2)

    # get positions and velocities in Cartesian coordinates
    Xs,Ys,Zs,vXs,vYs,vZs = utils.cylindrical_to_cartesian(Rs, Phis, Zs, vRs, vPhis, vZs)

    # randomly rotate the vectors by Euler rotations to get a mock projected offset, assuming observer is in z-hat direction
    vecs = np.transpose([Xs,Ys,Zs])   # (Nsamples x Ndim)
    rot_vecs = utils.euler_rot(vecs, np.ones(len(vecs))*(2*np.pi*np.random.random()), axis='X')
    rot_vecs = utils.euler_rot(rot_vecs, np.ones(len(vecs))*(2*np.pi*np.random.random()), axis='Y')
    rot_vecs = utils.euler_rot(rot_vecs, np.ones(len(vecs))*(2*np.pi*np.random.random()), axis='Z')

    Rproj_offsets = np.sqrt(rot_vecs[:,0]**2 + rot_vecs[:,1]**2)

    return Xs,Ys,Zs,vXs,vYs,vZs,R_offsets,Rproj_offsets

