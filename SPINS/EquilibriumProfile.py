#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 09:34:46 2025

@author: ttaczak
"""
from pathlib import Path
from spins_io import (
    validate_input,
    sample_openmc_sources,
    read_input_profile
)
from spins_compute.get_equilibrium_data import (
    get_equilibrium_data
)
from spins_compute.compute_profiles import (
    get_source_profile,
    get_profile_positions,
    get_neutrons_from_distribution,
    get_global_neutron_rate
)
from spins_plotting import (
    psi_contourplot,
    plot_sources,
    plot_LCFS,
    plot_source_profile
)
from ufiles import UFILE
import math
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ["OPENMC_CROSS_SECTIONS"] = "/usr/local/lib/endfb-vii.1-hdf5/cross_sections.xml"
try:
    import openmc
except ImportError:
    None


class EquilibriumProfile():
    '''
    Provides a neutron source profile from an equilibrium
    state file and a density and temperature profile
    from the plasma midplane.
    '''
    def __init__(self,
                 input_dir, 
                 output_dir, 
                 eqdsk_fname, 
                 ni_profile_fname, 
                 Ti_profile_fname,
                 distribution_samples = 1e4,
                 theta=[0.0,math.pi/2],
                 input_type="TRANSP", 
                 Zeff=1.0, 
                 verbose=True, 
                 wall_file = None):
        
        try:
            import openmc
            self.has_openmc = True
        except ImportError:
            self.has_openmc = False
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.input_data = None
        self.ni_spline = None
        self.Ti_spline = None
        self.source_spline = None
        self.verbose = verbose
        self.R = None
        self.Z = None
        self.psi_rectspline = None
        self.psin = None
        self.num_samples = None
        self.locations = None
        self.strengths = None
        self.LCFS = None
        self.wall_data = None
        self.distribution_samples = int(distribution_samples)
        self.source_distribution = None
        self.global_neutron_rate = None
        
        self.theta = self._standardize_theta_input(theta)
        
        
        eqdsk_path = self.input_dir / Path(eqdsk_fname)
        ni_profile_path = self.input_dir / Path(ni_profile_fname)
        Ti_profile_path = self.input_dir / Path(Ti_profile_fname)
        if wall_file is not None: 
            wall_path = self.input_dir / Path(wall_file)
        else:
            wall_path = None
        
        self._validate_inputs(eqdsk_path,
                              ni_profile_path,
                              Ti_profile_path,
                              wall_path)
        
        self._load_equilibrium(eqdsk_path,wall_path)
        self._load_input_data(ni_profile_path,Ti_profile_path)
        self._compute_1D_profile_splines(Zeff, input_type)
        self._compute_3D_source_profile(n_desired=self.distribution_samples)
        self._compute_global_neutron_rate()
        
    def _standardize_theta_input(self,theta):
        if not isinstance(theta, np.ndarray):
            theta = np.array([theta])
        if not theta.shape == (2,):
            if theta.shape == (1,):
                theta = np.insert(theta,0,0)
            else:
                raise ValueError("Unrecognized shape of theta. Expects either a float or np.array with num.shape == (1,), or num.shape == (2,)")
        return theta
    
    def _compute_global_neutron_rate(self):
        return get_global_neutron_rate(self.source_distribution,self.theta,self.LCFS)
    
    def get_openmc_source_profile(self):
        if self.has_openmc:
            # create a list of openmc source objects
            my_openmc_sources = []
             
            # define an isotropic angle object in openmc
            angle = openmc.stats.Isotropic()
             
            # loop through the source file to get x,y,z,s,T
            for source in self.source_distribution:
                # check to make sure you are reading the correct numpy file
                assert source.shape == (5,)
             
                # define the space for your 
                space = openmc.stats.Point((source[0],source[1],source[2]))
                strength = source[3]
                energy = openmc.stats.muir(e0=2400000.0, m_rat=4.0, kt=source[4]) # m_rat=4.0 for D-D and m_rat=5.0 for D-T fusion
             
                source = openmc.IndependentSource(space=space,angle=angle,energy=energy,strength=strength)
                my_openmc_sources.append(source)
             
            return my_openmc_sources
        else:
            np.save("openmc_sources", self.source_distribution)
            print("Save source distribution file as openmc_sources.npy with the sources \
                  as (samples,4) matrix for (samples,[x,y,z,strength,temp])")
            return None
    
    def _validate_inputs(self,eqdsk_path, ni_profile_path, Ti_profile_path, wall_file):
        validate_input(eqdsk_path)
        validate_input(ni_profile_path)
        validate_input(Ti_profile_path)
        validate_input(wall_file)
        
    def _load_equilibrium(self,eqdsk_path,wall_file, plot = False):
        '''
        Extracts the equilibrium data from the EQDSK provided by user. From the EQDSK,
        the plasma domain, range, psi profile, and last closed flux surface (LCFS) are
        used to define the profile attributes.

        Parameters
        ----------
        plot : BOOL, optional
            PLOTS THE EQUILIBRIUM AND DEVICE WALL IF TRUE. The default is False.

        Returns
        -------
        None.

        '''
        self.R, self.Z, self.psi_rectspline, self.psin, self.LCFS, self.wall_data = get_equilibrium_data(eqdsk_path,self.verbose,wall_file)
        plot_LCFS(self.LCFS, self.psin, self.R, self.Z, self.wall_data)
        # psi_contourplot(self.R,self.Z,self.psin, self.LCFS)
        if self.verbose:
            print("Equilibrium data loaded.")
            
    def _load_input_data(self,ni_profile_path,Ti_profile_path):
        '''
        Reads input profiles from provided files. Accepts .txt, .csv, .npy, and
        .dat files. The data must be in the form of an (m,2) matrix with m datapoints
        in [rho] and [ni or Ti] coordinates.

        Parameters
        ----------
        ni_profile_path : PATH
            FILE PATH OF THE ni INPUT DATA.
        Ti_profile_path : PATH
            FILE PATH OF THE Ti INPUT DATA..

        Returns
        -------
        None.

        '''
        ni_profile = read_input_profile(ni_profile_path)
        Ti_profile = read_input_profile(Ti_profile_path)
        self.input_data = {"ne": ni_profile, "Ti": Ti_profile}

    def _compute_1D_profile_splines(self, Zeff, input_type):
        '''
        Spline fits the user-defined ni and Ti profiles. Then, the source profiles are
        defined using either the Saddler-Van-Belle formula from [1] Fausser, 2012
        or the NRL plasma formulary for D-D fusion. 

        Parameters
        ----------
        Zeff : FLOAT
            AVERAGE PLASMA CHARGE. ASSUMING QUASINEUTRALITY, THIS IS USED TO CONVERT
            ELECTRON DENSITY (THOMPSON SCATTERING) TO ION DENSITY. SET TO 1.0 IF
            ZEFF ALREADY ACCOUNTED FOR.
        input_type : STRING
            THE FORMAT OF THE DATA BEING PASSED IN. CURRENTLY THE ONLY ALLOWABLE INPUTS
            ARE 'experimental' AND 'TRANSP'

        Returns
        -------
        None.

        '''
        self.ni_spline, self.Ti_spline, self.source_spline = get_source_profile(self.input_data, Zeff)
        
        # plot_source_profile(self.ni_spline,"ni Profile","$[s^{-1}m^{-3}]$")
        # plot_source_profile(self.Ti_spline,"Ti Profile","$[eV]$")
        # plot_source_profile(self.source_spline,"Emission Profile","$[n/ \mathrm{s}/ \mathrm{m}^3]$")
        if self.verbose:
            print("Profiles computed.")
            
    def _compute_3D_source_profile(self, n_desired=1000, oversample_factor = 2):
        '''
        Defines the 3D probability distribution of the OpenMC source. Uses the LCFS
        as a bound and creates a uniformly distributed source profile using cell
        rejection. Defines the source_distribution, which is a (sample,5) matrix 
        where source_distribution[n,:] represents the (x,y,z,strength,temperature) 
        description of a sampled source location.

        Parameters
        ----------
        n_desired : INT, optional
            DESIRED NUMBER OF SAMPLED USED TO CREATE THE SOURCE DISTRIBUTION.
            IMPORTANTLY, THIS VALUE IS INDEPENDENT OF THE NEUTRONS ACTUALLY
            USED IN OPENMC. The default is 1000.
        oversample_factor : INT, optional
            A FACTOR USED FOR BATCHES DURING CELL REJECTION. WITH A FACTOR
            OF 2, THIS MEANS 2*N_DESIRED SOURCES WILL BE SAMPLED DURING
            CELL REJECTION PHASE. ONLY CHANGE IF CELL REJECTION IS TAKING
            LONGER THAN DESIRED. The default is 2.

        Returns
        -------
        None.

        '''
        self.source_distribution = get_profile_positions(n_desired,oversample_factor,self.psi_rectspline,self.source_spline,self.Ti_spline, self.R, self.Z, self.LCFS, self.theta)
        plot_sources(self.source_distribution,self.psi_rectspline, self.R, self.Z, self.psin, "3D_profile", wall_data=self.wall_data, plot_name="Neutron Source Rate Density Differences")
        if self.verbose:
            print("Profiles computed.")
            
    def plot_source_profile(self):
        plot_source_profile(self.source_spline)
        
    def plot_density_and_temperature_profiles(self):
        plot_source_profile(self.ni_spline)
        plot_source_profile(self.Ti_spline)


