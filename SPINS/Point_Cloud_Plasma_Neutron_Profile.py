#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:51:43 2024

@author: ttaczak
"""

import re
import ufiles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
from pathlib import Path
from scipy import interpolate
from scipy.interpolate import interp1d
from matplotlib.collections import PolyCollection
from shapely.geometry import Point, Polygon
import geopandas as gpd
from matplotlib.colors import Normalize
from skimage import measure
from matplotlib.path import Path as PTH
import pickle
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib.cm import ScalarMappable
from numpy.polynomial.polynomial import Polynomial
from scipy import interpolate as interp
import ufiles
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from matplotlib.patches import PathPatch
from scipy.interpolate import splrep, splev
from scipy.optimize import root_scalar
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# Set font to Times New Roman for LaTeX
latex_style_times = {
    'font.family': 'serif',
    'font.serif': ['Times'],
    'text.usetex': True,
}

# Apply the font settings
rcParams.update(latex_style_times)

#plt.rcParams['font.family'] = 'Times'
np.random.seed(2)

class EquilibriumProfile():
    '''
    Provides a neutron source profile from an equilibrium
    state file and a density and temperature profile
    from the plasma midplane.
    '''
    def __init__(
            self,
            equilibrium_fpath,
            profiles_fpath,
            data_folder_path,
            num_contours = 20,
            samples = int(1e5),
            error_res = 10,
            verbose = True,
            dpi = 200,
            READ_IN = False,
            ):
        
        self.equ_path = equilibrium_fpath
        self.profiles_path = profiles_fpath
        
        try:
            fname = open(equilibrium_fpath)
        except FileNotFoundError:
            print(equilibrium_fpath)
            print('Equilibrium state file doesnt exist.')
        finally:
            fname.close()
        
        try:
            fname = open(profiles_fpath)
        except FileNotFoundError:
            print('Profiles file doesnt exist.')
        finally:
            fname.close()
          
        self.dpi = dpi
        self.start_time = time.time()
        self.data_folder_path = data_folder_path
        self.wall_data = np.loadtxt(data_folder_path + "/MASTU_wall.txt")/1000
        
        # Create psi Mesh from contours
        self.num_contours = num_contours
        self.poloidal_res = 30
        self.profiles = pd.read_csv(profiles_fpath, sep = '\t', header = None, index_col = 0)
        self.verbose = verbose
        self.psi_rz, self.R, self.Z = self.extract_equilibrium_data(equilibrium_fpath, nc = self.num_contours, plot = False)
        self.min_psi = 0.00001
        self.max_psi = np.max(self.psi_rz)
        self.psi_rz_norm = (self.psi_rz - self.max_psi) / (0.0 - self.max_psi)
        i, j = np.unravel_index(np.argmax(self.psi_rz), self.psi_rz.shape)
        self.max_psi_indices = (i,j)
        self.max_psi_location = np.array([self.R[i,j], self.Z[i,j]])
        self.psi_rectspline = interpolate.RectBivariateSpline(self.R[0,:],self.Z[:,0], self.psi_rz)
        self.min_dist_contours = self.trim_equilibrium(plot = False)
        self.rescale_contours()
        self.LCFS = self.min_dist_contours[0]
        
        self.psi_mesh = self.define_psi_mesh(read_in = False)
        self.plot_LCFS()
        self.max_contour_dist = self.get_max_contour_dist()
        
        # Define plasma parameters
        self.Zeff = 1.01
        self.Ti_spline, self.Ni_spline, exp_plotting_data = self.make_exp_1D_profiles("rho", plot=False)
        splines, transp_plotting_data = self.make_TRANSP_profiles("rho", plot=False)
        #################
        self.Ti_spline_transp, self.Ni_spline_transp = splines
        ################
        # Plot both profiles at the same time
        self.plot_exp_with_transp(exp_plotting_data, transp_plotting_data)
        
        # Fit plasma profiles with 1D profiles
        self.source_spline = self.get_source_profile(plot=False,type_xs="DD")
        
        self.plot_exp_and_transp_source_profiles()
        
        # Uniformly sample from the tokamak volume
        smoothing_width = 0.1
        num_samples = [1e2,5e2,1e3,5e3,1e4,5e4,1e5]#,1e6]#,5e6,1e7,1e8]
        reses = [50]#,60,70,80,90]
        FoM_max = np.zeros((len(num_samples),len(reses)))
        for j,res in enumerate(reses):
            for i,sample in enumerate(num_samples):
                self.vol_uniform_sampled_points = self.get_vol_uniform_sampled_points(int(sample),alpha=1.0,plot=False, read_in = True)
                self.neutron_source_rate_3d = self.get_neutron_source_rate(self.vol_uniform_sampled_points, res=res,smoothing_width=smoothing_width, sample_type="SPINS", plot=False,read_in = False)
                # self.neutron_source_rate_3d = self.get_neutron_source_rate_unstructured(self.vol_uniform_sampled_points, res=res, sample_type="SPINS", plot=False,read_in = True)
                
                self.uniform_sources = self.get_uniform_sampling_sources(int(sample), read_in = True)
                self.neutron_source_rate_2d = self.get_neutron_source_rate(self.uniform_sources, res=res,smoothing_width=smoothing_width,sample_type="OMCPS",plot=False,read_in = False)
                # self.neutron_source_rate_2d = self.get_neutron_source_rate_unstructured(self.uniform_sources, res=res,sample_type="OMCPS",plot=False,read_in = True)
                # print(f"Finished uniform for samples = {int(sample)}")
                
                # spins_fname = f"Data/sampled_rates/SPINS_rates_r{res}_s{int(sample)}.npy"
                # omcps_fname = f"Data/sampled_rates/OMCPS_rates_r{res}_s{int(sample)}.npy"
                # self.neutron_source_rate_3d = np.load(spins_fname)
                # self.neutron_source_rate_2d = np.load(omcps_fname)
                self.quantity_of_merit, FoM = self.get_quantity_of_merit(self.neutron_source_rate_3d, self.neutron_source_rate_2d)
                resolution=100
                self.plot_figure_of_merit(self.quantity_of_merit,res=resolution, show_inline=True)
                FoM_max[i,j] = FoM

        # Create meshgrid for plotting
        R, S = np.meshgrid(np.array(reses), np.array(num_samples))
        
        # Plotting
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # FoM_log = np.log10(FoM_max)
        # Surface plot
        ax.plot_surface(R, S, FoM_max, cmap='viridis', edgecolor='k', linewidth=0.5)
        # ax.plot_surface(R, S, FoM_log, cmap='viridis', edgecolor='k', linewidth=0.5)
        
        # Labels
        ax.set_xlabel("Resolution")
        ax.set_ylabel("Sample Index")
        ax.set_zlabel("Quantity")
        ax.set_title("3D Mesh Plot of Quantity vs. Resolution and Sample Index")
        
        plt.tight_layout()
        plt.show()
        
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        
        ax.scatter(num_samples, FoM_max[:,-1])
        
        # Labels
        ax.set_xlabel("Samples")
        ax.set_ylabel("Mean Figure of Merit Value")
        ax.set_title("Figure of Merit vs Sampling Rate")
        
        plt.tight_layout()
        plt.show()


        # self.vol_uniform_sampled_points = self.get_vol_uniform_sampled_points(samples,alpha=1.0,plot=False)
        # self.plot_sources(self.vol_uniform_sampled_points, "Sampled_Sources_3D_Uniform", plot_name="New Sampling")
        # self.neutron_source_rate_3d = self.get_neutron_source_rate(self.vol_uniform_sampled_points, res=error_res, smoothing_factor=2, sample_type="SPINS", plot=False,read_in = True)

        # # Uniformly sample over R to create sources
        # # num_samples = [1e2,5e2,1e3,5e3,1e4,5e4,1e5,1e6]#,5e6,1e7,1e8]
        # # for sample in num_samples:
        # #     self.uniform_sources = self.get_uniform_sampling_sources(int(sample), read_in = True)
        # #     print(f"Finished uniform for samples = {int(sample)}")
        
        # self.uniform_sources = self.get_uniform_sampling_sources(samples, read_in = True)
        # self.plot_sources(self.uniform_sources, "Sampled_Sources_2D_Uniform", plot_name="Previous Sampling")
        # self.neutron_source_rate_2d = self.get_neutron_source_rate(self.uniform_sources, res=error_res,sample_type="OMCPS",plot=False,read_in = True)
        # strengths = self.neutron_source_rate_3d[:,3]
        # # strengths = np.divide(np.abs(self.neutron_source_rate_3d[:,3]-self.neutron_source_rate_2d[:,3]),self.neutron_source_rate_3d[:,3])
        
        # self.plot_point_cloud(self.neutron_source_rate_3d[:,0:3].T, strengths, alpha=0.5)
        # # print(np.max(strengths))
        
        # num_samples = [1e2,5e2,1e3,5e3,1e4,5e4,1e5]#,1e6]#,5e6,1e7,1e8]
        # reses = [10,20,30,40,50]#,60,70,80,90]
        # for res in reses:
        #     for sample in num_samples:
        #         spins_fname = f"Data/sampled_rates/SPINS_rates_r{res}_s{int(sample)}.npy"
        #         omcps_fname = f"Data/sampled_rates/OMCPS_rates_r{res}_s{int(sample)}.npy"
        #         self.neutron_source_rate_3d = np.load(spins_fname)
        #         self.neutron_source_rate_2d = np.load(omcps_fname)
        #         self.quantity_of_merit = self.get_quantity_of_merit(self.neutron_source_rate_3d, self.neutron_source_rate_2d)
        #         resolution=100
        #         self.plot_figure_of_merit(self.quantity_of_merit,res=resolution, sigma=resolution/75, show_inline=True)
            
        # self.self_consistent_scaling(self.vol_uniform_sampled_points)
        #self.self_consistent_scaling(self.uniform_sources)
        #self.self_consistent_scaling(self.neutron_source_rate_3d)
        #self.self_consistent_scaling(self.neutron_source_rate_2d)
        
        end_time = time.time()
        print("Total time taken:".ljust(20) + f"{end_time-self.start_time:.2}s")
        print("Status:".ljust(20) + "Done")
        
    def get_quantity_of_merit(self,  data_1, data_2):
        
        x1,y1,z1,s1 = data_1.T
        x2,y2,z2,s2 = data_2.T
        
        differences = s1-s2 #np.abs(s1-s2)
        norm_differences = differences/np.sum(differences) * data_1.shape[0]
        
        # NOTE: The indep grids have the same x1=x2, z1=z2
        x,z = self.xyz_to_xz_plane(data_1[:,0:3])
        
        psi_vals = self.get_psi(np.column_stack((x, z)))
        rho_vals = self.get_rho_from_psi(psi_vals)
        strength = interp.splev(rho_vals,self.source_spline)
        strength_norm = strength/np.sum(strength)
        
        
        # FoM = diff_strength*abs_strength*circumference 
        quantity_of_merit = norm_differences*strength_norm*(np.pi * x**2)
        
        # print(f"Max quantity of merit: {np.max(quantity_of_merit):.2e}")
        # print(f"Mean quantity of merit: {np.mean(quantity_of_merit):.2e}")
        print(f"Res: {data_1.shape[0]},{data_2.shape[0]}".ljust(20) + f"{np.mean(quantity_of_merit):.2e}")
        
        return np.column_stack((x,z,quantity_of_merit)), np.mean(quantity_of_merit)
        
    def get_uniform_sampling_sources(self, samples, read_in = False, plot = False):
        fname = f"sampled_profiles/uniform_sources_{samples}.npy"
        path = Path("Data/"+fname)
        
        if read_in and path.exists():
            # print(f"Reading in file from {path}")
            uniform_sources = np.load(path)
        else:
            uniform_sources = np.zeros((samples,4)) # (num particles,(x,y,z,s))
            # Sample angles
            xz_angles = np.zeros((samples,))
            for i in range(samples):
                xz_angles[i] = np.random.rand() * 2 * np.pi
            
            # # Sample a random point along r for each angle
            # count = 0
            
            fitting_points = 300
            r_vals = np.linspace(0, 1, fitting_points)
            uniform_sources = np.empty((len(xz_angles), 4))
            
            for i, xz_angle in enumerate(xz_angles):
            
                # Predefine point
                point_to_be_rotated = self.max_psi_location + np.array([1.01 * self.max_contour_dist, 0.0])
            
                # Rotate and intersect
                rotated_point = self.rotate_point(point_to_be_rotated, self.max_psi_location, xz_angle)
                contour_intersect = self.find_intersection(rotated_point, self.max_psi_location, self.psi_mesh[0,:,:])
            
                # Sample rho and get psi
                rho_sample = np.random.rand()
                psi_sample = (rho_sample * self.max_psi) - self.min_psi
            
                # Linear path between max and intersection
                profile_x = np.linspace(self.max_psi_location[0], contour_intersect[0], num=fitting_points)
                profile_y = np.linspace(self.max_psi_location[1], contour_intersect[1], num=fitting_points)
                path_points = np.stack([profile_x, profile_y], axis=1)
            
                # Interpolate ψ along that path
                psi_profile = self.get_psi(path_points)
            
                # Fit spline to psi_profile(r) - psi_sample
                spline = interp.splrep(r_vals, psi_profile - psi_sample, k=3)
            
                # Root of spline = 0 in [0,1]
                try:
                    sol = root_scalar(lambda r: interp.splev(r, spline), bracket=[0.0, 1.0], method='brentq')
                    r_sampled = sol.root if sol.converged else np.nan
                except ValueError:
                    r_sampled = np.nan
            
                # Get sampled position
                intersection_vector = contour_intersect - self.max_psi_location
                sampled_pos = self.max_psi_location + r_sampled * intersection_vector
                x, y, z = self.xz_plane_to_xyz(sampled_pos)
            
                # Interpolate source strength
                strength = interp.splev(1 - rho_sample, self.source_spline)
            
                # Store result
                uniform_sources[i, :] = [x, y, z, strength]
            
                if i % 10000 == 0:
                    #print(f"Finished splining fit iteration: {i}")
                    print("Status:".ljust(20) + f"Finished splining fit iteration: {i}")
                    print()
            # for i,xz_angle in enumerate(xz_angles):
            #     # Define the point to be rotated to be max_psi shifted just beyond the last contour dist
            #     point_to_be_rotated = np.copy(self.max_psi_location)
            #     point_to_be_rotated[0] = point_to_be_rotated[0] + 1.01*self.max_contour_dist
                
            #     # rotate point to find intersection
            #     rotated_point = self.rotate_point(point_to_be_rotated,self.max_psi_location,xz_angle)
                
            #     # find intersection of rotated point with contour plot
            #     contour_intersect = self.find_intersection(rotated_point, self.max_psi_location, self.psi_mesh[0,:,:])
                
            #     # Randomly sample rho
            #     rho_sample = np.random.rand()
                
            #     # Convert to psi
            #     psi_sample = (rho_sample * self.max_psi) - self.min_psi
                
            #     # Create a polyfit for psi as a function of rho along difference vector
            #     fitting_points = 300
                
            #     # define x and y for samples
            #     profile_x = np.linspace(self.max_psi_location[0],contour_intersect[0], num = fitting_points)
            #     profile_y = np.linspace(self.max_psi_location[1],contour_intersect[1], num = fitting_points)
                
            #     # calculate psi at given locations
            #     psi_profile = self.get_psi(np.stack([profile_x, profile_y], axis=1))
                
            #     # polyfit psi_samples as a function from 0 to 1 (normalized r value) shifted by rho_sample
            #     coeff = np.polynomial.polynomial.polyfit(np.linspace(0,1, num=fitting_points), psi_profile - psi_sample, deg = 50)
            #     f = Polynomial(coeff)
                
            #     # Invert the function by finding the root of the shifted profile
            #     roots = f.roots()
            #     # The x-axis intersection is thus your rho sample
            #     r_sampled = self.clamp(self.trim_roots(roots),0.0,.999999)
                
                
            #     xx = np.linspace(0,1,num=fitting_points)
            #     yy = f(xx)
                
            #     # Find the vector representing the change in position to the intersection
            #     intersection_vector = contour_intersect - self.max_psi_location
                
            #     pos_change = r_sampled * intersection_vector
            #     sampled_pos = self.max_psi_location + pos_change
            #     x,y,z = self.xz_plane_to_xyz(sampled_pos)
                
            #     # temperature = self.Ti_spline(1-rho_sample)
            #     strength = interp.splev(1-rho_sample, self.source_spline)
                
            #     uniform_sources[i,:] = np.array([x,y,z,strength])
                
            #     if i%10000==0: print("Finished splining fit iteration:", i)
                
            if plot: self.plot_point_cloud(uniform_sources[:,0:3].T, strength)
            
            self.save_data(fname,uniform_sources)
        return uniform_sources
        

    
    def plot_figure_of_merit(self, data, res = 100, sigma=1, show_inline=True):
        '''
        Define the figure of merit to compare the two methods of sampling.
        THe relevant quantity is the difference in local neutron rate density
        times the source strength times the number of points sampled at a given
        location (ie. the radius) when projected onto a flat profile
        '''
        x,z,quantity_of_merit = data.T
        
        x_domain = np.linspace(np.min(self.LCFS[:,1]), np.max(self.LCFS[:,1]), num=res)
        z_domain = np.linspace(np.min(self.LCFS[:,0]), np.max(self.LCFS[:,0]), num=res)
        # Create a regular grid
        grid_x, grid_z = np.meshgrid(x_domain,z_domain)
        
        # Interpolate scattered data onto the grid
        strength = interp.griddata((x, z), quantity_of_merit, (grid_x, grid_z), method='cubic')
        strength = gaussian_filter(strength, sigma=sigma)
        
        # Define a polygon (a simple triangle for example)
        LCFS_poly = Polygon(self.LCFS)
        
        # # Create a mask by checking if each grid point is inside the polygon
        # mask = np.array([[LCFS_poly.contains(Point(z_, x_)) for x_, z_ in zip(row_x, row_z)] 
        #                  for row_x, row_z in zip(grid_x, grid_z)])
        
        # # Apply the mask by setting values outside the polygon to NaN
        # strength[~mask] = np.nan
        
        x_pad = 0.1
        y_pad = 0.2
        
        fig, ax = plt.subplots(figsize=(4,8),dpi = self.dpi)
        dpi_scaling_factor = self.dpi / 100
        
        psi_min = -0.01
        levels = np.linspace(psi_min, 1.0, 15)  # 10 levels between 30 and max(data)
        
        contour = ax.contour(self.R, self.Z, self.psi_rz_norm, levels = levels, cmap='viridis', alpha = 0.6, vmin=-0.1)        
        sc = ax.contourf(grid_x, grid_z, strength, cmap='magma', vmin=np.nanmin(strength), vmax=np.nanmax(strength), alpha=1.0)
        plt.plot(self.LCFS[:,1], self.LCFS[:,0], color="black", linewidth=3, label="LCFS")
        
        # Use make_axes_locatable to create the first colorbar
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="15%", pad=0.05)  # Colorbar for the first contour
        cbar1 = plt.colorbar(sc, cax=cax1)
        
        # adjust the tick mark parameters
        cbar1.ax.tick_params(labelsize=8, labelrotation=0,direction="in",colors="#222222")
        
        # Adjsut the color of the tick marks
        for label in cbar1.ax.get_yticklabels():
            label.set_color("#555555")
        
        formatter = ScalarFormatter()
        formatter.set_scientific("on")
        formatter.set_useOffset(True)
        
        # Apply the formatter to the colorbar
        cbar1.ax.yaxis.set_major_formatter(formatter)
        
        # Format the tick labels with one decimal place
        cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # Move the tick labels inside the colorbar
        cbar1.ax.yaxis.set_tick_params(labelsize=8, pad=-21, direction='in')  # Negative pad moves labels inside the bar

        cbar1.update_ticks()
        
        # Set the colorbar label and keep it inside and centered
        cbar1.set_label(r'Quantity of Merit', labelpad=8, rotation=270, fontsize=10, verticalalignment='center')

        # Create the second colorbar for the second contour plot
        levels = np.linspace(0.0, 1.0, num=6)
        
        # Use ScalarMappable to create a filled colorbar
        norm = Normalize(vmin=psi_min, vmax=1)
        sm = ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])  # Dummy array for the ScalarMappable
        
        cax2 = divider.append_axes("right", size="15%", pad=0.20)  # Place the second colorbar closer
        cbar2 = plt.colorbar(contour, cax=cax2)
        
        cbar2.ax.tick_params(labelsize=8, labelrotation=270, direction="in",size=0,pad = 0,labelleft=False, labelright=True)
        cbar2.set_ticks(levels)
        
        # Set the colorbar label and keep it inside and centered
        cbar2.set_label(r'$\psi_{norm}$', labelpad=0, rotation=270, fontsize=12, verticalalignment='center')

        ax.set_xlim(np.min(self.psi_mesh[:,:,0])-x_pad, np.max(self.psi_mesh[:,:,0])+x_pad)
        ax.set_ylim(np.min(self.psi_mesh[:,:,1])-y_pad, np.max(self.psi_mesh[:,:,1])+y_pad)
        ax.set_xlabel(r"Major Radius, $R[m]$")
        ax.set_ylabel(r"Height, $Z[m]$")
        ax.legend()
        
        self.set_plot_aspect_ratio(ax)
        self.adjust_label_size_and_padding(ax)
        ax.set_title('Figure of Merit', ha='center', fontsize=16, pad=15)
        # self.save_plot(fig, "FigureOfMerit_Final")
        if show_inline: plt.show()
        plt.close()
        
        
        
        
        # fig, ax = plt.subplots(figsize=(4,8),dpi = self.dpi)
        # dpi_scaling_factor = self.dpi / 100 
        
        # psi_min = -0.01
        # levels = np.linspace(psi_min, 1.0, 15)  # 10 levels between 30 and max(data)
        
        # contour = ax.contour(self.R, self.Z, self.psi_rz_norm, levels = levels, cmap='viridis', alpha = 0.6, vmin=-0.1)        
        # sc = ax.contourf(grid_x, grid_z, strength, cmap='magma', vmin=np.nanmin(strength), vmax=np.nanmax(strength), alpha=1.0)
        # plt.plot(self.LCFS[:,1], self.LCFS[:,0], color="black", linewidth=3, label="LCFS")
        
        # # Use make_axes_locatable to create the first colorbar
        # divider = make_axes_locatable(ax)
        # cax1 = divider.append_axes("right", size="15%", pad=0.05)  # Colorbar for the first contour
        # cbar1 = plt.colorbar(sc, cax=cax1)
        
        # # adjust the tick mark parameters
        # cbar1.ax.tick_params(labelsize=8, labelrotation=0,direction="in",colors="#555555")
        
        # # Adjsut the color of the tick marks
        # for label in cbar1.ax.get_yticklabels():
        #     label.set_color("#555555")
        
        # formatter = ScalarFormatter()
        # formatter.set_scientific("on")
        # formatter.set_useOffset(True)
        
        # # Apply the formatter to the colorbar
        # cbar1.ax.yaxis.set_major_formatter(formatter)
        
        # # Format the tick labels with one decimal place
        # cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # # Move the tick labels inside the colorbar
        # cbar1.ax.yaxis.set_tick_params(labelsize=8, pad=-21, direction='in')  # Negative pad moves labels inside the bar

        # cbar1.update_ticks()
        
        # # Set the colorbar label and keep it inside and centered
        # cbar1.set_label(r'Quantity of Merit', labelpad=8, rotation=270, fontsize=10, verticalalignment='center')

        # # Create the second colorbar for the second contour plot
        # levels = np.linspace(0.0, 1.0, num=6)
        
        # # Use ScalarMappable to create a filled colorbar
        # norm = Normalize(vmin=psi_min, vmax=1)
        # sm = ScalarMappable(norm=norm, cmap='viridis')
        # sm.set_array([])  # Dummy array for the ScalarMappable
        
        # cax2 = divider.append_axes("right", size="15%", pad=0.20)  # Place the second colorbar closer
        # cbar2 = plt.colorbar(contour, cax=cax2)
        
        # cbar2.ax.tick_params(labelsize=8, labelrotation=270, direction="in",size=0,pad = 0,labelleft=False, labelright=True)
        # cbar2.set_ticks(levels)
        
        # # Set the colorbar label and keep it inside and centered
        # cbar2.set_label(r'$\psi_{norm}$', labelpad=0, rotation=270, fontsize=12, verticalalignment='center')

        # ax.set_xlim(np.min(self.psi_mesh[:,:,0])-x_pad, np.max(self.psi_mesh[:,:,0])+x_pad)
        # ax.set_ylim(np.min(self.psi_mesh[:,:,1])-y_pad, np.max(self.psi_mesh[:,:,1])+y_pad)
        # ax.set_xlabel(r"Major radius, $R[m]$")
        # ax.set_ylabel(r"Height, $Z[m]$")
        # ax.legend()
        
        # self.set_plot_aspect_ratio(ax)
        # self.adjust_label_size_and_padding(ax)
        # ax.set_title('Figure of Merit', ha='center', fontsize=16, pad=15)
        # self.save_plot(fig, "FigureOfMerit_Final_brightness")
        # if show_inline: plt.show()
        # plt.close()
        
    def make_exp_1D_profiles(self, x_axis = "rho", plot = False):
        
        def average_repeats(data):
            x_current = data[0,0]
            x_count = 0
            y_sum = 0
            data_avg = []
            for i in data:
                if i[0] == x_current:
                    x_count += 1
                    y_sum += i[1]
                else:
                    data_avg.append(np.array([x_current, y_sum/x_count]))
                    x_count = 1
                    x_current = i[0]
                    y_sum = i[1]
            
            return np.array(data_avg)
        
        def bin_data(data, bin_size):
            # Calculate the number of bins
            num_bins = len(data) // bin_size
            
            # Trim the data if necessary to ensure it can be divided into full bins
            trimmed_data = data[:num_bins * bin_size]
            
            # Reshape the data into bins and compute the mean of each bin
            binned_data = trimmed_data.reshape(num_bins, bin_size).mean(axis=1)
            
            return binned_data
        
        
        
        ### CXRS data for Ti ###
        
        Ti_fname = "Ti_data.pickle"
        CXRS_data_loc = self.data_folder_path + "/" + Ti_fname
        profile_time = 0.500
        
        Ti_data_list = []
        with (open(CXRS_data_loc, "rb")) as openfile:
            while True:
                try:
                    Ti_data_list.append(pickle.load(openfile))
                except EOFError:
                    break
        
        Ti_data_dict = Ti_data_list[0]
        Ti_time = Ti_data_dict['time']
        Ti_data = Ti_data_dict['Ti']
        Ti_R = Ti_data_dict['R']
        
        time_closest_index = np.argmin(np.abs(Ti_time - profile_time))
        time_range = 20
        pad = 10

        Ti_smoothed = np.sum(Ti_data_dict['Ti'][time_closest_index-int(time_range/2):time_closest_index+int(time_range/2),:],axis = 0)/time_range
        shift = 0.14
        Ti_x = (Ti_R - (self.max_psi_location[0] + shift))/(np.max(self.LCFS[:,1]) - (self.max_psi_location[0]+shift))
        Ti_y = Ti_smoothed
        
        Ti_spline = interp.splrep(Ti_x, Ti_y, k=1)
        
        
        ### TS data for n_i ###
        
        TS_data_loc = self.data_folder_path + "/" + "TSdata_49394.txt"
        column_lables = pd.read_csv(TS_data_loc, skiprows=[0, 0], nrows=1, sep=r'\s+').columns
        df = pd.read_csv(TS_data_loc, skiprows=[0, 1], names=column_lables, sep=r'\s+')
        df.Ne = df.Ne.astype(float)
        
        Ni_profile_unsmoothed = df[~np.isnan(df['Ne'])][['psi_n','Ne']].to_numpy()
        Ni_profile_unsmoothed[:,1] = np.copy(Ni_profile_unsmoothed[:,1])
        
        # Apply moving average to smooth
        Ni_trimmed = np.array([rn for rn in Ni_profile_unsmoothed if rn[0] <= 1.00])
        Ni_trimmed = Ni_trimmed[Ni_trimmed[:,0].argsort(),:]
        Ni_trimmed[:,0] = Ni_trimmed[:,0] / (self.max_psi_location[0])
        Ni_trimmed_avg = average_repeats(Ni_trimmed)
        
        # Bin datapoints to smooth for interpolation
        bins = 100
        Ni_x_binned = bin_data(Ni_trimmed_avg[:,0], int(Ni_trimmed_avg[:,0].shape[0]/bins) )
        Ni_y_binned = bin_data(Ni_trimmed_avg[:,1], int(Ni_trimmed_avg[:,1].shape[0]/bins) )
        
        # Smooth binned points
        Ni_y_binned_smooth = np.copy(Ni_y_binned)
        
        repeats = 20
        for i in range(repeats):
            Ni_y_binned_smooth = pd.Series(np.copy(Ni_y_binned_smooth)).rolling(window=3, center=True).mean().to_numpy()
            Ni_y_binned_smooth[0] = Ni_y_binned[0]
            Ni_y_binned_smooth[-1] = Ni_y_binned[-1]
            
        Ni_x = Ni_x_binned
        Ni_y = Ni_y_binned_smooth
        
        Ni_spline = interp.splrep(Ni_x, Ni_y, k=5)
        
        plotting_data = (Ti_x, Ti_y, Ti_time, Ti_data_dict, Ti_spline, Ni_x, Ni_y, Ni_x_binned, Ni_y_binned, Ni_spline, Ni_trimmed_avg, time_closest_index)
        
        if plot: self.plot_exp_profile(Ti_x, Ti_y, Ti_time, Ti_data_dict, Ti_spline, Ni_x, Ni_y, Ni_x_binned, Ni_y_binned, Ni_spline, Ni_trimmed_avg, time_closest_index)
            
        return Ti_spline, Ni_spline, plotting_data
        
    def plot_exp_with_transp(self, exp_plotting_data, transp_plotting_data, time_range=20, pad=10):
        t_plotting_data, trans_profile_time = transp_plotting_data
        
        fig, ax = plt.subplots(1,(len(transp_plotting_data)),figsize = (8,4),dpi=300)
        fig.suptitle("Ion Density and Temperature Profiles")
        
        Ti_x, Ti_y, Ti_time, Ti_data_dict, Ti_spline, Ni_x, Ni_y, Ni_x_binned, Ni_y_binned, Ni_spline, Ni_trimmed_avg, time_closest_index = exp_plotting_data
        
        xnew = np.linspace(0, max(Ni_x), num = 1000)
        
        # fig, ax = plt.subplots(1,2,figsize = (8,4),dpi=self.dpi)
        
        cmap = plt.get_cmap("Reds")
        
        Ti_xnew = np.linspace(0,1,1000)
        Ti_ynew = interp.splev(Ti_xnew,Ti_spline)
        
        Ni_xnew = np.linspace(0,1,1000)
        Ni_ynew = interp.splev(Ni_xnew,Ni_spline)
        
        norm = Normalize(vmin=-pad, vmax=time_range+pad)
        Ti_x_within_domain_mask = (Ti_x > 0) & (Ti_x < 1)
        
        # Plot Ion Temperature Profile
        for i in range(int(time_range)):
            x = Ti_x[Ti_x_within_domain_mask]
            y = Ti_data_dict['Ti'][int(time_closest_index-time_range/2 + i),Ti_x_within_domain_mask]
            color = cmap(norm(i))
            if i == 0 or i == int(time_range - 1):
                ax[0].scatter(x, y, label=f't={Ti_time[int(time_closest_index-time_range/2 + i)]:1.2}s', color = color,s=7, alpha=0.25)
                #ax[0].plot(x, y, '-', label=f't={Ti_time[int(time_closest_index-time_range/2 + i)]:1.2}s', color = color)
            else:
                ax[0].scatter(x, y, color = color,s=7, alpha=0.25)
                #ax[0].plot(x, y, '-', color = color)
        
        smooth_label = r"$\overline{T}_{i, exp}$" #+ f", t={Ti_time[int(time_closest_index)]:1.2}s"
        ax[0].scatter(Ti_x[Ti_x_within_domain_mask], Ti_y[Ti_x_within_domain_mask], label=smooth_label, color = "red", s = 15)
        ax[0].plot(xnew, Ti_ynew, '-', c='red', label=r"$\overline{T}_{i, exp}(\rho)$")
        ax[0].legend()
        ax[0].set_xlabel(r"$\rho$")
        ax[0].set_ylabel(r"$T_i \ [eV]$")
        
        cmap = plt.get_cmap("Blues")
        
        # Plot Ion Density Profile
        ax[1].scatter(Ni_trimmed_avg[:,0], Ni_trimmed_avg[:,1], color='lightsteelblue', s=3, label=r"Raw $n_{i}$")
        ax[1].scatter(Ni_x_binned, Ni_y_binned, c='cornflowerblue', label=r"Smoothed $n_i$",s=15)
        ax[1].plot(Ni_xnew, Ni_ynew, '-', c='midnightblue', label=r"$n_{i}(\rho)$",linewidth=3)
        ax[1].set_xlabel(r"$\rho$")
        ax[1].set_ylabel(r"$n_i \ [m^{-3}]$")
        ax[1].legend()
        
        for i,info in enumerate(t_plotting_data):
            data_x, data_y, data_spline, times, color, measurement = info
            
            time_closest_index = np.argmin(np.abs(times - trans_profile_time))
            
            cmap = plt.get_cmap(f"{color}")
            norm = Normalize(vmin=0, vmax=2 * np.max(data_y))
            
            filled_marker_style = dict(marker='o', linestyle=':', markersize=15,
                                       color='darkgrey',
                                       markerfacecolor='tab:blue',
                                       markerfacecoloralt='lightsteelblue',
                                       markeredgecolor='brown')
            
            color=cmap(norm(np.max(data_y)))
            if i == 0:
                smooth_label = r"$T_{i, transp}$" #+ f", t={times[int(time_closest_index)]:1.2}s"
                ax[i].plot(xnew, interp.splev(xnew,data_spline), '-', c=color, label=r"$T_{i, transp}(\rho)$")
            elif i == 1:
                smooth_label = r"$n_{i, transp}$" #+ f", t={times[int(time_closest_index)]:1.2}s"
                ax[i].plot(xnew, interp.splev(xnew,data_spline), '-', c=color, label=r"$n_{i, transp}(\rho)$")
            
            ax[i].scatter(data_x, data_y, label=smooth_label, facecolors='none', color = color, s = 15, alpha=.5)
            ax[i].legend()
            ax[i].set_xlabel(r"$\rho$")
            ax[i].set_ylabel(rf"{measurement}")
        
        
        self.save_plot(fig,"EXP_AND_TRANSP_Profiles",dpi = self.dpi)
        
        plt.tight_layout()
        plt.show()
                
    
    
        
               
    def self_consistent_scaling(self, sources):
        LCFS_polygon = Polygon(self.LCFS)
        center = LCFS_polygon.centroid
        volume = 2*np.pi*center.y*LCFS_polygon.area * 1/2 # factor of 1/2 for the pi rotation
        surface_area = 4*np.pi**2 * center.y * LCFS_polygon.length
        avg_neutron_rate_density = (np.sum(sources[:,3])/sources.shape[0])
        total_neutrons = avg_neutron_rate_density * volume
        print(f"Total neutron rate {total_neutrons:.2e} s-1")
        print(f"Average neutron rate density {avg_neutron_rate_density:.2e} m-3 s-1")
        return avg_neutron_rate_density
    
    def get_neutron_source_rate_unstructured(self,sampled_points,res = 50,sample_type="SPINS", smoothing_factor = 1, plot=False, read_in = True):
        '''
        Uses an independent grid of points throughout the tokamak to determine
        the neutron source density. This accounts for both the weight/strength 
        of the source as well as the sampling density.

        Parameters
        ----------
        sampled_points : numpy.array
            Contains the x,y,z coordinates with the source strengths of n 
            sampled points; shape of (n,4)
        res : INT, optional
            Resolution of the independent grid across the tokamak.
            The default is 50.

        Returns
        -------
        Numpy array of shape (res,res,res) that contains the neutron source
        density at the given location.

        '''
        
        fname = f"sampled_rates_unstructured/{sample_type}_rates_r{res}_s{sampled_points.shape[0]}.npy"
        path = Path("Data/"+fname)
        
        if read_in and path.exists():
            # print(f"Reading in file from {path}")
            sources_with_rate_density = np.load(path)
        else:
            x_samples, y_samples, z_samples, strength_samples = sampled_points.T
            x_error_grid = np.random.rand(res**3) * 2 * np.max(self.R) -  np.max(self.R)
            y_error_grid = np.random.rand(res**3) * np.max(self.R)
            z_error_grid = np.random.rand(res**3) * (np.max(self.Z)-np.min(self.Z)) - np.min(self.Z)
            
            x_mesh,y_mesh,z_mesh = np.meshgrid(x_error_grid,y_error_grid,z_error_grid)
            
            x = np.random.rand(res**3) * 2 * np.max(self.R) -  np.max(self.R)
            y = np.random.rand(res**3) * np.max(self.R)
            z = np.random.rand(res**3) * (np.max(self.Z)-np.min(self.Z)) - np.min(self.Z)
            
            # if plot: self.plot_point_cloud((x,y,z), np.ones_like(x))
            points = np.column_stack((x, y, z))
            
            inside_tokamak_mask = self.mask_points_outside_plasma(points, self.LCFS)
            points_in_tokamak = np.column_stack((x[inside_tokamak_mask], y[inside_tokamak_mask], z[inside_tokamak_mask]))
            if plot: self.plot_point_cloud(points_in_tokamak.T, np.ones_like(x[inside_tokamak_mask]))
            
            dx_error_grid = smoothing_factor*(x_error_grid[1]-x_error_grid[0])
            dy_error_grid = smoothing_factor*(y_error_grid[1]-y_error_grid[0])
            dz_error_grid = smoothing_factor*(z_error_grid[1]-z_error_grid[0])
            ds = (dx_error_grid,dy_error_grid,dz_error_grid)
            
            
            neutron_rate_density = np.zeros((points_in_tokamak.shape[0],))
            for i,independent_grid_point in enumerate(points_in_tokamak):
                gridpoint_mask = self.get_ellipsoid_mask(independent_grid_point, sampled_points, ds)
                
                if strength_samples[gridpoint_mask].shape[0] == 0:
                    neutron_rate_density[i] = 0
                else:
                    neutron_rate_density[i] = np.sum(strength_samples[gridpoint_mask])/strength_samples[gridpoint_mask].shape[0]
            
            sources_with_rate_density = np.column_stack((points_in_tokamak, neutron_rate_density))
            if plot: self.plot_point_cloud(points_in_tokamak.T, sources_with_rate_density)
            self.save_data(fname, sources_with_rate_density)
        
        return sources_with_rate_density
     
    def get_neutron_source_rate(self,sampled_points,res = 50,sample_type="SPINS", smoothing_width = 1, plot=False, read_in = True):
        '''
        Uses an independent grid of points throughout the tokamak to determine
        the neutron source density. This accounts for both the weight/strength 
        of the source as well as the sampling density.

        Parameters
        ----------
        sampled_points : numpy.array
            Contains the x,y,z coordinates with the source strengths of n 
            sampled points; shape of (n,4)
        res : INT, optional
            Resolution of the independent grid across the tokamak.
            The default is 50.

        Returns
        -------
        Numpy array of shape (res,res,res) that contains the neutron source
        density at the given location.

        '''
        
        fname = f"sampled_rates/{sample_type}_rates_r{res}_s{sampled_points.shape[0]}.npy"
        path = Path("Data/"+fname)
        
        if read_in and path.exists():
            # print(f"Reading in file from {path}")
            sources_with_rate_density = np.load(path)
        else:
            x_samples, y_samples, z_samples, strength_samples = sampled_points.T
            x_error_grid = np.linspace(-np.max(self.R),np.max(self.R),num=res)
            y_error_grid = np.linspace(0,np.max(self.R),num=res)
            z_error_grid = np.linspace(np.min(self.Z),np.max(self.Z),num=res)
            
            x_mesh,y_mesh,z_mesh = np.meshgrid(x_error_grid,y_error_grid,z_error_grid)
            
            x=x_mesh.flatten()
            y=y_mesh.flatten()
            z=z_mesh.flatten()
            # if plot: self.plot_point_cloud((x,y,z), np.ones_like(x))
            points = np.column_stack((x, y, z))
            
            inside_tokamak_mask = self.mask_points_outside_plasma(points, self.LCFS)
            points_in_tokamak = np.column_stack((x[inside_tokamak_mask], y[inside_tokamak_mask], z[inside_tokamak_mask]))
            if plot: self.plot_point_cloud(points_in_tokamak.T, np.ones_like(x[inside_tokamak_mask]))
            
            dx = (x_error_grid[1]-x_error_grid[0]) #smoothing_factor*
            dy = (y_error_grid[1]-y_error_grid[0]) #smoothing_factor*
            dz = (z_error_grid[1]-z_error_grid[0]) #smoothing_factor*
            ds = (smoothing_width/dx, smoothing_width/dy, smoothing_width/dz)
            
            
            neutron_rate_density = np.zeros((points_in_tokamak.shape[0],))
            for i,independent_grid_point in enumerate(points_in_tokamak):
                gridpoint_mask = self.get_ellipsoid_mask(independent_grid_point, sampled_points, ds)
                
                if strength_samples[gridpoint_mask].shape[0] == 0:
                    neutron_rate_density[i] = 0
                else:
                    neutron_rate_density[i] = np.sum(strength_samples[gridpoint_mask])/strength_samples[gridpoint_mask].shape[0]
            
            sources_with_rate_density = np.column_stack((points_in_tokamak, neutron_rate_density))
            if plot: self.plot_point_cloud(points_in_tokamak.T, sources_with_rate_density)
            self.save_data(fname, sources_with_rate_density)
        
        return sources_with_rate_density
        
    def get_ellipsoid_mask(self,grid_point, samples, ds):
        """
        Compute boolean masks for each grid point indicating which sample points fall 
        within the defined ellipsoidal radius.
    
        Parameters:
        grid_point : (1, 3) array
            Independent grid point with coordinates (x, y, z).
        samples : (N, 4) array
            Sampled locations from sources with shape (N, 4) where columns are (sx, sy, sz, strength).
        ds : (3,) array or tuple
            Semi-axes (dx, dy, dz) defining the ellipsoid shape.
    
        Returns:
        gridpoint_masks : (M, N) boolean array
            Each row represents a grid point and each column represents a sample.
        """
        # Unpack grid points
        cx, cy, cz = grid_point.T  # Shape (1,)
        
        # Unpack sample points
        sx, sy, sz, strengths = samples.T  # Shape (N,)
        
        # Unpack ellipsoid radii
        dx, dy, dz = ds
    
        # Compute squared normalized distances using broadcasting
        mask = ((-1*sx+cx)**2 / dx**2 +
                (-1*sy+cy)**2 / dy**2 +
                (-1*sz+cz)**2 / dz**2) < 1.0  # Shape (M, N)
    
        return mask


    def find_local_sampled_points(self, grid_points, samples, ds=(1,1,1)):
        cx,cy,cz = grid_points # cheker points, independent grid
        sx,sy,sz,strengths = samples.T # sampled locations from sources
        dx,dy,dz = ds
        
        gridpoint_masks = []
        for x,y,z in zip(cx,cy,cz):
            samples_around_gridpoint = (((x - sx)**2)/dx**2 + ((y - sy)**2)/dy**2 + ((z - sz)**2)/dz**2) < 1.0
            gridpoint_masks.append(samples_around_gridpoint)
        return gridpoint_masks
   
    def get_vol_uniform_sampled_points(self, samples_inp, alpha=0.3, plot = False, read_in = True):
        fname = f"sampled_profiles/SPINS_profile_{int(samples_inp)}.npy"
        path = Path("Data/" + fname)
        
        if read_in and path.exists():
            # print(f"Reading in file from {path}")
            strengths = np.load(path)
        else:
            # samples inp is the desired number of reported samples, samples is the volumetric sampling
            # that may or may not be in the tokamak
            samples = samples_inp * 10
            
            # Randomly sample an x,y,z coordinates for each samples
            points = np.random.rand(samples, 3) # in the shape of [x0,y0,z0; x1,y2,z1; ...]
            
            # Shift the samples to be in a volume represented by the max points of half a torus
            points[:,0] = np.multiply(points[:,0], 2*np.max(self.R)) - np.max(self.R)
            points[:,1] = np.multiply(points[:,1], np.max(self.R))
            points[:,2] = np.multiply(points[:,2], np.max(self.Z) - np.min(self.Z)) - np.max(self.Z)
            x,y,z = points.T
    
            # Calculate radius of each point for sanity check
            r = np.sqrt(x**2 + y**2)
        
            mask = self.mask_points_outside_plasma(points, self.LCFS)
            
            points_in_plasma = points[mask,:]
            x,y,z=points_in_plasma.T
            points_psi = self.get_psi(np.vstack((r[mask],z)).T)
            points_rho = self.get_rho_from_psi(points_psi)
            points_strength = interp.splev(points_rho,self.source_spline)
    
            # if plot: self.plot_point_cloud((x,y,z), points_psi)
            # if plot: self.plot_point_cloud((x,y,z), points_rho)
            if plot: self.plot_point_cloud((x,y,z), points_strength,alpha)
            
            #print("Finished finding volumetrically uniform sampled points")
            print("Status:".ljust(20) + "Finished volumetrically uniform sampling")
            print()
            
            # if plot:
            #     plt.scatter(points_rho, points_strength)
            #     plt.ylim(-1869592524-100,0)
            strengths = np.column_stack((points_in_plasma,points_strength))[:samples_inp,:]
            # strengths[:,3] = strengths[:,3]/np.sum(strengths[:,3])
                
            self.save_data(fname, strengths)
        return strengths

    
    def xz_plane_to_xyz(self,xz, angle = np.pi):
        x,z = xz
        sampled_angle = angle*np.random.rand()
    
        cos_vals = np.cos(sampled_angle)
        sin_vals = np.sin(sampled_angle)
        
        # Rotate the samples to the x-z plane to be able to check if the point is within the largest contour
        x_rotated = x * cos_vals
        y_rotated = x * sin_vals
        z_rotated = z
        
        return (x_rotated,y_rotated,z_rotated)
    
    def plot_LCFS(self):
        fig, ax = plt.subplots(dpi=1200)
        vmax = 1.5
        # contour plot of psi
        contour = plt.contourf(self.R, self.Z, self.psi_rz_norm, vmax=vmax, levels=self.num_contours, cmap='Blues_r')
        
        #ax.imshow(self.psi_rz, cmap=plt.cm.gray)
        ax.plot(self.LCFS[:,1], self.LCFS[:,0],c='k', linewidth=2, label="LCFS")
        ax.set_title(f'Equilibrium and LCFS')
        # ax.set_xlabel(r"$R [m]")
        # ax.set_ylabel(r"$z [m]")
        ax.legend()
        
        divider = make_axes_locatable(ax)
        
        # Create the second colorbar for the second contour plot
        levels = np.linspace(0,vmax, num=7)
        
        # Use ScalarMappable to create a filled colorbar
        norm = Normalize(vmin=np.min(self.psi_rz_norm), vmax=vmax)#np.max(self.psi_rz_norm))
        sm = ScalarMappable(norm=norm, cmap='Blues_r')
        sm.set_array([])  # Dummy array for the ScalarMappable
        
        cax2 = divider.append_axes("right", size="15%", pad=0.20)  # Place the second colorbar closer
        cbar2 = plt.colorbar(contour, cax=cax2)
        
        cbar2.ax.tick_params(labelsize=8, labelrotation=-90, direction="out",size=0,pad = 0,labelleft=False, labelright=True)
        cbar2.set_ticks(levels)
        cbar2.ax.set_ylim(0, vmax)
        
        # Set the colorbar label and keep it inside and centered
        cbar2.set_label(r'$\psi_{norm}$', labelpad=-13, rotation=270, fontsize=12, verticalalignment='center')
        
        wall_x = self.wall_data[:,0]
        wall_y = self.wall_data[:,1]
        wall_x_closed = np.append(wall_x, wall_x[0])
        wall_y_closed = np.append(wall_y, wall_y[0])
        LCFS = np.column_stack((wall_y_closed,wall_x_closed))
        
        ax.plot(wall_x_closed, wall_y_closed, color='k', linewidth=1)
        
        # Create and apply clip path from LCFS
        lcfs_path = PTH(LCFS[:, [1, 0]])  # Assuming LCFS[:,1] is R and LCFS[:,0] is Z
        patch = PathPatch(lcfs_path, transform=ax.transData)
            
        # for artist in contour.get_children():
        #     artist.set_clip_path(patch)
        for coll in contour.collections:
            coll.set_clip_path(patch)
            
        finer_contours = 50
        #    Use more contour levels for finer detail
        fine_levels = np.linspace(np.min(self.psi_rz_norm), np.max(self.psi_rz_norm), finer_contours)  # 50 finer levels
        closest_to_zero = fine_levels[np.argmin(np.abs(fine_levels - 0))]

        # 2. Plot the contour line at this level, black color
        contour_zero = ax.contour(
            self.R, self.Z, self.psi_rz_norm,
            levels=[1],
            colors='black',
            linewidths=0.5
        )
        for coll in contour_zero.collections:
            coll.set_clip_path(patch)
        # for artist in contour_zero.get_children():
        #     artist.set_clip_path(patch)
        
        
        self.set_plot_aspect_ratio(ax)
        self.adjust_label_size_and_padding(ax)
        plt.show()
            
    def plot_point_cloud(self, points, strengths, alpha = 0.3):
        x,y,z = points
        z_LCFS, x_LCFS = self.LCFS[:,0], self.LCFS[:,1]
        # Normalize Data/Log data
        norm = Normalize(vmin=np.min(strengths), vmax=np.max(strengths))
        # norm = LogNorm(min(var),max(var))
        cmap = plt.get_cmap("viridis")
        
        # Create the plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Uniform Sampling Source Strength")
        nonzero_mask = strengths != 0
        colobar = ax.scatter3D(x[nonzero_mask],y[nonzero_mask],z[nonzero_mask], c=strengths[nonzero_mask], cmap=cmap, norm=norm, linestyle='-', lw=1, alpha=alpha, edgecolors='none')
        #colobar = ax.scatter3D(x,y,z, c=strengths, cmap=cmap, norm=norm, linestyle='-', lw=1, alpha=alpha, edgecolors='none')
        ax.scatter(x_LCFS,np.zeros_like(x_LCFS), z_LCFS, color='k', s=0.2)
        ax.scatter(-x_LCFS,np.zeros_like(x_LCFS), z_LCFS, color='k', s=0.2)
        if x.shape[0] != 1:
            ax.set_box_aspect(aspect=(np.ptp(x), np.ptp(y), np.ptp(z)))
        #ax.view_init(10,-45,30)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")
        fig.colorbar(colobar)
        plt.show()
        
    def xyz_to_xz_plane(self,points):
        x,y,z = points.T
        
        # mask = np.greater_equal(y,np.zeros_like(y))
        angles = np.arctan2(y,x) # if mask else np.arctan2(x,-y)
        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles-np.pi)
        
        # Rotate the samples to the x-z plane to be able to check if the point is within the largest contour
        x_rotated = x * cos_vals - y * sin_vals
        # y_rotated = x * sin_vals + y * cos_vals
        return x_rotated, z
    

    def rotate_point(self, point, center, angle):
        # Extract coordinates
        x, y = point[0], point[1]
        x_c, y_c = center
        
        # Translate point to origin (relative to the center)
        translated_point = np.array([x - x_c, y - y_c])
        
        # Create a rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle),  np.cos(angle)]])
        
        # Apply the rotation matrix to the translated point
        rotated_point = rotation_matrix.dot(translated_point)
        
        # Translate the point back to the original center
        final_point = rotated_point + np.array([x_c, y_c])
        
        return final_point
    
    def mask_points_outside_plasma(self, points, last_contour):
        x,y,z = points.T

        x_rotated, z = self.xyz_to_xz_plane(points)
        
        rotated_points = np.vstack((z,x_rotated)).T
        
        mask = self.points_in_polygon(rotated_points, self.LCFS)
        
        return mask
    
    
    # Function to calculate the slope and intercept of a line passing through two points
    def line_equation(self,p1, p2):
        if p2[0] == p1[0]:  # vertical line
            return float('inf'), p1[0]  # Return infinity as slope and x-intercept
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        intercept = p1[1] - slope * p1[0]
        return slope, intercept
    
    # Step 3: Function to find intersections between an arbitrary line and contours
    def find_intersection(self,p1, p2, contour):
        intersection = np.array((2,))
        # Calculate the slope and intercept of the arbitrary line
        m_line, b_line = self.line_equation(p1, p2)
        
        for i in range(contour.shape[0]):
            # Points on the contour segment
            x1, y1 = contour[i]
            x2, y2 = np.roll(contour,1,axis=0)[i]
            
            # Calculate the slope and intercept of the contour segment
            m_contour, b_contour = self.line_equation([x1, y1], [x2, y2])
            
            # Solve for the intersection point between the two lines
            intersect_x = (b_contour - b_line) / (m_line - m_contour)
            intersect_y = m_line * intersect_x + b_line

            # Check if the intersection point is within the bounds of the contour segment and the arbitrary line segment
            if min(x1, x2) <= intersect_x <= max(x1, x2) and min(p1[0], p2[0]) <= intersect_x <= max(p1[0], p2[0]):
                if min(y1, y2) <= intersect_y <= max(y1, y2) and min(p1[1], p2[1]) <= intersect_y <= max(p1[1], p2[1]):
                    intersection = np.array([intersect_x, intersect_y])
    
        return intersection
    
    def trim_roots(self,roots, Min=0,Max=1.0):
        nonzero_reals = np.isreal(roots)
        # Returns any in range
        for is_real, root in zip(nonzero_reals,roots):
            rroot = np.real(root)
            if is_real and rroot > Min and rroot < Max:
                return rroot
        
        real_roots = []
        # Returns the closest real root to the range
        for i,is_real in enumerate(nonzero_reals):
            if is_real:
                real_roots.append(np.real(roots[i]))
        
        if len(real_roots) == 0:
            real_roots.append(0.999999999)
        
        min_dist_to_range = []
        for root in real_roots:
            min_dist_to_range.append(min(np.abs(root-Min), np.abs(root-Max)))
        #min_dist_to_rannge = [min(np.abs(root-Min), np.abs(root-Max)) for root in real_roots]
        return real_roots[np.argmin(min_dist_to_range)]
                
    def clamp(self, value, Min, Max):
        return max(Min, min(value, Max))
        
    def get_max_contour_dist(self):
        outer_contour = self.psi_mesh[0,:,:]
        distances = []
        for point in outer_contour:
            distances.append(np.linalg.norm(point-self.max_psi_location))
        #index_of_max_dist = np.argmax(distances)
        return np.max(distances)

    def get_temperatures(self,points):
        psi = self.get_psi(points)
        rho = self.get_rho_from_psi(psi)
        return self.Ti_spline(rho)

    def plot_sources(self, sources_to_plot, fname, plot_name = "Neutron Source Rate Density Differences"):
        x_pad = 0.1
        y_pad = 0.2
        
        x,z = self.xyz_to_xz_plane(sources_to_plot[:,0:3])
        strengths = sources_to_plot[:,3]
        
        fig, ax = plt.subplots(figsize=(4,8),dpi = self.dpi)
        
        levels = np.linspace(0, 1.0, 15)  # 10 levels between 30 and max(data)

        contour = ax.contour(self.R, self.Z, self.psi_rz_norm, levels = levels, cmap='viridis_r', alpha = 0.5, vmin=-0.1)        
        #sc = ax.scatter(sources_to_plot[:,0], sources_to_plot[:,1], c=sources_to_plot[:,2], cmap='viridis', vmin=np.min(sources_to_plot[:,2]), vmax=np.max(sources_to_plot[:,2]))
        sc = ax.scatter(x,z, c=np.abs(strengths), cmap='viridis', vmin=np.min(strengths), vmax=np.max(strengths))
        
        # Use make_axes_locatable to create the first colorbar
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="15%", pad=0.05)  # Colorbar for the first contour
        cbar1 = plt.colorbar(sc, cax=cax1)
        
        cbar1.ax.tick_params(labelsize=8, labelrotation=0,direction="in")
        #cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1e'))
        
        formatter = ScalarFormatter()
        formatter.set_scientific("on")
        formatter.set_useOffset(True)
        
        # Apply the formatter to the colorbar
        cbar1.ax.yaxis.set_major_formatter(formatter)
        
        # Format the tick labels with one decimal place
        cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # Move the tick labels inside the colorbar
        cbar1.ax.yaxis.set_tick_params(labelsize=8, pad=-21, direction='in')  # Negative pad moves labels inside the bar

        cbar1.update_ticks()
        
        # Set the colorbar label and keep it inside and centered
        cbar1.set_label(r'Volumetric Source Strength $[n/m^3]$', labelpad=8, rotation=270, fontsize=10, verticalalignment='center')

        # Create the second colorbar for the second contour plot
        levels = np.linspace(0.0, 1.0, num=6)
        
        # Use ScalarMappable to create a filled colorbar
        norm = Normalize(vmin=0.0, vmax=1)
        sm = ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])  # Dummy array for the ScalarMappable
        
        cax2 = divider.append_axes("right", size="15%", pad=0.20)  # Place the second colorbar closer
        cbar2 = plt.colorbar(contour, cax=cax2)
        
        cbar2.ax.tick_params(labelsize=8, labelrotation=270, direction="in",size=0,pad = 0,labelleft=False, labelright=True)
        cbar2.set_ticks(levels)
        
        # Set the colorbar label and keep it inside and centered
        cbar2.set_label(r'$\psi_{norm}$', labelpad=0, rotation=270, fontsize=12, verticalalignment='center')

        
        ax.set_xlim(np.min(self.psi_mesh[:,:,0])-x_pad, np.max(self.psi_mesh[:,:,0])+x_pad)
        ax.set_ylim(np.min(self.psi_mesh[:,:,1])-y_pad, np.max(self.psi_mesh[:,:,1])+y_pad)
        ax.set_xlabel(r"Major radius, $R [m]$")
        ax.set_xlabel(r"Radius, $R[m]$")
        
        self.set_plot_aspect_ratio(ax)
        self.adjust_label_size_and_padding(ax)
        ax.set_title(f'{plot_name}', ha='center', fontsize=16, pad=15)
        self.save_plot(fig, fname)
        #plt.show()
    
    def save_data(self,fname,arr):
        loc = "Data/" + fname
        np.save(loc,arr)

    def get_rho_from_psi(self, psi):
        # (1 - (psi - min_psi) * max_rho / max_psi) + min_rho
        return 1.0 - (psi-self.min_psi) / self.max_psi
    
    def plot_exp_and_transp_source_profiles(self,interp_points = 100,type_xs="DD"):
        # #psi_fine_source = np.linspace(self.min_psi, self.max_psi, interp_points+1)
        # rho_fine_source = np.linspace(0.0, 1.0, interp_points+1)
        # Ti_transp = interp.splev(rho_fine_source,self.Ti_spline_transp)
        # Ni_transp = interp.splev(rho_fine_source,self.Ni_spline_transp)
        
        # if type_xs=="DD":
        #     Source_transp = (Ni_transp)**2 * self.DD_xs(Ti_transp/1000)
        # elif type_xs=="DT":
        #     Source_transp = (Ni_transp)**2 * self.DT_xs(Ti_transp/1000)
        
        # S_x_transp = np.linspace(0.0, 1.0, num = Source_transp.shape[0])
        # S_y_transp = Source_transp
        # S_interp_transp = interp.splrep(S_x_transp, S_y_transp, k=1)
        # S_xnew_transp = np.linspace(0,1,1000)
        # S_ynew_transp = interp.splev(S_xnew_transp,S_interp_transp)
        
        # S_x_exp = np.linspace(0.0, 1.0, num = Source_transp.shape[0])
        # Source_exp = interp.splev(S_x_exp,self.source_spline)
        
        # fig, ax = plt.subplots()
        # #ax.scatter(S_x_transp, S_y_transp, edgecolors='purple', facecolors='none', marker='o', label='Source Profile')
        # ax.plot(S_xnew_transp, S_ynew_transp, color = 'plum', label='TRANSP Source')
        # ax.plot(S_x_exp, Source_exp, color = 'mediumorchid', label='Experimental Source')
        # ax.set_xlabel(r'$\rho$')
        # ax.set_ylabel(r'Neutron source $[n/s]$')
        # ax.set_title(r'Fitted Source Profile vs $\rho$')
        # ax.legend()
        # --- Compute interpolated TRANSP profile ---
        rho_fine_source = np.linspace(0.0, 1.0, interp_points + 1)
        Ti_transp = interp.splev(rho_fine_source, self.Ti_spline_transp)
        Ni_transp = interp.splev(rho_fine_source, self.Ni_spline_transp)
        
        if type_xs == "DD":
            Source_transp = Ni_transp**2 * self.DD_xs(Ti_transp / 1e3)  # keV
        elif type_xs == "DT":
            Source_transp = Ni_transp**2 * self.DT_xs(Ti_transp / 1e3)
        
        # --- Smooth profile using linear spline for plotting ---
        S_x_transp = np.linspace(0.0, 1.0, Source_transp.shape[0])
        S_interp_transp = interp.splrep(S_x_transp, Source_transp, k=1)
        S_xnew_transp = np.linspace(0.0, 1.0, 1000)
        S_ynew_transp = interp.splev(S_xnew_transp, S_interp_transp)
        
        # --- Evaluate experimental spline ---
        S_x_exp = S_x_transp.copy()
        Source_exp = interp.splev(S_x_exp, self.source_spline)
        
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(6, 4))
        
        ax.plot(S_xnew_transp, S_ynew_transp, color='#C9A0DC', lw=2,linestyle='--', label='TRANSP Source')
        ax.plot(S_x_exp, Source_exp, color='#5B2C6F', lw=2, label='Experimental Source')
        
        ax.set_xlabel(r'Normalized radius $\rho$', fontsize=12)
        ax.set_ylabel(r'Neutron source $[n/\mathrm{s}/\mathrm{m}^3]$', fontsize=12)
        ax.set_title(r'Comparison of Neutron Source Profiles', fontsize=13)
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(fontsize=10, frameon=False, loc='upper right')
        
        ax.grid(True, linestyle=':', linewidth=0.6)
        fig.tight_layout()
        # fig.savefig('source_profile_comparison.pdf', dpi=300)  # Uncomment to save
        self.save_plot(fig, "1D_Source_Profile_both", dpi = self.dpi)
        plt.show()

        
    def get_source_profile(self, interp_points = 100, plot = False, type_xs="DD"):

        #psi_fine_source = np.linspace(self.min_psi, self.max_psi, interp_points+1)
        rho_fine_source = np.linspace(0.0, 1.0, interp_points+1)
        Ti = interp.splev(rho_fine_source,self.Ti_spline)
        Ni = interp.splev(rho_fine_source,self.Ni_spline)
        
        if type_xs=="DD":
            Source = (Ni)**2 * self.DD_xs(Ti/1000)
        elif type_xs=="DT":
            Source = (Ni)**2 * self.DT_xs(Ti/1000)
        
        S_x = np.linspace(0.0, 1.0, num = Source.shape[0])
        S_y = Source
        S_interp = interp.splrep(S_x, S_y, k=1)
        S_xnew = np.linspace(0,1,1000)
        S_ynew = interp.splev(S_xnew,S_interp)

        if plot:
            fig, ax = plt.subplots()
            ax.scatter(S_x, S_y, edgecolors='purple', facecolors='none', marker='o', label='Source Profile')
            ax.plot(S_xnew, S_ynew, label='Fitted Source', color = 'Purple')
            ax.set_xlabel(r'$\rho$')
            ax.set_ylabel(r'Neutron source $[n/s]$')
            ax.set_title(r'Fitted Source Profile vs $\rho$')
            ax.legend()
            
            self.save_plot(fig, "1D_Source_Profile", dpi = self.dpi)
            plt.show()
            
        return S_interp
    
    
    def plot_TRANSP_data(self, plotting_data, profile_time, plot=False):
        
        fig, axs = plt.subplots(1,(len(plotting_data)),figsize = (8,4),dpi=300)
        fig.suptitle("Ion Density and Temperature Profiles")
        
        for i,info in enumerate(plotting_data):
            data_x, data_y, data_spline, times, color, measurement = info
            
            time_closest_index = np.argmin(np.abs(times - profile_time))
            
            xnew = np.linspace(0, max(data_x), num = 1000)
            
            cmap = plt.get_cmap(f"{color}")
            
            pad = 2 * np.max(data_y)
            norm = Normalize(vmin=0, vmax=pad)
            
            smooth_label = r"$T$ (eV)" + f", t={times[int(time_closest_index)]:1.2}s"
            #color=cmap(norm(time_range+4))
            color=cmap(norm(np.max(data_y)))
            axs[i].scatter(data_x, data_y, label=smooth_label, color = color, s = 15)
            axs[i].plot(xnew, interp.splev(xnew,data_spline), '-', c=color, label="Ti Interpolation")
            axs[i].legend()
            axs[i].set_xlabel(r"$\rho$")
            axs[i].set_ylabel(rf"{measurement}")
            
        plt.tight_layout()
        plt.show()
    
    def make_TRANSP_profiles(self, x_axis = "rho", plot = False):
        
        profile_time = 0.53
        
        u_Te = ufiles.UFILE(fin="Data/OMFITTER49392.txt")
        u_Ne = ufiles.UFILE(fin="Data/NER49392.txt")
        u_Ti = ufiles.UFILE(fin="Data/OMFITTI249392.txt")
        
        u_files = [u_Ti, u_Ne]
        scales  = [1000,1e6 * self.Zeff]
        measurements = [r"$T_i [eV]$",r"$n_i [m^{-3}]$"]
        colors = ["OrRd", "GnBu"]
        
        splines = []
        plotting_data = []
        for i,(file, color, measurement) in enumerate(zip(u_files,colors,measurements)):
            times=file["X"]["data"]
            rho=file["Y"]["data"]
            data=file["f"]["data"]*scales[i]
            
            time_closest_index = np.argmin(np.abs(times - profile_time))
            time_range = 6
            
            # take closest time range of data
            data_t_trimmed = data[time_closest_index-int(time_range/2):time_closest_index+int(time_range/2),:]
            time_range = data_t_trimmed.shape[0]
            # temporally average profile 
            data_t_average = np.sum(data_t_trimmed,axis = 0)/time_range
            
            shift = 0.15
            # trim off excess data beyond the LCFS (rho > 1.0)
            data_rho = np.stack((rho,data_t_average),axis=1) #np.array([rt for rt in Ti_profile if rt[0] <= 1.00])
            data_rho_trimmed = np.array([rn for rn in data_rho if rn[0] <= 1.00 - shift]) # trims off pedestal region
            data_rho_trimmed = data_rho_trimmed[data_rho_trimmed[:,0].argsort(),:]
            data_x = data_rho_trimmed[:,0]/(1.00 - shift)
            data_y = data_rho_trimmed[:,1]
            
            # create spline fit of temporally averaged profile
            data_spline = interp.splrep(data_x, data_y, k = 1) #,s = Ti_y[0]**2.15 )
            splines.append(data_spline)
            
            plotting_data.append((data_x, data_y, data_spline, times, color, measurement))
            
        if plot: self.plot_TRANSP_data(plotting_data, profile_time)
        
        plotting = (plotting_data, profile_time)
        
        return tuple(splines), plotting
    
    def plot_exp_profile(self, Ti_x, Ti_y, Ti_time, Ti_data_dict, Ti_spline, Ni_x, Ni_y, Ni_x_binned, Ni_y_binned, Ni_spline, Ni_trimmed_avg, time_closest_index, time_range=20, pad=10):
        
        ### Plotting ###
        
        xnew = np.linspace(0, max(Ni_x), num = 1000)
        
        fig, ax = plt.subplots(1,2,figsize = (8,4),dpi=self.dpi)
        
        fig.suptitle("Ion Temperature and Density Profiles")
        
        cmap = plt.get_cmap("Reds")
        
        Ti_xnew = np.linspace(0,1,1000)
        Ti_ynew = interp.splev(Ti_xnew,Ti_spline)
        
        Ni_xnew = np.linspace(0,1,1000)
        Ni_ynew = interp.splev(Ni_xnew,Ni_spline)
        
        norm = Normalize(vmin=-pad, vmax=time_range+pad)
        Ti_x_within_domain_mask = (Ti_x > 0) & (Ti_x < 1)
        # Plot Ion Temperature Profile
        for i in range(int(time_range)):
            x = Ti_x[Ti_x_within_domain_mask]
            y = Ti_data_dict['Ti'][int(time_closest_index-time_range/2 + i),Ti_x_within_domain_mask]
            color = cmap(norm(i+pad))
            if i == 0 or i == int(time_range - 1):
                ax[0].scatter(x, y, label=rf'$t$={Ti_time[int(time_closest_index-time_range/2 + i)]:1.2}s', color = color,s=7, alpha=0.25)
                #ax[0].plot(x, y, '-', label=f't={Ti_time[int(time_closest_index-time_range/2 + i)]:1.2}s', color = color)
            else:
                ax[0].scatter(x, y, color = color,s=7, alpha=0.25)
                #ax[0].plot(x, y, '-', color = color)
        
        smooth_label = r"$\overline{T}_{time \ avg}$" + rf", $t$={Ti_time[int(time_closest_index)]:1.2}s"
        ax[0].scatter(Ti_x[Ti_x_within_domain_mask], Ti_y[Ti_x_within_domain_mask], label=smooth_label, color = "red", s = 15)
        ax[0].plot(xnew, Ti_ynew, '-', c='red', label=r"$T_i$ Interpolation")
        ax[0].legend()
        ax[0].set_xlabel(r"$\rho$")
        ax[0].set_ylabel(r"$T_i \ [eV]$")
        
        cmap = plt.get_cmap("Blues")
        
        # Plot Ion Density Profile
        ax[1].scatter(Ni_trimmed_avg[:,0], Ni_trimmed_avg[:,1], color='lightsteelblue', s=3, label="Scaled TS data")
        ax[1].scatter(Ni_x_binned, Ni_y_binned, c='cornflowerblue', label=r"$n_i$ binned data",s=15)
        ax[1].plot(Ni_xnew, Ni_ynew, '-', c='midnightblue', label=r"$n_i$ Smoothed Interpolation",linewidth=3)
        ax[1].set_xlabel(r"$\rho$")
        ax[1].set_ylabel(r"$n_i \ [m^{-3}]$")
        ax[1].legend()
        plt.tight_layout()
        plt.show()
        
        self.save_plot(fig,"NiTiProfiles",dpi = self.dpi)



    
    def convert_r_to_rho(self, r):
        points = np.stack([r,np.zeros(r.shape)],axis=1)
        psi_values = self.get_psi(points)
        return 1.0 - (psi_values-self.min_psi) * 1.0 / self.max_psi + 0.0
    

    def get_psi(self, points_to_evaluate_psi):
        
        if len(points_to_evaluate_psi.shape) == 1:
            points_to_evaluate_psi = points_to_evaluate_psi[np.newaxis, :]
            
        x = points_to_evaluate_psi[:, 0]
        y = points_to_evaluate_psi[:, 1]
        
        x_scale = max(self.R[0,:]) - min(self.R[0,:])
        y_scale = max(self.Z[:,0]) - min(self.Z[:,0])
        
        x_flipped = (x - min(self.R[0,:])) * y_scale/x_scale + min(self.Z[:,0]) 
        y_flipped = (y - min(self.Z[:,0])) * x_scale/y_scale + min(self.R[0,:])

        return self.psi_rectspline.ev(y_flipped, x_flipped)

    def DD_xs(self,T):
        '''
        DD cross sections from the NRL formulary p45 year 2013.
        Applicable for low energies ( T<25keV )

        Parameters
        ----------
        T : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        sigma_v_DD = 2.33e-14 * (T)**(-2/3)*np.exp(-18.76*(T)**(-1/3))*1e-6
        return sigma_v_DD

    def DT_xs(self,ion_temperature):
        """Sadler–Van Belle formula
        Ref : https://doi.org/10.1016/j.fusengdes.2012.02.025

        Args:
            ion_temperature (float, ndarray): ion temperature in keV

        Returns:
            float, ndarray: the DT cross section at the given temperature
        """

        ion_temperature = np.asarray(ion_temperature)

        c = [
            2.5663271e-18,
            19.983026,
            2.5077133e-2,
            2.5773408e-3,
            6.1880463e-5,
            6.6024089e-2,
            8.1215505e-3,
        ]

        U = 1 - ion_temperature * (
            c[2] + ion_temperature * (c[3] - c[4] * ion_temperature)
        ) / (1.0 + ion_temperature * (c[5] + c[6] * ion_temperature))

        val = (
            c[0]
            * np.exp(-c[1] * (U / ion_temperature) ** (1 / 3))
            / (U ** (5 / 6) * ion_temperature ** (2 / 3))
        )
        return val

    def get_psi(self, points_to_evaluate_psi):
        
        if len(points_to_evaluate_psi.shape) == 1:
            points_to_evaluate_psi = points_to_evaluate_psi[np.newaxis, :]
            
        x = points_to_evaluate_psi[:, 0]
        y = points_to_evaluate_psi[:, 1]
        
        x_scale = max(self.R[0,:]) - min(self.R[0,:])
        y_scale = max(self.Z[:,0]) - min(self.Z[:,0])
        
        x_flipped = (x - min(self.R[0,:])) * y_scale/x_scale + min(self.Z[:,0]) 
        y_flipped = (y - min(self.Z[:,0])) * x_scale/y_scale + min(self.R[0,:])

        return self.psi_rectspline.ev(y_flipped, x_flipped)
    
    def rescale_contours(self):
        for contour in self.min_dist_contours:
            contour[:,0] = contour[:,0]/self.psi_rz.shape[1] * (np.max(self.Z) - np.min(self.Z)) + np.min(self.Z)
            contour[:,1] = contour[:,1]/self.psi_rz.shape[0] * (np.max(self.R) - np.min(self.R)) + np.min(self.R)
        
        
    def define_psi_mesh(self, spline_inter_points = 100, read_in = False):
        '''
        Create a mesh for psi that has an equal number of points for each of the contours
        
        args:
            resolution (int): number of pixels in the interpolated contours
        '''
        interp_points = spline_inter_points
        
        path = Path("Data/psi_mesh.npy")
        
        if read_in and path.exists():
            print(f"Reading in file from {path}")
            psi_mesh = np.load(path)
            
        else:
            # Initialize psi mesh
            psi_mesh = np.zeros((self.num_contours+1, self.poloidal_res, 2))
            
            for i,contour in enumerate(self.min_dist_contours):
                psi_mesh[i,:,:] = self.interpolate_contour(contour, self.poloidal_res)
             
            psi_mesh[-1,:,0:2] = self.max_psi_location
                
            self.save_data("psi_mesh.npy", psi_mesh)
        
        
        
        return psi_mesh
    
    
    def define_psi_values(self):
        psi_values = np.zeros(self.psi_mesh.shape[0:2])
        for i in range(self.psi_mesh.shape[0]):
            for j in range(self.psi_mesh.shape[1]):
                x_scale = max(self.R[0,:]) - min(self.R[0,:])
                y_scale = max(self.Z[:,0]) - min(self.Z[:,0])
                
                x_flipped = (self.psi_mesh[i,j,0] - min(self.R[0,:])) * y_scale/x_scale + min(self.Z[:,0]) 
                y_flipped = (self.psi_mesh[i,j,1] - min(self.Z[:,0])) * x_scale/y_scale + min(self.R[0,:])
    
                psi_values[i,j] = self.psi_rectspline.ev(y_flipped, x_flipped)
        
        return psi_values
        
    def plot_mesh(self, indices):
        fig, ax = plt.subplots()
        for i in indices:
            ax.plot(self.psi_mesh[i,:,0], self.psi_mesh[i,:,1])
        ax.set_title(f"Sample Mesh, chords = {indices}")
        self.set_plot_aspect_ratio(ax)
        self.adjust_label_size_and_padding(ax,10,5,1)
        self.save_plot(fig,"New_mesh_generator_example",dpi=self.dpi)
        plt.show()
    
    def interpolate_contour(self, contour, num_points):
        """Interpolate a contour to have a fixed number of points."""
        x, y = contour[:, 0], contour[:, 1]
        t = np.linspace(0, 1, len(x))
        t_new = np.linspace(0, 1, num_points)
        
        interp_x = interp1d(t, x, kind='linear')
        interp_y = interp1d(t, y, kind='linear')
        
        x_new = interp_x(t_new)
        y_new = interp_y(t_new)
        
        return np.flip(np.vstack((x_new, y_new)).T,axis=1)


    def distance(self, p1, p2):
        """Calculate the Euclidean distance between two points p1 and p2."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
            
    def find_center_point(self):
        # NEED TO CHANGE LATER TO THROW AN ERROR AND FIND THE ACTUAL CENTERPOINT
        try:
            return np.array([np.mean(self.min_dist_contours['0'][0][:,0]), np.mean(self.min_dist_contours['0'][0][:,1])])
        except:
            print("self.min_dist_contours is not yet defined")

    def trim_equilibrium(self, plot = False):
        '''
        Limits the equilibrium to only the N'th contours around the maximum psi vlaue
        '''
        inner_contour_pad = 0.99
        #contour_values = np.linspace(self.min_psi, self.max_psi * inner_contour_pad, num = self.num_contours)
        contour_values = np.linspace(self.min_psi, self.max_psi * inner_contour_pad, num = self.num_contours)
        
        min_dist_contours = []
        center = np.array(self.max_psi_indices)[np.newaxis,:] #self.max_psi_location[np.newaxis,:]
        
        # Set the value at which you want to find contours
        for contour_value in contour_values:
        
            # Find contours at the given value
            contours = measure.find_contours(self.psi_rz, contour_value)
            
            # Plot all found contours
            for contour in contours:
                if self.points_in_polygon(center, contour)[0]:
                    min_dist_contours.append(contour)
   
        return min_dist_contours
    
    def points_in_polygon(self, points, polygon_coords):
        """
        Determine if multiple points are inside a polygon.
        """
        # Convert polygon to Path object
        polygon_path = PTH(polygon_coords)
        
        if not isinstance(points, np.ndarray):
            # Convert points to a NumPy array for efficient checking
            points_np = np.array(points)
        else:
            points_np = points
        
        # Use Path's contains_points method to check all points at once
        return polygon_path.contains_points(points_np)
    
    def set_plot_aspect_ratio(self,ax):
        """
        Adjust aspect ratio to physical coordinates
        """
        ax.set_aspect('equal', adjustable='box')
        
    def adjust_label_size_and_padding(self, ax, fontsize=10, labelpad=5, tickpad=2):
        """
        Adjust the font size of axis labels and make them tighter to the axis.
        
        Args:
        ax (matplotlib.axes.Axes): The axis object of the plot.
        fontsize (int): The font size for the axis labels.
        labelpad (int): Padding between the axis labels and the axis.
        tickpad (int): Padding between the ticks and tick labels.
        """
        # Set the font size and padding for axis labels
        ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize, labelpad=labelpad)
        ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize, labelpad=labelpad)
        
        # Set the font size and padding for tick labels
        ax.tick_params(axis='both', which='major', labelsize=fontsize, pad=tickpad)
        
    def save_plot(self, fig, filename, file_format='png', dpi=300, bbox_inches='tight'):
        """
        Save a Matplotlib plot to a file to /Images folder.
        
        Args:
        fig (matplotlib.figure.Figure): The figure object to save.
        filename (str): The name of the file (without extension or Images folder).
        file_format (str): The format to save the plot in ('png', 'pdf', 'svg', etc.).
        dpi (int): The resolution of the saved figure in dots per inch (DPI).
        bbox_inches (str or None): Bounding box in inches. 'tight' ensures no whitespace around the plot.
        """
        # Save the figure with the specified parameters
        loc = "Images/" + filename
        fig.savefig(f"{loc}.{file_format}", format=file_format, dpi=self.dpi, bbox_inches=bbox_inches)
        #print(f"Plot saved as {filename}.{file_format}")
        print("Plot saved:".ljust(20) + f"{filename}.{file_format}")

       
    def extract_equilibrium_data(self, equ_fpath, nc = 30, verbose = False, plot = False):
        """Extracts the equilibrium contours from the equilibrium files. Stores
        the contours as a mesh represented by self.psi_mesh to be accessed 
        during the creation of the neutron source profile.

        Args:
            string: file path/name containing equilibrium data ("*.equ")
            bool  : verbose

        Returns:
            float, ndarray: psi, magnetic potential [units]
        """
        with open(equ_fpath, 'r') as file:
            lines = file.readlines()
            
                
        # Initialize variables
        jm = 0                      # no. of grid points in radial direction
        km = 0                      # no. of grid points in vertial direction
        r = []                      # radial   coordinates of grid points   [m]
        z = []                      # vertical coordinates of grid points   [m]
        psib = 0.0                  # psi at plasma boundary           [Wb/rad]
        psi = []                    # flux per radian at grid points   [Wb/rad]    
        btf = 0.0                   # toroidal magnetic field               [T]
        rtf = 0.0                   # major radius at which btf is specified[m]
        psi_rz = np.zeros((1))      # 2D equilibrium mesh
        
        # Compile regex patterns
        jm_pattern = re.compile(r'jm\s*=\s*(\d+)\s*;')
        km_pattern = re.compile(r'km\s*=\s*(\d+)\s*;')
        psib_pattern = re.compile(r'psib\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)')
        btf_pattern = re.compile(r'btf\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)')
        rtf_pattern = re.compile(r'rtf\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)')
        
        # Read the first section
        for line in lines:
            if jm_pattern.search(line):
                jm = int(jm_pattern.search(line).group(1))
            elif km_pattern.search(line):
                km = int(km_pattern.search(line).group(1))
            elif psib_pattern.search(line):
                psib = float(psib_pattern.search(line).group(1))
            elif btf_pattern.search(line):
                btf = float(btf_pattern.search(line).group(1))
            elif rtf_pattern.search(line):
                rtf = float(rtf_pattern.search(line).group(1))
        
        # Read the r, z, and psi data
        r_data_start = False
        z_data_start = False
        psi_data_start = False
        
        for line in lines:
            if line.strip().startswith('r('):
                r_data_start = True
                continue
            if line.strip().startswith('z('):
                z_data_start = True
                r_data_start = False
                continue
            if line.strip().startswith('((psi('):
                psi_data_start = True
                z_data_start = False
                continue
        
            if r_data_start:
                r.extend([float(val) for val in line.split()])
            elif z_data_start:
                z.extend([float(val) for val in line.split()])
            elif psi_data_start:
                psi.extend([float(val) for val in line.split()])
        
        # Convert lists to numpy arrays for convenience
        r = np.array(r)
        z = np.array(z)
        psi = np.array(psi)
        
        if verbose:
            print(f'jm: {jm}')
            print(f'km: {km}')
            print(f'psib: {psib}')
            print(f'btf: {btf}')
            print(f'rtf: {rtf}')
            print(f'r: {r}')
            print(f'z: {z}')
            print(f'r shape: {r.shape}')
            print(f'z shape: {z.shape}')
            print(f'psi: {psi}')
        
        # Define a 2D profile for psi
        psirz_reshape = np.reshape(psi, (r.shape[0], r.shape[0]), order='F').T
        
        # Create a meshgrid for r and z
        self.R, self.Z = np.meshgrid(r, z)
        if plot:
            
            # Plot the contour
            plt.figure(figsize=(10, 8))
            contour = plt.contourf(self.R, self.Z, psirz_reshape, levels=self.num_contours, cmap='Blues')
            plt.colorbar(contour)
            
            # Add labels and title
            plt.xlabel('Radial Coordinate (r)')
            plt.ylabel('Vertical Coordinate (v)')
            plt.title('Contour Plot of Psi')
            
            # Show the plot
            plt.show()
        
        return psirz_reshape, self.R, self.Z
    
    def set_verbose(self, new_verbose):
        self.verbose = new_verbose

# Get the current script's directory
current_dir = __file__

parent_dir = '/'.join(current_dir.split('/')[:-2])

data_folder_name = 'Experimental_Data'

data_folder_path = parent_dir + '/' + data_folder_name

profile_fname = "Profiles.txt"

psi_fname = "equ_49392_symm_X4_crop.equ"

profile_path = data_folder_path + '/' + profile_fname
psi_path = data_folder_path + '/' + psi_fname

profile = EquilibriumProfile(psi_path, profile_path, data_folder_path)

