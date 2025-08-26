# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as interp

def get_sources(self, samples_inp, alpha=0.3, plot = False):
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
    
    print("Finished finding volumetrically uniform sampled points")
    
    # if plot:
    #     plt.scatter(points_rho, points_strength)
    #     plt.ylim(-1869592524-100,0)
        
    return np.column_stack((points_in_plasma,points_strength))[:samples_inp,:]