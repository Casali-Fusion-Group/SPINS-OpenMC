# -*- coding: utf-8 -*-
from spins_compute.get_density_profile_1D import get_density_profile_1D
from spins_compute.get_temperature_profile_1D import get_temperature_profile_1D
from spins_utils import (
    xyz_to_xz_plane,
)
from scipy import interpolate as interp
import numpy as np
from pathlib import Path
from matplotlib.path import Path as m_path
import matplotlib.pyplot as plt
import pickle
import math
from shapely.geometry import Polygon

def get_source_profile(input_data, Zeff, inp_type="TRANSP", interp_points = 100, plot = False, type_xs="DD"):
    ni_prof = np.multiply(input_data["ne"],Zeff)
    Ti_prof = input_data["Ti"]
    ni_spline = interp.splrep(ni_prof[:,0], ni_prof[:,1], k = 1)
    Ti_spline = interp.splrep(Ti_prof[:,0], Ti_prof[:,1], k = 1)
        
    rho_fine_source = np.linspace(0.0, 1.0, interp_points+1)
    ni = interp.splev(rho_fine_source,ni_spline)
    Ti = interp.splev(rho_fine_source,Ti_spline)
    
    if type_xs=="DD":
        Source = 1/2*(ni/1e6)**2 * DD_xs(Ti/1000)
    elif type_xs=="DT":
        Source = (ni/1e6)**2 * DT_xs(Ti/1000)
    
    S_x = np.linspace(0.0, 1.0, num = Source.shape[0])
    S_y = Source
    S_interp = interp.splrep(S_x, S_y, k=1)
        
    return ni_spline, Ti_spline, S_interp

def get_global_neutron_rate(sources,theta,LCFS):
    LCFS_polygon = Polygon(LCFS*100) # Convert to cm since volumetric source rate is in cm
    center = LCFS_polygon.centroid
    volume = 2*np.pi*center.x*LCFS_polygon.area# factor of 1/2 for the pi rotation
    avg_neutron_rate_density = np.mean(sources[:,3])
    total_neutrons = avg_neutron_rate_density * volume
    print(f"Total neutron rate {total_neutrons:.2e} s-1")
    print(f"Average neutron rate density {avg_neutron_rate_density*1e6:.2e} m-3 s-1")
    return avg_neutron_rate_density

# def make_EXP_profiles(Ti_path, Ni_path, LCFS, profile_time, x_axis = "rho", plot = False):
#     ### CXRS data for Ti ###
#     CXRS_data_loc = Ti_path
    
#     Ti_data_list = []
#     with (open(CXRS_data_loc, "rb")) as openfile:
#         while True:
#             try:
#                 Ti_data_list.append(pickle.load(openfile))
#             except EOFError:
#                 break
    
#     Ti_data_dict = Ti_data_list[0]
#     Ti_time = Ti_data_dict['time']
#     Ti_data = Ti_data_dict['Ti']
#     Ti_R = Ti_data_dict['R']
    
#     time_closest_index = np.argmin(np.abs(Ti_time - profile_time))
#     time_range = 20
#     pad = 10

#     Ti_smoothed = np.sum(Ti_data_dict['Ti'][time_closest_index-int(time_range/2):time_closest_index+int(time_range/2),:],axis = 0)/time_range
#     shift = 0.14
#     Ti_x = (Ti_R - (max_psi_location[0] + shift))/(np.max(LCFS[:,1]) - (max_psi_location[0]+shift))
#     Ti_y = Ti_smoothed
    
#     Ti_spline = interp.splrep(Ti_x, Ti_y, k=1)
    
    
#     ### TS data for n_i ###
    
#     TS_data_loc = self.data_folder_path + "/" + "TSdata_49394.txt"
#     column_lables = pd.read_csv(TS_data_loc, skiprows=[0, 0], nrows=1, sep=r'\s+').columns
#     df = pd.read_csv(TS_data_loc, skiprows=[0, 1], names=column_lables, sep=r'\s+')
#     df.Ne = df.Ne.astype(float)
    
#     Ni_profile_unsmoothed = df[~np.isnan(df['Ne'])][['psi_n','Ne']].to_numpy()
#     Ni_profile_unsmoothed[:,1] = np.copy(Ni_profile_unsmoothed[:,1])
    
#     # Apply moving average to smooth
#     Ni_trimmed = np.array([rn for rn in Ni_profile_unsmoothed if rn[0] <= 1.00])
#     Ni_trimmed = Ni_trimmed[Ni_trimmed[:,0].argsort(),:]
#     Ni_trimmed[:,0] = Ni_trimmed[:,0] / (self.max_psi_location[0])
#     Ni_trimmed_avg = average_repeats(Ni_trimmed)
    
#     # Bin datapoints to smooth for interpolation
#     bins = 100
#     Ni_x_binned = bin_data(Ni_trimmed_avg[:,0], int(Ni_trimmed_avg[:,0].shape[0]/bins) )
#     Ni_y_binned = bin_data(Ni_trimmed_avg[:,1], int(Ni_trimmed_avg[:,1].shape[0]/bins) )
    
#     # Smooth binned points
#     Ni_y_binned_smooth = np.copy(Ni_y_binned)
    
#     repeats = 20
#     for i in range(repeats):
#         Ni_y_binned_smooth = pd.Series(np.copy(Ni_y_binned_smooth)).rolling(window=3, center=True).mean().to_numpy()
#         Ni_y_binned_smooth[0] = Ni_y_binned[0]
#         Ni_y_binned_smooth[-1] = Ni_y_binned[-1]
        
#     Ni_x = Ni_x_binned
#     Ni_y = Ni_y_binned_smooth
    
#     Ni_spline = interp.splrep(Ni_x, Ni_y, k=5)
    
#     plotting_data = (Ti_x, Ti_y, Ti_time, Ti_data_dict, Ti_spline, Ni_x, Ni_y, Ni_x_binned, Ni_y_binned, Ni_spline, Ni_trimmed_avg, time_closest_index)
    
#     if plot: self.plot_exp_profile(Ti_x, Ti_y, Ti_time, Ti_data_dict, Ti_spline, Ni_x, Ni_y, Ni_x_binned, Ni_y_binned, Ni_spline, Ni_trimmed_avg, time_closest_index)
        
#     return Ti_spline, Ni_spline, plotting_data

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

def get_neutrons_from_distribution(source_profile, samples, normalize = True):
    coords = source_profile[:, [0,1,2,4]]
    probs = source_profile[:, 3]

    if normalize:
        probs = probs / np.sum(probs)

    # Defensive check
    if not np.isclose(np.sum(probs), 1.0, rtol=1e-5):
        raise ValueError("Probabilities must sum to 1.0")

    indices = np.random.choice(len(coords), size=samples, p=probs)
    return coords[indices]

def get_profile_positions(n_desired, oversample_factor,psi_spline,source_spline,Ti_spline, R, Z, LCFS, theta):
    # Randomly sample an x,y,z coordinates for each samples
    num_samples = n_desired * oversample_factor
    accepted_points = []
    
    while len(accepted_points) < n_desired: 
        candidate_points = sample_cell_range(num_samples, R, Z, theta, LCFS)
        inside_mask = points_inside_cell_mask(candidate_points, theta, LCFS)
        valid_candidates = candidate_points[inside_mask,:]
        accepted_points.append(valid_candidates)
        
        num_samples = max(n_desired - sum(len(x) for x in accepted_points), 1000)

    n_locations = np.vstack(accepted_points)[:n_desired]
    
    points_in_plasma = n_locations
    x,y,z=points_in_plasma.T
    
    # Calculate radius of each point for sanity check
    r = np.sqrt(x**2 + y**2)
    points_psi = get_psi(np.column_stack((r,z)), R, Z, psi_spline)
    points_rho = get_rho_from_psi(points_psi)
    points_strength = interp.splev(points_rho,source_spline)
    points_temperature = interp.splev(points_rho,Ti_spline)

    print("Finished finding volumetrically uniform sampled points")
        
    return np.column_stack((points_in_plasma,points_strength,points_temperature))

def calculate_xy_extrema(x0, y0, theta_start, theta_end):
    r = np.hypot(x0, y0)
    phi = np.arctan2(y0, x0)  # in radians
    
    # Global angles where extrema may occur
    theta_possible_extrema = [0 * math.pi, 0.5 * math.pi, 1.0 * math.pi, 1.5 * math.pi]
    
    # Adjust arc limits to be in [a, b], ensuring proper direction
    tmin = theta_start + phi
    tmax = theta_end + phi
    if tmax < tmin:
        tmax += 2 * np.pi  # wraparound
    
    # Add boundary angles
    thetas = [tmin, tmax]
    
    # Add any critical angles that fall within the arc
    for t in theta_possible_extrema:
        t_adj = t  # ensure in correct 2π cycle
        while t_adj < tmin:
            t_adj += 2 * np.pi
        if tmin <= t_adj <= tmax:
            thetas.append(t_adj)

    # Evaluate x = r cos(theta), y = r sin(theta)
    xs = [r * np.cos(t) for t in thetas]
    ys = [r * np.sin(t) for t in thetas]
    
    return [min(xs), max(xs),min(ys),max(ys)]

def sample_cell_range(samples, R, Z, theta, LCFS):
    points = np.random.rand(samples, 3) # in the shape of [x0,y0,z0; x1,y2,z1; ...]
    
    x0_inner_arc = np.min(LCFS[:,0])
    y0_inner_arc = 0.0
    
    x0_outer_arc = np.max(LCFS[:,0])
    y0_outer_arc = 0.0
    
    theta_start = theta[0]
    theta_end = theta[1]
    
    inner_extrema = calculate_xy_extrema(x0_inner_arc,y0_inner_arc,theta_start,theta_end)
    outer_extrema = calculate_xy_extrema(x0_outer_arc,y0_outer_arc,theta_start,theta_end)
    
    x_min = min([inner_extrema[0],outer_extrema[0]])
    x_max = max([inner_extrema[1],outer_extrema[1]])
    x_range = x_max - x_min
    y_min = min([inner_extrema[2],outer_extrema[2]])
    y_max = max([inner_extrema[3],outer_extrema[3]])
    y_range = y_max - y_min
    z_min = np.min(LCFS[:,1])
    z_max = np.max(LCFS[:,1])
    z_range = z_max - z_min
    
    # Shift the samples to be in a volume represented by the max points of half a torus
    points[:,0] = np.multiply(points[:,0], x_range) + x_min
    points[:,1] = np.multiply(points[:,1], y_range) + y_min
    points[:,2] = np.multiply(points[:,2], z_range) + z_min
    
    return points
    
def points_inside_cell_mask(points, theta, LCFS):
    x,y,z = points.T

    x_rotated, z = xyz_to_xz_plane(points)
    
    rotated_points = np.column_stack((x_rotated,z))
    
    # Convert LCFS to Path object
    polygon_path = m_path(LCFS)
    polygon_mask = polygon_path.contains_points(rotated_points)
    
    points_theta = np.arctan2(points[:,1], points[:,0]) % (2 * math.pi)
    theta_mask = np.logical_and(points_theta > theta[0], points_theta < theta[1])
    
    cell_rejection_mask = np.logical_and(polygon_mask,theta_mask)
    
    return cell_rejection_mask

def points_in_polygon(points, theta, polygon_coords):
    """
    Determine if multiple points are inside a polygon.
    """
    # Convert polygon to Path object
    polygon_path = m_path(polygon_coords)
    
    points_theta = np.arctan2(points[:,1], points[:,0])
    theta_mask = np.logical_and(points_theta > theta[0], points_theta < theta[1])
    
    # Use Path's contains_points method to check all points at once
    return polygon_path.contains_points(points[theta_mask])

def get_psi(eval_points, R, Z, psi_rectspline):
    
    if len(eval_points.shape) == 1:
        eval_points = eval_points[np.newaxis, :]
        
    x = eval_points[:, 0]
    y = eval_points[:, 1]
    
    x_scale = max(R[0,:]) - min(R[0,:])
    y_scale = max(Z[:,0]) - min(Z[:,0])
    
    x_flipped = (x - min(R[0,:])) * y_scale/x_scale + min(Z[:,0]) 
    y_flipped = (y - min(Z[:,0])) * x_scale/y_scale + min(R[0,:])

    return psi_rectspline.ev(y_flipped, x_flipped)

def get_rho_from_psi(psi):
    return psi


def make_TRANSP_profiles(u_Ne, u_Ti, Zeff, x_axis = "rho", plot = False):
    
    profile_time = 0.500
    
    u_files = [u_Ne, u_Ti]
    scales  = [1e6/Zeff,1000]
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
        
        # trim off excess data beyond the LCFS (rho > 1.0)
        data_rho = np.stack((rho,data_t_average),axis=1) 
        data_rho_trimmed = np.array([rn for rn in data_rho if rn[0] <= 1.00]) # trims off pedestal region
        data_rho_trimmed = data_rho_trimmed[data_rho_trimmed[:,0].argsort(),:]
        data_x = data_rho_trimmed[:,0]
        data_y = data_rho_trimmed[:,1]
        
        # create spline fit of temporally averaged profile
        data_spline = interp.splrep(data_x, data_y, k = 1)
        splines.append(data_spline)
        
        plotting_data.append((data_x, data_y, data_spline, times, color, measurement))
    
    plotting = (plotting_data, profile_time)
    
    return tuple(splines), plotting

def DD_xs(T):
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
    sigma_v_DD = 2.33e-14 * (T)**(-2/3)*np.exp(-18.76*(T)**(-1/3))
    return sigma_v_DD

def DT_xs_nrl(T):
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
    sigma_v_DD = 3.68e-12 * (T)**(-2/3)*np.exp(-19.94*(T)**(-1/3))
    return sigma_v_DD

def DT_xs(ion_temperature):
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