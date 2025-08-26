# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from spins_utils import (
    xyz_to_xz_plane,
    set_plot_aspect_ratio,
    adjust_label_size_and_padding,
    save_plot,
)
from GEQDSK_Reader import GEQDSK
import scipy.interpolate as interp
from matplotlib.path import Path as PTH
from matplotlib.patches import PathPatch

def plot_source_positions(gfile_path, sources,save_name=None, wall_file=None, debug=False, dpi=1200):
    g = GEQDSK(gfile_path)
    min_psi = np.min(g.psi)
    max_psi = np.max(g.psi)
    psin = (g.psi - max_psi) / (g.psiedge - max_psi)
    # print(g)
    R, Z = np.meshgrid(g.R_grid,g.Z_grid)
    
    if debug:
        
        # Plot the contour
        plt.figure(figsize=(5, 2.5))
        contour = plt.contourf(g.R_grid, g.Z_grid, psin, cmap='Blues_r')
        plt.colorbar(contour)
        
        # Add labels and title
        plt.xlabel('R (m)')
        plt.ylabel('Z (m)')
        plt.title('Contour Plot of Psi')
        
        # Show the plot
        plt.show()
        
    LCFS = np.column_stack((g.Z_LCFS,g.R_LCFS))
    LCFS_cleaned = cut_peanut_shaped_loop(g.R_LCFS, g.Z_LCFS)
    
    x_pad = 0.1
    y_pad = 0.2
    
    x,z = xyz_to_xz_plane(sources[:,0:3])
    strengths = sources[:,3]
    
    fig, ax = plt.subplots(figsize=(4,8),dpi = dpi)
    levels = np.linspace(0.0, .99, 15)  # 10 levels between 30 and max(data)
    
    contour = ax.contour(R, Z, psin, levels = levels, cmap='viridis_r', alpha = 0.5, vmin=-0.1)        
    #sc = ax.scatter(sources_to_plot[:,0], sources_to_plot[:,1], c=sources_to_plot[:,2], cmap='viridis', vmin=np.min(sources_to_plot[:,2]), vmax=np.max(sources_to_plot[:,2]))
    sc = ax.scatter(x,z,s=0.1)
    
        
    if wall_file is not None:
        wall_data = np.loadtxt(wall_file)/1000
        
        wall_x = wall_data[:,0]
        wall_y = wall_data[:,1]
        wall_x_closed = np.append(wall_x, wall_x[0])
        wall_y_closed = np.append(wall_y, wall_y[0])
        LCFS = np.column_stack((wall_y_closed,wall_x_closed))
        
        ax.plot(wall_x_closed, wall_y_closed, color='k', linewidth=1)
        
        # Create and apply clip path from LCFS
        # lcfs_path = PTH(LCFS[:, [1, 0]])  # Assuming LCFS[:,1] is R and LCFS[:,0] is Z
        lcfs_path = PTH(np.array([wall_x, wall_y]).T)  # Assuming LCFS[:,1] is R and LCFS[:,0] is Z
        patch = PathPatch(lcfs_path, transform=ax.transData)
            
        # for artist in contour.get_children():
        #     artist.set_clip_path(patch)
        contour.set_clip_path(patch)
    
        # # 2. Plot the contour line at this level, black color
        # contour_zero = ax.contour(
        #     R, Z, psin,
        #     levels=[1],
        #     colors='black',
        #     linewidths=5,
        #     label="LCFS"
        # )
        # contour_zero.set_clip_path(patch)
        # ax.legend()
    
    # # Use make_axes_locatable to create the first colorbar
    # divider = make_axes_locatable(ax)
    # cax1 = divider.append_axes("right", size="15%", pad=0.05)  # Colorbar for the first contour
    # cbar1 = plt.colorbar(sc, cax=cax1)
    
    # cbar1.ax.tick_params(labelsize=8, labelrotation=0,direction="in")
    # #cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1e'))
    
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
    # cbar1.set_label(r'Volumetric Source Strength $[n/m^3]$', labelpad=8, rotation=270, fontsize=10, verticalalignment='center')

    # # Create the second colorbar for the second contour plot
    # levels = np.linspace(0.0, 1.0, num=6)
    
    # # Use ScalarMappable to create a filled colorbar
    # norm = Normalize(vmin=0.0, vmax=1)
    # sm = ScalarMappable(norm=norm, cmap='viridis')
    # sm.set_array([])  # Dummy array for the ScalarMappable
    
    # cax2 = divider.append_axes("right", size="15%", pad=0.20)  # Place the second colorbar closer
    # cbar2 = plt.colorbar(contour, cax=cax2)
    
    # cbar2.ax.tick_params(labelsize=8, labelrotation=270, direction="in",size=0,pad = 0,labelleft=False, labelright=True)
    # cbar2.set_ticks(levels)
    
    # # Set the colorbar label and keep it inside and centered
    # cbar2.set_label(r'$\psi_{norm}$', labelpad=0, rotation=270, fontsize=12, verticalalignment='center')

    
    ax.set_xlim(np.min(R)-x_pad, np.max(R)+x_pad)
    ax.set_ylim(np.min(Z)-y_pad, np.max(Z)+y_pad)
    ax.set_xlabel(r"Major radius, $R [m]$")
    ax.set_xlabel(r"Radius, $R[m]$")
    
    set_plot_aspect_ratio(ax)
    adjust_label_size_and_padding(ax)
    ax.set_title(f'SPINS Source Profile', ha='center', fontsize=16, pad=15)
    if save_name is not None:
        save_plot(fig, save_name)
    plt.show()

def plot_sources(sources_to_plot,psi_mesh, R, Z, psin, fname, wall_data = None, dpi=300, plot_name = "Neutron Source Profile"):
    x_pad = 0.1
    y_pad = 0.2
    
    x,z = xyz_to_xz_plane(sources_to_plot[:,0:3])
    strengths = sources_to_plot[:,3]
    
    fig, ax = plt.subplots(figsize=(4,8),dpi = dpi)
    levels = np.linspace(0.0, 1.0, 15)  # 10 levels between 30 and max(data)

    contour = ax.contour(R, Z, psin, levels = levels, cmap='viridis_r', alpha = 0.5, vmin=-0.1)        
    #sc = ax.scatter(sources_to_plot[:,0], sources_to_plot[:,1], c=sources_to_plot[:,2], cmap='viridis', vmin=np.min(sources_to_plot[:,2]), vmax=np.max(sources_to_plot[:,2]))
    sc = ax.scatter(x,z, c=np.abs(strengths), cmap='viridis', vmin=np.min(strengths), vmax=np.max(strengths))
    
        
    if wall_data is not None:
        wall_x = wall_data[:,0]
        wall_y = wall_data[:,1]
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
    
        # 2. Plot the contour line at this level, black color
        contour_zero = ax.contour(
            R, Z, psin,
            levels=[1],
            colors='black',
            linewidths=0.5
        )
        for coll in contour_zero.collections:
            coll.set_clip_path(patch)
    
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

    
    ax.set_xlim(np.min(R)-x_pad, np.max(R)+x_pad)
    ax.set_ylim(np.min(Z)-y_pad, np.max(Z)+y_pad)
    ax.set_xlabel(r"Major radius, $R [m]$")
    ax.set_xlabel(r"Radius, $R[m]$")
    
    set_plot_aspect_ratio(ax)
    adjust_label_size_and_padding(ax)
    ax.set_title(f'{plot_name}', ha='center', fontsize=16, pad=15)
    save_plot(fig, fname)
    plt.show()
    
def plot_LCFS(LCFS, psin, R, Z, wall_data,num_contours=15):
    x_pad = 0.1
    y_pad = 0.2
    
    fig, ax = plt.subplots(dpi=1200)
    vmax = 1.5
    # contour plot of psi
    contour = plt.contourf(R, Z, psin, vmax=vmax, levels=num_contours, cmap='Blues_r')
    
    #ax.imshow(psi_rz, cmap=plt.cm.gray)
    ax.plot(LCFS[:,0], LCFS[:,1],c='k', linewidth=2, label="LCFS")
    ax.set_title(f'Equilibrium and LCFS')
    # ax.set_xlabel(r"$R [m]")
    # ax.set_ylabel(r"$z [m]")
    ax.legend()
    
    divider = make_axes_locatable(ax)
    
    # Create the second colorbar for the second contour plot
    levels = np.linspace(0,vmax, num=7)
    
    # Use ScalarMappable to create a filled colorbar
    norm = Normalize(vmin=np.min(psin), vmax=vmax)#np.max(psi_rz_norm))
    sm = ScalarMappable(norm=norm, cmap='Blues_r')
    sm.set_array([])  # Dummy array for the ScalarMappable
    
    cax2 = divider.append_axes("right", size="15%", pad=0.20)  # Place the second colorbar closer
    cbar2 = plt.colorbar(contour, cax=cax2)
    
    cbar2.ax.tick_params(labelsize=8, labelrotation=-90, direction="out",size=0,pad = 0,labelleft=False, labelright=True)
    cbar2.set_ticks(levels)
    cbar2.ax.set_ylim(0, vmax)
    
    # Set the colorbar label and keep it inside and centered
    cbar2.set_label(r'$\psi_{norm}$', labelpad=-13, rotation=270, fontsize=12, verticalalignment='center')
    
    if wall_data is not None:
        wall_x = wall_data[:,0]
        wall_y = wall_data[:,1]
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
    
        # 2. Plot the contour line at this level, black color
        contour_zero = ax.contour(
            R, Z, psin,
            levels=[1],
            colors='black',
            linewidths=0.5
        )
        for coll in contour_zero.collections:
            coll.set_clip_path(patch)
    # for artist in contour_zero.get_children():
    #     artist.set_clip_path(patch)
    
    
    ax.set_xlim(np.min(R)-x_pad, np.max(R)+x_pad)
    ax.set_ylim(np.min(Z)-y_pad, np.max(Z)+y_pad)
    ax.set_xlabel(r"Major radius, $R [m]$")
    ax.set_xlabel(r"Radius, $R[m]$")
    
    set_plot_aspect_ratio(ax)
    adjust_label_size_and_padding(ax)
    plt.show()
    
def psi_contourplot(R, Z, psin, LCFS, num_contours = 30):
    '''
    Plot the magnetic potential provided from an equdsk.

    Parameters
    ----------
    R : numpy array
        R COORDINATES OF PSI MESH. FROM EQDSK.
    Z : numpy array
        Z COORDINATES OF PSI MESH. FROM EQDSK.
    psi : numpy array
        MAGNETIC POTENTIAL. FROM EQDSK.
    num_contours : INT, optional
        NUMBER OF CONTOURS TO USE FOR PLOT. The default is 30.

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(R, Z, psin, levels=num_contours, cmap='Blues')
    plt.plot(LCFS[:,0], LCFS[:,1])
    plt.colorbar(contour)
    
    # Add labels and title
    plt.xlabel('R (m)')
    plt.ylabel('Z (m)')
    plt.title(r'Magnetic Potential ($\psi$)')
    
    # Show the plot
    plt.show()
    
def plot_source_profile(source_spline,plot_name,units, interp_points = 100,type_xs="DD"):
    # --- Evaluate experimental spline ---
    S_x_exp = np.linspace(0.0, 1.0, interp_points)
    Source_exp = interp.splev(S_x_exp, source_spline)
    
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # ax.plot(S_xnew_transp, S_ynew_transp, color='#C9A0DC', lw=2,linestyle='--', label='TRANSP Source')
    ax.plot(S_x_exp, Source_exp, color='#5B2C6F', lw=2, label='Experimental Source')
    
    ax.set_xlabel(r'Normalized radius $\rho$', fontsize=12)
    ax.set_ylabel(rf'{units}', fontsize=12)
    ax.set_title(plot_name, fontsize=13)
    
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(fontsize=10, frameon=False, loc='upper right')
    
    ax.grid(True, linestyle=':', linewidth=0.6)
    fig.tight_layout()
    # fig.savefig('source_profile_comparison.pdf', dpi=300)  # Uncomment to save
    
    plt.show()


def cut_peanut_shaped_loop(R, Z, dist_threshold=0.05):
    """
    Detect peanut-shaped pinch in a closed loop and cut it open,
    keeping the bigger of the two resulting loops, and ensure the
    returned loop is closed by repeating the first point at the end.

    Parameters
    ----------
    R : 1D np.array
        Radial coordinates of the closed contour.
    Z : 1D np.array
        Vertical coordinates of the closed contour.
    dist_threshold : float
        Distance threshold to identify a pinch.

    Returns
    -------
    np.array of shape (N+1, 2)
        The coordinates of the final closed loop (either original or bigger open loop),
        with the first point appended at the end to close the loop explicitly.
    """
    points = np.column_stack((Z, R))
    N = len(points)

    # Compute all pairwise distances, ignoring neighbors and self
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    for i in range(N):
        dists[i, i] = np.inf
        dists[i, (i-1) % N] = np.inf
        dists[i, (i+1) % N] = np.inf

    # Find all pairs below threshold
    pinch_indices = np.where(dists < dist_threshold)

    # Filter pairs to exclude neighbors, keep pairs far apart in index
    valid_pairs = [(i, j) for i, j in zip(*pinch_indices) if abs(i - j) > 5 and abs(i - j) < N - 5]

    if not valid_pairs:
        # No pinch detected: return original closed loop, ensure closure
        closed_loop = np.vstack([points, points[0]])
        return closed_loop

    # Find pair with minimum distance (strongest pinch)
    distances = [dists[i, j] for i, j in valid_pairs]
    min_idx = np.argmin(distances)
    i_cut, j_cut = valid_pairs[min_idx]

    # Ensure i_cut < j_cut
    if j_cut < i_cut:
        i_cut, j_cut = j_cut, i_cut

    # Two candidate loops:
    loop1 = points[i_cut : j_cut + 1]
    loop2 = np.vstack((points[j_cut:], points[:i_cut + 1]))

    # Keep the bigger loop (by number of points)
    if len(loop1) >= len(loop2):
        chosen_loop = loop1
    else:
        chosen_loop = loop2

    # Close the loop by appending the first point at the end
    closed_loop = np.vstack([chosen_loop, chosen_loop[0]])

    return closed_loop

if __name__=="__main__":
    gfile_path = "/Users/ttaczak/Desktop/Research/2-OpenMC/1-TOFE_paper/Experimental_Data/g49392.00350"
    source_loc = "/Users/ttaczak/Desktop/Research/2-OpenMC/1-TOFE_paper/Source_Creation/Data/sampled_profiles/SPINS_profile_1000.npy"
    wall_file = "/Users/ttaczak/Desktop/Research/2-OpenMC/1-TOFE_paper/Experimental_Data/MASTU_wall.txt"
    sources = np.load(source_loc)
    plot_source_positions(gfile_path, sources,save_name=None, wall_file=wall_file, debug=False, dpi=1200)

