#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 11:31:37 2025

@author: ttaczak
"""
import numpy as np

def xyz_to_xz_plane(points):
    x,y,z = points.T
    
    # mask = np.greater_equal(y,np.zeros_like(y))
    angles = np.arctan2(y,x) # if mask else np.arctan2(x,-y)
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles-np.pi)
    
    # Rotate the samples to the x-z plane to be able to check if the point is within the largest contour
    x_rotated = x * cos_vals - y * sin_vals
    # y_rotated = x * sin_vals + y * cos_vals
    return x_rotated, z

def set_plot_aspect_ratio(ax):
    """
    Adjust aspect ratio to physical coordinates
    """
    ax.set_aspect('equal', adjustable='box')
    
def adjust_label_size_and_padding(ax, fontsize=10, labelpad=5, tickpad=2):
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
    
def save_plot(fig, filename, file_format='png', dpi=300, bbox_inches='tight'):
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
    fig.savefig(f"{loc}.{file_format}", format=file_format, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Plot saved as {filename}.{file_format}")