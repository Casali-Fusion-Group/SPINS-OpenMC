#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:51:43 2024

@author: ttaczak
"""

import numpy as np
from scipy import interpolate as interp
import pandas as pd
from pathlib import Path

def get_density_profile_1D(self, Zeff, x_axis = "rho", plot = False):

    ### TS data for n_i ###
    
    TS_data_loc = str(self.ni_profile_path)
    column_lables = pd.read_csv(TS_data_loc, skiprows=[0, 0], nrows=1, sep=r'\s+').columns
    df = pd.read_csv(TS_data_loc, skiprows=[0, 1], names=column_lables, sep=r'\s+')
    df.Ne = df.Ne.astype(float)
    
    Ni_profile_unsmoothed = df[~np.isnan(df['Ne'])][['psi_n','Ne']].to_numpy()
    Ni_profile_unsmoothed[:,1] = np.copy(Ni_profile_unsmoothed[:,1]) / Zeff
    
    # Apply moving average to smooth
    Ni_trimmed = np.array([rn for rn in Ni_profile_unsmoothed if rn[0] <= 1.00])
    Ni_trimmed = Ni_trimmed[Ni_trimmed[:,0].argsort(),:]
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
    
    Ni_plotting_data = (Ni_x, Ni_y, Ni_x_binned, Ni_y_binned, Ni_spline, Ni_trimmed_avg)
        
    return Ni_spline, Ni_plotting_data

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