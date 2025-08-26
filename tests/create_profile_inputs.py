#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 13:35:35 2025

@author: ttaczak
"""
import numpy as np
from scipy import interpolate as interp
from ufiles import UFILE
import pandas as pd
from pathlib import Path


# Define the position of the data directory
current_dir = Path(__file__).resolve()
grandparent_dir = current_dir.parent.parent

# Define the paths for where the input data is stored.
input_dir = grandparent_dir / Path('SPINS_inputs')
output_dir = grandparent_dir / Path('SPINS_outputs')
eqdsk_path = input_dir / Path("equ_49392_symm_X4_crop.equ")
ni_path = input_dir / Path("TSdata_49394.txt")
Ti_path = input_dir / Path("Ti_data.pickle")
wall_file = input_dir / Path("MASTU_wall.txt")

from scipy import interpolate as interp
from ufiles import UFILE
import pandas as pd

def make_TRANSP_profiles(u_Ne, u_Ti, Zeff,output_dir, x_axis = "rho", plot = False):
    
    profile_time = 0.500
    
    u_files = [u_Ne, u_Ti]
    scales  = [1e6/Zeff,1000]
    measurements = [r"$T_i [eV]$",r"$n_i [m^{-3}]$"]
    colors = ["OrRd", "GnBu"]
    out_names = ["TRANSP_Ti_profile", "TRANSP_ni_profile"]
    
    splines = []
    plotting_data = []
    for i,(file, color, measurement, out_name) in enumerate(zip(u_files,colors,measurements,out_names)):
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
        output_profile = data_rho_trimmed[data_rho_trimmed[:,0].argsort(),:]
        
        output_fname = output_dir / Path(out_name)
        # As a pandas DataFrame (pickle or csv are typical for DF)
        df = pd.DataFrame(output_profile, columns=["rho", "prof"])
        df.to_pickle(f".pkl")     # Binary DataFrame format
        df.to_csv(f"{output_fname}.csv", index=False, header=False)  # CSV
        
        # As a .txt file
        np.savetxt(f"{output_fname}.txt", output_profile)
        
        # As a .dat file (same as txt, just extension)
        np.savetxt(f"{output_fname}.dat", output_profile)
        
        # As a .npy file
        np.save(f"{output_fname}.npy", output_profile)
        
        # # create spline fit of temporally averaged profile
        # data_spline = interp.splrep(data_x, data_y, k = 1)
        # splines.append(data_spline)
        
        # plotting_data.append((data_x, data_y, data_spline, times, color, measurement))
        
u_Ne = UFILE(input_dir / "NER49392.txt")
u_Ti = UFILE(input_dir / "OMFITTI249392.txt")
Zeff = 1.5

make_TRANSP_profiles(u_Ne, u_Ti, Zeff, output_dir)








