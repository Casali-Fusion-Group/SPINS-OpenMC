#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 09:23:23 2025

@author: ttaczak
"""
    
from EquilibriumProfile import EquilibriumProfile 
from pathlib import Path

if __name__=="__main__":
    # Get the current script's directory as well as the two above this
    current_dir = Path(__file__).resolve()
    grandparent_dir = current_dir.parent.parent
    
    # Define folder and file names
    loc_input_data = grandparent_dir / Path('MASTU_Data')
    fname_profile = "Profiles.txt"
    fname_eqdsk = "equ_49392_symm_X4_crop.equ"
    TS_fname = "TSdata_49394.txt"
    Ti_fname = "Ti_data.pickle"
    
    # Create neutron source profile
    profile = EquilibriumProfile(loc_input_data, fname_eqdsk, TS_fname, Ti_fname, verbose=False)
    #profile.plot_all()
