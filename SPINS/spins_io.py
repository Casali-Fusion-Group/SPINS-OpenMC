#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 09:34:46 2025

@author: ttaczak
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
try:
    import openmc
    from openmc import IndependentSource
    have_openmc = True
except ImportError:
    have_openmc = False

def sample_openmc_sources(n_sources):
    if have_openmc:
        sources = []
        for i in range(n_sources.shape[0]):
            x, y, z, s, Ti = n_sources[i,:]
            point_source = IndependentSource()
            point_source.energy = openmc.stats.muir(e0=14080000.0,m_rat = 4.0, kt = Ti)
            point_source.space = openmc.stats.Point(xyz=(x,y,z))
            point_source.angle = openmc.stats.Isotropic()
            point_source.strength = s
            
            sources.append(point_source)
        return sources
    else:
        loc = "../" + "SPINS_sources.npy"
        np.save(loc,n_sources)
        
        print(f"OPENMC NOT FOUND! Creating NumPy file with source information at {loc}.")
        return None
    

def validate_input(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Expected a file at '{path}', but it does not exist or is not a file.")

def read_input_profile(data):
    """
    Accepts a flexible input (list of lists, numpy array, DataFrame, or filepath)
    and returns a 2D numpy array.
    """
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError("Only 2D arrays are supported.")
        return data

    elif isinstance(data, pd.DataFrame):
        return data.values

    elif isinstance(data, list):
        return np.array(data)

    elif isinstance(data, str) or isinstance(data, Path):
        path = Path(data)
        ext = path.suffix.lower()
        
        if ext == '.csv':
            return pd.read_csv(path, header=None).values
        elif ext in ['.txt', '.dat']:
            return np.loadtxt(path)
        elif ext == '.npy':
            arr = np.load(path)
            if arr.ndim != 2:
                raise ValueError("Loaded array is not 2D.")
            return arr
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    else:
        raise TypeError("Unsupported input type. Must be array, DataFrame, list, or file path.")


def save_profiles_to_json(profiles: dict, filename: str):
    """
    Save multiple (x, y) profile pairs to a JSON file.

    Parameters:
    - profiles: dict of the form {'profile_name': {'x': [...], 'y': [...]} }
    - filename: output JSON filename
    """
    # Convert all arrays to lists (if not already)
    serializable_profiles = {}
    for name, profile in profiles.items():
        x, y = profile["x"], profile["y"]
        serializable_profiles[name] = {
            "x": list(x),
            "y": list(y)
        }

    # Save to JSON
    with open(filename, "w") as f:
        json.dump(serializable_profiles, f, indent=2)
    print(f"Profiles saved to {filename}")