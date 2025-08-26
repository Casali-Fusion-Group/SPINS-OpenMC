#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:39:17 2024

@author: ttaczak
"""

import vtk
import numpy as np
import matplotlib.pyplot as plt

# Example points and colors

sources = np.load("sources.npy")

#%%
points = sources[:,0:2]
strengths = sources[:,3]  

# convert to RGB values
strengths = strengths/np.max(strengths)
cmap = plt.get_cmap('viridis')
rgb_values = [cmap(strength)[:3] for strength in strengths]  # Take only RGB (not alpha)

# Create a VTK points object
vtk_points = vtk.vtkPoints()
for p in points:
    print(tuple(p))
    vtk_points.InsertNextPoint(tuple(p))

# Create a VTK polydata object
polydata = vtk.vtkPolyData()
polydata.SetPoints(vtk_points)

# Create a VTK color array for the points
vtk_colors = vtk.vtkUnsignedCharArray()
vtk_colors.SetNumberOfComponents(3)  # RGB values have 3 components
vtk_colors.SetName("Colors")

# Add the colors to the color array
for color in rgb_values:
    # Convert color to 0-255 range for VTK
    vtk_colors.InsertNextTuple3(int(color[0] * 255), int(color[0] * 128), int(color[0] * 128))

# Assign colors to the points
polydata.GetPointData().SetScalars(vtk_colors)

# Write the data to a VTK file
writer = vtk.vtkPolyDataWriter()
writer.SetFileName("colored_points.vtk")
writer.SetInputData(polydata)
writer.Write()