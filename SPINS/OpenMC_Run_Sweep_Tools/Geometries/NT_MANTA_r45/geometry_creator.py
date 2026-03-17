import numpy as np
import openmc
from pathlib import Path

inner_boundary = np.load("NT_inner_boundary.npy")*100
outer_boundary = np.load("NT_outer_boundary.npy")*100
breeder_boundary = np.load("NT_breeder_boundary.npy")*100

materials_fname = "/home/btaczak/ISFNT-15/Materials/materials.xml"
materials = openmc.Materials.from_xml(materials_fname)

for mat in materials:
    if mat.name == "tungsten":
        tungsten = mat
    elif mat.name == "breeder":
        breeder_mat = mat

z_divs = 158.03081447773055
r_hemispheres = 525.0
x_min, x_max = -1000.0, 1000.0
y_min, y_max = -1000.0, 1000.0
z_min, z_max = -1000.0, 1000.0

upper_div_plane = openmc.ZPlane(z0=z_divs)
lower_div_plane = openmc.ZPlane(z0=-z_divs)
r_divider_cylinder = openmc.ZCylinder(r=r_hemispheres)
bounding_box = openmc.model.RectangularParallelepiped(x_min, x_max, y_min, y_max, z_min, z_max, boundary_type = 'vacuum')


cells = []

inner_wall = openmc.model.Polygon(points=inner_boundary, basis='rz')
outer_wall = openmc.model.Polygon(points=outer_boundary, basis='rz')
breeder_wall = openmc.model.Polygon(points=breeder_boundary, basis='rz')

plasma = -inner_wall  # polygon extruded in Y
upper_div = +inner_wall & -outer_wall & +upper_div_plane
lower_div = +inner_wall & -outer_wall & -lower_div_plane
outer_first_wall = +inner_wall & -outer_wall & +lower_div_plane & -upper_div_plane & +r_divider_cylinder
inner_first_wall = +inner_wall & -outer_wall & +lower_div_plane & -upper_div_plane & -r_divider_cylinder

upper_div_breeder = +outer_wall & -breeder_wall & +upper_div_plane
lower_div_breeder = +inner_wall & -breeder_wall & -lower_div_plane
outer_breeder = +outer_wall & -breeder_wall & +lower_div_plane & -upper_div_plane & +r_divider_cylinder
inner_breeder = +outer_wall & -breeder_wall & +lower_div_plane & -upper_div_plane & -r_divider_cylinder
bounding_region = -bounding_box & +breeder_wall & +outer_wall 

plasma = openmc.Cell(name=f"plasma", fill=None, region=plasma)
upper_div_cell = openmc.Cell(name=f"upper_div", fill=tungsten, region=upper_div)
lower_div_cell = openmc.Cell(name=f"lower_div", fill=tungsten, region=lower_div)
outer_first_wall_cell = openmc.Cell(name=f"outer_first_wall", fill=tungsten, region=outer_first_wall)
inner_first_wall_cell = openmc.Cell(name=f"inner_first_wall", fill=tungsten, region=inner_first_wall)
upper_div_breeder_cell = openmc.Cell(name=f"upper_div_breeder", fill=tungsten, region=upper_div_breeder)
lower_div_breeder_cell = openmc.Cell(name=f"lower_div_breeder", fill=tungsten, region=lower_div_breeder)
outer_breeder_cell = openmc.Cell(name=f"outer_breeder", fill=breeder_mat, region=outer_breeder)
inner_breeder_cell = openmc.Cell(name=f"inner_breeder", fill=breeder_mat, region=inner_breeder)
bounding_cell = openmc.Cell(name="bounds", fill=None, region=bounding_region)
cells=[plasma, 
    upper_div_cell, 
    lower_div_cell,
    outer_first_wall_cell,
    inner_first_wall_cell,
    upper_div_breeder_cell,
    lower_div_breeder_cell,
    outer_breeder_cell,
    inner_breeder_cell,
    bounding_cell]

geometry = openmc.Geometry(openmc.Universe(cells=cells))
geometry.remove_redundant_surfaces()
geometry.export_to_xml()