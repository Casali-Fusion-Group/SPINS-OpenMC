import numpy as np
import openmc
from pathlib import Path

# Materials
tungsten = openmc.Material(name='tungsten')
tungsten.add_element('W', 1.0)  # natural tungsten
tungsten.set_density('g/cm3', 19.25)  # density of solid tungsten

enrichment = 0.6
breeder_mat = openmc.Material(name="breeder")
breeder_mat.add_nuclide('Li6', enrichment, percent_type='ao')
breeder_mat.add_nuclide('Li7', (1-enrichment), percent_type='ao')
breeder_mat.set_density('g/cm3', 0.534)

materials = openmc.Materials([tungsten,breeder_mat])
materials.export_to_xml()