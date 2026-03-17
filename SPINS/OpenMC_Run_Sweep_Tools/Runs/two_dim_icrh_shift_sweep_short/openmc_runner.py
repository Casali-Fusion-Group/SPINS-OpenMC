import openmc
geometry = openmc.Geometry.from_xml("geometry.xml")
materials = openmc.Materials.from_xml("materials.xml")
settings = openmc.Settings.from_xml("settings.xml")
tallies = openmc.Tallies.from_xml("tallies.xml")
model = openmc.model.Model(geometry=geometry,materials=materials,settings=settings,tallies=tallies)
model.export_to_xml()
model.run()