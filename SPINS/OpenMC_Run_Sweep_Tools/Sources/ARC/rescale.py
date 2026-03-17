import numpy as np

names = [
    "ARC_Default_5.npy",
    "ARC_Default_100.npy",
    "ARC_Default_1000.npy",
    "ARC_20_current_1000.npy",
    "ARC_20_density_1000.npy",
    "ARC_20_temp_1000.npy",
]

for name in names:
    sources = np.load(name)
    old_xmax = sources[:,0].max()
    old_zmax = sources[:,1].max()
    sources[:,0] = sources[:,0]*100
    sources[:,1] = sources[:,1]*100
    new_xmax = sources[:,0].max()
    new_zmax = sources[:,1].max()
    np.save(name,sources)
    print(f"Finished rescaling {name}")
    print(f"Max(x): {old_xmax} -> {new_xmax}")
    print(f"Max(z): {old_zmax} -> {new_zmax}")