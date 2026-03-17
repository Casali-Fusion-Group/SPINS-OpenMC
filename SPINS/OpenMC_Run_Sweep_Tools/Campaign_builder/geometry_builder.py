"""
geometry_builder.py
===================
OpenMC model builder for a single parametric run.
Copied into each run folder by build_campaign.py and executed there by the
HPC job script (openmc_runner.py / qsub).

Usage
-----
    python geometry_builder.py
    (no arguments — reads run_config.yaml from current working directory)

Exports geometry.xml, materials.xml, settings.xml, tallies.xml to cwd.
Does NOT run OpenMC.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import openmc
import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants — must match original script exactly
# ---------------------------------------------------------------------------
arc_radius   = 365    # cm
manta_radius = 455    # cm
length_icrh  = 300    # cm

x_min, x_max = -1000.0, 1000.0
y_min, y_max = -1000.0, 1000.0
z_min, z_max = -1000.0, 1000.0


# ---------------------------------------------------------------------------
# Geometry lookup table — copied verbatim from original script
# ---------------------------------------------------------------------------

def get_divider_planes(run: dict) -> tuple[float, float]:
    """Return (z_divs, r_hemisphere) for the given run parameters.

    Copied verbatim from the original campaign script.

    Args:
        run: Dict containing at least 'radius' and 'triang' keys.

    Returns:
        (z_divs, r_hemisphere) tuple of floats.

    Raises:
        ValueError: If radius or triang combination is not recognised.
    """
    z_divs = None
    r_hemisphere = None
    print(run["radius"])
    print(run["triang"])
    if run["radius"] == manta_radius:
        z_divs = 158.03081447773055
        if run["triang"] == -0.5:
            r_hemisphere = 525.5
        elif run["triang"] == 0.5:
            r_hemisphere = 400
        else:
            raise ValueError("Sorry, triang not recognized to find z_divs and r_hemispheres")
    elif run["radius"] == arc_radius:
        z_divs = 158.03081447773055
        if run["triang"] == 0.5:
            r_hemisphere = 280
        else:
            raise ValueError("Sorry, triang not recognized to find z_divs and r_hemispheres")
    else:
        raise ValueError("Sorry, radius not recognized to find z_divs and r_hemispheres")

    return (z_divs, r_hemisphere)


# ---------------------------------------------------------------------------
# Source builder — copied verbatim from original script
# ---------------------------------------------------------------------------

def make_ring_sources(sources_fname: Path) -> list[openmc.IndependentSource]:
    """Build a list of ring sources from a .npy source data file.

    Copied verbatim from the original campaign script.

    Args:
        sources_fname: Path to the .npy file containing ring source data.
                       Columns: [r, z, neutron_line_density, ...]

    Returns:
        List of openmc.IndependentSource objects with cylindrical spatial
        distributions, isotropic angles, and Muir energy spectra.
    """
    ring_source_data = np.load(sources_fname)

    # Normalize based on arclength of ring source
    line_density_neutron_emission = ring_source_data[:, 0] * 2 * np.pi * ring_source_data[:, 2]
    weights = line_density_neutron_emission / np.sum(line_density_neutron_emission)

    sources = []
    for ring, w in zip(ring_source_data, weights):
        source = openmc.IndependentSource()
        radius = openmc.stats.Discrete([ring[0]], [1])
        z_values = openmc.stats.Discrete([ring[1]], [1])
        angle = openmc.stats.Uniform(a=0., b=2 * np.pi)
        source.space = openmc.stats.CylindricalIndependent(
            r=radius, phi=angle, z=z_values, origin=(0.0, 0.0, 0.0)
        )
        source.angle = openmc.stats.Isotropic()
        source.energy = openmc.stats.muir(e0=14.08e6, m_rat=5.0, kt=19000.0)  # CHANGE kt LATER TO ACTUAL VALUES!
        source.particle = "neutron"
        source.strength = w
        sources.append(source)

    return sources


# ---------------------------------------------------------------------------
# Main model builder
# ---------------------------------------------------------------------------

def build_model() -> None:
    """Read run_config.yaml from cwd, build the OpenMC model, export to XML.

    Reads all parameters from run_config.yaml in the current working
    directory. Reconstructs all derived geometry quantities. Exports
    geometry.xml, materials.xml, settings.xml, and tallies.xml.

    Raises:
        FileNotFoundError: If run_config.yaml is not found in cwd.
        ValueError: If a tally cell name is not found among constructed cells.
    """
    cwd = Path.cwd()
    config_path = cwd / "run_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"run_config.yaml not found in {cwd}. "
            "geometry_builder.py must be run from within a run folder."
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Extract parameters
    # ------------------------------------------------------------------
    params = cfg["parameters"]
    radius            = params["radius"]
    triang            = params["triang"]
    center_shift_icrh = params["center_shift_icrh"]
    width_icrh        = params["width_icrh"]
    source_device     = params["source_device"]
    source_scenario   = params["source_scenario"]

    paths_cfg       = cfg["paths"]
    geometries_dir  = Path(paths_cfg["geometries_dir"])
    sources_dir     = Path(paths_cfg["sources_dir"])
    materials_file  = Path(paths_cfg["materials_file"])

    sim_cfg    = cfg["simulation"]
    tally_cfg  = cfg["tallies"]

    log.info("Building geometry for run_id=%s", cfg.get("run_id", "<unknown>"))
    log.debug("Parameters: %s", params)

    # ------------------------------------------------------------------
    # Derived geometry quantities (never stored in config)
    # ------------------------------------------------------------------
    icrh_x_location = center_shift_icrh
    icrh_y_location = np.sqrt(radius**2 - center_shift_icrh**2)
    log.debug("icrh_x_location = %s, icrh_y_location = %s",
              icrh_x_location, icrh_y_location)

    (z_divs, r_hemispheres) = get_divider_planes(params)
    log.debug("z_divs = %s, r_hemisphere = %s", z_divs, r_hemispheres)

    # ------------------------------------------------------------------
    # Source file path — constructed from parameters, NOT stored in config
    # ------------------------------------------------------------------
    sources_fname = (
        sources_dir / source_device / f"triang_{triang}" / f"{source_scenario}_1000.npy"
    )
    log.debug("sources_fname: %s", sources_fname)
    sources = make_ring_sources(sources_fname)

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------
    materials = openmc.Materials.from_xml(str(materials_file))
    tungsten = None
    breeder_mat = None
    for mat in materials:
        if mat.name == "tungsten":
            tungsten = mat
        elif mat.name == "breeder":
            breeder_mat = mat
    if tungsten is None:
        raise ValueError("Material 'tungsten' not found in materials.xml")
    if breeder_mat is None:
        raise ValueError("Material 'breeder' not found in materials.xml")

    # ------------------------------------------------------------------
    # Geometry .npy file names (symlinked into cwd by build_campaign.py)
    # ------------------------------------------------------------------
    geometry_fname_root   = f"radius_{radius}_triang_{triang}_"
    breeder_boundary_fname = cwd / (geometry_fname_root + "breeder_boundary.npy")
    inner_boundary_fname   = cwd / (geometry_fname_root + "inner_boundary.npy")
    outer_boundary_fname   = cwd / (geometry_fname_root + "outer_boundary.npy")

    inner_boundary_points   = np.load(inner_boundary_fname)
    outer_boundary_points   = np.load(outer_boundary_fname)
    breeder_boundary_points = np.load(breeder_boundary_fname)

    # ------------------------------------------------------------------
    # Surfaces — exactly as in original script
    # ------------------------------------------------------------------
    upper_div_plane    = openmc.ZPlane(z0=z_divs)
    lower_div_plane    = openmc.ZPlane(z0=-z_divs)
    r_divider_cylinder = openmc.ZCylinder(r=r_hemispheres)
    bounding_box       = openmc.model.RectangularParallelepiped(
        x_min, x_max, y_min, y_max, z_min, z_max, boundary_type="vacuum"
    )

    icrh_duct_region = openmc.model.RectangularPrism(
        width_icrh, width_icrh, axis="y", origin=(icrh_x_location, 0)
    )
    icrh_front_wall = openmc.YPlane(y0=icrh_y_location)
    icrh_back_wall  = openmc.YPlane(y0=icrh_y_location + length_icrh)

    inner_wall   = openmc.model.Polygon(points=inner_boundary_points,   basis="rz")
    outer_wall   = openmc.model.Polygon(points=outer_boundary_points,   basis="rz")
    breeder_wall = openmc.model.Polygon(points=breeder_boundary_points, basis="rz")

    # ------------------------------------------------------------------
    # Regions — exactly as in original script
    # ------------------------------------------------------------------
    icrh_region = -icrh_duct_region & +icrh_front_wall & -icrh_back_wall

    plasma_region        = -inner_wall & ~icrh_region
    upper_div_region     = +inner_wall & -breeder_wall & +upper_div_plane
    lower_div_region     = +inner_wall & -breeder_wall & -lower_div_plane
    outer_first_wall_reg = (+inner_wall & -outer_wall & +lower_div_plane
                            & -upper_div_plane & +r_divider_cylinder & ~icrh_region)
    inner_first_wall_reg = (+inner_wall & -outer_wall & +lower_div_plane
                            & -upper_div_plane & -r_divider_cylinder)
    outer_breeder_reg    = (+outer_wall & -breeder_wall & +lower_div_plane
                            & -upper_div_plane & +r_divider_cylinder & ~icrh_region)
    inner_breeder_reg    = (+outer_wall & -breeder_wall & +lower_div_plane
                            & -upper_div_plane & -r_divider_cylinder)
    bounding_region      = -bounding_box & +breeder_wall & ~icrh_region

    # ------------------------------------------------------------------
    # Cells — names must match tally_cells in run_config.yaml exactly
    # ------------------------------------------------------------------
    plasma_cell           = openmc.Cell(name="plasma",           fill=None,        region=plasma_region)
    upper_div_cell        = openmc.Cell(name="upper_div",        fill=tungsten,    region=upper_div_region)
    lower_div_cell        = openmc.Cell(name="lower_div",        fill=tungsten,    region=lower_div_region)
    outer_first_wall_cell = openmc.Cell(name="outer_first_wall", fill=tungsten,    region=outer_first_wall_reg)
    inner_first_wall_cell = openmc.Cell(name="inner_first_wall", fill=tungsten,    region=inner_first_wall_reg)
    outer_breeder_cell    = openmc.Cell(name="outer_breeder",    fill=breeder_mat, region=outer_breeder_reg)
    inner_breeder_cell    = openmc.Cell(name="inner_breeder",    fill=breeder_mat, region=inner_breeder_reg)
    bounding_cell         = openmc.Cell(name="bounds",           fill=None,        region=bounding_region)
    icrh_duct_cell        = openmc.Cell(name="icrh_duct",        fill=None,        region=icrh_region)

    cells = [
        plasma_cell,
        upper_div_cell,
        lower_div_cell,
        outer_first_wall_cell,
        inner_first_wall_cell,
        outer_breeder_cell,
        inner_breeder_cell,
        bounding_cell,
        icrh_duct_cell,
    ]

    # Lookup map for tally construction
    cell_map: dict[str, openmc.Cell] = {c.name: c for c in cells}

    # ------------------------------------------------------------------
    # Geometry object
    # ------------------------------------------------------------------
    geometry = openmc.Geometry(openmc.Universe(cells=cells))
    geometry.remove_redundant_surfaces()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    settings = openmc.Settings()
    settings.run_mode = sim_cfg["run_mode"]
    settings.batches  = sim_cfg["batches"]
    settings.particles = sim_cfg["particles"]
    settings.source   = sources

    # ------------------------------------------------------------------
    # Tallies
    # ------------------------------------------------------------------
    tallies_list = openmc.Tallies()
    energy_bins_eV = np.array(tally_cfg["energy_groups_MeV"]) * 1e6
    energy_filter  = openmc.EnergyFilter(energy_bins_eV)
    scores         = tally_cfg["scores"]

    # Per-cell energy-group tallies
    for cell_name in tally_cfg["tally_cells"]:
        if cell_name not in cell_map:
            available = list(cell_map.keys())
            raise ValueError(
                f"Tally cell '{cell_name}' not found among constructed cells.\n"
                f"Available cell names: {available}"
            )
        cell_filter = openmc.CellFilter([cell_map[cell_name]])
        tally = openmc.Tally(name=f"{cell_name}_tally")
        tally.filters = [cell_filter, energy_filter]
        tally.scores  = scores
        tallies_list.append(tally)
        log.debug("Created tally '%s' for cell '%s'", tally.name, cell_name)

    # icrh_duct_end_current tally — preserved verbatim from original script
    neutron_particle_filter = openmc.ParticleFilter(["neutron"])
    icrh_back_wall_filter   = openmc.SurfaceFilter(icrh_back_wall)
    icrh_cell_filter        = openmc.CellFromFilter([icrh_duct_cell])

    icrh_duct_end_current = openmc.Tally(name="icrh_duct_end_current")
    icrh_duct_end_current.scores  = ["current"]
    icrh_duct_end_current.filters = [
        neutron_particle_filter, icrh_back_wall_filter, icrh_cell_filter
    ]
    tallies_list.append(icrh_duct_end_current)

    # ------------------------------------------------------------------
    # Export XML
    # ------------------------------------------------------------------
    model = openmc.model.Model(
        geometry=geometry,
        materials=materials,
        settings=settings,
        tallies=tallies_list,
    )
    model.export_to_xml()
    log.info("Geometry XML exported successfully to %s", cwd)
    print(f"Geometry XML exported successfully to {cwd}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    build_model()
