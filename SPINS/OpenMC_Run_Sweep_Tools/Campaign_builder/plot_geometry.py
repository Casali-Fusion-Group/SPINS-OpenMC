"""
plot_geometry.py
================
Plots the OpenMC geometry for each unique (radius, triang) combination
in the campaign, using OpenMC's built-in model.plot() in the RZ plane.

For each unique (radius, triang) parent folder:
  1. Finds one representative run folder inside it
  2. Reads run_config.yaml and calls geometry_builder.py to export XML
     (if XML files are not already present)
  3. Loads the model and generates an RZ-plane plot
  4. Overlays text labels for the inboard and outboard cells
  5. Saves to base_dir/plots_geometry/geometry_R{r}_T{t}.png

Usage
-----
    python plot_geometry.py --config campaign_config.yaml
    python plot_geometry.py --config campaign_config.yaml --width 1200 --height 1200
    python plot_geometry.py --config campaign_config.yaml --rebuild-xml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load campaign_config.yaml.

    Args:
        config_path: Path to campaign_config.yaml.

    Returns:
        Parsed config dict.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Campaign walking
# ---------------------------------------------------------------------------

def find_unique_parent_folders(base_dir: Path, parent_params: list[str]) -> list[Path]:
    """Find all unique parent folders created by the parent_folder_params hierarchy.

    For default config this means all base_dir/radius_*/triang_*/ directories
    that contain at least one run_config.yaml somewhere inside them.

    Args:
        base_dir:      Campaign base directory.
        parent_params: Ordered list of parent folder parameter names
                       (e.g. ["radius", "triang"]).

    Returns:
        Sorted list of unique parent folder Paths.
    """
    depth = len(parent_params)
    parent_folders: set[Path] = set()

    for run_cfg_path in base_dir.rglob("run_config.yaml"):
        # Walk up 'depth' levels from the run folder
        candidate = run_cfg_path.parent
        for _ in range(depth):
            candidate = candidate.parent
        # Verify it sits directly under base_dir at the right depth
        try:
            rel = candidate.relative_to(base_dir)
            if len(rel.parts) == depth:
                parent_folders.add(candidate)
        except ValueError:
            pass

    return sorted(parent_folders)


def pick_representative_run(parent_folder: Path) -> Path | None:
    """Return the first run folder found inside a parent folder.

    Prefers a run that already has geometry XML files present.

    Args:
        parent_folder: The radius_*/triang_*/ parent directory.

    Returns:
        Path to a run folder (containing run_config.yaml), or None.
    """
    # Prefer a run that already has geometry.xml
    for run_cfg in sorted(parent_folder.rglob("run_config.yaml")):
        run_path = run_cfg.parent
        if (run_path / "geometry.xml").exists():
            log.info("Using run with existing XML: %s", run_path)
            return run_path

    # Fall back to any run folder
    for run_cfg in sorted(parent_folder.rglob("run_config.yaml")):
        return run_cfg.parent

    return None


def parse_parent_params(parent_folder: Path, base_dir: Path) -> dict[str, Any]:
    """Extract parameter values from a parent folder path.

    E.g. base_dir/radius_455/triang_-0.5 → {"radius": 455, "triang": -0.5}

    Args:
        parent_folder: The parent folder path.
        base_dir:      Campaign base directory.

    Returns:
        Dict of {param_name: value} parsed from folder name segments.
    """
    rel   = parent_folder.relative_to(base_dir)
    params: dict[str, Any] = {}
    for part in rel.parts:
        if "_" in part:
            idx = part.index("_")
            key = part[:idx]
            val_str = part[idx+1:]
            try:
                val: Any = int(val_str)
            except ValueError:
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
            params[key] = val
    return params


# ---------------------------------------------------------------------------
# XML building
# ---------------------------------------------------------------------------

def ensure_xml(run_path: Path, rebuild: bool = False) -> bool:
    """Ensure geometry.xml, materials.xml, settings.xml exist in run_path.

    Calls geometry_builder.py if XML files are missing or --rebuild-xml set.

    Args:
        run_path: Path to the run folder.
        rebuild:  If True, always regenerate XML even if it exists.

    Returns:
        True if XML files are present after this call, False on failure.
    """
    xml_files = ["geometry.xml", "materials.xml", "settings.xml", "tallies.xml"]
    all_present = all((run_path / f).exists() for f in xml_files)

    if all_present and not rebuild:
        log.info("XML files already present in %s", run_path)
        return True

    builder = run_path / "geometry_builder.py"
    if not builder.exists():
        log.warning("geometry_builder.py not found in %s — cannot build XML", run_path)
        return False

    log.info("Running geometry_builder.py in %s ...", run_path)
    result = subprocess.run(
        [sys.executable, str(builder)],
        cwd=str(run_path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error(
            "geometry_builder.py failed in %s:\nSTDOUT: %s\nSTDERR: %s",
            run_path, result.stdout, result.stderr,
        )
        return False

    log.info("geometry_builder.py succeeded in %s", run_path)
    return all((run_path / f).exists() for f in xml_files)


# ---------------------------------------------------------------------------
# Cell colour map
# ---------------------------------------------------------------------------

# Maps cell name → (fill_color_hex, alpha)
CELL_COLORS: dict[str, tuple[str, float]] = {
    "plasma":           ("#FFD700", 0.6),   # gold
    "inner_first_wall": ("#1f77b4", 0.85),  # blue  — inboard
    "outer_first_wall": ("#d62728", 0.85),  # red   — outboard
    "inner_breeder":    ("#aec7e8", 0.75),  # light blue — inboard
    "outer_breeder":    ("#ffbb78", 0.75),  # light orange — outboard
    "upper_div":        ("#2ca02c", 0.75),  # green
    "lower_div":        ("#2ca02c", 0.75),  # green
    "icrh_duct":        ("#9467bd", 0.6),   # purple
    "bounds":           ("#ffffff", 0.0),   # transparent
}

# RGB tuples (0-255) for OpenMC's color dict
def _hex_to_rgb255(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Geometry plot
# ---------------------------------------------------------------------------

def plot_geometry_rz(
    run_path: Path,
    parent_params: dict[str, Any],
    cfg: dict[str, Any],
    output_dir: Path,
    plot_width_px: int,
    plot_height_px: int,
    dpi: int,
) -> Path | None:
    """Generate an RZ-plane OpenMC geometry plot for one representative run.

    Loads the model from XML in run_path, calls model.plot() with a colour
    map that highlights inboard vs outboard cells, then adds a matplotlib
    legend and saves the final annotated figure.

    Args:
        run_path:       Run folder containing geometry XML files.
        parent_params:  Dict of parent parameter values (radius, triang).
        cfg:            Campaign config dict.
        output_dir:     Directory to save output plots.
        plot_width_px:  Width of the OpenMC plot in pixels.
        plot_height_px: Height of the OpenMC plot in pixels.
        dpi:            Output figure DPI.

    Returns:
        Path to the saved PNG, or None on failure.
    """
    import openmc

    radius = parent_params.get("radius", "?")
    triang = parent_params.get("triang", "?")
    label  = make_run_label(parent_params)

    # Change into run directory so OpenMC finds the XML files
    orig_cwd = Path.cwd()
    os.chdir(run_path)

    try:
        # Load model
        geometry  = openmc.Geometry.from_xml("geometry.xml")
        materials = openmc.Materials.from_xml("materials.xml")
        settings  = openmc.Settings.from_xml("settings.xml")
        tallies   = openmc.Tallies.from_xml("tallies.xml")
        model     = openmc.model.Model(
            geometry=geometry,
            materials=materials,
            settings=settings,
            tallies=tallies,
        )

        # Build OpenMC color map from CELL_COLORS
        all_cells = geometry.get_all_cells()
        color_by_cell: dict[openmc.Cell, tuple[int, int, int]] = {}
        for cell in all_cells.values():
            if cell.name in CELL_COLORS:
                hex_c, _ = CELL_COLORS[cell.name]
                color_by_cell[cell] = _hex_to_rgb255(hex_c)

        # Determine plot bounds from the run_config radius
        plot_extent = float(radius) * 2.2 if isinstance(radius, (int, float)) else 1200.0
        half = plot_extent / 2.0

        # Determine y extent to ensure visualization of icrh duct
        length_substring = 18
        ind = str(run_path).find("center_shift_icrh_")
        # print(f"RUN PATH: {str(run_path)}")
        # print(f"CENTER SHIFT: {str(run_path)[ind+length_substring:ind+length_substring+2]}")
        center_shift_icrh = int(str(run_path)[ind+length_substring:ind+length_substring+2])
        print(center_shift_icrh)
        icrh_x_location = center_shift_icrh
        icrh_y_location = np.sqrt(radius**2 - center_shift_icrh**2)

        # RZ plot: basis='xz', origin at (half, 0, 0) to show full torus cross-section
        # We use basis='xz' so x→R and z→Z (standard poloidal cross-section view)
        openmc_plot = openmc.Plot()
        openmc_plot.basis       = "xz"
        openmc_plot.origin      = (0, icrh_y_location + 1, 0.0)
        openmc_plot.width       = (plot_extent, plot_extent)
        openmc_plot.pixels      = (plot_width_px, plot_height_px)
        openmc_plot.color_by    = "cell"
        openmc_plot.colors      = color_by_cell
        openmc_plot.filename    = f"geometry_{label}"

        plots = openmc.Plots([openmc_plot])
        plots.export_to_xml()

        # Run OpenMC in plot mode
        openmc.plot_geometry()

        # The PNG is written to run_path/geometry_{label}.png
        raw_png = run_path / f"geometry_{label}.png"
        if not raw_png.exists():
            # OpenMC sometimes appends .png automatically
            candidates = list(run_path.glob(f"geometry_{label}*.png"))
            if candidates:
                raw_png = candidates[0]
            else:
                log.error("OpenMC did not produce a PNG for %s", label)
                return None

        # ----------------------------------------------------------------
        # Annotate with matplotlib: load the PNG, add legend + labels
        # ----------------------------------------------------------------
        raw_img = plt.imread(str(raw_png))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(
            raw_img,
            extent=[-half, half, -half, half],
            origin="upper",
        )

        # Label inboard and outboard regions with arrows
        ib_cells = cfg["tallies"]["inboard_cells"]
        ob_cells = cfg["tallies"]["outboard_cells"]

        # Approximate label positions in R-Z space
        # Inboard: small R, Z=0
        # Outboard: large R, Z=0
        inboard_r  = float(radius) * 0.3 if isinstance(radius, (int, float)) else half * 0.3
        outboard_r = float(radius) * 0.85 if isinstance(radius, (int, float)) else half * 0.85

        # ax.annotate(
        #     "INBOARD\n(" + ", ".join(ib_cells) + ")",
        #     xy=(inboard_r - half, 0),
        #     xytext=(inboard_r - half - half * 0.25, half * 0.35),
        #     fontsize=9, color="white", fontweight="bold",
        #     ha="center",
        #     arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
        #     bbox=dict(boxstyle="round,pad=0.3", facecolor="#1f77b4", alpha=0.7),
        # )
        # ax.annotate(
        #     "OUTBOARD\n(" + ", ".join(ob_cells) + ")",
        #     xy=(outboard_r - half, 0),
        #     xytext=(outboard_r - half + half * 0.2, half * 0.35),
        #     fontsize=9, color="white", fontweight="bold",
        #     ha="center",
        #     arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
        #     bbox=dict(boxstyle="round,pad=0.3", facecolor="#d62728", alpha=0.7),
        # )

        # Legend
        legend_handles = []
        for cell_name, (hex_c, alpha) in CELL_COLORS.items():
            if cell_name in ("bounds",):
                continue
            label_str = cell_name.replace("_", " ").title()
            if cell_name in ib_cells:
                label_str += " [IB]"
            elif cell_name in ob_cells:
                label_str += " [OB]"
            legend_handles.append(
                mpatches.Patch(
                    facecolor=hex_c, alpha=alpha,
                    edgecolor="black", linewidth=0.5,
                    label=label_str,
                )
            )
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            fontsize=8,
            framealpha=0.85,
            title="Cells",
            title_fontsize=9,
        )

        ax.set_xlabel("R (cm)", fontsize=11)
        ax.set_ylabel("Z (cm)", fontsize=11)
        ax.set_title(
            f"Geometry — RZ Plane\n"
            f"radius={radius} cm, triang={triang}",
            fontsize=13,
        )

        out_png = output_dir / f"geometry_{label}.png"
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", out_png)
        print(f"Saved {out_png}")
        return out_png

    except Exception as exc:
        log.error("Failed to plot geometry for %s: %s", label, exc, exc_info=True)
        return None

    finally:
        os.chdir(orig_cwd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_all_run_folders(base_dir: Path) -> list[Path]:
    """Find every run folder in the campaign (any folder containing run_config.yaml).

    Args:
        base_dir: Campaign base directory.

    Returns:
        Sorted list of run folder Paths.
    """
    return sorted(p.parent for p in base_dir.rglob("run_config.yaml"))


def parse_run_params(run_path: Path) -> dict[str, Any]:
    """Read all parameters from run_config.yaml in a run folder.

    Args:
        run_path: Path to the run folder.

    Returns:
        Parameters dict from run_config.yaml, or empty dict on failure.
    """
    cfg_path = run_path / "run_config.yaml"
    try:
        with open(cfg_path) as f:
            run_cfg = yaml.safe_load(f)
        return run_cfg.get("parameters", {})
    except Exception as exc:
        log.warning("Could not read run_config.yaml in %s: %s", run_path, exc)
        return {}


def make_run_label(params: dict[str, Any]) -> str:
    """Build a short filename-safe label from all run parameters.

    Args:
        params: Full parameters dict for a run.

    Returns:
        String like "R455_T-0.5_cshift0_w50_MANTA_default"
    """
    return (
        f"R{params.get('radius', '?')}"
        f"_T{params.get('triang', '?')}"
        f"_cshift{params.get('center_shift_icrh', '?')}"
        f"_w{params.get('width_icrh', '?')}"
        f"_{params.get('source_device', '?')}"
        f"_{params.get('source_scenario', '?')}"
    )


def main() -> None:
    """Entry point for plot_geometry.py."""
    parser = argparse.ArgumentParser(
        description="Plot OpenMC RZ geometry for every run in the campaign."
    )
    parser.add_argument("--config", default="campaign_config.yaml")
    parser.add_argument("--width",  type=int, default=1000,
                        help="OpenMC plot width in pixels (default: 1000)")
    parser.add_argument("--height", type=int, default=1000,
                        help="OpenMC plot height in pixels (default: 1000)")
    parser.add_argument("--rebuild-xml", action="store_true",
                        help="Re-run geometry_builder.py even if XML files exist")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s [%(levelname)s] %(message)s")
    elif args.verbose:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING,
                            format="%(asctime)s [%(levelname)s] %(message)s")

    config_path = Path(args.config).expanduser().resolve()
    cfg         = load_config(config_path)
    base_dir    = Path(cfg["paths"]["base_dir"]).expanduser().resolve()
    dpi         = cfg["plot"].get("dpi", 150)

    output_dir = base_dir / "plots_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_folders = find_all_run_folders(base_dir)
    print(f"\nFound {len(run_folders)} run folders.")
    print(f"Output directory: {output_dir}\n")

    saved: list[Path] = []

    for i, run_path in enumerate(run_folders, 1):
        params = parse_run_params(run_path)
        if not params:
            print(f"  [{i}/{len(run_folders)}] SKIP (no params): {run_path.name}")
            continue

        label = make_run_label(params)
        print(f"  [{i}/{len(run_folders)}] {label}")

        ok = ensure_xml(run_path, rebuild=args.rebuild_xml)
        if not ok:
            print(f"    WARNING: Could not build XML — skipping")
            continue

        result = plot_geometry_rz(
            run_path       = run_path,
            parent_params  = params,
            cfg            = cfg,
            output_dir     = output_dir,
            plot_width_px  = args.width,
            plot_height_px = args.height,
            dpi            = dpi,
        )
        if result:
            saved.append(result)

    print(f"\nDone. {len(saved)}/{len(run_folders)} geometry plots saved to {output_dir}")


if __name__ == "__main__":
    main()