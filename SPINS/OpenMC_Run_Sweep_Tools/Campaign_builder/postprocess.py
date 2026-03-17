"""
postprocess.py
==============
Post-processing script for the OpenMC parametric campaign.
Reads statepoint results across all completed run folders and generates
comparison plots in three analysis modes.

Usage
-----
    # Extract results from all statepoints and build campaign_results.json:
    python postprocess.py --config campaign_config.yaml --mode sensitivity \\
                          --param center_shift_icrh

    python postprocess.py --config campaign_config.yaml --mode compare \\
                          --runs run_id_1 run_id_2 run_id_3

    python postprocess.py --config campaign_config.yaml --mode heatmap \\
                          --x-param center_shift_icrh --y-param width_icrh \\
                          --cell outer_first_wall --score flux

Modes
-----
sensitivity  Show how one parameter drives outputs while others are fixed.
compare      Side-by-side comparison of explicitly chosen run combinations.
heatmap      2D parameter-interaction heatmap for a chosen output variable.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml

log = logging.getLogger(__name__)

try:
    import openmc
    OPENMC_AVAILABLE = True
except ImportError:
    OPENMC_AVAILABLE = False
    log.warning("openmc not importable — statepoint extraction disabled.")


# =============================================================================
# Config / path helpers
# =============================================================================

def load_config(config_path: Path) -> dict[str, Any]:
    """Load the campaign YAML config.

    Args:
        config_path: Path to campaign_config.yaml.

    Returns:
        Parsed config dict.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_base_dir(cfg: dict[str, Any]) -> Path:
    """Resolve the base_dir from the campaign config.

    Args:
        cfg: Campaign config dict.

    Returns:
        Absolute resolved Path.
    """
    return Path(cfg["paths"]["base_dir"]).expanduser().resolve()


def apply_style(cfg: dict[str, Any]) -> None:
    """Apply matplotlib style from plot config.

    Args:
        cfg: Campaign config dict.
    """
    style = cfg["plot"].get("style", "seaborn-v0_8-whitegrid")
    try:
        plt.style.use(style)
    except OSError:
        log.warning("Matplotlib style '%s' not available, using default.", style)
    plt.rcParams.update({"font.size": cfg["plot"].get("font_size", 11)})


def make_plot_dir(base_dir: Path, mode: str) -> Path:
    """Create and return a timestamped plot output directory.

    Args:
        base_dir: Campaign base directory.
        mode:     Analysis mode name.

    Returns:
        Created directory path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = base_dir / f"plots_{mode}_{ts}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    log.info("Plot output directory: %s", plot_dir)
    return plot_dir


def save_fig(fig: plt.Figure, plot_dir: Path, stem: str, dpi: int) -> None:
    """Save a matplotlib figure as both PNG.

    Args:
        fig:      Figure to save.
        plot_dir: Output directory.
        stem:     Filename stem (no extension).
        dpi:      Resolution in dots per inch.
    """
    for ext in ("png",):
        path = plot_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        log.info("Saved %s", path)
    plt.close(fig)


# =============================================================================
# Statepoint extraction
# =============================================================================

def find_latest_statepoint(run_path: Path) -> Path | None:
    """Find the statepoint file with the highest batch number in run_path.

    Args:
        run_path: Run folder path.

    Returns:
        Path to the latest statepoint .h5 file, or None if not found.
    """
    sps = sorted(run_path.glob("statepoint.*.h5"))
    return sps[-1] if sps else None


def extract_run_results(
    run_path: Path,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract tally results from a run's statepoint file.

    Args:
        run_path: Path to the run folder.
        cfg:      Campaign config dict (provides tally metadata).

    Returns:
        Results dict, or None if extraction failed.
    """
    if not OPENMC_AVAILABLE:
        log.error("openmc is not available — cannot extract statepoint results.")
        return None

    run_config_path = run_path / "run_config.yaml"
    if not run_config_path.exists():
        log.warning("No run_config.yaml in %s — skipping.", run_path)
        return None

    with open(run_config_path) as f:
        run_cfg = yaml.safe_load(f)

    sp_path = find_latest_statepoint(run_path)
    if sp_path is None:
        log.warning("No statepoint file found in %s — skipping.", run_path)
        return None

    tally_cfg      = run_cfg.get("tallies", cfg["tallies"])
    tally_cells    = tally_cfg["tally_cells"]
    scores         = tally_cfg["scores"]
    energy_bins_eV = (np.array(tally_cfg["energy_groups_MeV"]) * 1e6).tolist()
    group_labels   = tally_cfg["energy_group_labels"]

    try:
        sp = openmc.StatePoint(str(sp_path))
    except Exception as exc:
        log.warning("Failed to open statepoint %s: %s", sp_path, exc)
        return None

    cells_data: dict[str, Any] = {}

    for cell_name in tally_cells:
        tally_name = f"{cell_name}_tally"
        try:
            tally = sp.get_tally(name=tally_name)
        except Exception:
            log.warning("Tally '%s' not found in %s — skipping cell.", tally_name, sp_path)
            continue

        cells_data[cell_name] = {}
        for score in scores:
            try:
                df = tally.get_pandas_dataframe(scores=[score])
                mean   = df["mean"].values.tolist()
                std_dev = df["std. dev."].values.tolist()
            except Exception as exc:
                log.warning("Failed to extract score '%s' for tally '%s': %s",
                            score, tally_name, exc)
                mean = std_dev = [0.0] * (len(energy_bins_eV) - 1)
            cells_data[cell_name][score] = {"mean": mean, "std_dev": std_dev}

    # icrh_duct_end_current — scalar net current
    icrh_current: dict[str, float] = {"mean": 0.0, "std_dev": 0.0}
    try:
        icrh_tally = sp.get_tally(name="icrh_duct_end_current")
        df = icrh_tally.get_pandas_dataframe(scores=["current"])
        icrh_current = {
            "mean":    float(df["mean"].sum()),
            "std_dev": float(np.sqrt((df["std. dev."] ** 2).sum())),
        }
    except Exception as exc:
        log.warning("icrh_duct_end_current tally not found or failed: %s", exc)

    result = {
        "run_id":                run_cfg.get("run_id", run_path.name),
        "parameters":            run_cfg["parameters"],
        "energy_bins_eV":        energy_bins_eV,
        "energy_group_labels":   group_labels,
        "cells":                 cells_data,
        "icrh_duct_end_current": icrh_current,
    }

    # Save per-run results.json
    results_path = run_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved results.json to %s", results_path)

    return result


def collect_all_results(
    base_dir: Path,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Walk base_dir and extract results from all completed run folders.

    Args:
        base_dir: Campaign base directory.
        cfg:      Campaign config dict.

    Returns:
        List of result dicts for all successfully processed runs.
    """
    all_results: list[dict[str, Any]] = []

    for run_config_path in sorted(base_dir.rglob("run_config.yaml")):
        run_path = run_config_path.parent
        sp = find_latest_statepoint(run_path)
        if sp is None:
            log.debug("No statepoint in %s — skipping.", run_path)
            continue

        # Try loading cached results.json first
        cached = run_path / "results.json"
        if cached.exists():
            try:
                with open(cached) as f:
                    result = json.load(f)
                log.debug("Loaded cached results.json from %s", run_path)
                all_results.append(result)
                continue
            except Exception:
                pass  # Fall through to re-extract

        result = extract_run_results(run_path, cfg)
        if result is not None:
            all_results.append(result)

    log.info("Collected results from %d completed runs.", len(all_results))

    # Write aggregated campaign_results.json
    agg_path = base_dir / "campaign_results.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Wrote aggregated results to %s", agg_path)

    return all_results


# =============================================================================
# Result query helpers
# =============================================================================

def get_total_flux(result: dict, cell_name: str) -> tuple[float, float]:
    """Return total integrated flux (mean, std_dev) for a cell.

    Args:
        result:    Run result dict.
        cell_name: Cell name key.

    Returns:
        (mean, std_dev) summed over all energy groups.
    """
    cell = result["cells"].get(cell_name, {})
    flux = cell.get("flux", {"mean": [], "std_dev": []})
    mean    = sum(flux["mean"])
    std_dev = float(np.sqrt(sum(s**2 for s in flux["std_dev"])))
    return mean, std_dev


def get_group_flux(result: dict, cell_name: str) -> tuple[list, list]:
    """Return per-group flux arrays (means, std_devs) for a cell.

    Args:
        result:    Run result dict.
        cell_name: Cell name key.

    Returns:
        (means_list, std_devs_list) each of length n_energy_groups.
    """
    cell = result["cells"].get(cell_name, {})
    flux = cell.get("flux", {"mean": [], "std_dev": []})
    return flux["mean"], flux["std_dev"]


def get_score_total(result: dict, cell_name: str, score: str) -> tuple[float, float]:
    """Return total score value (mean, std_dev) integrated over all groups.

    Args:
        result:    Run result dict.
        cell_name: Cell name key.
        score:     Score name (e.g. "damage-energy", "heating").

    Returns:
        (mean, std_dev) summed over all energy groups.
    """
    cell = result["cells"].get(cell_name, {})
    sc   = cell.get(score, {"mean": [], "std_dev": []})
    mean    = sum(sc["mean"])
    std_dev = float(np.sqrt(sum(s**2 for s in sc["std_dev"])))
    return mean, std_dev


def most_common_value(results: list[dict], param: str) -> Any:
    """Return the most frequently occurring value of a parameter.

    Args:
        results: List of run result dicts.
        param:   Parameter name.

    Returns:
        Most common value across all results.
    """
    vals = [r["parameters"].get(param) for r in results if param in r["parameters"]]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def filter_results(
    results: list[dict],
    fixed: dict[str, Any],
    free_param: str | None = None,
) -> list[dict]:
    """Filter results to those matching all fixed parameter values.

    Args:
        results:    List of run result dicts.
        fixed:      Dict of {param: value} to match exactly.
        free_param: If given, this parameter is NOT filtered (it is the sweep axis).

    Returns:
        Filtered list.
    """
    out = []
    for r in results:
        p = r["parameters"]
        match = all(
            k == free_param or str(p.get(k)) == str(v)
            for k, v in fixed.items()
        )
        if match:
            out.append(r)
    return out


def group_by_param(
    results: list[dict],
    param: str,
) -> dict[Any, list[dict]]:
    """Group results by their value of a single parameter.

    Args:
        results: List of run result dicts.
        param:   Parameter name to group by.

    Returns:
        Dict mapping param_value -> list of matching run result dicts.
    """
    groups: dict[Any, list[dict]] = {}
    for r in results:
        val = r["parameters"].get(param)
        groups.setdefault(val, []).append(r)
    return groups


def group_labels_for(results: list[dict]) -> list[str]:
    """Extract energy group labels from the first result.

    Args:
        results: List of run result dicts.

    Returns:
        Energy group label strings.
    """
    if results:
        return results[0].get("energy_group_labels", [])
    return []


def energy_groups_for_flag(flag: str, n_groups: int) -> list[int]:
    """Return group indices corresponding to a --groups flag value.

    Args:
        flag:     One of all|thermal|epithermal|fast|highfast.
        n_groups: Total number of energy groups.

    Returns:
        List of 0-based group indices.
    """
    mapping = {
        "all":        list(range(n_groups)),
        "thermal":    [0],
        "epithermal": [1],
        "fast":       [2],
        "highfast":   [3],
    }
    return mapping.get(flag, list(range(n_groups)))


# =============================================================================
# MODE 1: sensitivity
# =============================================================================

def mode_sensitivity(
    results: list[dict],
    cfg: dict[str, Any],
    args: argparse.Namespace,
    plot_dir: Path,
) -> None:
    """Generate sensitivity plots varying one parameter.

    Fixes all other parameters at their most common value (or values
    specified via --fix KEY=VALUE). Plots total flux, per-group flux,
    total damage-energy, and ICRH duct current vs the swept parameter.

    Args:
        results:  All collected run result dicts.
        cfg:      Campaign config dict.
        args:     Parsed CLI arguments.
        plot_dir: Directory for output figures.
    """
    param      = args.param
    score      = getattr(args, "score", "flux")
    groups_flag = getattr(args, "groups", "all")
    side       = getattr(args, "side", "both")
    dpi        = cfg["plot"]["dpi"]
    figsize    = cfg["plot"]["figsize"]
    ib_color   = cfg["plot"]["inboard_color"]
    ob_color   = cfg["plot"]["outboard_color"]
    ib_cells   = cfg["tallies"]["inboard_cells"]
    ob_cells   = cfg["tallies"]["outboard_cells"]

    all_params = list(cfg["parameters"].keys())

    # Build fixed-parameter dict
    fixed: dict[str, Any] = {}
    for fix_str in (getattr(args, "fix", None) or []):
        k, v = fix_str.split("=", 1)
        fixed[k.strip()] = v.strip()
    for p in all_params:
        if p != param and p not in fixed:
            fixed[p] = most_common_value(results, p)

    log.info("Sensitivity mode: sweeping '%s', fixed params: %s", param, fixed)

    # Filter and group
    relevant = filter_results(results, fixed, free_param=param)
    if not relevant:
        log.warning("No results match the fixed parameter constraints for sensitivity.")
        return

    grouped   = group_by_param(relevant, param)
    param_vals = sorted(grouped.keys(), key=lambda v: (str(type(v)), v))
    x          = np.arange(len(param_vals))
    group_lbls = group_labels_for(relevant)
    n_groups   = len(group_lbls)
    grp_indices = energy_groups_for_flag(groups_flag, n_groups)

    # ----------------------------------------------------------------
    # Chart A: Total integrated flux — inboard vs outboard
    # ----------------------------------------------------------------
    def _total_flux_for_cellset(cell_set: list[str], run: dict) -> tuple[float, float]:
        means = [get_total_flux(run, c)[0] for c in cell_set]
        stds  = [get_total_flux(run, c)[1] for c in cell_set]
        return sum(means), float(np.sqrt(sum(s**2 for s in stds)))

    ib_means, ib_stds, ob_means, ob_stds = [], [], [], []
    for val in param_vals:
        runs = grouped[val]
        ib_m = np.mean([_total_flux_for_cellset(ib_cells, r)[0] for r in runs])
        ib_s = np.mean([_total_flux_for_cellset(ib_cells, r)[1] for r in runs])
        ob_m = np.mean([_total_flux_for_cellset(ob_cells, r)[0] for r in runs])
        ob_s = np.mean([_total_flux_for_cellset(ob_cells, r)[1] for r in runs])
        ib_means.append(ib_m); ib_stds.append(ib_s)
        ob_means.append(ob_m); ob_stds.append(ob_s)

    fig, ax = plt.subplots(figsize=figsize)
    w = 0.35
    if side in ("inboard", "both"):
        ax.bar(x - w/2, ib_means, w, yerr=ib_stds, capsize=4,
               color=ib_color, alpha=0.9, label="Inboard (sum)")
    if side in ("outboard", "both"):
        ax.bar(x + w/2, ob_means, w, yerr=ob_stds, capsize=4,
               color=ob_color, alpha=0.9, label="Outboard (sum)")
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in param_vals])
    ax.set_xlabel(param); ax.set_ylabel("Total Integrated Flux")
    ax.set_title(f"Sensitivity: Total Flux vs {param}")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    save_fig(fig, plot_dir, f"sensitivity_A_total_flux_vs_{param}", dpi)

    # ----------------------------------------------------------------
    # Chart B: Per-group flux for each tally cell
    # ----------------------------------------------------------------
    cmap = plt.get_cmap(cfg["plot"]["colormap"])
    for cell_name in cfg["tallies"]["tally_cells"]:
        fig, ax = plt.subplots(figsize=figsize)
        selected_lbls = [group_lbls[i] for i in grp_indices]
        x_grp = np.arange(len(grp_indices))
        bar_w = 0.8 / max(len(param_vals), 1)

        for p_idx, val in enumerate(param_vals):
            runs = grouped[val]
            grp_means, grp_stds = [], []
            for g_idx in grp_indices:
                all_m = [get_group_flux(r, cell_name)[0][g_idx]
                         for r in runs if len(get_group_flux(r, cell_name)[0]) > g_idx]
                all_s = [get_group_flux(r, cell_name)[1][g_idx]
                         for r in runs if len(get_group_flux(r, cell_name)[1]) > g_idx]
                grp_means.append(np.mean(all_m) if all_m else 0.0)
                grp_stds.append(np.mean(all_s)  if all_s else 0.0)

            offset = (p_idx - len(param_vals)/2 + 0.5) * bar_w
            ax.bar(x_grp + offset, grp_means, bar_w, yerr=grp_stds, capsize=3,
                   color=cmap(p_idx / max(len(param_vals)-1, 1)),
                   alpha=0.85, label=f"{param}={val}")

        ax.set_xticks(x_grp); ax.set_xticklabels(selected_lbls, rotation=20, ha="right")
        ax.set_xlabel("Energy Group"); ax.set_ylabel("Flux")
        ax.set_title(f"Sensitivity: Flux Spectrum ({cell_name}) vs {param}")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.tight_layout()
        safe_cell = cell_name.replace(" ", "_")
        save_fig(fig, plot_dir, f"sensitivity_B_spectrum_{safe_cell}_vs_{param}", dpi)

    # ----------------------------------------------------------------
    # Chart C: Total damage-energy (NWL proxy) per cell
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    for c_idx, cell_name in enumerate(cfg["tallies"]["tally_cells"]):
        c_means, c_stds = [], []
        for val in param_vals:
            runs = grouped[val]
            ms = [get_score_total(r, cell_name, "damage-energy")[0] for r in runs]
            ss = [get_score_total(r, cell_name, "damage-energy")[1] for r in runs]
            c_means.append(np.mean(ms)); c_stds.append(np.mean(ss))
        offset = (c_idx - len(cfg["tallies"]["tally_cells"])/2 + 0.5) * (0.8/len(cfg["tallies"]["tally_cells"]))
        ax.bar(x + offset, c_means,
               0.8/len(cfg["tallies"]["tally_cells"]),
               yerr=c_stds, capsize=3,
               color=cmap(c_idx / max(len(cfg["tallies"]["tally_cells"])-1, 1)),
               alpha=0.85, label=cell_name)
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in param_vals])
    ax.set_xlabel(param); ax.set_ylabel("Total Damage-Energy (NWL proxy)")
    ax.set_title(f"Sensitivity: Damage-Energy vs {param}")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    save_fig(fig, plot_dir, f"sensitivity_C_damage_energy_vs_{param}", dpi)

    # ----------------------------------------------------------------
    # Chart D: ICRH duct end current vs swept parameter
    # ----------------------------------------------------------------
    icrh_means, icrh_stds = [], []
    for val in param_vals:
        runs = grouped[val]
        ms = [r["icrh_duct_end_current"]["mean"]    for r in runs]
        ss = [r["icrh_duct_end_current"]["std_dev"] for r in runs]
        icrh_means.append(np.mean(ms)); icrh_stds.append(np.mean(ss))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, icrh_means, 0.5, yerr=icrh_stds, capsize=4,
           color=cfg["plot"]["inboard_color"], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in param_vals])
    ax.set_xlabel(param); ax.set_ylabel("ICRH Duct End Current (n/s)")
    ax.set_title(f"Sensitivity: ICRH Duct End Current vs {param}")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    save_fig(fig, plot_dir, f"sensitivity_D_icrh_current_vs_{param}", dpi)

    log.info("Sensitivity mode complete. %d charts saved to %s", 4, plot_dir)


# =============================================================================
# MODE 2: compare
# =============================================================================

def mode_compare(
    results: list[dict],
    cfg: dict[str, Any],
    args: argparse.Namespace,
    plot_dir: Path,
) -> None:
    """Generate side-by-side comparison plots for explicitly chosen runs.

    Args:
        results:  All collected run result dicts.
        cfg:      Campaign config dict.
        args:     Parsed CLI arguments (must include args.runs).
        plot_dir: Directory for output figures.
    """
    run_ids    = args.runs
    cells_arg  = getattr(args, "cells", None) or cfg["tallies"]["tally_cells"]
    dpi        = cfg["plot"]["dpi"]
    figsize    = cfg["plot"]["figsize"]

    results_by_id = {r["run_id"]: r for r in results}
    selected: list[dict] = []
    for rid in run_ids:
        if rid not in results_by_id:
            log.warning("run_id '%s' not found in results — skipping.", rid)
        else:
            selected.append(results_by_id[rid])

    if not selected:
        log.error("No valid run_ids found. Aborting compare mode.")
        return

    # Find which parameters differ across selected runs
    all_params = list(cfg["parameters"].keys())
    differing  = [
        p for p in all_params
        if len(set(str(r["parameters"].get(p)) for r in selected)) > 1
    ]
    short_labels = [
        "\n".join(f"{p}={r['parameters'].get(p)}" for p in differing)
        or r["run_id"]
        for r in selected
    ]
    cmap = plt.get_cmap(cfg["plot"]["colormap"])
    colors = [cmap(i / max(len(selected)-1, 1)) for i in range(len(selected))]
    x = np.arange(len(selected))

    # ----------------------------------------------------------------
    # Chart A: Total flux per run, grouped by cell
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    n_cells = len(cells_arg)
    bar_w   = 0.8 / max(n_cells, 1)
    for c_idx, cell_name in enumerate(cells_arg):
        means = [get_total_flux(r, cell_name)[0] for r in selected]
        stds  = [get_total_flux(r, cell_name)[1] for r in selected]
        offset = (c_idx - n_cells/2 + 0.5) * bar_w
        ax.bar(x + offset, means, bar_w, yerr=stds, capsize=3,
               color=cmap(c_idx / max(n_cells-1, 1)),
               alpha=0.85, label=cell_name)
    ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("Total Integrated Flux"); ax.set_title("Compare: Total Flux per Run")
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    save_fig(fig, plot_dir, "compare_A_total_flux", dpi)

    # ----------------------------------------------------------------
    # Chart B: Flux spectrum per cell (step-function overlay)
    # ----------------------------------------------------------------
    group_lbls = group_labels_for(selected)
    for cell_name in cells_arg:
        fig, ax = plt.subplots(figsize=figsize)
        bins_eV = np.array(selected[0]["energy_bins_eV"]) if selected else np.array([])
        for r_idx, r in enumerate(selected):
            means, stds = get_group_flux(r, cell_name)
            if not means:
                continue
            m = np.array(means); s = np.array(stds)
            if len(bins_eV) == len(m) + 1:
                xs = np.repeat(bins_eV, 2)[1:-1]
                ys = np.repeat(m, 2)
                yl = np.repeat(np.maximum(m - s, 0), 2)
                yh = np.repeat(m + s, 2)
            else:
                xs = np.arange(len(m)); ys = m; yl = m-s; yh = m+s
            ax.plot(xs, ys, color=colors[r_idx], linewidth=2,
                    label=short_labels[r_idx])
            ax.fill_between(xs, yl, yh, color=colors[r_idx], alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel("Energy (eV)"); ax.set_ylabel("Flux")
        ax.set_title(f"Compare: Flux Spectrum — {cell_name}")
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.tight_layout()
        safe_cell = cell_name.replace(" ", "_")
        save_fig(fig, plot_dir, f"compare_B_spectrum_{safe_cell}", dpi)

    # ----------------------------------------------------------------
    # Chart C: Damage-energy per run per cell
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    for c_idx, cell_name in enumerate(cells_arg):
        means = [get_score_total(r, cell_name, "damage-energy")[0] for r in selected]
        stds  = [get_score_total(r, cell_name, "damage-energy")[1] for r in selected]
        offset = (c_idx - n_cells/2 + 0.5) * bar_w
        ax.bar(x + offset, means, bar_w, yerr=stds, capsize=3,
               color=cmap(c_idx / max(n_cells-1, 1)),
               alpha=0.85, label=cell_name)
    ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("Total Damage-Energy (NWL proxy)"); ax.set_title("Compare: Damage-Energy per Run")
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    save_fig(fig, plot_dir, "compare_C_damage_energy", dpi)

    # ----------------------------------------------------------------
    # Chart D: ICRH duct end current per run
    # ----------------------------------------------------------------
    icrh_m = [r["icrh_duct_end_current"]["mean"]    for r in selected]
    icrh_s = [r["icrh_duct_end_current"]["std_dev"] for r in selected]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, icrh_m, 0.5, yerr=icrh_s, capsize=4,
           color=cfg["plot"]["inboard_color"], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("ICRH Duct End Current (n/s)"); ax.set_title("Compare: ICRH Duct End Current")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    save_fig(fig, plot_dir, "compare_D_icrh_current", dpi)

    # ----------------------------------------------------------------
    # Markdown table to stdout
    # ----------------------------------------------------------------
    param_cols  = differing or all_params
    cell_cols   = cells_arg[:3]  # limit width
    header_cols = (["run_id"] + param_cols
                   + [f"flux_{c[:10]}" for c in cell_cols]
                   + ["icrh_current"])
    header = " | ".join(f"{c:<18}" for c in header_cols)
    sep    = " | ".join("-" * 18 for _ in header_cols)
    print(f"\n{'='*len(header)}")
    print("COMPARISON TABLE")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for r in selected:
        p = r["parameters"]
        row_vals = (
            [r["run_id"]]
            + [str(p.get(pc, "N/A")) for pc in param_cols]
            + [f"{get_total_flux(r, c)[0]:.3e}" for c in cell_cols]
            + [f"{r['icrh_duct_end_current']['mean']:.3e}"]
        )
        print(" | ".join(f"{v:<18}" for v in row_vals))
    print()


# =============================================================================
# MODE 3: heatmap
# =============================================================================

def mode_heatmap(
    results: list[dict],
    cfg: dict[str, Any],
    args: argparse.Namespace,
    plot_dir: Path,
) -> None:
    """Generate 2D parameter heatmaps for a chosen output variable.

    Args:
        results:  All collected run result dicts.
        cfg:      Campaign config dict.
        args:     Parsed CLI arguments.
        plot_dir: Directory for output figures.
    """
    x_param    = args.x_param
    y_param    = args.y_param
    score      = getattr(args, "output_var", None) or getattr(args, "score", "flux")
    cell_name  = getattr(args, "cell", None) or cfg["tallies"]["outboard_cells"][0]
    dpi        = cfg["plot"]["dpi"]
    all_params = list(cfg["parameters"].keys())

    # Build fixed-parameter dict
    fixed: dict[str, Any] = {}
    for fix_str in (getattr(args, "fix", None) or []):
        k, v = fix_str.split("=", 1)
        fixed[k.strip()] = v.strip()
    for p in all_params:
        if p not in (x_param, y_param) and p not in fixed:
            fixed[p] = most_common_value(results, p)

    relevant = filter_results(results, fixed, free_param=None)
    # Allow x and y to vary
    relevant = [
        r for r in results
        if all(
            p in (x_param, y_param) or str(r["parameters"].get(p)) == str(fixed.get(p))
            for p in all_params
        )
    ]

    x_vals = sorted(set(r["parameters"].get(x_param) for r in relevant),
                    key=lambda v: (str(type(v)), v))
    y_vals = sorted(set(r["parameters"].get(y_param) for r in relevant),
                    key=lambda v: (str(type(v)), v))

    if not x_vals or not y_vals:
        log.warning("No data for heatmap with x=%s, y=%s", x_param, y_param)
        return

    # Build lookup
    lookup: dict[tuple, dict] = {
        (r["parameters"].get(x_param), r["parameters"].get(y_param)): r
        for r in relevant
    }

    def _get_val(r: dict) -> float:
        if score == "flux":
            return get_total_flux(r, cell_name)[0]
        return get_score_total(r, cell_name, score)[0]

    def _build_matrix() -> np.ndarray:
        mat = np.full((len(y_vals), len(x_vals)), np.nan)
        for j, xv in enumerate(x_vals):
            for i, yv in enumerate(y_vals):
                r = lookup.get((xv, yv))
                if r is not None:
                    mat[i, j] = _get_val(r)
        return mat

    def _plot_heatmap(mat: np.ndarray, title: str, stem: str, cmap_name: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vmin = np.nanmin(mat); vmax = np.nanmax(mat)
        im = ax.imshow(mat, aspect="auto", cmap=cmap_name, origin="lower",
                       vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=f"{score} ({cell_name})")
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(v) for v in x_vals], rotation=20, ha="right")
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([str(v) for v in y_vals])
        ax.set_xlabel(x_param); ax.set_ylabel(y_param)
        ax.set_title(title)
        mid = (vmin + vmax) / 2 if not np.isnan(vmin) else 0
        for i in range(len(y_vals)):
            for j in range(len(x_vals)):
                val = mat[i, j]
                txt = f"{val:.2e}" if not np.isnan(val) else "N/A"
                color = "white" if val > mid else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=color if not np.isnan(val) else "grey")
        fig.tight_layout()
        save_fig(fig, plot_dir, stem, dpi)

    mat = _build_matrix()
    safe_cell = cell_name.replace(" ", "_")
    _plot_heatmap(mat, f"Heatmap: {score} ({cell_name})\n{x_param} vs {y_param}",
                  f"heatmap_{safe_cell}_{score}_{x_param}_vs_{y_param}", "viridis")

    # Inboard vs outboard heatmaps
    for side_cell, cmap_name in [
        (cfg["tallies"]["inboard_cells"][0],  "Blues"),
        (cfg["tallies"]["outboard_cells"][0], "Reds"),
    ]:
        side_mat = np.full((len(y_vals), len(x_vals)), np.nan)
        for j, xv in enumerate(x_vals):
            for i, yv in enumerate(y_vals):
                r = lookup.get((xv, yv))
                if r is not None:
                    if score == "flux":
                        side_mat[i, j] = get_total_flux(r, side_cell)[0]
                    else:
                        side_mat[i, j] = get_score_total(r, side_cell, score)[0]
        safe_sc = side_cell.replace(" ", "_")
        _plot_heatmap(
            side_mat,
            f"Heatmap: {score} ({side_cell})\n{x_param} vs {y_param}",
            f"heatmap_{safe_sc}_{score}_{x_param}_vs_{y_param}",
            cmap_name,
        )

    # Ratio heatmap (outboard / inboard)
    ob_cell = cfg["tallies"]["outboard_cells"][0]
    ib_cell = cfg["tallies"]["inboard_cells"][0]
    ratio_mat = np.full((len(y_vals), len(x_vals)), np.nan)
    for j, xv in enumerate(x_vals):
        for i, yv in enumerate(y_vals):
            r = lookup.get((xv, yv))
            if r is not None:
                ob_v = get_total_flux(r, ob_cell)[0] if score == "flux" else get_score_total(r, ob_cell, score)[0]
                ib_v = get_total_flux(r, ib_cell)[0] if score == "flux" else get_score_total(r, ib_cell, score)[0]
                ratio_mat[i, j] = ob_v / ib_v if ib_v != 0 else np.nan
    _plot_heatmap(
        ratio_mat,
        f"Ratio Heatmap: {ob_cell} / {ib_cell} ({score})\n{x_param} vs {y_param}",
        f"heatmap_ratio_{score}_{x_param}_vs_{y_param}",
        "RdBu_r",
    )

    # Optional user-defined ratio
    ratio_cells = getattr(args, "ratio_cells", None)
    if ratio_cells and len(ratio_cells) == 2:
        cell_a, cell_b = ratio_cells
        user_ratio = np.full((len(y_vals), len(x_vals)), np.nan)
        for j, xv in enumerate(x_vals):
            for i, yv in enumerate(y_vals):
                r = lookup.get((xv, yv))
                if r is not None:
                    va = get_total_flux(r, cell_a)[0] if score == "flux" else get_score_total(r, cell_a, score)[0]
                    vb = get_total_flux(r, cell_b)[0] if score == "flux" else get_score_total(r, cell_b, score)[0]
                    user_ratio[i, j] = va / vb if vb != 0 else np.nan
        _plot_heatmap(
            user_ratio,
            f"Ratio Heatmap: {cell_a} / {cell_b} ({score})\n{x_param} vs {y_param}",
            f"heatmap_ratio_{cell_a}_{cell_b}_{score}",
            "RdBu_r",
        )


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for postprocess.py.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Post-process OpenMC campaign results and generate comparison plots."
    )
    parser.add_argument("--config", default="campaign_config.yaml",
                        help="Path to campaign_config.yaml")
    parser.add_argument("--mode", required=True,
                        choices=["sensitivity", "compare", "heatmap"],
                        help="Analysis mode")
    parser.add_argument("--score", default="flux",
                        choices=["flux", "heating", "damage-energy"],
                        help="Score to plot (default: flux)")
    parser.add_argument("--groups", default="all",
                        choices=["all", "thermal", "epithermal", "fast", "highfast"],
                        help="Energy groups to include (default: all)")
    parser.add_argument("--side", default="both",
                        choices=["inboard", "outboard", "both"],
                        help="Which side to show in sensitivity mode (default: both)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug",   action="store_true")

    # sensitivity
    parser.add_argument("--param", help="Parameter to sweep in sensitivity mode")
    parser.add_argument("--fix", nargs="*", metavar="KEY=VALUE",
                        help="Fix parameters at specific values, e.g. --fix radius=455")

    # compare
    parser.add_argument("--runs", nargs="*", metavar="RUN_ID",
                        help="run_ids to compare")
    parser.add_argument("--cells", nargs="*", metavar="CELL",
                        help="Cells to include in compare mode (default: all)")

    # heatmap
    parser.add_argument("--x-param", dest="x_param",
                        help="X-axis parameter for heatmap")
    parser.add_argument("--y-param", dest="y_param",
                        help="Y-axis parameter for heatmap")
    parser.add_argument("--cell",   help="Cell name for heatmap output variable")
    parser.add_argument("--output-var", dest="output_var",
                        choices=["flux", "heating", "damage-energy"],
                        help="Output variable for heatmap (overrides --score)")
    parser.add_argument("--ratio-cells", dest="ratio_cells", nargs=2,
                        metavar=("CELL_A", "CELL_B"),
                        help="Two cells for a custom ratio heatmap")

    return parser


def main() -> None:
    """Entry point for postprocess.py."""
    parser = build_parser()
    args   = parser.parse_args()

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
    base_dir    = resolve_base_dir(cfg)
    apply_style(cfg)

    # Collect / extract all results
    results = collect_all_results(base_dir, cfg)

    if not results:
        log.error("No completed runs found under %s. Exiting.", base_dir)
        return

    plot_dir = make_plot_dir(base_dir, args.mode)

    if args.mode == "sensitivity":
        if not args.param:
            parser.error("--mode sensitivity requires --param <parameter_name>")
        mode_sensitivity(results, cfg, args, plot_dir)

    elif args.mode == "compare":
        if not args.runs:
            parser.error("--mode compare requires --runs run_id_1 run_id_2 ...")
        mode_compare(results, cfg, args, plot_dir)

    elif args.mode == "heatmap":
        if not args.x_param or not args.y_param:
            parser.error("--mode heatmap requires --x-param and --y-param")
        mode_heatmap(results, cfg, args, plot_dir)

    print(f"\nDone. Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()