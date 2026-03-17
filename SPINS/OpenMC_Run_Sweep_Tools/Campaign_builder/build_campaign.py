"""
build_campaign.py
=================
Pure setup script for the OpenMC parametric campaign.
Creates run folders, geometry symlinks, per-run config files, and the
campaign manifest. Does NOT build geometry, run OpenMC, or call qsub.

Usage
-----
    python build_campaign.py [--config campaign_config.yaml]
                             [--dry-run]
                             [--verbose]
                             [--debug]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections.abc import Iterator
from itertools import product
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load and return the campaign YAML config.

    Args:
        config_path: Path to campaign_config.yaml.

    Returns:
        Parsed config dict.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    log.debug("Loaded campaign config from %s", config_path)
    return cfg


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Expand and resolve all paths from the paths block.

    Args:
        cfg: Full campaign config dict.

    Returns:
        Dict mapping path key -> absolute resolved Path.
    """
    return {k: Path(v).expanduser().resolve() for k, v in cfg["paths"].items()}


# ---------------------------------------------------------------------------
# Exclusion rule evaluator
# ---------------------------------------------------------------------------

def _evaluate_condition(combo_val: Any, condition: Any) -> bool:
    """Evaluate a single condition against a combo value.

    Args:
        combo_val: The value from the current parameter combination.
        condition: Either a bare value (exact match) or a dict like
                   {op: value} where op in {eq, neq, gt, gte, lt, lte}.

    Returns:
        True if the condition is satisfied.
    """
    if isinstance(condition, dict):
        op, ref = next(iter(condition.items()))
        # Coerce to matching numeric type when possible
        if isinstance(combo_val, (int, float)) and isinstance(ref, (int, float)):
            combo_val = type(ref)(combo_val)
        ops = {
            "eq":  lambda a, b: a == b,
            "neq": lambda a, b: a != b,
            "gt":  lambda a, b: a >  b,
            "gte": lambda a, b: a >= b,
            "lt":  lambda a, b: a <  b,
            "lte": lambda a, b: a <= b,
        }
        if op not in ops:
            raise ValueError(f"Unknown exclusion operator '{op}'. "
                             f"Supported: {list(ops)}")
        return ops[op](combo_val, ref)
    else:
        # Bare value — exact equality with type coercion
        if isinstance(combo_val, (int, float)) and isinstance(condition, (int, float)):
            return type(condition)(combo_val) == condition
        return combo_val == condition


def is_excluded(
    combo: dict[str, Any],
    rules: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any] | None]:
    """Check whether a parameter combination matches any exclusion rule.

    A combination is excluded when ALL conditions in a single rule are
    satisfied simultaneously.

    Args:
        combo: Parameter combination dict {param_name: value}.
        rules: List of exclusion rule dicts from campaign_config.yaml.
               Each rule may contain a "reason" key (informational only)
               plus one or more parameter conditions.

    Returns:
        (True, matched_rule) if excluded, (False, None) otherwise.
    """
    for rule in rules:
        conditions = {k: v for k, v in rule.items() if k != "reason"}
        if all(
            k in combo and _evaluate_condition(combo[k], v)
            for k, v in conditions.items()
        ):
            return True, rule
    return False, None


# ---------------------------------------------------------------------------
# Combination generation
# ---------------------------------------------------------------------------

def generate_combinations(
    parameters: dict[str, list[Any]],
) -> Iterator[dict[str, Any]]:
    """Yield all Cartesian-product combinations as dicts.

    Args:
        parameters: Ordered dict mapping param name -> list of values.

    Yields:
        Dict {param_name: value} for each combination.
    """
    keys = list(parameters.keys())
    for combo_vals in product(*parameters.values()):
        yield dict(zip(keys, combo_vals))


# ---------------------------------------------------------------------------
# Folder naming — replicates existing script exactly
# ---------------------------------------------------------------------------

def build_run_paths(
    combo: dict[str, Any],
    base_dir: Path,
    parent_folder_params: list[str],
    parameters: dict[str, list[Any]],
) -> tuple[Path, str]:
    """Compute (run_path, run_name) for a given parameter combination.

    Replicates the folder-naming logic from the existing script:
      - parent dirs: base_dir / radius_{r} / triang_{t} / ...
      - run_name:    remaining params joined with "_"

    Args:
        combo:                 Parameter combination dict.
        base_dir:              Campaign base directory.
        parent_folder_params:  Params used as nested parent directories.
        parameters:            Full ordered parameter dict (for ordering).

    Returns:
        (run_path, run_name) tuple.
    """
    parent_path = base_dir
    for param in parent_folder_params:
        parent_path = parent_path / f"{param}_{combo[param]}"

    run_name_components = [
        f"{param}_{combo[param]}"
        for param in parameters
        if param not in parent_folder_params
    ]
    run_name = "_".join(run_name_components)
    run_path = parent_path / run_name
    return run_path, run_name


# ---------------------------------------------------------------------------
# Symlink creation
# ---------------------------------------------------------------------------

def create_geometry_symlinks(
    combo: dict[str, Any],
    run_path: Path,
    geometries_dir: Path,
    dry_run: bool = False,
) -> None:
    """Create .npy geometry boundary symlinks in the run folder.

    Replicates the symlink logic from the existing script exactly.

    Args:
        combo:         Parameter combination dict (needs radius, triang).
        run_path:      Destination run folder.
        geometries_dir: Source directory for .npy files.
        dry_run:       If True, log actions but create nothing.
    """
    root = f"radius_{combo['radius']}_triang_{combo['triang']}_"
    suffixes = ["breeder_boundary.npy", "inner_boundary.npy", "outer_boundary.npy"]

    for suffix in suffixes:
        fname = root + suffix
        src = geometries_dir / fname
        dst = run_path / fname

        if dst.exists() or dst.is_symlink():
            log.debug("Symlink already exists, skipping: %s", dst)
            continue

        if not src.exists():
            log.warning("Source geometry file not found, skipping symlink: %s", src)
            raise ValueError(f"Source geometry file not found, skipping symlink: {str(src)}")
            continue

        if dry_run:
            log.info("[DRY-RUN] Would symlink %s -> %s", dst, src)
        else:
            os.symlink(src, dst, target_is_directory=False)
            log.debug("Created symlink %s -> %s", dst, src)


# ---------------------------------------------------------------------------
# Per-run config writer
# ---------------------------------------------------------------------------

def write_run_config(
    combo: dict[str, Any],
    run_id: str,
    run_path: Path,
    paths: dict[str, Path],
    cfg: dict[str, Any],
    dry_run: bool = False,
) -> None:
    """Write run_config.yaml into the run folder.

    Args:
        combo:    Parameter combination dict.
        run_id:   String identifier for this run.
        run_path: Absolute path to the run folder.
        paths:    Resolved paths dict from campaign config.
        cfg:      Full campaign config dict.
        dry_run:  If True, log but do not write.
    """
    run_cfg = {
        "run_id":   run_id,
        "run_path": str(run_path),
        "status":   "pending",
        "parameters": {k: combo[k] for k in cfg["parameters"]},
        "paths": {
            "geometries_dir": str(paths["geometries_dir"]),
            "materials_dir":  str(paths["materials_dir"]),
            "sources_dir":    str(paths["sources_dir"]),
            "materials_file": str(paths["materials_file"]),
        },
        "simulation": cfg["simulation"],
        "tallies":    cfg["tallies"],
    }

    out_path = run_path / "run_config.yaml"
    if dry_run:
        log.info("[DRY-RUN] Would write run_config.yaml to %s", run_path)
    else:
        with open(out_path, "w") as f:
            yaml.dump(run_cfg, f, default_flow_style=False, sort_keys=False)
        log.debug("Wrote run_config.yaml to %s", out_path)


# ---------------------------------------------------------------------------
# Script copying
# ---------------------------------------------------------------------------

def copy_scripts(
    run_path: Path,
    geometry_builder_src: Path,
    dry_run: bool = False,
) -> None:
    """Copy geometry_builder.py into the run folder.

    Args:
        run_path:              Destination run folder.
        geometry_builder_src:  Absolute path to geometry_builder.py.
        dry_run:               If True, log but do not copy.
    """
    if not geometry_builder_src.exists():
        log.warning("geometry_builder.py not found at %s — skipping copy",
                    geometry_builder_src)
        return

    dst = run_path / geometry_builder_src.name
    if dry_run:
        log.info("[DRY-RUN] Would copy %s -> %s", geometry_builder_src, dst)
    else:
        shutil.copy2(geometry_builder_src, dst)
        log.debug("Copied %s -> %s", geometry_builder_src, dst)


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

def write_manifest(
    base_dir: Path,
    config_path: Path,
    valid_runs: list[dict[str, Any]],
    excluded_runs: list[dict[str, Any]],
    total_combinations: int,
    dry_run: bool = False,
) -> None:
    """Write campaign_manifest.json to base_dir.

    Args:
        base_dir:           Campaign base directory.
        config_path:        Absolute path to campaign_config.yaml.
        valid_runs:         List of valid run entry dicts.
        excluded_runs:      List of excluded run entry dicts.
        total_combinations: Total before exclusion filtering.
        dry_run:            If True, log but do not write.
    """
    manifest = {
        "base_dir":          str(base_dir),
        "campaign_config":   str(config_path),
        "total_combinations": total_combinations,
        "excluded_count":    len(excluded_runs),
        "valid_count":       len(valid_runs),
        "valid_runs":        valid_runs,
        "excluded_runs":     excluded_runs,
    }
    out_path = base_dir / "campaign_manifest.json"
    if dry_run:
        log.info("[DRY-RUN] Would write campaign_manifest.json to %s", base_dir)
    else:
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info("Wrote campaign_manifest.json to %s", out_path)


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def print_summary_table(
    valid_runs: list[dict[str, Any]],
    excluded_runs: list[dict[str, Any]],
    total: int,
    param_keys: list[str],
) -> None:
    """Print a formatted summary table of all valid runs to stdout.

    Args:
        valid_runs:  List of valid run entry dicts.
        excluded_runs: List of excluded run entry dicts.
        total:       Total combinations before filtering.
        param_keys:  Ordered list of parameter names.
    """
    col_widths = {
        "idx":    5,
        "run_id": 80,
        **{k: max(len(k), 10) for k in param_keys},
        "path":   60,
    }

    header = (
        f"{'#':<{col_widths['idx']}} "
        + " ".join(f"{k:<{col_widths[k]}}" for k in param_keys)
        + f"  {'run_id':<{col_widths['run_id']}}  path"
    )
    sep = "-" * len(header)

    log.info("\n%s\nCAMPAIGN BUILD SUMMARY\n%s", sep, sep)
    log.info(header)
    log.info(sep)

    for i, entry in enumerate(valid_runs, 1):
        p = entry["parameters"]
        row = (
            f"{i:<{col_widths['idx']}} "
            + " ".join(f"{str(p[k]):<{col_widths[k]}}" for k in param_keys)
            + f"  {entry['run_id']:<{col_widths['run_id']}}  {entry['run_path']}"
        )
        log.info(row)

    log.info(sep)
    log.info(
        "TOTAL: %d valid runs | %d excluded | %d total combinations",
        len(valid_runs), len(excluded_runs), total,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for build_campaign.py."""
    parser = argparse.ArgumentParser(
        description="Build OpenMC campaign run folders from campaign_config.yaml."
    )
    parser.add_argument(
        "--config", default="campaign_config.yaml",
        help="Path to campaign_config.yaml (default: campaign_config.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print all actions but create no files or folders.",
    )
    parser.add_argument("--verbose", action="store_true", help="Set logging to INFO.")
    parser.add_argument("--debug",   action="store_true", help="Set logging to DEBUG.")
    args = parser.parse_args()

    # Configure logging
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
    cfg = load_config(config_path)
    paths = resolve_paths(cfg)

    base_dir              = paths["base_dir"]
    geometries_dir        = paths["geometries_dir"]
    geometry_builder_src  = paths["geometry_builder_script"] \
                            if paths["geometry_builder_script"].is_absolute() \
                            else (config_path.parent / cfg["paths"]["geometry_builder_script"]).resolve()

    parameters           = cfg["parameters"]
    parent_folder_params = cfg["parent_folder_params"]
    exclusion_rules      = cfg.get("exclusions", [])
    param_keys           = list(parameters.keys())

    if not args.dry_run:
        base_dir.mkdir(parents=True, exist_ok=True)

    valid_runs: list[dict[str, Any]]    = []
    excluded_runs: list[dict[str, Any]] = []
    total_count = 0

    for combo in generate_combinations(parameters):
        total_count += 1
        excluded, matched_rule = is_excluded(combo, exclusion_rules)

        if excluded:
            log.debug(
                "EXCLUDED [%s]: %s",
                matched_rule.get("reason", "no reason given"), combo,
            )
            excluded_runs.append({
                "parameters":   combo,
                "matched_rule": matched_rule,
            })
            continue

        run_path, run_name = build_run_paths(
            combo, base_dir, parent_folder_params, parameters
        )
        run_id = run_name

        # Create run folder
        if not args.dry_run:
            run_path.mkdir(parents=True, exist_ok=True)
            log.debug("Created run folder: %s", run_path)
        else:
            log.info("[DRY-RUN] Would create folder: %s", run_path)

        # Geometry symlinks
        create_geometry_symlinks(combo, run_path, geometries_dir, args.dry_run)

        # run_config.yaml
        write_run_config(combo, run_id, run_path, paths, cfg, args.dry_run)

        # Copy geometry_builder.py
        copy_scripts(run_path, geometry_builder_src, args.dry_run)

        valid_runs.append({
            "run_id":     run_id,
            "run_path":   str(run_path),
            "parameters": {k: combo[k] for k in parameters},
            "status":     "pending",
        })

    # Write manifest
    write_manifest(
        base_dir, config_path, valid_runs, excluded_runs, total_count, args.dry_run
    )

    # Summary table (always printed regardless of log level)
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  CAMPAIGN BUILD COMPLETE")
    print(f"  Config : {config_path}")
    print(f"  Base   : {base_dir}")
    print(sep)
    print(f"  {'#':<5} {'run_id'}")
    print(f"  {'-'*5} {'-'*70}")
    for i, entry in enumerate(valid_runs, 1):
        print(f"  {i:<5} {entry['run_id']}")
    print(sep)
    print(f"  {len(valid_runs)} valid runs | "
          f"{len(excluded_runs)} excluded | "
          f"{total_count} total combinations")
    if args.dry_run:
        print("  [DRY-RUN] No files or folders were created.")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
