"""
submit_campaign.py
==================
HPC submission script for the OpenMC parametric campaign.
Replicates finish_prepping_run_dir() from the original script:
  - Templates the qsub shell script (RUNNAME / RUNDIR substitution)
  - Copies openmc_runner.py into each run folder
  - Checks for the RUN_SUBMITTED sentinel file
  - Calls qsub and touches the sentinel on success

Usage
-----
    python submit_campaign.py [--config campaign_config.yaml]
                              [--run-ids id1 id2 ...]
                              [--resubmit]
                              [--verbose]
                              [--debug]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """Load the campaign YAML config.

    Args:
        config_path: Path to campaign_config.yaml.

    Returns:
        Parsed config dict.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_manifest(base_dir: Path) -> dict[str, Any]:
    """Load campaign_manifest.json from base_dir.

    Args:
        base_dir: Campaign base directory.

    Returns:
        Parsed manifest dict.

    Raises:
        FileNotFoundError: If manifest does not exist.
    """
    manifest_path = base_dir / "campaign_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"campaign_manifest.json not found at {manifest_path}.\n"
            "Run build_campaign.py first."
        )
    with open(manifest_path) as f:
        return json.load(f)


def load_run_config(run_path: Path) -> dict[str, Any]:
    """Load run_config.yaml from a run folder.

    Args:
        run_path: Path to the run folder.

    Returns:
        Parsed run config dict.
    """
    cfg_path = run_path / "run_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.yaml not found in {run_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def update_run_status(run_path: Path, status: str) -> None:
    """Update the status field in run_config.yaml.

    Args:
        run_path: Path to the run folder.
        status:   New status string (e.g. "submitted", "pending").
    """
    cfg_path = run_path / "run_config.yaml"
    with open(cfg_path) as f:
        run_cfg = yaml.safe_load(f)
    run_cfg["status"] = status
    with open(cfg_path, "w") as f:
        yaml.dump(run_cfg, f, default_flow_style=False, sort_keys=False)
    log.debug("Updated status to '%s' in %s", status, cfg_path)


# ---------------------------------------------------------------------------
# Submission logic — replicates finish_prepping_run_dir() verbatim
# ---------------------------------------------------------------------------

def submit_run(
    run_entry: dict[str, Any],
    base_dir: Path,
    openmc_runner_template: Path,
    openmc_runner_script: Path,
) -> str:
    """Prepare and submit a single run to the HPC scheduler via qsub.

    Replicates finish_prepping_run_dir() from the original script exactly:
      1. Copy the shell template to openmc_run.sh in the run folder
      2. Replace RUNNAME and RUNDIR tokens in the shell script
      3. Copy openmc_runner.py into the run folder
      4. Check for RUN_SUBMITTED sentinel — skip if present
      5. Call qsub; touch sentinel and update status on success

    Args:
        run_entry:              Entry dict from campaign_manifest.json.
        base_dir:               Campaign base directory.
        openmc_runner_template: Path to openmc_run_template.sh.
        openmc_runner_script:   Path to openmc_runner.py.

    Returns:
        One of: "submitted", "already_submitted", "failed"
    """
    run_id   = run_entry["run_id"]
    run_path = Path(run_entry["run_path"])
    params   = run_entry["parameters"]

    # Short run name used in shell script — mirrors original script
    run_name_short = (
        f"{params['source_device']}_R{params['radius']}_T{params['triang']}"
    )

    log.info("Preparing submission for run: %s", run_name_short)

    # ----------------------------------------------------------------
    # Step 1 & 2: Copy and template the shell script
    # ----------------------------------------------------------------
    run_script_dst = run_path / "openmc_run.sh"

    if not openmc_runner_template.exists():
        log.error("Shell template not found: %s", openmc_runner_template)
        return "failed"

    shutil.copy(openmc_runner_template, run_script_dst)
    log.debug("Copied %s -> %s", openmc_runner_template, run_script_dst)

    with open(run_script_dst) as f:
        run_script = f.read()

    run_script = run_script.replace("RUNNAME", run_name_short)
    run_script = run_script.replace("RUNDIR",  str(run_path))

    with open(run_script_dst, "w") as f:
        f.write(run_script)
    log.debug("Templated openmc_run.sh for %s", run_name_short)

    # ----------------------------------------------------------------
    # Step 3: Copy openmc_runner.py
    # ----------------------------------------------------------------
    if not openmc_runner_script.exists():
        log.warning("openmc_runner.py not found at %s — skipping copy",
                    openmc_runner_script)
    else:
        shutil.copy(openmc_runner_script, run_path / openmc_runner_script.name)
        log.debug("Copied openmc_runner.py to %s", run_path)

    # ----------------------------------------------------------------
    # Step 4: Check sentinel file
    # ----------------------------------------------------------------
    sentinel = run_path / "RUN_SUBMITTED"
    if sentinel.exists():
        log.info("RUN %s ALREADY SUBMITTED! SKIPPING SUBMISSION.", run_name_short)
        return "already_submitted"

    # ----------------------------------------------------------------
    # Step 5: Call qsub
    # ----------------------------------------------------------------
    log.info("SUBMITTING %s!!", run_name_short)
    result = subprocess.run(
        ["qsub", "openmc_run.sh"],
        cwd=str(run_path),
        capture_output=True,
        text=True,
    )

    log.debug("Return code: %d", result.returncode)
    log.debug("STDOUT: %s", result.stdout)
    log.debug("STDERR: %s", result.stderr)

    if result.returncode != 0:
        log.error(
            "qsub FAILED for %s (exit code %d).\nSTDERR: %s",
            run_name_short, result.returncode, result.stderr,
        )
        return "failed"

    job_id = result.stdout.strip()
    log.info("Submitted job: %s  →  job_id: %s", run_name_short, job_id)
    print(f"Submitted job: {job_id}")

    sentinel.touch()
    update_run_status(run_path, "submitted")
    return "submitted"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for submit_campaign.py."""
    parser = argparse.ArgumentParser(
        description="Submit OpenMC campaign runs to HPC scheduler via qsub."
    )
    parser.add_argument(
        "--config", default="campaign_config.yaml",
        help="Path to campaign_config.yaml (default: campaign_config.yaml)",
    )
    parser.add_argument(
        "--run-ids", nargs="*", metavar="RUN_ID",
        help="Only submit the specified run_ids.",
    )
    parser.add_argument(
        "--resubmit", action="store_true",
        help="Also re-submit runs with status 'submitted' (in addition to 'pending').",
    )
    parser.add_argument("--verbose", action="store_true", help="Set logging to INFO.")
    parser.add_argument("--debug",   action="store_true", help="Set logging to DEBUG.")
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

    base_dir = Path(cfg["paths"]["base_dir"]).expanduser().resolve()
    openmc_runner_template = (
        Path(cfg["paths"]["openmc_runner_template"]).expanduser().resolve()
        if Path(cfg["paths"]["openmc_runner_template"]).is_absolute()
        else (base_dir / cfg["paths"]["openmc_runner_template"]).resolve()
    )
    openmc_runner_script = (
        Path(cfg["paths"]["openmc_runner_script"]).expanduser().resolve()
        if Path(cfg["paths"]["openmc_runner_script"]).is_absolute()
        else (base_dir / cfg["paths"]["openmc_runner_script"]).resolve()
    )

    manifest = load_manifest(base_dir)

    # Determine which statuses are eligible
    eligible_statuses = {"pending"}
    if args.resubmit:
        eligible_statuses.add("submitted")

    # Filter runs
    runs_to_submit: list[dict[str, Any]] = []
    for entry in manifest["valid_runs"]:
        run_path = Path(entry["run_path"])
        try:
            run_cfg = load_run_config(run_path)
            status  = run_cfg.get("status", "pending")
        except FileNotFoundError:
            log.warning("run_config.yaml missing for %s — treating as pending",
                        entry["run_id"])
            status = "pending"

        if status not in eligible_statuses:
            log.debug("Skipping run %s (status=%s)", entry["run_id"], status)
            continue

        if args.run_ids and entry["run_id"] not in args.run_ids:
            continue

        runs_to_submit.append(entry)

    if not runs_to_submit:
        print("No runs to submit. All runs may already be submitted or complete.")
        return

    log.info("Submitting %d run(s)...", len(runs_to_submit))

    n_submitted        = 0
    n_already_submitted = 0
    n_failed           = 0

    for entry in runs_to_submit:
        outcome = submit_run(
            entry, base_dir, openmc_runner_template, openmc_runner_script
        )
        if outcome == "submitted":
            n_submitted += 1
        elif outcome == "already_submitted":
            n_already_submitted += 1
        else:
            n_failed += 1

    # Summary
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  SUBMISSION SUMMARY")
    print(sep)
    print(f"  Submitted        : {n_submitted}")
    print(f"  Already submitted: {n_already_submitted}")
    print(f"  Failed           : {n_failed}")
    print(f"  Total processed  : {len(runs_to_submit)}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
