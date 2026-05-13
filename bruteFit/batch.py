from __future__ import annotations

import csv
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

os.environ.setdefault("BRUTEFIT_HEADLESS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/brutefit-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/brutefit-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from dateutil import rrule, tz  # noqa: F401

import matplotlib
import numpy as np

matplotlib.use("Agg")

from processrecord import fileutils as fh
from processrecord.container import ProcessRecord

from .dataFitting import fit_models_headless
from .fitConfig import FitConfig
from .gaussianModels import component_peak_height
from .transition_matching import TransitionGuess

DEFAULT_RUN_NAMES = {"B", "Q0", "Q1", "Q0_Q1"}

# Batch mode is intentionally a thin orchestration layer around the same processing,
# guessing, and fitting functions used by the GUI. That keeps the two workflows from
# drifting apart: if peak guessing improves in the preview dialog, the tree runner
# gets the same behavior unless we explicitly fork it.


def discover_run_dirs(root: str | Path, run_names: set[str] | None = None) -> list[Path]:
    """Find spectroscopy run folders under a compound/run directory layout."""
    root = Path(root)
    allowed = run_names or DEFAULT_RUN_NAMES
    return sorted(
        path
        for path in root.glob("*/*")
        if path.is_dir() and path.name in allowed
    )


def fitconfig_to_dict(fc: FitConfig) -> dict[str, Any]:
    return {
        fc_field.name: getattr(fc, fc_field.name)
        for fc_field in fields(fc)
        if not fc_field.name.startswith("_")
    }


def _require_metadata(metadata: dict[str, Any], run_dir: Path) -> tuple[str, float, float, float]:
    """
    Batch processing must be reproducible, so we require the same processing metadata
    the GUI lets the user review manually. Missing keys should fail loudly here rather
    than silently producing a processed spectrum with default concentration/pathlength.
    """
    missing = [
        key for key in ("lims_ID", "concentration_MOL_L", "pathlength_cm", "field_B")
        if key not in metadata
    ]
    if missing:
        raise ValueError(
            f"{run_dir} metadata is missing required key(s): {', '.join(missing)}"
        )
    return (
        str(metadata["lims_ID"]),
        float(metadata["concentration_MOL_L"]),
        float(metadata["pathlength_cm"]),
        float(metadata["field_B"]),
    )


def process_run_folder(run_dir: str | Path, output_dir: str | Path | None = None):
    """
    Convert one raw run folder into the normalized processed DataFrame used by fitting.

    Output goes into brutefit_batch/ inside the run folder. Keeping derived files next
    to the source data makes review easy while keeping the code repository clean.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "brutefit_batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    pos_df, neg_df, abs_df, sticks_df, basename, metadata, metadata_path = fh.read_pos_neg_abs_from_folder(str(run_dir))
    if pos_df is None or neg_df is None or abs_df is None:
        raise RuntimeError(f"Could not read required raw CSV files from {run_dir}")

    lims, concentration, pathlength, field = _require_metadata(metadata, run_dir)
    proc = ProcessRecord(
        (pos_df, neg_df, abs_df, sticks_df, basename or run_dir.name),
        lims,
        concentration,
        pathlength,
        field,
    )
    proc.clean_data()
    processed_df = proc.get_merged_df()
    processed_path = output_dir / "processed.csv"
    processed_df.to_csv(processed_path, index=False)
    return processed_df, metadata, metadata_path, processed_path


def write_auto_guesses_csv(transitions: list[TransitionGuess], output_path: str | Path) -> None:
    """
    Save the exact initial guesses that seeded fitting.

    This is a reproducibility artifact: paired/unpaired status, auto/manual origin,
    height, and area-amplitude are all retained so a collaborator can reconstruct why
    a final A/D or B/D value did or did not exist for a transition.
    """
    output_path = Path(output_path)
    rows = []
    for transition in transitions:
        for source, peak in (("abs", transition.abs_peak), ("mcd", transition.mcd_peak)):
            if peak is None:
                continue
            rows.append(
                {
                    "transition_id": transition.transition_id,
                    "status": transition.status,
                    "source": source,
                    "origin": peak.origin,
                    "label": peak.label or "",
                    "center": peak.center,
                    "sigma": peak.sigma,
                    "height": peak.height,
                    "amplitude": peak.amplitude,
                    "match_distance": transition.match_distance,
                }
            )

    fieldnames = [
        "transition_id",
        "status",
        "source",
        "origin",
        "label",
        "center",
        "sigma",
        "height",
        "amplitude",
        "match_distance",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _result_metric_value(bf_result, bundle, metric: str) -> float:
    if metric == "residual_rms":
        return float(bf_result._residual_rms(bundle.mcd_result) + bf_result._residual_rms(bundle.abs_result))
    if metric == "combo":
        return float(bf_result._combo_metric(bundle.mcd_result) + bf_result._combo_metric(bundle.abs_result))
    return float(getattr(bundle.mcd_result, metric))


def _param_value(result, prefix: str | None, suffix: str) -> float | None:
    if not prefix:
        return None
    param = result.params.get(f"{prefix}{suffix}")
    if param is None:
        return None
    return float(param.value)


def write_top_fit_summary_csv(bf_result, output_path: str | Path, n: int = 3, metric: str = "redchi") -> None:
    """
    Save transition-level values for the top fits.

    The fitting models use area-amplitude parameters internally. We also write peak
    heights for visual sanity checks, but ratios are computed from fitted amplitudes
    to stay consistent with the Gaussian model normalization.
    """
    output_path = Path(output_path)
    rows = []
    bundles = bf_result.n_best_results(n=n, metric=metric)
    for rank, bundle in enumerate(bundles, start=1):
        metric_value = _result_metric_value(bf_result, bundle, metric)
        for spec in bundle.transition_specs:
            mcd_amp = _param_value(bundle.mcd_result, spec.mcd_prefix, "amplitude")
            abs_amp = _param_value(bundle.abs_result, spec.abs_prefix, "amplitude")
            mcd_center = _param_value(bundle.mcd_result, spec.mcd_prefix, "center")
            abs_center = _param_value(bundle.abs_result, spec.abs_prefix, "center")
            mcd_sigma = _param_value(bundle.mcd_result, spec.mcd_prefix, "sigma")
            abs_sigma = _param_value(bundle.abs_result, spec.abs_prefix, "sigma")
            mcd_label = "" if spec.mcd_prefix is None else spec.mcd_prefix[0]
            ratio = None
            if spec.status == "paired" and mcd_amp is not None and abs_amp not in {None, 0.0}:
                ratio = abs(mcd_amp / abs_amp)

            rows.append(
                {
                    "rank": rank,
                    "metric": metric,
                    "metric_value": metric_value,
                    "mcd_redchi": float(bundle.mcd_result.redchi),
                    "abs_redchi": float(bundle.abs_result.redchi),
                    "mcd_bic": float(bundle.mcd_result.bic),
                    "abs_bic": float(bundle.abs_result.bic),
                    "transition_id": spec.transition_id,
                    "status": spec.status,
                    "ratio_label": spec.ratio_label or "",
                    "mcd_label": mcd_label,
                    "mcd_value": mcd_amp,
                    "d_value": abs_amp,
                    "ratio": ratio,
                    "mcd_center": mcd_center,
                    "d_center": abs_center,
                    "mcd_sigma": mcd_sigma,
                    "d_sigma": abs_sigma,
                    "mcd_height": (
                        None if mcd_amp is None or mcd_center is None or mcd_sigma is None
                        else component_peak_height(mcd_amp, mcd_center, mcd_sigma, label=mcd_label or None)
                    ),
                    "d_height": (
                        None if abs_amp is None or abs_center is None or abs_sigma is None
                        else component_peak_height(abs_amp, abs_center, abs_sigma)
                    ),
                    "match_distance": spec.match_distance,
                    "mcd_prefix": spec.mcd_prefix or "",
                    "abs_prefix": spec.abs_prefix or "",
                }
            )

    fieldnames = [
        "rank",
        "metric",
        "metric_value",
        "mcd_redchi",
        "abs_redchi",
        "mcd_bic",
        "abs_bic",
        "transition_id",
        "status",
        "ratio_label",
        "mcd_label",
        "mcd_value",
        "d_value",
        "ratio",
        "mcd_center",
        "d_center",
        "mcd_sigma",
        "d_sigma",
        "mcd_height",
        "d_height",
        "match_distance",
        "mcd_prefix",
        "abs_prefix",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_top_fit_svgs(bf_result, output_dir: str | Path, n: int = 3, metric: str = "redchi") -> list[Path]:
    """Export the ranked fit figures as SVGs for lab notebook / manuscript review."""
    output_dir = Path(output_dir)
    figures = bf_result.get_plot_figs(n=n, metric=metric)
    output_paths = []
    for rank, fig in enumerate(figures, start=1):
        output_path = output_dir / f"top_fit_{rank:02d}.svg"
        fig.savefig(output_path, format="svg", bbox_inches="tight")
        output_paths.append(output_path)
    return output_paths


def batch_fit_run(
    run_dir: str | Path,
    fc: FitConfig | None = None,
    top_n: int = 3,
    metric: str = "redchi",
    processes: int = 1,
) -> dict[str, Any]:
    """Run the full process -> guess -> fit -> export pipeline for one folder."""
    run_dir = Path(run_dir)
    output_dir = run_dir / "brutefit_batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    fc = fc or FitConfig()
    processed_df, metadata, metadata_path, processed_path = process_run_folder(run_dir, output_dir=output_dir)
    bf_result, fc, transitions = fit_models_headless(processed_df, fc=fc, processes=processes)

    auto_guess_path = output_dir / "auto_guesses.csv"
    summary_path = output_dir / "top_fit_summary.csv"
    manifest_path = output_dir / "batch_run.json"
    write_auto_guesses_csv(transitions, auto_guess_path)
    write_top_fit_summary_csv(bf_result, summary_path, n=top_n, metric=metric)
    svg_paths = save_top_fit_svgs(bf_result, output_dir, n=top_n, metric=metric)

    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "metadata_path": metadata_path,
        "metadata": metadata,
        "fit_config": fitconfig_to_dict(fc),
        "top_n": int(top_n),
        "metric": metric,
        "outputs": {
            "processed": str(processed_path),
            "auto_guesses": str(auto_guess_path),
            "summary": str(summary_path),
            "svgs": [str(path) for path in svg_paths],
        },
        "fit_count": len(bf_result.fit_bundles),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def batch_fit_tree(
    root: str | Path,
    top_n: int = 3,
    metric: str = "redchi",
    processes: int = 1,
    run_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Run batch fitting over every discovered run folder below a dataset root."""
    manifests = []
    for run_dir in discover_run_dirs(root, run_names=run_names):
        print(f"\n=== Batch fitting {run_dir} ===")
        manifests.append(
            batch_fit_run(
                run_dir,
                fc=FitConfig(),
                top_n=top_n,
                metric=metric,
                processes=processes,
            )
        )
    return manifests
