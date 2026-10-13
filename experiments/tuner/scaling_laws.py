"""Scaling law fitting and plotting for llmtuna-backed multi-scale studies.

Fits ``linregress`` on log-log over each scale's best result, which ``Study``
returns as ``{"scale_label", "best_cfg", "best_value", "all_trials"}``.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


def fit_scaling_law(flops_list: List[float], values_list: List[float]) -> Dict:
    """Fit ``log10(value) = slope * log10(flops) + intercept``."""
    log_flops = np.log10(flops_list)
    log_values = np.log10(values_list)
    result = linregress(log_flops, log_values)

    def predict(flops: float) -> float:
        return 10 ** (result.slope * np.log10(flops) + result.intercept)

    return {
        "slope": result.slope,
        "intercept": result.intercept,
        "r_squared": result.rvalue ** 2,
        "predict": predict,
    }


def fit_all_scaling_laws(scale_results: List[Dict]) -> Dict[str, Dict]:
    """Fit scaling laws for the metric + each hparam across FLOP scales.

    Returns ``{name: {slope, intercept, r_squared, predict}}``, with one entry
    per hparam whose best value is positive at every scale, and ``{}`` for
    fewer than 2 scales.
    """
    if len(scale_results) < 2:
        return {}

    flops = [float(r["scale_label"]) for r in scale_results]
    laws = {}

    metric_values = [r["best_value"] for r in scale_results]
    if all(v is not None and v > 0 for v in metric_values):
        laws["min_val_bpb"] = fit_scaling_law(flops, metric_values)

    param_names = list(scale_results[0]["best_cfg"].keys()) if scale_results[0].get("best_cfg") else []
    for name in param_names:
        values = [
            r["best_cfg"].get(name) if r.get("best_cfg") else None
            for r in scale_results
        ]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
               for v in values):
            laws[name] = fit_scaling_law(flops, values)

    return laws


def plot_scaling_laws(
    scale_results: List[Dict],
    scaling_laws: Dict[str, Dict],
    output_dir: Path,
) -> List[Path]:
    """Generate scaling law plots and save as PNGs. Returns saved paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    flops = [float(r["scale_label"]) for r in scale_results]

    # One plot per fitted law.
    n_laws = len(scaling_laws)
    if n_laws == 0:
        return saved

    fig, axes = plt.subplots(nrows=n_laws, ncols=1, figsize=(12, 5 * n_laws))
    axes = np.atleast_1d(axes)

    for idx, (name, law) in enumerate(scaling_laws.items()):
        ax = axes[idx]

        if name == "min_val_bpb":
            y_values = [r["best_value"] for r in scale_results]
        else:
            y_values = [r["best_cfg"][name] for r in scale_results]

        ax.scatter(flops, y_values, color="blue", s=100, zorder=5, label="Data")

        x_smooth = np.logspace(np.log10(min(flops)), np.log10(max(flops)), 100)
        y_smooth = [law["predict"](x) for x in x_smooth]
        ax.plot(x_smooth, y_smooth, color="red", linewidth=2, label="Fit")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("FLOPs")
        ax.set_ylabel(name)
        eq = f"log10({name}) = {law['slope']:.3f} * log10(FLOPs) + {law['intercept']:.3f}"
        ax.set_title(f"{name} vs FLOPs (R² = {law['r_squared']:.3f})")
        ax.text(0.05, 0.95, eq, transform=ax.transAxes, fontsize=10,
                verticalalignment="top")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    path = output_dir / "scaling_law_fits.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    saved.append(path)

    # Per-parameter scatter of every trial attempt.
    param_names = [name for name in scaling_laws if name != "min_val_bpb"]
    if not param_names:
        return saved

    has_trials = any("all_trials" in r and r["all_trials"] for r in scale_results)
    if not has_trials:
        return saved

    fig, axes = plt.subplots(nrows=len(param_names), ncols=1,
                             figsize=(12, 5 * len(param_names)))
    axes = np.atleast_1d(axes)

    unique_flops = sorted(set(flops))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_flops)))
    flops_color_map = dict(zip(unique_flops, colors))

    for idx, name in enumerate(param_names):
        ax = axes[idx]
        for r in scale_results:
            if not r.get("all_trials"):
                continue
            flops_val = float(r["scale_label"])
            color = flops_color_map[flops_val]

            trial_params = [t["cfg"].get(name) for t in r["all_trials"] if name in t["cfg"]]
            trial_metrics = [t["value"] for t in r["all_trials"] if name in t["cfg"]]

            if trial_params:
                ax.scatter(trial_params, trial_metrics, alpha=0.5, color=color,
                           label=f"FLOPs={flops_val:.1e}")

            ax.scatter(r["best_cfg"].get(name), r["best_value"],
                       color=color, marker="*", s=200, edgecolor="black",
                       linewidth=1.5, zorder=10)

        if name in ("embedding_lr", "unembedding_lr", "matrix_lr", "weight_decay",
                    "prepretrain_abs_flops", "prepretrain_pct_flops"):
            ax.set_xscale("log")

        ax.set_xlabel(name)
        ax.set_ylabel("min_val_bpb")
        ax.set_title(f"Parameter Attempts: {name}")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    path = output_dir / "parameter_attempts.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    saved.append(path)

    return saved


def format_scaling_law_table(scaling_laws: Dict[str, Dict]) -> str:
    """Format scaling laws as a text table for logging."""
    lines = [
        f"{'Parameter':<30} | {'Slope':>8} | {'Intercept':>10} | {'R²':>6}",
        f"{'-' * 30}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 6}",
    ]
    for name, fit in scaling_laws.items():
        lines.append(
            f"{name:<30} | {fit['slope']:>8.4f} | "
            f"{fit['intercept']:>10.4f} | {fit['r_squared']:>6.3f}"
        )
    return "\n".join(lines)
