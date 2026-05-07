"""Command-line entry point for Experiment 3 (privacy vs utility)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common.io_utils import ensure_dir, timestamp_slug
from experiments.e3_privacy.privacy_simulation import PrivacyParameters, PrivacyResult, PrivacySimulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedChainLLM E3 privacy simulations.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/e3_default.json"),
        help="Path to JSON config.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments"),
        help="Base directory that contains results/ and plots/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override RNG seed (defaults to config).",
    )
    return parser.parse_args()


def load_parameters(config_path: Path) -> Tuple[PrivacyParameters, Dict]:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    params = PrivacyParameters(
        base_accuracy=raw["simulation"]["base_accuracy"],
        base_training_time=raw["simulation"]["base_training_time"],
        alpha_scale=raw["simulation"]["alpha_scale"],
        noise_penalty_scale=raw["simulation"]["noise_penalty_scale"],
        training_time_jitter=raw["simulation"]["training_time_jitter"],
    )
    return params, raw


def run_sweeps(
    params: PrivacyParameters,
    config: Dict,
    seed: int,
) -> List[PrivacyResult]:
    results: List[PrivacyResult] = []
    run_id = 0

    for sensitivity in config["sensitivity_levels"]:
        for encryption in config["encryption_strategies"]:
            for epsilon in config["epsilon_values"]:
                sim = PrivacySimulation(params=params, seed=seed + run_id)
                results.append(sim.run_configuration(sensitivity=sensitivity, encryption=encryption, epsilon=epsilon))
                run_id += 1

    return results


def build_plots(df: pd.DataFrame, config: Dict, plots_dir: Path) -> List[Path]:
    plot_paths: List[Path] = []

    # Figure 1: Accuracy vs epsilon for each sensitivity (averaged across encryption).
    acc_fig, ax = plt.subplots(figsize=(6, 4))
    for sensitivity, group in df.groupby("sensitivity"):
        averaged = (
            group.groupby("epsilon")["accuracy"]
            .mean()
            .reindex(config["epsilon_values"])
        )
        ax.plot(
            averaged.index,
            averaged.values,
            marker="o",
            label=sensitivity.capitalize(),
        )
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs ε across sensitivity tiers")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    path_accuracy = plots_dir / "e3_accuracy_vs_epsilon.png"
    acc_fig.tight_layout()
    acc_fig.savefig(path_accuracy, dpi=300)
    plt.close(acc_fig)
    plot_paths.append(path_accuracy)

    # Figure 2: Training time vs encryption method (averaged across epsilon).
    time_fig, ax = plt.subplots(figsize=(6, 4))
    encryption_methods = config["encryption_strategies"]
    bar_width = 0.22
    base_positions = np.arange(len(encryption_methods))
    grouped = list(df.groupby("sensitivity"))
    for idx, (sensitivity, group) in enumerate(grouped):
        means = [group[group["encryption"] == enc]["training_time"].mean() for enc in encryption_methods]
        offsets = base_positions + (idx - (len(grouped) - 1) / 2) * bar_width
        ax.bar(offsets, means, width=bar_width, label=sensitivity.capitalize())
    ax.set_xticks(base_positions)
    ax.set_xticklabels([enc.upper() for enc in encryption_methods])
    ax.set_ylabel("Training time (hrs, normalized)")
    ax.set_title("Training time vs encryption strategy")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    path_time = plots_dir / "e3_training_time_vs_encryption.png"
    time_fig.tight_layout()
    time_fig.savefig(path_time, dpi=300)
    plt.close(time_fig)
    plot_paths.append(path_time)

    # Figure 3: Privacy risk vs accuracy scatter.
    scatter_fig, ax = plt.subplots(figsize=(6, 4))
    for sensitivity, group in df.groupby("sensitivity"):
        ax.scatter(
            group["privacy_risk_score"],
            group["accuracy"],
            label=sensitivity.capitalize(),
            alpha=0.8,
        )
    ax.set_xlabel("Privacy risk score (1/ε)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Privacy-utility trade-off")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    path_tradeoff = plots_dir / "e3_privacy_risk_tradeoff.png"
    scatter_fig.tight_layout()
    scatter_fig.savefig(path_tradeoff, dpi=300)
    plt.close(scatter_fig)
    plot_paths.append(path_tradeoff)

    return plot_paths


def main() -> None:
    args = parse_args()
    params, config = load_parameters(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 0)

    results_dir = ensure_dir(args.output_root / "results")
    plots_dir = ensure_dir(args.output_root / "plots")
    timestamp = timestamp_slug()

    results = run_sweeps(params=params, config=config, seed=seed)
    df = pd.DataFrame([result.to_record() for result in results])

    csv_path = results_dir / f"e3_privacy_metrics_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    plot_paths = build_plots(df=df, config=config, plots_dir=plots_dir)

    print("E3 privacy-utility simulation complete.")
    print(f"Metrics CSV: {csv_path}")
    for path in plot_paths:
        print(f"Figure saved: {path}")


if __name__ == "__main__":
    main()

