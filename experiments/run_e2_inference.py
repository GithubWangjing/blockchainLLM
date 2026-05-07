"""Command-line entry point for Experiment 2 (decentralized inference)."""

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
from experiments.e2_inference.inference_simulation import (
    InferenceParameters,
    InferenceResult,
    InferenceSimulation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedChainLLM E2 inference simulations.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/e2_default.json"),
        help="Path to the JSON config file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments"),
        help="Base directory where results/ and plots/ live.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override RNG seed (defaults to config value).",
    )
    return parser.parse_args()


def load_parameters(config_path: Path) -> Tuple[InferenceParameters, Dict]:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    params = InferenceParameters(
        base_latency_ms=raw["simulation"]["base_latency_ms"],
        base_accuracy=raw["simulation"]["base_accuracy"],
        verification_overhead_ms=raw["simulation"]["verification_overhead_ms"],
        jitter_std_fraction=raw["simulation"]["jitter_std_fraction"],
        requests_per_config=raw["simulation"]["requests_per_config"],
    )
    return params, raw


def run_sweeps(
    params: InferenceParameters,
    config: Dict,
    seed: int,
) -> List[InferenceResult]:
    results: List[InferenceResult] = []
    run_id = 0

    for pruning in config["sweeps"]["pruning_levels"]:
        for quantization in config["sweeps"]["quantization_modes"]:
            for nodes in config["sweeps"]["edge_nodes"]:
                sim = InferenceSimulation(params=params, seed=seed + run_id)
                results.extend(sim.run_configuration(pruning=pruning, quantization=quantization, num_edge_nodes=nodes))
                run_id += 1

    return results


def build_plots(
    df: pd.DataFrame,
    config: Dict,
    plots_dir: Path,
) -> List[Path]:
    plot_paths: List[Path] = []

    # Figure 1: Latency vs edge nodes for decentralized strategy, grouped by compression config.
    decentralized = df[df["strategy"] == "decentralized"].copy()
    decentralized["compression_label"] = decentralized.apply(
        lambda row: f"prune {int(row['pruning'] * 100)}% + {row['quantization'].upper()}",
        axis=1,
    )

    lat_fig, ax = plt.subplots(figsize=(7, 4))
    for label, group in decentralized.groupby("compression_label"):
        ordered = group.sort_values("num_edge_nodes")
        ax.plot(
            ordered["num_edge_nodes"],
            ordered["mean_latency_ms"],
            marker="o",
            label=label,
        )
    ax.set_xlabel("Edge nodes (count)")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Decentralized latency vs edge scale")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    path_latency = plots_dir / "e2_latency_vs_nodes.png"
    lat_fig.tight_layout()
    lat_fig.savefig(path_latency, dpi=300)
    plt.close(lat_fig)
    plot_paths.append(path_latency)

    # Figure 2: Accuracy heatmap (pruning x quantization) using decentralized results.
    heatmap_source = decentralized[decentralized["num_edge_nodes"] == max(config["sweeps"]["edge_nodes"])]
    pivot = heatmap_source.pivot_table(
        index="pruning",
        columns="quantization",
        values="accuracy",
        aggfunc=np.mean,
    )

    acc_fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=pivot.values.min(), vmax=pivot.values.max())
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([col.upper() for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(idx * 100)}%" for idx in pivot.index])
    ax.set_xlabel("Quantization")
    ax.set_ylabel("Pruning level")
    ax.set_title("Accuracy vs compression (decentralized)")
    for (i, j), value in np.ndenumerate(pivot.values):
        ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white")
    acc_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    path_acc = plots_dir / "e2_accuracy_heatmap.png"
    acc_fig.tight_layout()
    acc_fig.savefig(path_acc, dpi=300)
    plt.close(acc_fig)
    plot_paths.append(path_acc)

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

    csv_path = results_dir / f"e2_inference_metrics_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    plot_paths = build_plots(df=df, config=config, plots_dir=plots_dir)

    print("E2 decentralized inference simulation complete.")
    print(f"Metrics CSV: {csv_path}")
    for path in plot_paths:
        print(f"Figure saved: {path}")


if __name__ == "__main__":
    main()

