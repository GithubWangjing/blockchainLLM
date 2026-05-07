"""Command-line entry point for Experiment 1 (dynamic sharding)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg", force=True)  # headless backend for CLI use
import matplotlib.pyplot as plt
import pandas as pd

from experiments.common.io_utils import ensure_dir, timestamp_slug
from experiments.e1_sharding.sharding_simulation import (
    DynamicShardParams,
    ShardingSimulation,
    SimulationParameters,
    SimulationResult,
    Strategy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedChainLLM E1 sharding simulations.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/e1_default.json"),
        help="Path to a JSON config file with simulation parameters.",
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
        help="Override RNG seed (defaults to value in the config).",
    )
    return parser.parse_args()


def load_parameters(config_path: Path) -> Tuple[SimulationParameters, Dict]:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    dyn = raw["simulation"]["dynamic"]
    dynamic_params = DynamicShardParams(
        min_shards=dyn["min_shards"],
        max_shards=dyn["max_shards"],
        scale_up_threshold=dyn["scale_up_threshold"],
        scale_down_threshold=dyn["scale_down_threshold"],
        grace_period=dyn["grace_period"],
    )

    params = SimulationParameters(
        duration=raw["simulation"]["duration"],
        time_step=raw["simulation"]["time_step"],
        block_time=raw["simulation"]["block_time"],
        block_size=raw["simulation"]["block_size"],
        consensus_delay=raw["simulation"]["consensus_delay"],
        base_latency_overhead=raw["simulation"]["base_latency_overhead"],
        static_shards=raw["simulation"]["static_shards"],
        dynamic=dynamic_params,
    )

    return params, raw


def run_sweeps(
    params: SimulationParameters,
    config: Dict,
    seed: int,
) -> Tuple[List[SimulationResult], Dict[Tuple[Strategy, float, int], SimulationResult]]:
    results: List[SimulationResult] = []
    index: Dict[Tuple[Strategy, float, int], SimulationResult] = {}
    run_id = 0

    for arrival_rate in config["sweeps"]["arrival_rates"]:
        for node_count in config["sweeps"]["node_counts"]:
            for strategy in ("none", "static", "dynamic"):
                run_seed = seed + run_id
                sim = ShardingSimulation(params=params, seed=run_seed)
                result = sim.run(arrival_rate=arrival_rate, node_count=node_count, strategy=strategy)
                results.append(result)
                index[(strategy, arrival_rate, node_count)] = result
                run_id += 1

    return results, index


def build_plots(
    df: pd.DataFrame,
    index: Dict[Tuple[Strategy, float, int], SimulationResult],
    config: Dict,
    plots_dir: Path,
) -> List[Path]:
    plot_paths: List[Path] = []
    arrival_rates = config["sweeps"]["arrival_rates"]
    node_counts = config["sweeps"]["node_counts"]
    ref_nodes = node_counts[len(node_counts) // 2]
    ref_arrival = arrival_rates[len(arrival_rates) // 2]

    # Throughput vs arrival rate (fixed node count).
    throughput_fig, ax = plt.subplots(figsize=(6, 4))
    for strategy in ("none", "static", "dynamic"):
        subset = df[(df["node_count"] == ref_nodes) & (df["strategy"] == strategy)]
        subset = subset.sort_values("arrival_rate")
        ax.plot(subset["arrival_rate"], subset["throughput_tps"], marker="o", label=strategy.title())
    ax.set_xlabel("Arrival rate (tx/sec)")
    ax.set_ylabel("Throughput (TPS)")
    ax.set_title(f"Throughput vs load (node count = {ref_nodes})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    path_tps = plots_dir / "e1_tps_vs_arrival.png"
    throughput_fig.tight_layout()
    throughput_fig.savefig(path_tps, dpi=300)
    plt.close(throughput_fig)
    plot_paths.append(path_tps)

    # Latency vs node count (fixed arrival rate).
    latency_fig, ax = plt.subplots(figsize=(6, 4))
    for strategy in ("none", "static", "dynamic"):
        subset = df[(df["arrival_rate"] == ref_arrival) & (df["strategy"] == strategy)]
        subset = subset.sort_values("node_count")
        ax.plot(subset["node_count"], subset["avg_latency"], marker="s", label=strategy.title())
    ax.set_xlabel("Node count")
    ax.set_ylabel("Average confirmation latency (s)")
    ax.set_title(f"Latency vs membership (arrival rate = {ref_arrival} tx/s)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    path_latency = plots_dir / "e1_latency_vs_nodes.png"
    latency_fig.tight_layout()
    latency_fig.savefig(path_latency, dpi=300)
    plt.close(latency_fig)
    plot_paths.append(path_latency)

    # Load variance vs time for the heaviest scenario.
    rep_arrival = max(arrival_rates)
    rep_nodes = max(node_counts)
    load_fig, ax = plt.subplots(figsize=(6, 4))
    time_axis = None
    for strategy in ("none", "static", "dynamic"):
        result = index[(strategy, rep_arrival, rep_nodes)]
        if time_axis is None:
            time_axis = (
                pd.Series(range(len(result.load_variance_series))) * config["simulation"]["time_step"]
            )
        ax.plot(time_axis, result.load_variance_series, label=strategy.title())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue length variance")
    ax.set_title("Shard load variance over time")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    path_load = plots_dir / "e1_load_variance_vs_time.png"
    load_fig.tight_layout()
    load_fig.savefig(path_load, dpi=300)
    plt.close(load_fig)
    plot_paths.append(path_load)

    return plot_paths


def main() -> None:
    args = parse_args()
    params, config = load_parameters(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 0)

    results_dir = ensure_dir(args.output_root / "results")
    plots_dir = ensure_dir(args.output_root / "plots")
    timestamp = timestamp_slug()

    results, index = run_sweeps(params=params, config=config, seed=seed)
    df = pd.DataFrame([result.to_record() for result in results])

    csv_path = results_dir / f"e1_sharding_metrics_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    plot_paths = build_plots(df=df, index=index, config=config, plots_dir=plots_dir)

    print("E1 dynamic sharding simulation complete.")
    print(f"Metrics CSV: {csv_path}")
    for path in plot_paths:
        print(f"Figure saved: {path}")


if __name__ == "__main__":
    main()

