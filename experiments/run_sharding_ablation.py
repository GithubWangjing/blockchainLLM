"""Static sharding, dynamic sharding, and unsharded blockchain simulation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reviewer_experiment_utils import RESULTS_DIR, TABLES_DIR, ensure_output_dirs, save_figure, style_matplotlib


@dataclass
class RunResult:
    strategy: str
    offered_load_tps: int
    validator_count: int
    static_shard_count: int
    throughput_tps: float
    mean_confirmation_latency_ms: float
    p95_confirmation_latency_ms: float
    queue_length_mean: float
    shard_count_mean: float
    shard_reconfiguration_count: int
    overload_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sharding ablation simulation.")
    parser.add_argument("--quick", action="store_true", help="Use fewer parameter combinations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def simulate_one(rng: np.random.Generator, strategy: str, load: int, validators: int, static_shards: int, quick: bool) -> tuple[RunResult, pd.DataFrame]:
    steps = 60 if quick else 240
    dt = 0.5
    min_shards, max_shards = 1, 12
    shards = 1 if strategy == "unsharded_baseline" else static_shards if strategy == "static_sharding" else 2
    queues = np.zeros(shards)
    high, low = 28.0, 5.0
    reconfigs = 0
    processed_total = 0.0
    latency_samples = []
    time_rows = []
    queue_means = []
    overload_steps = 0
    for step in range(steps):
        t = step * dt
        arrival_multiplier = 1.0 + (0.45 if strategy == "dynamic_sharding" and step > steps * 0.55 else 0.0)
        arrivals = rng.poisson(load * dt * arrival_multiplier)
        for _ in range(arrivals):
            queues[int(np.argmin(queues))] += 1
        validator_penalty = 1.0 + 0.035 * np.log2(max(validators, 2) / 4.0)
        cap_per_shard = max(1.0, (38.0 * dt) / validator_penalty)
        if strategy == "dynamic_sharding":
            cap_per_shard *= 0.96
        processed = np.minimum(queues, cap_per_shard)
        queues -= processed
        processed_total += float(processed.sum())
        base_latency = 115.0 * validator_penalty
        for q, p in zip(queues, processed):
            if p > 0:
                latency_samples.extend((base_latency + q * 7.5 + rng.normal(0, 9, int(max(1, p)))).clip(15).tolist())
        avg_q = float(np.mean(queues))
        queue_means.append(avg_q)
        if avg_q > high:
            overload_steps += 1
        if strategy == "dynamic_sharding":
            if avg_q > high and len(queues) < max_shards:
                queues = np.append(queues * len(queues) / (len(queues) + 1), 0.0)
                reconfigs += 1
            elif avg_q < low and len(queues) > min_shards and step % 6 == 0:
                queues = queues[:-1]
                reconfigs += 1
        time_rows.append({"time_step": step, "time_sec": t, "offered_load_tps": load * arrival_multiplier, "shard_count": len(queues), "queue_length_mean": avg_q, "strategy": strategy})
    duration = steps * dt
    throughput = processed_total / duration
    lat = np.asarray(latency_samples) if latency_samples else np.array([0.0])
    result = RunResult(
        strategy=strategy,
        offered_load_tps=load,
        validator_count=validators,
        static_shard_count=static_shards if strategy == "static_sharding" else 0,
        throughput_tps=float(min(throughput, load * 1.25)),
        mean_confirmation_latency_ms=float(np.mean(lat)),
        p95_confirmation_latency_ms=float(np.percentile(lat, 95)),
        queue_length_mean=float(np.mean(queue_means)),
        shard_count_mean=float(np.mean([r["shard_count"] for r in time_rows])),
        shard_reconfiguration_count=reconfigs,
        overload_rate=float(overload_steps / steps),
    )
    return result, pd.DataFrame(time_rows)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    loads = [50, 150, 300, 500] if args.quick else [50, 100, 150, 200, 300, 400, 500]
    validators = [4, 16, 64] if args.quick else [4, 8, 16, 32, 64]
    static_counts = [4] if args.quick else [2, 4, 8]
    rows = []
    time_dfs = []
    for load in loads:
        for val in validators:
            for strategy in ["unsharded_baseline", "static_sharding", "dynamic_sharding"]:
                counts = static_counts if strategy == "static_sharding" else [0]
                for sc in counts:
                    result, time_df = simulate_one(rng, strategy, load, val, sc or 4, args.quick)
                    rows.append(result.__dict__)
                    if load == max(loads) and val == validators[len(validators) // 2] and strategy == "dynamic_sharding":
                        time_dfs.append(time_df)
    return pd.DataFrame(rows), pd.concat(time_dfs, ignore_index=True)


def plot(df: pd.DataFrame, adaptation: pd.DataFrame) -> list[str]:
    style_matplotlib()
    outputs: list[str] = []
    ref_val = sorted(df["validator_count"].unique())[len(df["validator_count"].unique()) // 2]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for strategy, group in df[df["validator_count"] == ref_val].groupby("strategy", sort=False):
        agg = group.groupby("offered_load_tps")["throughput_tps"].mean().reset_index()
        ax.plot(agg["offered_load_tps"], agg["throughput_tps"], marker="o", label=strategy.replace("_", " "))
    ax.set_xlabel("Offered load (TPS)")
    ax.set_ylabel("Throughput (TPS)")
    ax.set_title("Throughput vs Offered Load")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    outputs += save_figure(fig, "fig_sharding_throughput_vs_load")

    ref_load = sorted(df["offered_load_tps"].unique())[len(df["offered_load_tps"].unique()) // 2]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for strategy, group in df[df["offered_load_tps"] == ref_load].groupby("strategy", sort=False):
        agg = group.groupby("validator_count")["p95_confirmation_latency_ms"].mean().reset_index()
        ax.plot(agg["validator_count"], agg["p95_confirmation_latency_ms"], marker="s", label=strategy.replace("_", " "))
    ax.set_xlabel("Validator count")
    ax.set_ylabel("P95 confirmation latency (ms)")
    ax.set_title("Confirmation Latency vs Validators")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    outputs += save_figure(fig, "fig_sharding_latency_vs_validators")

    fig, ax1 = plt.subplots(figsize=(6.4, 3.8))
    ax1.plot(adaptation["time_sec"], adaptation["shard_count"], color="#4C78A8", label="Shard count")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Shard count")
    ax2 = ax1.twinx()
    ax2.plot(adaptation["time_sec"], adaptation["offered_load_tps"], color="#F58518", alpha=0.7, label="Offered load")
    ax2.set_ylabel("Offered load (TPS)")
    ax1.set_title("Dynamic Shard Adaptation")
    ax1.grid(True, linestyle="--", alpha=0.3)
    outputs += save_figure(fig, "fig_dynamic_shard_adaptation")
    return outputs


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    df, adaptation = run(args)
    result_path = RESULTS_DIR / "sharding_ablation.csv"
    table_path = TABLES_DIR / "table_sharding_ablation.csv"
    df.to_csv(result_path, index=False)
    df.to_csv(table_path, index=False)
    for path in plot(df, adaptation):
        print(f"Wrote {path}")
    print(f"Wrote {result_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()

