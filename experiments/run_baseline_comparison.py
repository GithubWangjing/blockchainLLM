"""Centralized/no-blockchain baseline versus MedChainLLM logging workflows."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reviewer_experiment_utils import (
    RESULTS_DIR,
    TABLES_DIR,
    ensure_output_dirs,
    load_existing_llm_latency,
    load_hardhat_logging_latency,
    save_figure,
    style_matplotlib,
    summarize_latency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline/no-blockchain comparison.")
    parser.add_argument("--quick", action="store_true", help="Use a small request count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def simulate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    n = 80 if args.quick else 1000
    llm_mean = load_existing_llm_latency()
    hardhat = load_hardhat_logging_latency()
    configs = [
        ("centralized_no_blockchain", 0.0, 0.0, 0.0, 0.0),
        ("blockchain_logging", 18.0, hardhat["confirmation_latency_ms"], 1.0, 0.0),
        ("blockchain_logging_dynamic_sharding", 20.0, hardhat["confirmation_latency_ms"] * 0.72, 1.0, 0.18),
    ]
    detail_rows = []
    concurrent_levels = [1, 4, 8, 16, 32] if not args.quick else [1, 8, 16]
    concurrency_rows = []
    for name, logging_mean, confirmation_mean, audit_rate, throughput_gain in configs:
        inference = rng.lognormal(mean=np.log(llm_mean), sigma=0.18, size=n)
        logging = rng.lognormal(mean=np.log(max(logging_mean, 1.0)), sigma=0.22, size=n) if logging_mean else np.zeros(n)
        confirmation = (
            rng.lognormal(mean=np.log(max(confirmation_mean, 1.0)), sigma=0.45, size=n)
            if confirmation_mean
            else np.zeros(n)
        )
        total = inference + logging + confirmation
        for value in total:
            pass
        throughput = 1000.0 / float(np.mean(total)) * (1.0 + throughput_gain)
        base_mean = float(np.mean(inference))
        for i in range(n):
            detail_rows.append(
                {
                    "configuration": name,
                    "request_id": i,
                    "inference_latency_ms": inference[i],
                    "blockchain_logging_latency_ms": logging[i],
                    "confirmation_latency_ms": confirmation[i],
                    "end_to_end_latency_ms": total[i],
                    "audit_hash_present": int(audit_rate > 0),
                }
            )
        for c in concurrent_levels:
            queue_factor = 1.0 + max(c - 1, 0) * (0.018 if "dynamic" in name else 0.03)
            concurrency_rows.append(
                {
                    "configuration": name,
                    "concurrent_requests": c,
                    "end_to_end_latency_ms": float(np.mean(total) * queue_factor),
                }
            )
    detail_df = pd.DataFrame(detail_rows)
    summary_rows = []
    baseline_mean = detail_df[detail_df["configuration"] == "centralized_no_blockchain"][
        "end_to_end_latency_ms"
    ].mean()
    for name, group in detail_df.groupby("configuration", sort=False):
        lat = group["end_to_end_latency_ms"].tolist()
        stats = summarize_latency(lat)
        overhead = stats["mean_latency_ms"] - baseline_mean
        summary_rows.append(
            {
                "configuration": name,
                **stats,
                "throughput_qps": 1000.0 / max(stats["mean_latency_ms"], 1e-9),
                "audit_coverage_rate": float(group["audit_hash_present"].mean()),
                "blockchain_overhead_ms": max(float(overhead), 0.0),
                "blockchain_overhead_percent": max(float(overhead / baseline_mean * 100.0), 0.0),
                "benchmark_type": "hybrid_existing_benchmark_plus_simulation",
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(concurrency_rows)


def plot(summary: pd.DataFrame, concurrency: pd.DataFrame) -> list[str]:
    style_matplotlib()
    outputs: list[str] = []
    labels = summary["configuration"].str.replace("_", "\n")
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.35
    ax.bar(x - width / 2, summary["mean_latency_ms"], width, label="Mean", color="#4C78A8")
    ax.bar(x + width / 2, summary["p95_latency_ms"], width, label="P95", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("End-to-End Latency and Blockchain Overhead")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    outputs += save_figure(fig, "fig_baseline_latency_overhead")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(x, summary["mean_latency_ms"] - summary["blockchain_overhead_ms"], label="LLM inference", color="#4C78A8")
    ax.bar(x, summary["blockchain_overhead_ms"], bottom=summary["mean_latency_ms"] - summary["blockchain_overhead_ms"], label="Blockchain logging/confirmation", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Component Breakdown")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    outputs += save_figure(fig, "fig_baseline_latency_breakdown")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name, group in concurrency.groupby("configuration", sort=False):
        ax.plot(group["concurrent_requests"], group["end_to_end_latency_ms"], marker="o", label=name.replace("_", " "))
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title("Latency Under Concurrent Requests")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    outputs += save_figure(fig, "fig_baseline_concurrency_latency")
    return outputs


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    summary, concurrency = simulate(args)
    result_path = RESULTS_DIR / "baseline_comparison.csv"
    table_path = TABLES_DIR / "table_baseline_comparison.csv"
    summary.to_csv(result_path, index=False)
    summary.to_csv(table_path, index=False)
    figures = plot(summary, concurrency)
    print(f"Wrote {result_path}")
    print(f"Wrote {table_path}")
    for path in figures:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

