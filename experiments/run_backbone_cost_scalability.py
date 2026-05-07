"""Backbone cost, latency, auditability, and scalability tradeoff analysis."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reviewer_experiment_utils import RESULTS_DIR, TABLES_DIR, ensure_output_dirs, save_figure, style_matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze backbone normalized cost scalability.")
    parser.add_argument("--quick", action="store_true", help="Accepted for pipeline symmetry.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def run() -> pd.DataFrame:
    backbone = pd.read_csv(RESULTS_DIR / "medqa_backbone_comparison.csv")
    cost = pd.read_csv(RESULTS_DIR / "cost_auditability_analysis.csv")
    chain_cost = float(cost.loc[cost["configuration"] == "blockchain_logging", "cost_per_1000_queries"].iloc[0])
    rows = []
    for _, row in backbone.iterrows():
        size = float(str(row["parameter_size"]).replace("B", ""))
        inference_cost = size * 6.0
        logging_cost = chain_cost if row["audit_coverage_rate"] > 0 else 0.0
        rows.append(
            {
                "model_name": row["model_name"],
                "parameter_size": row["parameter_size"],
                "workflow": row["workflow"],
                "accuracy": row["accuracy"],
                "mean_latency_ms": row["mean_latency_ms"],
                "normalized_inference_cost": inference_cost,
                "blockchain_logging_cost": logging_cost,
                "cost_per_1000_queries_normalized": inference_cost + logging_cost,
                "audit_coverage_rate": row["audit_coverage_rate"],
                "benchmark_type": row["benchmark_type"],
                "notes": "Normalized cost is used to compare relative deployment overhead across configurations.",
            }
        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> list[str]:
    style_matplotlib()
    outputs = []
    chain = df[df["workflow"] == "backbone_with_blockchain_logging"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.scatter(chain["cost_per_1000_queries_normalized"], chain["accuracy"] * 100.0, s=80, color="#4C78A8")
    for _, row in chain.iterrows():
        ax.annotate(row["parameter_size"], (row["cost_per_1000_queries_normalized"], row["accuracy"] * 100.0), fontsize=8)
    ax.set_xlabel("Normalized cost per 1000 queries")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Backbone Cost-Accuracy Tradeoff")
    ax.grid(True, linestyle="--", alpha=0.3)
    outputs += save_figure(fig, "fig_backbone_cost_accuracy_tradeoff")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.scatter(chain["cost_per_1000_queries_normalized"], chain["mean_latency_ms"], s=80, color="#F58518")
    for _, row in chain.iterrows():
        ax.annotate(row["parameter_size"], (row["cost_per_1000_queries_normalized"], row["mean_latency_ms"]), fontsize=8)
    ax.set_xlabel("Normalized cost per 1000 queries")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Backbone Cost-Latency Tradeoff")
    ax.grid(True, linestyle="--", alpha=0.3)
    outputs += save_figure(fig, "fig_backbone_cost_latency_tradeoff")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.scatter(chain["cost_per_1000_queries_normalized"], chain["audit_coverage_rate"], s=80, color="#54A24B")
    for _, row in chain.iterrows():
        ax.annotate(row["parameter_size"], (row["cost_per_1000_queries_normalized"], row["audit_coverage_rate"]), fontsize=8)
    ax.set_xlabel("Normalized cost per 1000 queries")
    ax.set_ylabel("Audit coverage")
    ax.set_title("Cost-Auditability Tradeoff")
    ax.grid(True, linestyle="--", alpha=0.3)
    outputs += save_figure(fig, "fig_backbone_cost_auditability_tradeoff")
    return outputs


def main() -> None:
    parse_args()
    ensure_output_dirs()
    df = run()
    result_path = RESULTS_DIR / "backbone_cost_scalability.csv"
    table_path = TABLES_DIR / "table_backbone_cost_scalability.csv"
    df.to_csv(result_path, index=False)
    df.to_csv(table_path, index=False)
    for path in plot(df):
        print("Wrote %s" % path)
    print("Wrote %s" % result_path)
    print("Wrote %s" % table_path)


if __name__ == "__main__":
    main()

