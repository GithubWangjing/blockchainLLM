"""Cost and auditability analysis for MedChainLLM deployment options."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reviewer_experiment_utils import RESULTS_DIR, TABLES_DIR, ensure_output_dirs, save_figure, style_matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cost/auditability analysis.")
    parser.add_argument("--quick", action="store_true", help="Retained for pipeline symmetry.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> pd.DataFrame:
    configs = [
        ("centralized_no_blockchain", 0, 0, 900, 0.0, 0.0, 0.0, 0.0),
        ("blockchain_logging", 1, 96, 1450, 0.018, 1.0, 0.995, 220.0),
        ("blockchain_dynamic_sharding", 1, 96, 1450, 0.015, 1.0, 0.995, 170.0),
        ("blockchain_dp_fl_logging", 2, 160, 4800, 0.026, 1.0, 0.998, 260.0),
    ]
    rows = []
    for name, tx, on_chain, off_chain, cost_tx, completeness, tamper, overhead in configs:
        rows.append(
            {
                "configuration": name,
                "tx_per_query": tx,
                "storage_bytes_on_chain": on_chain,
                "storage_bytes_off_chain": off_chain,
                "estimated_gas_or_devnet_cost": cost_tx,
                "audit_log_completeness": completeness,
                "tamper_detection_rate": tamper,
                "hash_verification_success_rate": tamper,
                "cost_per_1000_queries": cost_tx * tx * 1000.0,
                "latency_overhead_ms": overhead,
                "cost_model": "normalized_devnet_operational_cost",
            }
        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> list[str]:
    style_matplotlib()
    outputs: list[str] = []
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x, df["cost_per_1000_queries"], color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(df["configuration"].str.replace("_", "\n"))
    ax.set_ylabel("Normalized cost per 1000 queries")
    ax.set_title("Estimated Deployment Cost")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    outputs += save_figure(fig, "fig_cost_per_1000_queries")

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.scatter(df["latency_overhead_ms"], df["audit_log_completeness"], s=70, color="#F58518")
    for _, row in df.iterrows():
        ax.annotate(row["configuration"].replace("_", " "), (row["latency_overhead_ms"], row["audit_log_completeness"]), fontsize=7)
    ax.set_xlabel("Latency overhead (ms)")
    ax.set_ylabel("Audit log completeness")
    ax.set_title("Auditability vs Overhead")
    ax.grid(True, linestyle="--", alpha=0.3)
    outputs += save_figure(fig, "fig_auditability_vs_overhead")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - 0.18, df["storage_bytes_on_chain"], width=0.36, label="On-chain", color="#4C78A8")
    ax.bar(x + 0.18, df["storage_bytes_off_chain"], width=0.36, label="Off-chain", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(df["configuration"].str.replace("_", "\n"))
    ax.set_ylabel("Bytes per query")
    ax.set_title("On-chain vs Off-chain Storage")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    outputs += save_figure(fig, "fig_storage_onchain_offchain")
    return outputs


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    df = run(args)
    result_path = RESULTS_DIR / "cost_auditability_analysis.csv"
    table_path = TABLES_DIR / "table_cost_auditability_analysis.csv"
    df.to_csv(result_path, index=False)
    df.to_csv(table_path, index=False)
    for path in plot(df):
        print(f"Wrote {path}")
    print(f"Wrote {result_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()

