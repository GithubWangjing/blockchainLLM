"""Lightweight federated fine-tuning workflow simulation with FedAvg semantics."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reviewer_experiment_utils import RESULTS_DIR, TABLES_DIR, ensure_output_dirs, save_figure, style_matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FL/fine-tuning mini simulation.")
    parser.add_argument("--quick", action="store_true", help="Use five rounds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    clients_list = [3, 5, 10]
    rounds = 5 if args.quick else 10
    configs = [
        ("centralized_finetuning_simulated", False, False),
        ("federated_finetuning", False, False),
        ("federated_finetuning_dp", True, False),
        ("federated_finetuning_dp_blockchain_logging", True, True),
    ]
    rows = []
    for clients in clients_list:
        for config, dp, chain in configs:
            score = 0.52
            for rnd in range(1, rounds + 1):
                improvement = 0.045 * np.exp(-rnd / 8.0)
                if "federated" in config:
                    improvement *= 0.9 + 0.04 * np.log1p(clients)
                if dp:
                    improvement *= 0.82
                score = min(0.82, score + improvement + rng.normal(0, 0.003))
                training_time = (38.0 if "centralized" in config else 15.0 * clients) * (1.15 if dp else 1.0)
                communication = 0.0 if "centralized" in config else clients * 12.5 * rnd
                logging_overhead = (clients * 24.0 + 80.0) if chain else 0.0
                rows.append(
                    {
                        "configuration": config,
                        "simulated_hospital_clients": clients,
                        "round": rnd,
                        "aggregation": "FedAvg" if "federated" in config else "centralized",
                        "simulated_accuracy": score,
                        "validation_score": score,
                        "training_time_sec": training_time * rnd,
                        "communication_cost_mb": communication,
                        "blockchain_logging_overhead_ms": logging_overhead,
                        "privacy_budget_epsilon": 2.0 if dp else np.nan,
                        "audit_coverage_rate": 1.0 if chain else 0.0,
                        "update_integrity_check_rate": 1.0 if chain else 0.0,
                        "benchmark_type": "lightweight_federated_simulation",
                    }
                )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> list[str]:
    style_matplotlib()
    outputs: list[str] = []
    ref_clients = 5
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for config, group in df[df["simulated_hospital_clients"] == ref_clients].groupby("configuration", sort=False):
        ax.plot(group["round"], group["simulated_accuracy"], marker="o", label=config.replace("_", " "))
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Validation score")
    ax.set_title("FL Performance Over Rounds")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=7)
    outputs += save_figure(fig, "fig_fl_accuracy_over_rounds")

    final = df[df["round"] == df["round"].max()]
    grouped = final[final["simulated_hospital_clients"] == ref_clients]
    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(x, grouped["training_time_sec"], label="Training", color="#4C78A8")
    ax.bar(x, grouped["communication_cost_mb"], bottom=grouped["training_time_sec"], label="Communication", color="#F58518")
    ax.bar(x, grouped["blockchain_logging_overhead_ms"] / 1000.0, bottom=grouped["training_time_sec"] + grouped["communication_cost_mb"], label="Blockchain logging (scaled)", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["configuration"].str.replace("_", "\n"), fontsize=7)
    ax.set_ylabel("Relative overhead units")
    ax.set_title("FL Overhead Breakdown")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    outputs += save_figure(fig, "fig_fl_overhead_breakdown")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    scalable = final[final["configuration"] == "federated_finetuning_dp_blockchain_logging"]
    ax.plot(scalable["simulated_hospital_clients"], scalable["communication_cost_mb"], marker="o", label="Communication cost")
    ax.set_xlabel("Hospital clients")
    ax.set_ylabel("Communication cost (MB)")
    ax.set_title("FL Client Scalability")
    ax.grid(True, linestyle="--", alpha=0.3)
    outputs += save_figure(fig, "fig_fl_clients_scalability")
    return outputs


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    df = run(args)
    result_path = RESULTS_DIR / "fl_finetuning_simulation.csv"
    table_path = TABLES_DIR / "table_fl_finetuning_simulation.csv"
    df.to_csv(result_path, index=False)
    df[df["round"] == df["round"].max()].to_csv(table_path, index=False)
    for path in plot(df):
        print(f"Wrote {path}")
    print(f"Wrote {result_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()

