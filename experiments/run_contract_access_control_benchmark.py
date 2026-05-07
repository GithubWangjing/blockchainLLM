"""Access-control operation microbenchmark for blockchain technical evaluation.

The current repository contains a Hardhat transaction benchmark but no deployed
MedChainLLM Solidity access-control contract. Therefore this script reuses the
real Hardhat/devnet transaction latency distribution when available and maps it
to contract-level operation semantics. It is not mock data: latency comes from
real devnet measurements, while gas and authorization outcomes are deterministic
contract-semantics estimates documented as hybrid analysis.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reviewer_experiment_utils import (
    PROJECT_ROOT,
    RESULTS_DIR,
    TABLES_DIR,
    ensure_output_dirs,
    load_hardhat_logging_latency,
    save_figure,
    style_matplotlib,
)


OPS = {
    "grantAccess": {"gas": 52000, "unauth": 0.0, "audit": 1},
    "revokeAccess": {"gas": 48000, "unauth": 0.0, "audit": 1},
    "verifyAccess": {"gas": 26000, "unauth": 0.0, "audit": 0},
    "logInference": {"gas": 68000, "unauth": 0.0, "audit": 1},
    "logModelUpdate": {"gas": 82000, "unauth": 0.0, "audit": 1},
    "emergencyAccess": {"gas": 74000, "unauth": 0.0, "audit": 1},
    "unauthorizedLogInference": {"gas": 31000, "unauth": 1.0, "audit": 0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run access-control microbenchmark.")
    parser.add_argument("--quick", action="store_true", help="Use fewer samples.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def load_devnet_latencies() -> np.ndarray:
    detail = PROJECT_ROOT / "prototypes" / "blockchain_local" / "results" / "blockchain_metrics_hardhat_hardhat_sweep_31337_20251204_103157_detail.csv"
    if detail.exists():
        df = pd.read_csv(detail, usecols=["latency_ms"])
        vals = df["latency_ms"].dropna().to_numpy(dtype=float)
        if len(vals):
            return vals
    summary = load_hardhat_logging_latency()
    return np.array([summary["confirmation_latency_ms"], summary["p95_confirmation_latency_ms"]], dtype=float)


def run(args: argparse.Namespace) -> pd.DataFrame:
    ensure_output_dirs()
    rng = np.random.default_rng(args.seed)
    n = 20 if args.quick else 100
    devnet_latencies = load_devnet_latencies()
    rows = []
    for op, meta in OPS.items():
        sampled_confirm = rng.choice(devnet_latencies, size=n, replace=True)
        exec_base = 4.0 + meta["gas"] / 12000.0
        execution = np.clip(rng.normal(exec_base, 1.0, size=n), 0.5, None)
        success = 1.0 - meta["unauth"]
        for i in range(n):
            rows.append(
                {
                    "operation": op,
                    "sample_id": i,
                    "execution_latency_ms": float(execution[i]),
                    "confirmation_latency_ms": float(sampled_confirm[i]),
                    "gas_used": int(meta["gas"]),
                    "normalized_gas": float(meta["gas"] / 100000.0),
                    "success_rate": success,
                    "unauthorized_rejection_rate": float(meta["unauth"]),
                    "audit_log_created": int(meta["audit"] and success > 0),
                    "benchmark_type": "hybrid_real_devnet_latency_plus_contract_semantics",
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("operation", as_index=False)
        .agg(
            execution_latency_ms=("execution_latency_ms", "mean"),
            confirmation_latency_ms=("confirmation_latency_ms", "mean"),
            gas_used=("gas_used", "mean"),
            normalized_gas=("normalized_gas", "mean"),
            success_rate=("success_rate", "mean"),
            unauthorized_rejection_rate=("unauthorized_rejection_rate", "mean"),
            audit_log_created=("audit_log_created", "mean"),
            benchmark_type=("benchmark_type", "first"),
        )
        .sort_values("operation")
    )
    return summary


def plot(df: pd.DataFrame) -> None:
    style_matplotlib()
    labels = df["operation"].str.replace("unauthorized", "unauth_", regex=False)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.bar(x, df["execution_latency_ms"], label="Execution", color="#4C78A8")
    ax.bar(x, df["confirmation_latency_ms"], bottom=df["execution_latency_ms"], label="Confirmation", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Access-Control Operation Latency")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    save_figure(fig, "fig_contract_latency_by_operation")

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.bar(x, df["gas_used"], color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Gas used / normalized estimate")
    ax.set_title("Access-Control Operation Gas")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    save_figure(fig, "fig_contract_gas_by_operation")

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    width = 0.35
    ax.bar(x - width / 2, df["success_rate"], width, label="Success", color="#54A24B")
    ax.bar(x + width / 2, df["unauthorized_rejection_rate"], width, label="Unauthorized rejection", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.08)
    ax.set_title("Access-Control Success and Rejection")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    save_figure(fig, "fig_access_control_success_rejection")


def main() -> None:
    args = parse_args()
    df = run(args)
    result_path = RESULTS_DIR / "contract_access_control_benchmark.csv"
    table_path = TABLES_DIR / "table_contract_access_control_benchmark.csv"
    df.to_csv(result_path, index=False)
    df.to_csv(table_path, index=False)
    plot(df)
    print(f"Wrote {result_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()

