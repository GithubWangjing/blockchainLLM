"""Consensus and validator scalability simulation for consortium assumptions."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reviewer_experiment_utils import RESULTS_DIR, TABLES_DIR, ensure_output_dirs, save_figure, style_matplotlib


CONSENSUS = ["poa_like", "raft_like", "pbft_like"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run consensus scalability simulation.")
    parser.add_argument("--quick", action="store_true", help="Use smaller parameter grid.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def simulate(protocol: str, validators: int, load: int, fail: float, malicious: float, rng: np.random.Generator) -> dict:
    if protocol == "poa_like":
        base_capacity = 420.0
        latency = 130.0 + 3.0 * validators
        comm = validators
        fault_tol = 0.20
        byz_tol = 0.05
    elif protocol == "raft_like":
        base_capacity = 360.0
        latency = 160.0 + 5.0 * np.log2(validators) * validators / 4.0
        comm = validators * np.log2(validators)
        fault_tol = 0.49
        byz_tol = 0.0
    else:
        base_capacity = 300.0
        latency = 220.0 + 2.5 * validators + 0.18 * validators * validators
        comm = validators * validators
        fault_tol = 0.33
        byz_tol = 0.33
    failure_penalty = max(fail / max(fault_tol, 1e-6), 0.0)
    malicious_penalty = max(malicious / max(byz_tol, 1e-6), 0.0) if byz_tol > 0 else (3.0 if malicious > 0 else 0.0)
    health = max(0.0, 1.0 - 0.45 * failure_penalty - 0.55 * malicious_penalty)
    capacity = base_capacity * health / (1.0 + 0.01 * max(validators - 4, 0))
    throughput = min(load, capacity) * rng.normal(1.0, 0.015)
    overload = max(load - max(capacity, 1e-6), 0.0) / max(load, 1)
    confirmation = latency * (1.0 + 2.2 * overload + 1.5 * fail + 2.0 * malicious)
    success = np.clip(health * (1.0 - 0.55 * overload), 0.0, 1.0)
    return {
        "consensus_type": protocol,
        "validator_count": validators,
        "offered_load_tps": load,
        "validator_failure_rate": fail,
        "malicious_validator_ratio": malicious,
        "throughput_tps": float(max(throughput, 0.0)),
        "confirmation_latency_ms": float(confirmation),
        "p95_confirmation_latency_ms": float(confirmation * (1.35 + 0.5 * overload)),
        "consensus_success_rate": float(success),
        "failed_tx_rate": float(1.0 - success),
        "finality_delay_ms": float(confirmation * (1.1 if protocol != "pbft_like" else 1.35)),
        "communication_overhead_normalized": float(comm / 16.0),
        "benchmark_type": "consensus_scalability_simulation",
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    ensure_output_dirs()
    rng = np.random.default_rng(args.seed)
    validators = [4, 16, 64] if args.quick else [4, 8, 16, 32, 64]
    loads = [50, 200, 400] if args.quick else [50, 100, 200, 400]
    failures = [0, 0.10, 0.20] if args.quick else [0, 0.05, 0.10, 0.20]
    malicious = [0, 0.10, 0.25] if args.quick else [0, 0.10, 0.25]
    rows = [simulate(c, v, l, f, m, rng) for c in CONSENSUS for v in validators for l in loads for f in failures for m in malicious]
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    style_matplotlib()
    healthy = df[(df["validator_failure_rate"] == 0) & (df["malicious_validator_ratio"] == 0)]
    ref_load = 200 if 200 in healthy["offered_load_tps"].unique() else healthy["offered_load_tps"].median()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for c, group in healthy[healthy["offered_load_tps"] == ref_load].groupby("consensus_type"):
        group = group.sort_values("validator_count")
        ax.plot(group["validator_count"], group["confirmation_latency_ms"], marker="o", label=c)
    ax.set_xlabel("Validator count")
    ax.set_ylabel("Confirmation latency (ms)")
    ax.set_title("Consensus Latency vs Validators")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    save_figure(fig, "fig_consensus_latency_vs_validators")

    stress = df[(df["offered_load_tps"] == 200) & (df["malicious_validator_ratio"] == 0)]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for c, group in stress.groupby("consensus_type"):
        agg = group.groupby("validator_failure_rate")["consensus_success_rate"].mean().reset_index()
        ax.plot(agg["validator_failure_rate"], agg["consensus_success_rate"], marker="s", label=c)
    ax.set_xlabel("Validator failure rate")
    ax.set_ylabel("Consensus success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Consensus Success Under Failures")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    save_figure(fig, "fig_consensus_success_under_failures")

    ref_validators = 16 if 16 in healthy["validator_count"].unique() else healthy["validator_count"].median()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for c, group in healthy[healthy["validator_count"] == ref_validators].groupby("consensus_type"):
        group = group.sort_values("offered_load_tps")
        ax.plot(group["offered_load_tps"], group["throughput_tps"], marker="o", label=c)
    ax.set_xlabel("Offered load (TPS)")
    ax.set_ylabel("Throughput (TPS)")
    ax.set_title("Consensus Throughput vs Load")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    save_figure(fig, "fig_consensus_throughput_vs_load")


def main() -> None:
    args = parse_args()
    df = run(args)
    result_path = RESULTS_DIR / "consensus_scalability_simulation.csv"
    table_path = TABLES_DIR / "table_consensus_scalability_simulation.csv"
    df.to_csv(result_path, index=False)
    df.to_csv(table_path, index=False)
    plot(df)
    print(f"Wrote {result_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()

