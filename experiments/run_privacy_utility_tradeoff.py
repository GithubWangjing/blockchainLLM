"""Standard Gaussian mechanism privacy-utility tradeoff simulation."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reviewer_experiment_utils import RESULTS_DIR, TABLES_DIR, ensure_output_dirs, gaussian_sigma, save_figure, style_matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gaussian mechanism privacy-utility sweep.")
    parser.add_argument("--quick", action="store_true", help="Retained for pipeline symmetry.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    epsilons = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0]
    tiers = {"low": 0.5, "medium": 1.0, "high": 1.5}
    delta = 1e-5
    base_accuracy = 0.78
    rows = []
    for tier, sensitivity in tiers.items():
        for epsilon in epsilons:
            sigma = gaussian_sigma(epsilon, delta, sensitivity)
            utility_penalty = min(0.48, 0.035 * np.log1p(sigma) + {"low": 0.0, "medium": 0.015, "high": 0.035}[tier])
            acc = max(0.05, base_accuracy - utility_penalty + rng.normal(0, 0.004))
            rows.append(
                {
                    "epsilon": epsilon,
                    "delta": delta,
                    "sigma": sigma,
                    "sensitivity_tier": tier,
                    "simulated_accuracy": acc,
                    "simulated_loss": -np.log(max(acc, 1e-6)),
                    "privacy_overhead": sigma / max(sensitivity, 1e-9),
                    "utility_drop_percent": (base_accuracy - acc) / base_accuracy * 100.0,
                    "benchmark_type": "standard_gaussian_mechanism_simulation",
                }
            )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> list[str]:
    style_matplotlib()
    outputs: list[str] = []
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for tier, group in df.groupby("sensitivity_tier", sort=False):
        ax.plot(group["epsilon"], group["simulated_accuracy"], marker="o", label=tier)
    ax.set_xscale("log")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Simulated accuracy")
    ax.set_title("Privacy-Utility Tradeoff")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Sensitivity")
    outputs += save_figure(fig, "fig_privacy_accuracy_vs_epsilon")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for tier, group in df.groupby("sensitivity_tier", sort=False):
        ax.plot(group["epsilon"], group["sigma"], marker="s", label=tier)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Gaussian noise scale sigma")
    ax.set_title("Noise Scale vs Epsilon")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Sensitivity")
    outputs += save_figure(fig, "fig_privacy_noise_vs_epsilon")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    pivot = df.groupby("sensitivity_tier")["utility_drop_percent"].mean().reindex(["low", "medium", "high"])
    ax.bar(pivot.index, pivot.values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_xlabel("Sensitivity tier")
    ax.set_ylabel("Mean utility drop (%)")
    ax.set_title("Utility Drop by Sensitivity Policy")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    outputs += save_figure(fig, "fig_privacy_utility_by_sensitivity")
    return outputs


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    df = run(args)
    result_path = RESULTS_DIR / "privacy_utility_tradeoff.csv"
    table_path = TABLES_DIR / "table_privacy_utility_tradeoff.csv"
    df.to_csv(result_path, index=False)
    df.to_csv(table_path, index=False)
    for path in plot(df):
        print(f"Wrote {path}")
    print(f"Wrote {result_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()

