"""MedQA workflow comparison with and without blockchain audit logging."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reviewer_experiment_utils import PROJECT_ROOT, RESULTS_DIR, TABLES_DIR, ensure_output_dirs, load_hardhat_logging_latency, load_medqa_existing, save_figure, style_matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedQA workflow comparison.")
    parser.add_argument("--quick", action="store_true", help="Use fewer simulated questions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    existing = load_medqa_existing()
    hardhat = load_hardhat_logging_latency()
    n = min(int(existing["num_questions"]), 30) if args.quick else int(existing["num_questions"])
    workflows = [
        ("qwen_only", 0.0, 0.0, 0.0),
        ("qwen_with_blockchain_logging", 18.0, hardhat["confirmation_latency_ms"], 1.0),
        ("qwen_with_audit_and_human_review_flag", 18.0, hardhat["confirmation_latency_ms"], 1.0),
    ]
    detail_rows = []
    summary_rows = []
    # Reuse the same Qwen answer outcomes for every workflow: blockchain
    # integration changes traceability and latency, not backbone accuracy.
    detail_path = PROJECT_ROOT / "prototypes" / "medical_task_eval" / "results" / "medqa_detail.csv"
    if detail_path.exists():
        detail = pd.read_csv(detail_path)
        qwen_correct = detail["is_correct"].astype(int).to_numpy()[:n]
        if len(qwen_correct) < n:
            qwen_correct = np.pad(qwen_correct, (0, n - len(qwen_correct)), constant_values=0)
    else:
        qwen_correct = rng.binomial(1, existing["accuracy"], n)
    base_confidence = rng.beta(2.0 + qwen_correct, 3.0, n)
    for name, log_mean, confirm_mean, audit_rate in workflows:
        inference = rng.lognormal(np.log(existing["mean_latency_ms"]), 0.24, n)
        logging = rng.lognormal(np.log(max(log_mean, 1.0)), 0.25, n) if log_mean else np.zeros(n)
        confirmation = rng.lognormal(np.log(max(confirm_mean, 1.0)), 0.45, n) if confirm_mean else np.zeros(n)
        total = inference + logging + confirmation
        correct = qwen_correct
        confidence = base_confidence
        review_flag = ((confidence < 0.34) | (total > np.percentile(total, 90))).astype(int) if "human_review" in name else np.zeros(n, dtype=int)
        for i in range(n):
            detail_rows.append(
                {
                    "workflow": name,
                    "question_id": f"medqa_{i:05d}",
                    "inference_latency_ms": inference[i],
                    "blockchain_logging_latency_ms": logging[i],
                    "confirmation_latency_ms": confirmation[i],
                    "end_to_end_latency_ms": total[i],
                    "is_correct": correct[i],
                    "confidence_score": confidence[i],
                    "review_flag": review_flag[i],
                    "audit_hash_present": int(audit_rate > 0),
                    "failed_case_type": "wrong_answer" if not correct[i] else "none",
                }
            )
        summary_rows.append(
            {
                "workflow": name,
                "accuracy": float(np.mean(correct)),
                "mean_latency_ms": float(np.mean(total)),
                "p95_latency_ms": float(np.percentile(total, 95)),
                "blockchain_overhead_ms": float(np.mean(logging + confirmation)),
                "audit_coverage_rate": audit_rate,
                "review_flag_rate": float(np.mean(review_flag)),
                "error_count": int(n - np.sum(correct)),
                "benchmark_type": "uses_existing_medqa_accuracy_latency_with_simulated_audit_overhead",
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def plot(summary: pd.DataFrame, detail: pd.DataFrame) -> list[str]:
    style_matplotlib()
    outputs: list[str] = []
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for workflow, group in detail.groupby("workflow", sort=False):
        ax.hist(group["end_to_end_latency_ms"], bins=18, alpha=0.45, label=workflow.replace("_", " "))
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Questions")
    ax.set_title("MedQA Latency Distribution")
    ax.legend()
    outputs += save_figure(fig, "fig_medqa_latency_distribution")

    x = np.arange(len(summary))
    fig, ax1 = plt.subplots(figsize=(6.4, 3.8))
    ax1.bar(x - 0.18, summary["accuracy"] * 100.0, width=0.36, label="Accuracy", color="#4C78A8")
    ax1.bar(x + 0.18, summary["audit_coverage_rate"] * 100.0, width=0.36, label="Audit coverage", color="#54A24B")
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["workflow"].str.replace("_", "\n"))
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Accuracy and Auditability")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax1.legend()
    outputs += save_figure(fig, "fig_medqa_accuracy_latency_tradeoff")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x, summary["mean_latency_ms"] - summary["blockchain_overhead_ms"], label="LLM inference", color="#4C78A8")
    ax.bar(x, summary["blockchain_overhead_ms"], bottom=summary["mean_latency_ms"] - summary["blockchain_overhead_ms"], label="Audit logging", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["workflow"].str.replace("_", "\n"))
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("MedQA Workflow Overhead Breakdown")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    outputs += save_figure(fig, "fig_medqa_overhead_breakdown")
    return outputs


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    summary, detail = run(args)
    result_path = RESULTS_DIR / "medqa_workflow_comparison.csv"
    detail_path = RESULTS_DIR / "medqa_workflow_detail.csv"
    table_path = TABLES_DIR / "table_medqa_workflow_comparison.csv"
    summary.to_csv(result_path, index=False)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(table_path, index=False)
    for path in plot(summary, detail):
        print(f"Wrote {path}")
    print(f"Wrote {result_path}")
    print(f"Wrote {detail_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
