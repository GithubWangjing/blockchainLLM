"""Error and invalid-output analysis for MedQA governance review flags."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reviewer_experiment_utils import REPORTS_DIR, RESULTS_DIR, TABLES_DIR, ensure_output_dirs, save_figure, style_matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MedQA invalid outputs and review flags.")
    parser.add_argument("--quick", action="store_true", help="Accepted for pipeline symmetry.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_detail() -> pd.DataFrame:
    path = RESULTS_DIR / "medqa_backbone_detail.csv"
    if path.exists():
        return pd.read_csv(path)
    fallback = RESULTS_DIR / "medqa_workflow_detail.csv"
    df = pd.read_csv(fallback)
    df["model_name"] = "Qwen/Qwen2.5-0.5B"
    df["invalid_output"] = 0
    return df


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_detail()
    review = df[df["workflow"].astype(str).str.contains("review", case=False, na=False)].copy()
    if review.empty:
        review = df.copy()
    if "failure_type" not in review:
        review["failure_type"] = np.where(review["is_correct"].astype(int) == 0, "wrong_answer", "none")
    review["wrong_answer"] = (review["is_correct"].astype(int) == 0).astype(int)
    review["latency_bucket"] = pd.qcut(review["end_to_end_latency_ms"], q=min(4, review["end_to_end_latency_ms"].nunique()), duplicates="drop")
    rows = []
    for model, group in review.groupby("model_name", sort=False):
        invalid = group["invalid_output"].astype(int)
        wrong = group["wrong_answer"].astype(int)
        flag = group["review_flag"].astype(int)
        target_failure = ((invalid == 1) | (wrong == 1)).astype(int)
        corr = group["end_to_end_latency_ms"].corr(wrong)
        common = group[group["failure_type"] != "none"]["failure_type"].mode()
        rows.append(
            {
                "model_name": model,
                "num_questions": int(len(group)),
                "invalid_output_rate": float(invalid.mean()),
                "wrong_answer_rate": float(wrong.mean()),
                "review_flag_rate": float(flag.mean()),
                "review_precision_proxy": float(target_failure[flag == 1].mean()) if int(flag.sum()) else 0.0,
                "latency_failure_correlation": float(corr) if not pd.isna(corr) else 0.0,
                "most_common_failure_type": common.iloc[0] if len(common) else "none",
            }
        )
    return pd.DataFrame(rows), review


def plot(summary: pd.DataFrame, detail: pd.DataFrame) -> list[str]:
    style_matplotlib()
    outputs = []
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(summary["model_name"].str.replace("Qwen/Qwen2.5-", "", regex=False), summary["invalid_output_rate"] * 100.0, color="#4C78A8")
    ax.set_xlabel("Model")
    ax.set_ylabel("Invalid output rate (%)")
    ax.set_title("Invalid Output by Model")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    outputs += save_figure(fig, "fig_medqa_invalid_output_by_model")

    by_type = detail.groupby("failure_type")["review_flag"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(by_type["failure_type"], by_type["review_flag"] * 100.0, color="#F58518")
    ax.set_xlabel("Failure type")
    ax.set_ylabel("Review flag rate (%)")
    ax.set_title("Review Flags by Failure Type")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    outputs += save_figure(fig, "fig_medqa_review_flag_by_failure_type")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    correct = detail[detail["is_correct"].astype(int) == 1]
    wrong = detail[detail["is_correct"].astype(int) == 0]
    ax.hist(correct["end_to_end_latency_ms"], bins=18, alpha=0.55, label="Correct", color="#54A24B")
    ax.hist(wrong["end_to_end_latency_ms"], bins=18, alpha=0.55, label="Wrong/invalid", color="#E45756")
    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("Questions")
    ax.set_title("Latency and Correctness")
    ax.legend()
    outputs += save_figure(fig, "fig_medqa_latency_correctness")
    return outputs


def write_report(summary: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    text = """# MedQA Error Analysis

This analysis treats review flags as a human-in-the-loop governance entry point. The system does not use LLM output directly as clinical decision support.

Blockchain logging records inference metadata, hashes, timestamps, and audit trajectories. It does not guarantee medical correctness and does not remove hallucination risk.

Observed failure categories include invalid answer formatting, wrong answers, low-confidence outputs, and unusually long generation latency. Hallucination and medical reasoning failures require clinician review, retrieval grounding, stronger backbones, and task-specific validation in future work.

Summary table:

{table}
""".format(table=_summary_table_text(summary))
    (REPORTS_DIR / "medqa_error_analysis.md").write_text(text, encoding="utf-8")


def _summary_table_text(summary: pd.DataFrame) -> str:
    try:
        return summary.to_markdown(index=False)
    except Exception:
        return "```csv\n" + summary.to_csv(index=False) + "```"


def main() -> None:
    parse_args()
    ensure_output_dirs()
    summary, detail = run()
    result_path = RESULTS_DIR / "medqa_error_analysis.csv"
    table_path = TABLES_DIR / "table_medqa_error_analysis.csv"
    summary.to_csv(result_path, index=False)
    summary.to_csv(table_path, index=False)
    write_report(summary)
    for path in plot(summary, detail):
        print("Wrote %s" % path)
    print("Wrote %s" % result_path)
    print("Wrote %s" % table_path)


if __name__ == "__main__":
    main()
