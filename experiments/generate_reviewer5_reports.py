"""Generate Reviewer 5 extension summary, manifest, and response mapping."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from experiments.reviewer_experiment_utils import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR, RESULTS_DIR, TABLES_DIR, ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Reviewer 5 extension reports.")
    parser.add_argument("--quick", action="store_true", help="Accepted for pipeline symmetry.")
    parser.add_argument("--seed", type=int, default=42, help="Seed recorded in report.")
    return parser.parse_args()


def rel(path):
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def collect_manifest():
    return {
        "results": sorted(rel(p) for p in RESULTS_DIR.glob("*.csv")),
        "tables": sorted(rel(p) for p in TABLES_DIR.glob("*.csv")),
        "figures": sorted(rel(p) for p in FIGURES_DIR.glob("*.*") if p.suffix.lower() in [".png", ".pdf", ".svg"]),
        "reports": sorted(rel(p) for p in REPORTS_DIR.glob("*.md")),
    }


def generate_mapping():
    rows = [
        {
            "reviewer5_comment": "Benchmark too limited to a single lightweight model.",
            "required_revision": "Compare multiple backbone sizes and separate backbone quality from blockchain workflow effects.",
            "new_experiment_or_text_revision": "experiments/run_medqa_backbone_comparison.py",
            "output_files": "results/medqa_backbone_comparison.csv; figures/fig_backbone_accuracy_latency_tradeoff.*",
            "manuscript_location": "Main Figure 3; MedQA workflow evaluation",
            "interpretation": "Absolute accuracy is backbone-dependent; blockchain preserves predictions while adding auditability.",
            "limitation_statement": "Reported backbone rows are real evaluations; unevaluated larger models should remain excluded until real predictions are available.",
        },
        {
            "reviewer5_comment": "No baseline.",
            "required_revision": "Report no-blockchain versus blockchain accuracy, latency, cost, and auditability.",
            "new_experiment_or_text_revision": "Backbone workflows plus prior baseline_comparison.",
            "output_files": "results/baseline_comparison.csv; results/medqa_backbone_comparison.csv",
            "manuscript_location": "Main Table 2",
            "interpretation": "Blockchain increases latency/cost but adds full audit coverage in the modeled workflow.",
            "limitation_statement": "Prototype overhead should be remeasured on production consortium infrastructure.",
        },
        {
            "reviewer5_comment": "No ablation.",
            "required_revision": "Keep previous sharding and workflow ablations and connect them to backbone study.",
            "new_experiment_or_text_revision": "No new code; integrate results/sharding_ablation.csv and backbone comparison.",
            "output_files": "results/sharding_ablation.csv; figures/fig_sharding_*",
            "manuscript_location": "Main Figure 2; Supplementary ablation",
            "interpretation": "Dynamic sharding improves scaling trends independent of LLM backbone.",
            "limitation_statement": "Sharding evidence remains simulation unless validated on a larger devnet.",
        },
        {
            "reviewer5_comment": "No real-world deployment metrics.",
            "required_revision": "Add end-to-end latency, throughput/cost proxies, audit coverage, and review flags.",
            "new_experiment_or_text_revision": "experiments/run_backbone_cost_scalability.py",
            "output_files": "results/backbone_cost_scalability.csv; figures/fig_backbone_cost_*",
            "manuscript_location": "Main Table 3",
            "interpretation": "Normalized deployment metrics show the cost-latency-accuracy frontier.",
            "limitation_statement": "Normalized cost is not a dollar estimate.",
        },
        {
            "reviewer5_comment": "No error analysis.",
            "required_revision": "Analyze invalid outputs, wrong answers, latency failures, and review flag behavior.",
            "new_experiment_or_text_revision": "experiments/run_medqa_error_analysis.py",
            "output_files": "results/medqa_error_analysis.csv; reports/medqa_error_analysis.md",
            "manuscript_location": "Main Figure 4 or Supplementary error analysis",
            "interpretation": "Review flags provide governance entry points for unreliable outputs.",
            "limitation_statement": "Review flag is a proxy and is not clinician adjudication.",
        },
        {
            "reviewer5_comment": "No cost analysis.",
            "required_revision": "Report normalized inference and blockchain logging cost across backbones.",
            "new_experiment_or_text_revision": "experiments/run_backbone_cost_scalability.py",
            "output_files": "results/backbone_cost_scalability.csv; figures/fig_backbone_cost_auditability_tradeoff.*",
            "manuscript_location": "Main Table 3",
            "interpretation": "Blockchain cost is bounded and becomes a smaller percentage as model inference dominates.",
            "limitation_statement": "Use normalized cost only unless real cloud pricing is added.",
        },
        {
            "reviewer5_comment": "LLM hallucination risk.",
            "required_revision": "Clarify that blockchain does not guarantee correctness and add human review flags.",
            "new_experiment_or_text_revision": "reports/medqa_error_analysis.md and manuscript text.",
            "output_files": "figures/fig_medqa_review_flag_by_failure_type.*",
            "manuscript_location": "Limitations and governance subsection",
            "interpretation": "Audit logs support traceability; hallucination requires review, grounding, and stronger backbones.",
            "limitation_statement": "No result is clinical validation.",
        },
        {
            "reviewer5_comment": "Why blockchain.",
            "required_revision": "Quantify auditability benefit against latency/cost overhead.",
            "new_experiment_or_text_revision": "Backbone comparison and cost/auditability analysis.",
            "output_files": "figures/fig_backbone_auditability_overhead.*; results/cost_auditability_analysis.csv",
            "manuscript_location": "System evaluation and discussion",
            "interpretation": "Blockchain adds tamper-evident logging and audit coverage, not higher answer accuracy.",
            "limitation_statement": "Requires governance of validators and key management.",
        },
        {
            "reviewer5_comment": "Scalability.",
            "required_revision": "Connect sharding scalability with backbone cost/latency scaling.",
            "new_experiment_or_text_revision": "experiments/run_backbone_cost_scalability.py plus prior sharding ablation.",
            "output_files": "results/backbone_cost_scalability.csv; results/sharding_ablation.csv",
            "manuscript_location": "Main Figure 2 and Supplementary scalability",
            "interpretation": "Backbone and blockchain scaling are separate bottlenecks that can be reported jointly.",
            "limitation_statement": "Large-scale figures are simulation/hybrid unless deployed on real hospital infrastructure.",
        },
        {
            "reviewer5_comment": "Limitations.",
            "required_revision": "Explicitly mark real, hybrid, simulation, and mock outputs.",
            "new_experiment_or_text_revision": "reports/experiment_summary_v2.md",
            "output_files": "results/experiment_manifest_v2.json; reports/backbone_extension_plan.md",
            "manuscript_location": "Limitations paragraph",
            "interpretation": "The evidence supports system feasibility and overhead accounting.",
            "limitation_statement": "Clinical validation and larger real benchmarks remain future work.",
        },
    ]
    path = TABLES_DIR / "table_reviewer5_response_mapping.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def generate_summary(seed):
    backbone = pd.read_csv(RESULTS_DIR / "medqa_backbone_comparison.csv")
    error = pd.read_csv(RESULTS_DIR / "medqa_error_analysis.csv")
    cost = pd.read_csv(RESULTS_DIR / "backbone_cost_scalability.csv")
    status_path = RESULTS_DIR / "medqa_backbone_evaluation_status.csv"
    status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    base = backbone[backbone["workflow"] == "backbone_only"]
    chain = backbone[backbone["workflow"] == "backbone_with_blockchain_logging"]
    real_models = sorted(base[base["benchmark_type"] == "real"]["model_name"].unique())
    failed_models = []
    if not status.empty:
        failed_models = sorted(status[status["evaluation_status"] != "completed"]["model_name"].unique())
    overhead_trend = chain[["parameter_size", "blockchain_overhead_percent"]].to_dict("records")
    hybrid_note = (
        "Blockchain workflow rows reuse the same real model predictions and add measured/hybrid devnet-derived "
        "logging overhead; correctness is intentionally unchanged across no-blockchain and blockchain workflows."
    )
    table = base[["model_name", "parameter_size", "num_questions", "accuracy", "mean_latency_ms", "benchmark_type"]].to_csv(index=False)
    text = """# Experiment Summary V2

Seed used for Reviewer 5 quick extension: `{seed}`.

## Reviewer 5 Focus

This extension adds a MedQA backbone comparison, error/invalid-output analysis, and normalized backbone cost scalability analysis. It reinforces the interpretation that MedChainLLM does not aim to improve intrinsic medical reasoning ability. Absolute MedQA accuracy is backbone-dependent; blockchain integration preserves model predictions while adding auditability, governance, and tamper-evident workflow logging.

## Real, Hybrid, Simulation, and Mock Status

- Real backbone evaluations included in formal manuscript outputs: `{real_models}`.
- Not evaluated / failed models: `{failed_models}`.
- Hybrid analyses: {hybrid_note}
- Exclusion rule: Only real backbone evaluations are included in the manuscript figures and tables. Mock or dry-run outputs, if any, were used solely for pipeline testing and excluded from scientific analysis.
- Simulation/hybrid analyses: normalized cost, review flag proxies, prior DP/FL simulations, and blockchain logging overhead estimates.
- No result should be described as clinical validation.

## Backbone Accuracy and Latency

```csv
{table}
```

Blockchain overhead percentages by backbone workflow:

```text
{overhead_trend}
```

## Error and Governance Summary

Invalid output, wrong answer, and review flag rates are summarized in `results/medqa_error_analysis.csv`. Review flags are a human-in-the-loop governance entry point, not automated clinical adjudication. Blockchain records inference process and audit trail; it does not guarantee medical correctness.

## Cost Scalability

`results/backbone_cost_scalability.csv` reports normalized inference cost, blockchain logging cost, cost per 1000 queries, audit coverage, latency, and accuracy. Normalized cost is used to compare relative deployment overhead across configurations.

## Updated Figure/Table Recommendations

- Main Figure 2: Sharding scalability and blockchain performance: `fig_sharding_throughput_vs_load`, `fig_sharding_latency_vs_validators`, `fig_dynamic_shard_adaptation`.
- Main Figure 3: MedQA backbone and workflow evaluation: `fig_backbone_accuracy_latency_tradeoff`, `fig_backbone_latency_breakdown`, `fig_backbone_auditability_overhead`.
- Main Figure 4: Privacy, FL, and governance tradeoffs: `fig_privacy_accuracy_vs_epsilon`, `fig_fl_accuracy_over_rounds`, `fig_medqa_review_flag_by_failure_type` or `fig_medqa_invalid_output_by_model`.
- Main Table 2: Baseline, ablation, and backbone comparison.
- Main Table 3: Cost, auditability, and deployment overhead.
- Supplementary: full MedQA detail, error analysis, cost scalability, FL logs, full DP parameters, and mock fallback notes.

## Remaining Limitations for Manuscript Text

- Models beyond the completed Qwen2.5-1.5B/3B/7B subset, such as 14B or proprietary clinical models, remain unevaluated unless real predictions are added.
- Clinical safety cannot be inferred from MedQA accuracy alone.
- Hallucination mitigation requires retrieval grounding, clinician review, stronger backbones, and prospective validation.
- Blockchain adds auditability and tamper evidence but depends on validator governance, key management, and consortium trust assumptions.
""".format(
        seed=seed,
        real_models=", ".join(real_models) if real_models else "none",
        failed_models=", ".join(failed_models) if failed_models else "none",
        hybrid_note=hybrid_note,
        table=table,
        overhead_trend=overhead_trend,
    )
    path = REPORTS_DIR / "experiment_summary_v2.md"
    path.write_text(text, encoding="utf-8")
    return path


def main():
    args = parse_args()
    ensure_output_dirs()
    mapping = generate_mapping()
    summary = generate_summary(args.seed)
    manifest = collect_manifest()
    manifest["tables"] = sorted(set(manifest["tables"] + [rel(mapping)]))
    manifest["reports"] = sorted(set(manifest["reports"] + [rel(summary)]))
    manifest_path = RESULTS_DIR / "experiment_manifest_v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Wrote %s" % summary)
    print("Wrote %s" % mapping)
    print("Wrote %s" % manifest_path)


if __name__ == "__main__":
    main()
