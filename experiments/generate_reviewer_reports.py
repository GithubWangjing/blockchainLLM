"""Generate engineering and reviewer-response reports from produced outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.reviewer_experiment_utils import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR, RESULTS_DIR, TABLES_DIR, ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reviewer-response reports and manifest.")
    parser.add_argument("--seed", type=int, default=42, help="Recorded for reproducibility notes.")
    return parser.parse_args()


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def collect_outputs() -> dict:
    outputs = {
        "results": sorted(rel(p) for p in RESULTS_DIR.glob("*.csv")),
        "tables": sorted(rel(p) for p in TABLES_DIR.glob("*.csv")),
        "figures": sorted(rel(p) for p in FIGURES_DIR.glob("*.*") if p.suffix.lower() in {".png", ".pdf", ".svg"}),
        "reports": sorted(rel(p) for p in REPORTS_DIR.glob("*.md")),
    }
    return outputs


def generate_mapping() -> Path:
    rows = [
        {
            "reviewer_comment": "Missing centralized/no-blockchain baseline and blockchain overhead.",
            "required_revision": "Add inference-only baseline, blockchain logging workflow, and dynamic-sharding workflow.",
            "experiment_or_text_revision": "experiments/run_baseline_comparison.py",
            "generated_outputs": "results/baseline_comparison.csv; figures/fig_baseline_latency_overhead.*",
            "manuscript_location_suggestion": "Main Table 2; Main Figure 2 or performance subsection",
        },
        {
            "reviewer_comment": "Dynamic sharding is not sufficiently justified.",
            "required_revision": "Compare unsharded, static sharding, and dynamic sharding under load/validator sweeps.",
            "experiment_or_text_revision": "experiments/run_sharding_ablation.py",
            "generated_outputs": "results/sharding_ablation.csv; figures/fig_sharding_throughput_vs_load.*; figures/fig_dynamic_shard_adaptation.*",
            "manuscript_location_suggestion": "Main Figure 2; Supplementary full ablation",
        },
        {
            "reviewer_comment": "MedQA only reports blockchain version and does not isolate auditability benefit.",
            "required_revision": "Compare Qwen-only, Qwen plus blockchain logging, and Qwen plus review flags.",
            "experiment_or_text_revision": "experiments/run_medqa_workflow_comparison.py",
            "generated_outputs": "results/medqa_workflow_comparison.csv; figures/fig_medqa_*",
            "manuscript_location_suggestion": "Main Figure 3; workflow evaluation subsection",
        },
        {
            "reviewer_comment": "Privacy/DP formulation is not standard.",
            "required_revision": "Use Gaussian mechanism sigma >= sensitivity * sqrt(2 ln(1.25/delta)) / epsilon and report utility tradeoff.",
            "experiment_or_text_revision": "experiments/run_privacy_utility_tradeoff.py",
            "generated_outputs": "results/privacy_utility_tradeoff.csv; figures/fig_privacy_*",
            "manuscript_location_suggestion": "Main Figure 4; privacy subsection",
        },
        {
            "reviewer_comment": "FL/fine-tuning claims lack experimental support.",
            "required_revision": "Add lightweight FedAvg/DP-FedAvg workflow simulation with blockchain update hashes.",
            "experiment_or_text_revision": "experiments/run_fl_finetuning_simulation.py",
            "generated_outputs": "results/fl_finetuning_simulation.csv; figures/fig_fl_*",
            "manuscript_location_suggestion": "Main Figure 4; Supplementary FL logs",
        },
        {
            "reviewer_comment": "Cost, auditability, and end-to-end workflow metrics are missing.",
            "required_revision": "Quantify tx/query, storage, normalized cost, tamper detection, and hash verification.",
            "experiment_or_text_revision": "experiments/run_cost_auditability_analysis.py",
            "generated_outputs": "results/cost_auditability_analysis.csv; figures/fig_cost_per_1000_queries.*; figures/fig_auditability_vs_overhead.*",
            "manuscript_location_suggestion": "Main Table 3; deployment overhead subsection",
        },
    ]
    path = TABLES_DIR / "table_reviewer_response_mapping.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def generate_engineering_report() -> Path:
    text = """# Engineering Report

## Current Project Structure

- `experiments/`: existing simulation package with `run_e1_sharding.py`, `run_e2_inference.py`, `run_e3_privacy.py`, configs, shared IO helpers, prior CSVs under `experiments/results/`, and prior PNG plots under `experiments/plots/`.
- `prototypes/blockchain/`: geth/docker prototype scaffold.
- `prototypes/blockchain_local/`: Hardhat local devnet benchmark, package files, generated transaction CSV summaries, and plots.
- `prototypes/llm/`: Qwen latency benchmark script and existing latency summary/detail CSVs.
- `prototypes/medical_task_eval/`: MedQA evaluation script, Qwen MedQA detail/summary CSVs, and accuracy/latency plots.
- `src/`: currently empty in this workspace scan.
- Added output roots: `results/`, `figures/`, `tables/`, `reports/`, `scripts/`.

## Existing Experiment Scripts

- `experiments/run_e1_sharding.py`: discrete-time simulation for none/static/dynamic sharding, writes timestamped CSV and three PNG plots.
- `experiments/run_e2_inference.py`: decentralized inference/compression simulation, writes latency and accuracy metrics.
- `experiments/run_e3_privacy.py`: older privacy-utility simulation using sensitivity/encryption/epsilon sweeps; useful but not a standard Gaussian DP mechanism.
- `prototypes/blockchain_local/run_benchmark.py`: real Hardhat devnet transaction benchmark with latency/throughput CSVs and plots.
- `prototypes/llm/run_latency_benchmark.py`: real Hugging Face Qwen latency benchmark.
- `prototypes/medical_task_eval/run_medqa_eval.py`: real MedQA-USMLE evaluation with Qwen, including accuracy and latency.

## Existing Data, Results, and Figures

- Prior simulation results: `experiments/results/*.csv`.
- Prior simulation plots: `experiments/plots/*.png`.
- Hardhat benchmark outputs: `prototypes/blockchain_local/results/*.csv` and `prototypes/blockchain_local/plots/*.png`.
- LLM latency outputs: `prototypes/llm/results/*.csv` and `prototypes/llm/plots/*.png`.
- MedQA outputs: `prototypes/medical_task_eval/results/medqa_summary.csv`, `prototypes/medical_task_eval/results/medqa_detail.csv`, and plots under `prototypes/medical_task_eval/plots/`.

## Capability Check

- Blockchain simulation: yes, under `experiments/e1_sharding/`.
- Hardhat/devnet benchmark: yes, under `prototypes/blockchain_local/`.
- LLM latency benchmark: yes, under `prototypes/llm/`.
- MedQA benchmark: yes, under `prototypes/medical_task_eval/`.
- Plotting scripts: yes, embedded in each runner; prior scripts saved PNG only, new reviewer scripts save PNG/PDF/SVG.

## One-command Run Status

The original project did not expose a single unified pipeline across all experiments. The new reviewer-response scripts each provide CLI entry points with `--quick` and `--seed`. A quick pipeline can be run by invoking the six new scripts followed by `experiments/generate_reviewer_reports.py`.

## Required Additions

- Baseline/no-blockchain comparison CSV/table and latency overhead figures.
- Sharding ablation CSV/table and throughput/latency/adaptation figures.
- MedQA workflow comparison CSV/table/detail and auditability/latency figures.
- Standard Gaussian DP privacy-utility CSV/table and tradeoff figures.
- FL/fine-tuning mini simulation CSV/table and workflow figures.
- Cost/auditability CSV/table and deployment overhead figures.
- `reports/experiment_summary.md`, `results/experiment_manifest.json`, and `tables/table_reviewer_response_mapping.csv`.
"""
    path = REPORTS_DIR / "engineering_report.md"
    path.write_text(text, encoding="utf-8")
    return path


def generate_summary(seed: int) -> Path:
    baseline = pd.read_csv(RESULTS_DIR / "baseline_comparison.csv")
    medqa = pd.read_csv(RESULTS_DIR / "medqa_workflow_comparison.csv")
    cost = pd.read_csv(RESULTS_DIR / "cost_auditability_analysis.csv")
    privacy = pd.read_csv(RESULTS_DIR / "privacy_utility_tradeoff.csv")
    eps_hi = privacy[privacy["epsilon"] == privacy["epsilon"].max()]["simulated_accuracy"].mean()
    eps_lo = privacy[privacy["epsilon"] == privacy["epsilon"].min()]["simulated_accuracy"].mean()
    overhead = baseline.loc[baseline["configuration"] == "blockchain_logging", "blockchain_overhead_percent"].iloc[0]
    medqa_acc = medqa["accuracy"].iloc[0]
    cost_1000 = cost.loc[cost["configuration"] == "blockchain_logging", "cost_per_1000_queries"].iloc[0]
    text = f"""# Experiment Summary

Seed used for the verified quick pipeline: `{seed}`.

## Added Experiments and Reviewer Coverage

1. `experiments/run_baseline_comparison.py`: addresses missing centralized/no-blockchain baseline and quantifies blockchain overhead.
2. `experiments/run_sharding_ablation.py`: addresses dynamic sharding evidence via unsharded/static/dynamic comparisons.
3. `experiments/run_medqa_workflow_comparison.py`: addresses MedQA with/without blockchain and auditability benefit.
4. `experiments/run_privacy_utility_tradeoff.py`: replaces the non-standard DP discussion with a Gaussian mechanism simulation.
5. `experiments/run_fl_finetuning_simulation.py`: supports FL/fine-tuning workflow feasibility claims.
6. `experiments/run_cost_auditability_analysis.py`: addresses cost, storage, tamper detection, and audit completeness.

## Commands

```bash
python -m experiments.run_baseline_comparison --quick --seed {seed}
python -m experiments.run_sharding_ablation --quick --seed {seed}
python -m experiments.run_medqa_workflow_comparison --quick --seed {seed}
python -m experiments.run_privacy_utility_tradeoff --quick --seed {seed}
python -m experiments.run_fl_finetuning_simulation --quick --seed {seed}
python -m experiments.run_cost_auditability_analysis --quick --seed {seed}
python -m experiments.generate_reviewer_reports --seed {seed}
```

## Main Result Summary

- Baseline comparison: blockchain logging adds approximately `{overhead:.1f}%` mean latency overhead in the quick hybrid run while increasing audit coverage from 0 to 1.
- MedQA workflow: Qwen-only and blockchain-integrated workflows share the same simulated correctness outcomes; accuracy is shown as a property of the lightweight Qwen backbone, not a blockchain effect. Quick-run accuracy is `{medqa_acc:.3f}`.
- Privacy tradeoff: average simulated accuracy rises from `{eps_lo:.3f}` at epsilon 0.1 to `{eps_hi:.3f}` at epsilon 8.0 as Gaussian noise decreases.
- Cost/auditability: blockchain logging uses normalized devnet cost `{cost_1000:.2f}` per 1000 queries with complete audit logs in the modeled setting.

## Benchmark vs Simulation Status

- Real benchmark inputs reused: existing Qwen latency CSVs, existing MedQA Qwen CSVs, and existing Hardhat/devnet transaction CSVs.
- Hybrid analyses: baseline comparison and MedQA workflow combine real benchmark summaries with simulated logging/review workflow components.
- Pure simulations: sharding ablation, Gaussian privacy-utility tradeoff, FL/fine-tuning simulation, and normalized cost/auditability analysis.
- FL note for manuscript: This is a lightweight federated simulation intended to validate workflow feasibility and overhead trends, not a clinical fine-tuning validation.

## Output Files

- CSV results: `results/*.csv`
- Tables: `tables/*.csv`
- Figures: `figures/*.png`, `figures/*.pdf`, `figures/*.svg`
- Reports: `reports/engineering_report.md`, `reports/experiment_summary.md`
- Manifest: `results/experiment_manifest.json`

## Suggested Manuscript Placement

- Main Figure 2: `fig_sharding_throughput_vs_load`, `fig_sharding_latency_vs_validators`, `fig_dynamic_shard_adaptation`.
- Main Figure 3: `fig_medqa_latency_distribution`, `fig_medqa_overhead_breakdown`, `fig_medqa_accuracy_latency_tradeoff`.
- Main Figure 4: `fig_privacy_accuracy_vs_epsilon`, `fig_fl_accuracy_over_rounds`, `fig_fl_overhead_breakdown`.
- Main Table 1: system configuration and threat assumptions, mostly text/table revision.
- Main Table 2: `table_baseline_comparison.csv` and `table_sharding_ablation.csv`.
- Main Table 3: `table_cost_auditability_analysis.csv`.
- Supplementary: full sharding sweep, FL per-round logs, privacy full parameter table, MedQA detail CSV, and cost model assumptions.

## Text-only Revisions Still Needed

- Precisely define the threat model, consortium-chain trust assumptions, key management, and what auditability can and cannot prove.
- Clarify that MedChainLLM targets secure workflow/auditability rather than improving medical QA accuracy.
- State that DP and FL experiments are simulations unless replaced with clinical fine-tuning runs.
- Discuss external validity limits of small-model MedQA and synthetic large-scale blockchain simulation.
"""
    path = REPORTS_DIR / "experiment_summary.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    mapping = generate_mapping()
    engineering = generate_engineering_report()
    summary = generate_summary(args.seed)
    outputs = collect_outputs()
    outputs["tables"] = sorted(set(outputs["tables"] + [rel(mapping)]))
    outputs["reports"] = sorted(set(outputs["reports"] + [rel(engineering), rel(summary)]))
    manifest = RESULTS_DIR / "experiment_manifest.json"
    manifest.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"Wrote {engineering}")
    print(f"Wrote {summary}")
    print(f"Wrote {mapping}")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
