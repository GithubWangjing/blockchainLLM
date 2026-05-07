# Experiment Summary

Seed used for the verified quick pipeline: `7`.

## Added Experiments and Reviewer Coverage

1. `experiments/run_baseline_comparison.py`: addresses missing centralized/no-blockchain baseline and quantifies blockchain overhead.
2. `experiments/run_sharding_ablation.py`: addresses dynamic sharding evidence via unsharded/static/dynamic comparisons.
3. `experiments/run_medqa_workflow_comparison.py`: addresses MedQA with/without blockchain and auditability benefit.
4. `experiments/run_privacy_utility_tradeoff.py`: replaces the non-standard DP discussion with a Gaussian mechanism simulation.
5. `experiments/run_fl_finetuning_simulation.py`: supports FL/fine-tuning workflow feasibility claims.
6. `experiments/run_cost_auditability_analysis.py`: addresses cost, storage, tamper detection, and audit completeness.

## Commands

```bash
python -m experiments.run_baseline_comparison --quick --seed 7
python -m experiments.run_sharding_ablation --quick --seed 7
python -m experiments.run_medqa_workflow_comparison --quick --seed 7
python -m experiments.run_privacy_utility_tradeoff --quick --seed 7
python -m experiments.run_fl_finetuning_simulation --quick --seed 7
python -m experiments.run_cost_auditability_analysis --quick --seed 7
python -m experiments.generate_reviewer_reports --seed 7
```

## Main Result Summary

- Baseline comparison: blockchain logging adds approximately `10.4%` mean latency overhead in the quick hybrid run while increasing audit coverage from 0 to 1.
- MedQA workflow: Qwen-only and blockchain-integrated workflows share the same simulated correctness outcomes; accuracy is shown as a property of the lightweight Qwen backbone, not a blockchain effect. Quick-run accuracy is `0.167`.
- Privacy tradeoff: average simulated accuracy rises from `0.632` at epsilon 0.1 to `0.744` at epsilon 8.0 as Gaussian noise decreases.
- Cost/auditability: blockchain logging uses normalized devnet cost `18.00` per 1000 queries with complete audit logs in the modeled setting.

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
