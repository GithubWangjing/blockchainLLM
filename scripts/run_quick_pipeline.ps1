chcp 65001 | Out-Null
$env:PYTHONIOENCODING = "utf-8"
python -m experiments.run_baseline_comparison --quick --seed 7
python -m experiments.run_sharding_ablation --quick --seed 7
python -m experiments.run_medqa_workflow_comparison --quick --seed 7
python -m experiments.run_privacy_utility_tradeoff --quick --seed 7
python -m experiments.run_fl_finetuning_simulation --quick --seed 7
python -m experiments.run_cost_auditability_analysis --quick --seed 7
python -m experiments.generate_reviewer_reports --seed 7
