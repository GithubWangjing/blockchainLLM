# Engineering Report

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
