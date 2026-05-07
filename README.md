# MedChainLLM Reviewer Experiments

This repository contains the experiment code, generated tables, figures, and
reports for:

**MedChainLLM: A Blockchain-Integrated Architecture for Secure, Scalable, and Privacy-Adaptive Medical Large Language Models**

The repository is organized to keep low-cost reproduction separate from
expensive real LLM inference.

## What Is Included

- `experiments/`: simulation, benchmark aggregation, MedQA workflow, backbone comparison, error analysis, and report generators.
- `results/`: generated CSV outputs used by the manuscript response.
- `tables/`: manuscript-ready CSV tables.
- `figures/`: paper-style PNG/PDF/SVG figures.
- `reports/`: engineering report, experiment summaries, reviewer mappings, and error analysis.
- `data/medqa/`: MedQA-USMLE 4-option JSONL subset source used for real backbone evaluation.
- `prototypes/`: earlier blockchain, LLM latency, and MedQA prototype benchmarks.

Large model files are intentionally **not tracked**. Download them only when
you need to rerun real backbone inference.

## Lowest-Cost Setup

Use this path if you only need to regenerate simulation-based results, reports,
and figures from lightweight scripts.

```bash
conda create -n blockchain python=3.10 -y
conda activate blockchain
pip install -r requirements.txt
```

Run the lightweight reviewer pipeline:

```bash
python -m experiments.run_baseline_comparison --quick --seed 7
python -m experiments.run_sharding_ablation --quick --seed 7
python -m experiments.run_medqa_workflow_comparison --quick --seed 7
python -m experiments.run_privacy_utility_tradeoff --quick --seed 7
python -m experiments.run_fl_finetuning_simulation --quick --seed 7
python -m experiments.run_cost_auditability_analysis --quick --seed 7
python -m experiments.generate_reviewer_reports --seed 7
```

On Windows, the same commands are wrapped in:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_quick_pipeline.ps1
```

## Real MedQA Backbone Evaluation

Install optional LLM dependencies:

```bash
pip install -r requirements-llm.txt
```

Download models. For low cost, start with 1.5B. The 3B model is much slower on
CPU.

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/Qwen2.5-1.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct
```

If disk space is limited in the repository directory, put models elsewhere and
set:

```bash
export MEDCHAIN_MODEL_ROOT=/path/to/MedChainLLM_models
```

Windows PowerShell:

```powershell
$env:MEDCHAIN_MODEL_ROOT="F:\MedChainLLM_models"
```

Run a real 50-question subset:

```bash
python -m experiments.run_medqa_backbone_comparison --quick --seed 7 --max-questions 50 --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct --cpu-dtype float16
python -m experiments.run_medqa_error_analysis --seed 7
python -m experiments.run_backbone_cost_scalability --seed 7
python -m experiments.generate_reviewer5_reports --seed 7
```

Important interpretation:

- MedChainLLM does not aim to improve intrinsic medical reasoning ability.
- MedQA accuracy is backbone-dependent.
- Blockchain logging preserves model predictions while adding auditability and tamper-evident workflow logging.
- Only real backbone evaluations should be used in manuscript figures/tables.
- No result is clinical validation.

## Current Real Backbone Results

The committed outputs include real 50-question subset results:

| Backbone | Accuracy | Mean latency |
| --- | ---: | ---: |
| Qwen2.5-0.5B | 0.20 | 2169.8 ms |
| Qwen2.5-1.5B-Instruct | 0.36 | 8104.3 ms |
| Qwen2.5-3B-Instruct | 0.44 | 94658.1 ms |

The 3B CPU latency is a local prototype measurement, not a GPU deployment
estimate.

## Server Reproduction From Scratch

These commands assume a Linux GPU server. They clone the repository, create the
environment, download the MedQA JSONL file, download Qwen backbones, and run the
Reviewer 5 backbone pipeline.

```bash
git clone https://github.com/GithubWangjing/blockchainLLM.git
cd blockchainLLM
conda env create -f environment.yml
conda activate blockchain
pip install -r requirements-llm.txt
```

Download the real MedQA-USMLE 4-option JSONL file:

```bash
mkdir -p data/medqa
wget -O data/medqa/GBaker_MedQA_USMLE_4options_test.json \
  https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf/resolve/main/test.json
```

If `wget` is unavailable:

```bash
python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve
Path("data/medqa").mkdir(parents=True, exist_ok=True)
urlretrieve(
    "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf/resolve/main/test.json",
    "data/medqa/GBaker_MedQA_USMLE_4options_test.json",
)
PY
```

Download the real Qwen backbones:

```bash
mkdir -p models
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir ./models/Qwen2.5-1.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
  --local-dir ./models/Qwen2.5-3B-Instruct
```

Run the 50-question real backbone subset:

```bash
python -m experiments.run_medqa_backbone_comparison \
  --quick \
  --seed 7 \
  --max-questions 50 \
  --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
  --cpu-dtype float16

python -m experiments.run_medqa_error_analysis --seed 7
python -m experiments.run_backbone_cost_scalability --seed 7
python -m experiments.generate_reviewer5_reports --seed 7
```

Run a larger 200-question subset on a GPU server:

```bash
python -m experiments.run_medqa_backbone_comparison \
  --seed 7 \
  --max-questions 200 \
  --models Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
  --merge-existing \
  --cpu-dtype float16

python -m experiments.run_medqa_error_analysis --seed 7
python -m experiments.run_backbone_cost_scalability --seed 7
python -m experiments.generate_reviewer5_reports --seed 7
```

One-command Linux helper:

```bash
bash scripts/server_run_reviewer5.sh 50
```

The argument is `max_questions`. For example, use `200` for a larger subset.

## Optional Larger Backbones

Use a GPU server for 7B or 14B models. These are not part of the minimum-cost
pipeline and should only be added to manuscript tables after real inference
completes.

Run 7B on the same 50-question subset and merge it with existing completed
0.5B/1.5B/3B results:

```bash
bash scripts/server_run_large_backbones.sh 50 Qwen/Qwen2.5-7B-Instruct
```

Run 7B and 14B on a 200-question subset:

```bash
bash scripts/server_run_large_backbones.sh 200 \
  Qwen/Qwen2.5-7B-Instruct \
  Qwen/Qwen2.5-14B-Instruct
```

Equivalent manual command after downloading the models:

```bash
python -m experiments.run_medqa_backbone_comparison \
  --seed 7 \
  --max-questions 200 \
  --models Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct \
  --merge-existing \
  --cpu-dtype float16
python -m experiments.run_medqa_error_analysis --seed 7
python -m experiments.run_backbone_cost_scalability --seed 7
python -m experiments.generate_reviewer5_reports --seed 7
```

## Blockchain Technical Evaluation

Reviewer 5 blockchain-specific evaluation can be regenerated with:

```bash
python -m experiments.run_contract_access_control_benchmark --quick --seed 7
python -m experiments.run_tamper_evidence_evaluation --quick --seed 7
python -m experiments.run_consensus_scalability_simulation --quick --seed 7
python -m experiments.generate_blockchain_technical_reports --quick --seed 7
```

Windows helper:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_blockchain_technical_evaluation.ps1
```

Outputs include:

- `results/contract_access_control_benchmark.csv`
- `results/tamper_evidence_evaluation.csv`
- `results/consensus_scalability_simulation.csv`
- `reports/blockchain_technical_evaluation.md`
- `reports/experiment_summary_v3.md`

Evidence status:

- Access-control latency: real Hardhat/devnet latency reused with deterministic contract semantics.
- Tamper evidence: real SHA-256 hash verification over workflow records.
- Consensus scalability: simulation.

## Notes on Secrets and Large Files

The repository excludes:

- `models/`
- `*.safetensors`, `*.bin`, `*.pt`, `*.pth`
- `node_modules/`
- generated blockchain keystore/nodekey/password files
- Python caches

Do not commit API keys or private model credentials.
