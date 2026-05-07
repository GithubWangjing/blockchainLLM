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

## Server Reproduction

For a GPU server:

```bash
git clone https://github.com/GithubWangjing/blockchainLLM.git
cd blockchainLLM
conda env create -f environment.yml
conda activate blockchain
pip install -r requirements-llm.txt
```

Then download models and run the real backbone commands above. Prefer GPU for
3B or larger models.

## Notes on Secrets and Large Files

The repository excludes:

- `models/`
- `*.safetensors`, `*.bin`, `*.pt`, `*.pth`
- `node_modules/`
- generated blockchain keystore/nodekey/password files
- Python caches

Do not commit API keys or private model credentials.

