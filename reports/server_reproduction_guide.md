# Server Reproduction Guide

This guide is intended for low-friction synchronization to a GPU server.

## Minimal Simulation/Report Reproduction

```bash
git clone https://github.com/GithubWangjing/blockchainLLM.git
cd blockchainLLM
conda env create -f environment.yml
conda activate blockchain
python -m experiments.run_baseline_comparison --quick --seed 7
python -m experiments.run_sharding_ablation --quick --seed 7
python -m experiments.generate_reviewer_reports --seed 7
```

## Real Backbone Reproduction From Scratch

```bash
pip install -r requirements-llm.txt
mkdir -p data/medqa
python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve
Path("data/medqa").mkdir(parents=True, exist_ok=True)
urlretrieve(
    "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf/resolve/main/test.json",
    "data/medqa/GBaker_MedQA_USMLE_4options_test.json",
)
PY
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/Qwen2.5-1.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct
python -m experiments.run_medqa_backbone_comparison --seed 7 --max-questions 50 --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct --cpu-dtype float16
python -m experiments.run_medqa_error_analysis --seed 7
python -m experiments.run_backbone_cost_scalability --seed 7
python -m experiments.generate_reviewer5_reports --seed 7
```

Use larger `--max-questions` values only when GPU memory and runtime budget are
available.

Or use:

```bash
bash scripts/server_run_reviewer5.sh 50
```
