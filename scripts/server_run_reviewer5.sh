#!/usr/bin/env bash
set -euo pipefail

MAX_QUESTIONS="${1:-50}"
SEED="${SEED:-7}"

echo "[1/5] Installing lightweight and LLM dependencies"
pip install -r requirements.txt
pip install -r requirements-llm.txt

echo "[2/5] Downloading MedQA-USMLE 4-option JSONL"
mkdir -p data/medqa
python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve

target = Path("data/medqa/GBaker_MedQA_USMLE_4options_test.json")
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists() and target.stat().st_size > 0:
    print(f"Dataset already exists: {target}")
else:
    urlretrieve(
        "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf/resolve/main/test.json",
        target,
    )
    print(f"Downloaded dataset to {target}")
PY

echo "[3/5] Downloading Qwen backbones"
mkdir -p models
if [ ! -f "models/Qwen2.5-1.5B-Instruct/config.json" ]; then
  huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir ./models/Qwen2.5-1.5B-Instruct
else
  echo "Qwen2.5-1.5B-Instruct already exists"
fi

if [ ! -f "models/Qwen2.5-3B-Instruct/config.json" ]; then
  huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
    --local-dir ./models/Qwen2.5-3B-Instruct
else
  echo "Qwen2.5-3B-Instruct already exists"
fi

echo "[4/5] Running real MedQA backbone comparison"
python -m experiments.run_medqa_backbone_comparison \
  --seed "${SEED}" \
  --max-questions "${MAX_QUESTIONS}" \
  --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
  --cpu-dtype float16

echo "[5/5] Regenerating error analysis, cost analysis, and reports"
python -m experiments.run_medqa_error_analysis --seed "${SEED}"
python -m experiments.run_backbone_cost_scalability --seed "${SEED}"
python -m experiments.generate_reviewer5_reports --seed "${SEED}"

echo "Done. Key outputs:"
echo "  results/medqa_backbone_comparison.csv"
echo "  results/medqa_error_analysis.csv"
echo "  results/backbone_cost_scalability.csv"
echo "  reports/experiment_summary_v2.md"
