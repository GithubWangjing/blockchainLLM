#!/usr/bin/env bash
set -euo pipefail

# Run optional larger real backbones on a GPU server.
# Usage examples:
#   bash scripts/server_run_large_backbones.sh 50 Qwen/Qwen2.5-7B-Instruct
#   bash scripts/server_run_large_backbones.sh 200 Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct

MAX_QUESTIONS="${1:-50}"
shift || true

if [ "$#" -eq 0 ]; then
  MODELS=("Qwen/Qwen2.5-7B-Instruct")
else
  MODELS=("$@")
fi

SEED="${SEED:-7}"
CPU_DTYPE="${CPU_DTYPE:-float16}"

pip install -r requirements.txt
pip install -r requirements-llm.txt

mkdir -p data/medqa models
python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve

target = Path("data/medqa/GBaker_MedQA_USMLE_4options_test.json")
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists() or target.stat().st_size == 0:
    urlretrieve(
        "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options-hf/resolve/main/test.json",
        target,
    )
print(f"MedQA file ready: {target}")
PY

for model in "${MODELS[@]}"; do
  local_name="${model#Qwen/}"
  local_dir="./models/${local_name}"
  if [ ! -f "${local_dir}/config.json" ]; then
    echo "Downloading ${model} to ${local_dir}"
    huggingface-cli download "${model}" --local-dir "${local_dir}"
  else
    echo "${model} already exists at ${local_dir}"
  fi
done

python -m experiments.run_medqa_backbone_comparison \
  --seed "${SEED}" \
  --max-questions "${MAX_QUESTIONS}" \
  --models "${MODELS[@]}" \
  --merge-existing \
  --cpu-dtype "${CPU_DTYPE}"

python -m experiments.run_medqa_error_analysis --seed "${SEED}"
python -m experiments.run_backbone_cost_scalability --seed "${SEED}"
python -m experiments.generate_reviewer5_reports --seed "${SEED}"

echo "Large-backbone run complete."
