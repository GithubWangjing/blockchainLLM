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
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export MEDCHAIN_MODEL_ROOT="${MEDCHAIN_MODEL_ROOT:-$PWD/models}"

pip install -r requirements.txt
pip install -r requirements-llm.txt

mkdir -p data/medqa models
python - <<'PY'
import os
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

target = Path("data/medqa/GBaker_MedQA_USMLE_4options_test.json")
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists() or target.stat().st_size == 0:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    try:
        subprocess.run(
            [
                "hf",
                "download",
                "GBaker/MedQA-USMLE-4-options-hf",
                "test.json",
                "--repo-type",
                "dataset",
                "--local-dir",
                str(target.parent),
            ],
            check=True,
        )
    except Exception as hf_exc:
        try:
            urlretrieve(
                f"{endpoint}/datasets/GBaker/MedQA-USMLE-4-options-hf/resolve/main/test.json",
                target,
            )
        except Exception as url_exc:
            raise SystemExit(
                "MedQA dataset is missing locally and could not be downloaded. "
                "The repository normally includes data/medqa/GBaker_MedQA_USMLE_4options_test.json; "
                "run `git checkout origin/main -- data/medqa/GBaker_MedQA_USMLE_4options_test.json` "
                "or copy this file from another machine, then rerun the script. "
                f"hf error: {hf_exc}; url error: {url_exc}"
            )
print(f"MedQA file ready: {target}")
PY

model_dir_ready() {
  local dir="$1"
  [ -f "${dir}/config.json" ] || return 1
  [ -f "${dir}/tokenizer.json" ] || [ -f "${dir}/tokenizer_config.json" ] || return 1
  find "${dir}" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) -print -quit 2>/dev/null | grep -q . || return 1
  if find "${dir}" -maxdepth 1 \( -name '*.incomplete' -o -name '*.lock' \) -print -quit 2>/dev/null | grep -q .; then
    return 1
  fi
  return 0
}

for model in "${MODELS[@]}"; do
  local_name="${model#Qwen/}"
  local_dir="./models/${local_name}"
  if ! model_dir_ready "${local_dir}"; then
    echo "Downloading ${model} to ${local_dir}"
    if command -v hf >/dev/null 2>&1; then
      hf download "${model}" --local-dir "${local_dir}"
    else
      huggingface-cli download "${model}" --local-dir "${local_dir}"
    fi
  else
    echo "${model} already exists at ${local_dir}"
  fi
done

python -m experiments.run_medqa_backbone_comparison \
  --seed "${SEED}" \
  --max-questions "${MAX_QUESTIONS}" \
  --local-medqa-json data/medqa/GBaker_MedQA_USMLE_4options_test.json \
  --models "${MODELS[@]}" \
  --merge-existing \
  --cpu-dtype "${CPU_DTYPE}"

python -m experiments.run_medqa_error_analysis --seed "${SEED}"
python -m experiments.run_backbone_cost_scalability --seed "${SEED}"
python -m experiments.generate_reviewer5_reports --seed "${SEED}"

echo "Large-backbone run complete."
