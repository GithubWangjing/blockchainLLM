## MedQA Medical Task Evaluation

Benchmark Qwen/Qwen2.5-0.5B on the MedQA-USMLE test split, capturing accuracy and latency metrics.

### Setup

```bash
pip install -r prototypes/medical_task_eval/requirements.txt
```

### Run the benchmark

```bash
python prototypes/medical_task_eval/run_medqa_eval.py
```

Key options:

- `--model-name`: Hugging Face model id (default `Qwen/Qwen2.5-0.5B`).
- `--batch-size`: Number of questions loaded per evaluation chunk.
- `--max-samples`: Cap the number of MedQA test questions (default: entire split).
- `--repetitions`: Repeat generation per question to smooth latency (first run counts for accuracy).
- `--dataset-id`, `--dataset-config`, `--dataset-split`: Override dataset source if the default mirrors are unavailable (set `--dataset-id` to `auto` to try known public copies).
- `--results-root`, `--plots-root`: Output directories for CSVs and plots.

The script produces:

- `medqa_detail.csv` (per-question accuracy + latency stats)
- `medqa_summary.csv` (overall accuracy, mean latency, p95 latency)
- `medqa_accuracy.png`
- `medqa_latency.png`

