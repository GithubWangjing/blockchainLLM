## LLM Latency Benchmark

This prototype measures end-to-end text generation latency for small causal language models using Hugging Face Transformers and PyTorch.

### Setup

```bash
pip install -r prototypes/llm/requirements.txt
```

### Run with defaults

```bash
python prototypes/llm/run_latency_benchmark.py
```

This loads `Qwen/Qwen2.5-0.5B`, detects `cuda` automatically when available, and benchmarks prompt lengths `[32, 128, 512, 1024]` with generation lengths `[8, 32, 64]`. Detailed and summary CSVs are written to `prototypes/llm/results/`, and plots to `prototypes/llm/plots/`.

### Customize prompts and generation lengths

Provide space-separated integers for prompt or generation lengths:

```bash
python prototypes/llm/run_latency_benchmark.py \
  --prompt-lengths 64 256 1024 \
  --gen-tokens 16 32 128 \
  --repetitions 30
```

Override output locations if desired with `--results-root` and `--plots-root`. The script always creates the target directories if they do not exist. Results include per-run latency CSVs, aggregated statistics, and two plots:

- `latency_vs_prompt_length_<timestamp>.png`
- `latency_vs_gen_tokens_<timestamp>.png`

### Troubleshooting

- The benchmark requires a modern PyTorch build that ships `torch.cuda.amp`, even if you are running on CPU. Install the versions pinned in `requirements.txt` (PyTorch ≥ 2.0, Transformers ≥ 4.35). If you are on CPU only, you can explicitly run:

  ```bash
  pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
  ```

- If your existing Anaconda environment pulls in much older Torch/Transformers releases, create a fresh virtualenv or conda env before installing these dependencies.

