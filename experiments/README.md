# MedChainLLM Simulation Experiments

All research experiments live inside this `experiments/` package. Each subdirectory encapsulates one family of simulations:

- `common/`: shared utilities (seeding, IO helpers, plotting styles).
- `configs/`: reusable JSON/YAML configuration snippets.
- `results/`: CSV outputs.
- `plots/`: publication-ready figures (PNG by default).
- `e1_sharding/`, `e2_inference/`, `e3_privacy/`: code and modules dedicated to the three experiments described in the MedChainLLM paper.

Typical usage pattern:

```bash
# Dynamic sharding
python -m experiments.run_e1_sharding --config experiments/configs/e1_default.json --output-root experiments

# Decentralized inference
python -m experiments.run_e2_inference --config experiments/configs/e2_default.json --output-root experiments

# Privacy-utility sweep
python -m experiments.run_e3_privacy --config experiments/configs/e3_default.json --output-root experiments
```

Each runner script prints a concise summary of generated CSVs and figures after completion.

