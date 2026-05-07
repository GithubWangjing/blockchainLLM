# MedChainLLM Local Hardhat Prototype

This alternative prototype keeps everything self-contained on a single machine using Hardhat's built-in devnet (no Docker needed). It mirrors the CSV/plot outputs from the Docker-based benchmark but drives a `npx hardhat node` instead.

## Layout

- `package.json`, `hardhat.config.js` – minimal Hardhat project (default mnemonic, 10 pre-funded accounts, chainId 31337).
- `run_benchmark.py` – Python script that submits native transfers, collects latency + throughput metrics, and emits CSV/plots.
- `results/`, `plots/` – storage for benchmark artifacts.

## Prerequisites

- Node.js 18+ and npm.
- Python 3.10+ with `pip`.

## 1. Install Hardhat dependencies

```bash
cd prototypes/blockchain_local
npm install
```

## 2. Start the local chain

```bash
cd prototypes/blockchain_local
npx hardhat node
```

> **Windows tip:** Hardhat loads `hardhat.config.js` from the current working directory. If you run the command from the repo root, use `npx hardhat --config prototypes/blockchain_local/hardhat.config.js node`. The `HHE3: No Hardhat config file found` error means the command was started from the wrong directory or before `npm install` completed.

Keep this process running; it exposes JSON-RPC at `http://127.0.0.1:8545` and prints the pre-funded accounts + private keys (derived from the mnemonic `test test test test test test test test test test test junk`).

## 3. Install benchmark dependencies

```bash
pip install -r prototypes/blockchain_local/requirements.txt
```

## 4. Run the benchmark

With the Hardhat node running in another terminal:

```bash
python prototypes/blockchain_local/run_benchmark.py --rpc-url http://127.0.0.1:8545 --tag hardhat_sweep
```

The defaults now sweep a 4x4 grid (transaction_counts = 200, 400, 800, 1200 and batch_sizes = 10, 25, 50, 100) and repeat each configuration three times.

To override any dimension, pass explicit values. Example:

```bash
python prototypes/blockchain_local/run_benchmark.py ^
  --rpc-url http://127.0.0.1:8545 ^
  --transaction-counts 300 600 ^
  --batch-sizes 20 40 ^
  --repetitions 5 ^
  --tag hardhat_custom
```

> **PowerShell note:** Use the single-line form above or escape newlines with the backtick character (`` ` ``). Appending `\` at the end of a line (as in Bash) passes a literal backslash to argparse, producing `invalid int value: '\'`.

Outputs (timestamped slug):

- `results/blockchain_metrics_<slug>_detail.csv` – one row per transaction.
- `results/blockchain_summary_<slug>_runs.csv` – one row per (tx_count, batch_size, repetition).
- `results/blockchain_summary_<slug>_mean.csv` – mean/std per (tx_count, batch_size).
- `plots/throughput_vs_load_<slug>.png`
- `plots/latency_vs_load_<slug>.png`
- `plots/latency_histogram_txXXXX_batchYYYY_<slug>.png` – highest-load scenario histogram.

By default, the benchmark deterministically derives the same accounts that Hardhat generates (mnemonic above). You can override the sender via `--private-key` or `--private-key-file`.

## Notes & Troubleshooting

- Ensure only one Hardhat node is bound to `8545` before starting another benchmark.
- If you want to reset the chain, just stop `npx hardhat node` and start it again (state is in-memory).
- The benchmark polls receipts aggressively; adjust `--poll-interval` or `--max-inflight` for slower machines.
- Seeing `HHE3: No Hardhat config file found`? Re-run `npm install` inside `prototypes/blockchain_local` and start `npx hardhat node` from that directory (or pass `--config prototypes/blockchain_local/hardhat.config.js` when calling it from elsewhere).

