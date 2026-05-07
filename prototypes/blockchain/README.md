# MedChainLLM Blockchain Prototype

This directory contains a reproducible 4-node Clique PoA devnet plus a Python benchmark that submits real transactions and records throughput/latency metrics.

## Components

- `docker-compose.yml` – spins up four `geth` nodes (one sealer) and an asset-builder container that prepares the genesis file, nodekeys, and funded dev accounts.
- `scripts/generate_assets.py` – deterministic asset generator (run automatically by the compose stack but can be executed manually).
- `scripts/start_geth.sh` – shared entrypoint invoked by every node container.
- `run_benchmark.py` – Python benchmark that drives load through the RPC endpoint, logs CSV metrics, and emits plots.
- `results/`, `plots/` – storage for benchmark artifacts.

## Prerequisites

- Docker Desktop (with `docker compose` CLI) and enough local disk/CPU for 4 lightweight containers.
- Python 3.10+ plus the packages listed in `requirements.txt`.

## 1. Generate devnet assets

Although `docker compose up` will run this automatically, you can pre-generate the files (addresses, node keys, genesis) via:

```bash
cd prototypes/blockchain
python scripts/generate_assets.py
```

The script writes everything under `prototypes/blockchain/config/`.

## 2. Launch the 4-node network

```bash
cd prototypes/blockchain
docker compose up -d
```

What happens:

- `asset_builder` installs the minimal Python deps and runs `scripts/generate_assets.py`.
- `node1` imports the signer key, exposes JSON-RPC on `8545/8546`, and seals blocks.
- `node2`–`node4` join as additional permissioned peers to mimic a small consortium.

Stop the network with `docker compose down` (add `-v` to wipe data directories).

## 3. Install benchmark dependencies

```bash
pip install -r prototypes/blockchain/requirements.txt
```

## 4. Run the benchmark

Example command (adjust batch sizes / RPC URL as needed):

```bash
python prototypes/blockchain/run_benchmark.py \
  --rpc-url http://localhost:8545 \
  --transaction-counts 200 400 \
  --batch-sizes 25 50 \
  --tag local
```

Outputs:

- Detailed per-transaction CSV: `prototypes/blockchain/results/blockchain_metrics_<slug>.csv`
- Scenario summary CSV: `prototypes/blockchain/results/blockchain_summary_<slug>.csv`
- Plots:
  - `plots/throughput_vs_batch_size_<slug>.png`
  - `plots/latency_histogram_<slug>.png`
  - `plots/latency_vs_load_<slug>.png`

The benchmark reads the funded sender key from `config/accounts/sender.key` (generated during step 1). Override with `--private-key` or `--private-key-file` if you prefer different credentials.

## Troubleshooting

- **Docker missing** – install Docker Desktop and ensure `docker compose version` succeeds.
- **RPC connection errors** – verify `node1` is running (`docker ps`) and ports `8545/8546` are free.
- **Missing config** – rerun `python scripts/generate_assets.py` to recreate the genesis and account files, then restart the compose stack.

