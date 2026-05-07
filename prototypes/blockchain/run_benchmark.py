"""Benchmark script for the MedChainLLM PoA devnet."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eth_account import Account
from tqdm import tqdm
from web3 import Web3
from web3.exceptions import TransactionNotFound

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
DEFAULT_SENDER_KEY = CONFIG_DIR / "accounts" / "sender.key"
DEFAULT_SEALER_ADDRESS = CONFIG_DIR / "accounts" / "sealer.address"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_slug() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


@dataclass
class TxRecord:
    scenario_id: str
    batch_size: int
    tx_count: int
    tx_hash: str
    send_time_iso: str
    confirm_time_iso: str
    block_number: int
    latency_ms: float
    send_monotonic: float
    confirm_monotonic: float


@dataclass
class ScenarioResult:
    scenario_id: str
    tx_count: int
    batch_size: int
    throughput_tps: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedChainLLM blockchain benchmark")
    parser.add_argument("--rpc-url", default="http://localhost:8545", help="JSON-RPC endpoint")
    parser.add_argument("--chain-id", type=int, default=4242, help="Chain ID for signing")
    parser.add_argument(
        "--transaction-counts",
        type=int,
        nargs="+",
        default=[200],
        help="Number of transactions to submit per scenario",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[25, 50, 100],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=200,
        help="Maximum number of outstanding transactions before polling for receipts",
    )
    parser.add_argument("--gas-limit", type=int, default=21000, help="Gas limit per transaction")
    parser.add_argument("--gas-price-gwei", type=float, default=1, help="Gas price for the devnet")
    parser.add_argument(
        "--value-wei",
        type=int,
        default=1,
        help="Transfer value (in wei) for each transaction",
    )
    parser.add_argument(
        "--private-key-file",
        type=Path,
        default=DEFAULT_SENDER_KEY,
        help="Path to the hex-encoded sender private key",
    )
    parser.add_argument(
        "--private-key",
        type=str,
        default=None,
        help="Override the sender private key via CLI",
    )
    parser.add_argument(
        "--recipient-address",
        type=str,
        default=None,
        help="Recipient address (default uses the same account for self-transfer)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds to wait between receipt polling attempts",
    )
    parser.add_argument("--tag", default="", help="Optional label for output artifacts")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for CSV outputs",
    )
    parser.add_argument(
        "--plots-root",
        type=Path,
        default=PLOTS_DIR,
        help="Directory for plots",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def poll_receipts(
    w3: Web3,
    pending: Dict[bytes, Dict[str, float]],
    scenario_id: str,
    batch_size: int,
    tx_count: int,
    poll_interval: float,
) -> List[TxRecord]:
    completed: List[TxRecord] = []
    if not pending:
        return completed

    waiting = True
    while waiting:
        to_remove: List[bytes] = []
        for tx_hash, meta in list(pending.items()):
            try:
                receipt = w3.eth.get_transaction_receipt(tx_hash)
            except TransactionNotFound:
                continue
            if receipt and receipt.blockNumber is not None:
                confirm_monotonic = time.perf_counter()
                latency_ms = (confirm_monotonic - meta["send_monotonic"]) * 1000.0
                completed.append(
                    TxRecord(
                        scenario_id=scenario_id,
                        batch_size=batch_size,
                        tx_count=tx_count,
                        tx_hash=tx_hash.hex(),
                        send_time_iso=meta["send_iso"],
                        confirm_time_iso=now_iso(),
                        block_number=receipt.blockNumber,
                        latency_ms=latency_ms,
                        send_monotonic=meta["send_monotonic"],
                        confirm_monotonic=confirm_monotonic,
                    )
                )
                to_remove.append(tx_hash)
        for tx_hash in to_remove:
            pending.pop(tx_hash, None)
        if completed:
            waiting = False
        else:
            time.sleep(poll_interval)
    return completed


def run_scenario(
    w3: Web3,
    account,
    recipient: str,
    args: argparse.Namespace,
    tx_total: int,
    batch_size: int,
) -> Tuple[ScenarioResult, List[TxRecord]]:
    scenario_id = f"tx{tx_total}_batch{batch_size}"
    gas_price = w3.to_wei(args.gas_price_gwei, "gwei")
    nonce = w3.eth.get_transaction_count(account.address)
    pending: Dict[bytes, Dict[str, float]] = {}
    all_records: List[TxRecord] = []
    sent = 0
    pbar = tqdm(total=tx_total, desc=f"[{scenario_id}] sending", leave=False)

    while sent < tx_total:
        to_send = min(batch_size, tx_total - sent)
        for _ in range(to_send):
            tx = {
                "nonce": nonce,
                "to": recipient,
                "value": args.value_wei,
                "gas": args.gas_limit,
                "gasPrice": gas_price,
                "chainId": args.chain_id,
            }
            signed = account.sign_transaction(tx)
            send_monotonic = time.perf_counter()
            tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
            pending[tx_hash] = {"send_monotonic": send_monotonic, "send_iso": now_iso()}
            nonce += 1
            sent += 1
            pbar.update(1)
            if len(pending) >= args.max_inflight:
                all_records.extend(
                    poll_receipts(
                        w3,
                        pending,
                        scenario_id,
                        batch_size,
                        tx_total,
                        args.poll_interval,
                    )
                )
        # After each batch, try to drain at least one receipt
        all_records.extend(
            poll_receipts(
                w3, pending, scenario_id, batch_size, tx_total, args.poll_interval
            )
        )

    pbar.close()

    while pending:
        all_records.extend(
            poll_receipts(
                w3, pending, scenario_id, batch_size, tx_total, args.poll_interval
            )
        )

    if not all_records:
        raise RuntimeError("Benchmark produced no confirmed transactions.")

    send_start = min(r.send_monotonic for r in all_records)
    confirm_end = max(r.confirm_monotonic for r in all_records)
    duration = max(confirm_end - send_start, 1e-9)
    latencies = [r.latency_ms for r in all_records]
    throughput = len(all_records) / duration

    scenario = ScenarioResult(
        scenario_id=scenario_id,
        tx_count=tx_total,
        batch_size=batch_size,
        throughput_tps=throughput,
        avg_latency_ms=float(np.mean(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        min_latency_ms=float(np.min(latencies)),
        max_latency_ms=float(np.max(latencies)),
    )
    return scenario, all_records


def save_csv(records: List[TxRecord], scenarios: List[ScenarioResult], output_slug: str, results_dir: Path) -> Tuple[Path, Path]:
    ensure_dir(results_dir)
    detail_path = results_dir / f"blockchain_metrics_{output_slug}.csv"
    summary_path = results_dir / f"blockchain_summary_{output_slug}.csv"

    with detail_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "scenario_id",
                "batch_size",
                "tx_count",
                "tx_hash",
                "send_time_iso",
                "confirm_time_iso",
                "block_number",
                "latency_ms",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.scenario_id,
                    record.batch_size,
                    record.tx_count,
                    record.tx_hash,
                    record.send_time_iso,
                    record.confirm_time_iso,
                    record.block_number,
                    f"{record.latency_ms:.4f}",
                ]
            )

    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "scenario_id",
                "tx_count",
                "batch_size",
                "throughput_tps",
                "avg_latency_ms",
                "p95_latency_ms",
                "p99_latency_ms",
                "min_latency_ms",
                "max_latency_ms",
            ]
        )
        for scenario in scenarios:
            writer.writerow(
                [
                    scenario.scenario_id,
                    scenario.tx_count,
                    scenario.batch_size,
                    f"{scenario.throughput_tps:.4f}",
                    f"{scenario.avg_latency_ms:.4f}",
                    f"{scenario.p95_latency_ms:.4f}",
                    f"{scenario.p99_latency_ms:.4f}",
                    f"{scenario.min_latency_ms:.4f}",
                    f"{scenario.max_latency_ms:.4f}",
                ]
            )

    return detail_path, summary_path


def plot_throughput(summary_df: pd.DataFrame, output_slug: str, plots_dir: Path) -> Path:
    ensure_dir(plots_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    for tx_count, group in summary_df.groupby("tx_count"):
        group = group.sort_values("batch_size")
        ax.plot(
            group["batch_size"],
            group["throughput_tps"],
            marker="o",
            label=f"{tx_count} tx",
        )
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (TPS)")
    ax.set_title("Throughput vs Batch Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = plots_dir / f"throughput_vs_batch_size_{output_slug}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_latency_hist(records_df: pd.DataFrame, output_slug: str, plots_dir: Path) -> Path:
    ensure_dir(plots_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(records_df["latency_ms"], bins=40, color="#2c7fb8", alpha=0.8)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title("Latency Distribution")
    ax.grid(True, alpha=0.2)
    path = plots_dir / f"latency_histogram_{output_slug}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_latency_vs_load(summary_df: pd.DataFrame, output_slug: str, plots_dir: Path) -> Path:
    ensure_dir(plots_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    for batch_size, group in summary_df.groupby("batch_size"):
        group = group.sort_values("tx_count")
        ax.plot(
            group["tx_count"],
            group["avg_latency_ms"],
            marker="s",
            label=f"batch {batch_size}",
        )
    ax.set_xlabel("Transactions per scenario")
    ax.set_ylabel("Avg latency (ms)")
    ax.set_title("Latency vs Load")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = plots_dir / f"latency_vs_load_{output_slug}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    ensure_dir(args.results_root)
    ensure_dir(args.plots_root)

    if not DEFAULT_SENDER_KEY.exists():
        raise FileNotFoundError(
            f"Missing sender key at {DEFAULT_SENDER_KEY}. Run scripts/generate_assets.py first."
        )

    if args.private_key:
        sender_key_hex = args.private_key
    else:
        sender_key_hex = load_text(args.private_key_file)
    sender_key_hex = sender_key_hex[2:] if sender_key_hex.startswith("0x") else sender_key_hex
    account = Account.from_key(bytes.fromhex(sender_key_hex))
    if args.recipient_address:
        recipient = args.recipient_address
    elif DEFAULT_SEALER_ADDRESS.exists():
        recipient = load_text(DEFAULT_SEALER_ADDRESS)
    else:
        recipient = account.address

    w3 = Web3(Web3.HTTPProvider(args.rpc_url))
    if not w3.is_connected():
        raise ConnectionError(f"Unable to reach RPC at {args.rpc_url}")

    scenario_results: List[ScenarioResult] = []
    records: List[TxRecord] = []

    for tx_total in args.transaction_counts:
        for batch_size in args.batch_sizes:
            scenario, tx_records = run_scenario(w3, account, recipient, args, tx_total, batch_size)
            scenario_results.append(scenario)
            records.extend(tx_records)
            print(
                f"[{scenario.scenario_id}] throughput={scenario.throughput_tps:.2f} TPS, "
                f"avg latency={scenario.avg_latency_ms:.2f} ms"
            )

    slug_parts = ["chain", str(args.chain_id), timestamp_slug()]
    if args.tag:
        slug_parts.insert(1, args.tag)
    output_slug = "_".join(slug_parts)

    detail_path, summary_path = save_csv(records, scenario_results, output_slug, args.results_root)

    records_df = pd.DataFrame([r.__dict__ for r in records])
    summary_df = pd.DataFrame([s.__dict__ for s in scenario_results])

    throughput_plot = plot_throughput(summary_df, output_slug, args.plots_root)
    histogram_plot = plot_latency_hist(records_df, output_slug, args.plots_root)
    latency_vs_load_plot = plot_latency_vs_load(summary_df, output_slug, args.plots_root)

    print("Benchmark complete.")
    print(f"Detail CSV: {detail_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Throughput plot: {throughput_plot}")
    print(f"Latency histogram: {histogram_plot}")
    print(f"Latency-vs-load plot: {latency_vs_load_plot}")


if __name__ == "__main__":
    main()

