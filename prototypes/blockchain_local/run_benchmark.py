"""Benchmark script for the Hardhat-based local devnet."""

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
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

HARDHAT_MNEMONIC = "test test test test test test test test test test test junk"
HARDHAT_CHAIN_ID = 31337

Account.enable_unaudited_hdwallet_features()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_slug() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TxRecord:
    scenario_id: str
    batch_size: int
    tx_count: int
    repetition: int
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
    repetition: int
    throughput_tps: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float


@dataclass
class ComboSummary:
    tx_count: int
    batch_size: int
    repetitions: int
    mean_throughput_tps: float
    std_throughput_tps: float
    mean_avg_latency_ms: float
    std_avg_latency_ms: float


def sample_std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def summarize_combos(scenarios: List[ScenarioResult]) -> List[ComboSummary]:
    grouped: Dict[Tuple[int, int], List[ScenarioResult]] = {}
    for scenario in scenarios:
        grouped.setdefault((scenario.tx_count, scenario.batch_size), []).append(scenario)

    summaries: List[ComboSummary] = []
    for (tx_count, batch_size), items in grouped.items():
        throughput_values = [s.throughput_tps for s in items]
        avg_latency_values = [s.avg_latency_ms for s in items]
        summaries.append(
            ComboSummary(
                tx_count=tx_count,
                batch_size=batch_size,
                repetitions=len(items),
                mean_throughput_tps=float(np.mean(throughput_values)),
                std_throughput_tps=sample_std(throughput_values),
                mean_avg_latency_ms=float(np.mean(avg_latency_values)),
                std_avg_latency_ms=sample_std(avg_latency_values),
            )
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hardhat devnet latency benchmark")
    parser.add_argument("--rpc-url", default="http://127.0.0.1:8545", help="Hardhat JSON-RPC URL")
    parser.add_argument("--chain-id", type=int, default=HARDHAT_CHAIN_ID, help="Chain ID for signing")
    parser.add_argument(
        "--transaction-counts",
        type=int,
        nargs="+",
        default=[200, 400, 800, 1200],
        help="Number of transactions per scenario",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100],
        help="How many transactions to send before polling confirmations",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of times to repeat each (tx_count, batch_size) scenario",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=200,
        help="Maximum outstanding tx before forcing receipt polling",
    )
    parser.add_argument("--gas-limit", type=int, default=21000, help="Gas limit per transaction")
    parser.add_argument("--gas-price-gwei", type=float, default=1, help="Gas price (in gwei)")
    parser.add_argument(
        "--value-wei",
        type=int,
        default=1,
        help="Value (wei) transferred per transaction",
    )
    parser.add_argument(
        "--private-key",
        type=str,
        default=None,
        help="Optional hex private key (0x...) overriding the default Hardhat mnemonic",
    )
    parser.add_argument(
        "--private-key-file",
        type=Path,
        default=None,
        help="Optional file containing a hex private key",
    )
    parser.add_argument(
        "--sender-index",
        type=int,
        default=0,
        help="HD wallet index to derive the sender (Hardhat mnemonic)",
    )
    parser.add_argument(
        "--recipient-index",
        type=int,
        default=1,
        help="HD wallet index to derive the recipient",
    )
    parser.add_argument(
        "--recipient-address",
        type=str,
        default=None,
        help="Optional explicit recipient address",
    )
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Seconds between receipt polls")
    parser.add_argument("--tag", default="", help="Optional output slug tag")
    parser.add_argument("--results-root", type=Path, default=RESULTS_DIR, help="Directory for CSVs")
    parser.add_argument("--plots-root", type=Path, default=PLOTS_DIR, help="Directory for plots")
    return parser.parse_args()


def derive_account(index: int):
    path = f"m/44'/60'/0'/0/{index}"
    return Account.from_mnemonic(HARDHAT_MNEMONIC, account_path=path)


def load_account(args: argparse.Namespace):
    if args.private_key:
        key_hex = args.private_key
    elif args.private_key_file and args.private_key_file.exists():
        key_hex = args.private_key_file.read_text(encoding="utf-8").strip()
    else:
        acct = derive_account(args.sender_index)
        return acct

    key_hex = key_hex[2:] if key_hex.startswith("0x") else key_hex
    return Account.from_key(bytes.fromhex(key_hex))


def load_recipient(args: argparse.Namespace, sender_address: str) -> str:
    if args.recipient_address:
        return Web3.to_checksum_address(args.recipient_address)
    if args.recipient_index == args.sender_index:
        return sender_address
    recipient = derive_account(args.recipient_index)
    return recipient.address


def poll_receipts(
    w3: Web3,
    pending: Dict[bytes, Dict[str, float]],
    scenario_id: str,
    batch_size: int,
    tx_count: int,
    repetition: int,
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
                        repetition=repetition,
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
    repetition: int,
) -> Tuple[ScenarioResult, List[TxRecord]]:
    scenario_id = f"tx{tx_total}_batch{batch_size}_rep{repetition}"
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
                        repetition,
                        args.poll_interval,
                    )
                )
        all_records.extend(
            poll_receipts(
                w3,
                pending,
                scenario_id,
                batch_size,
                tx_total,
                repetition,
                args.poll_interval,
            )
        )

    pbar.close()

    while pending:
        all_records.extend(
            poll_receipts(
                w3, pending, scenario_id, batch_size, tx_total, repetition, args.poll_interval
            )
        )

    if not all_records:
        raise RuntimeError("No confirmed transactions recorded.")

    send_start = min(r.send_monotonic for r in all_records)
    confirm_end = max(r.confirm_monotonic for r in all_records)
    duration = max(confirm_end - send_start, 1e-9)
    latencies = [r.latency_ms for r in all_records]
    throughput = len(all_records) / duration

    scenario = ScenarioResult(
        scenario_id=scenario_id,
        tx_count=tx_total,
        batch_size=batch_size,
        repetition=repetition,
        throughput_tps=throughput,
        avg_latency_ms=float(np.mean(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        p99_latency_ms=float(np.percentile(latencies, 99)),
        min_latency_ms=float(np.min(latencies)),
        max_latency_ms=float(np.max(latencies)),
    )
    return scenario, all_records


def save_csvs(
    records: List[TxRecord],
    scenarios: List[ScenarioResult],
    combo_summaries: List[ComboSummary],
    output_slug: str,
    results_dir: Path,
) -> Tuple[Path, Path, Path]:
    ensure_dir(results_dir)
    detail_path = results_dir / f"blockchain_metrics_{output_slug}_detail.csv"
    runs_path = results_dir / f"blockchain_summary_{output_slug}_runs.csv"
    mean_path = results_dir / f"blockchain_summary_{output_slug}_mean.csv"

    with detail_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "scenario_id",
                "tx_count",
                "batch_size",
                "repetition",
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
                    record.tx_count,
                    record.batch_size,
                    record.repetition,
                    record.tx_hash,
                    record.send_time_iso,
                    record.confirm_time_iso,
                    record.block_number,
                    f"{record.latency_ms:.4f}",
                ]
            )

    with runs_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "scenario_id",
                "tx_count",
                "batch_size",
                "repetition",
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
                    scenario.repetition,
                    f"{scenario.throughput_tps:.4f}",
                    f"{scenario.avg_latency_ms:.4f}",
                    f"{scenario.p95_latency_ms:.4f}",
                    f"{scenario.p99_latency_ms:.4f}",
                    f"{scenario.min_latency_ms:.4f}",
                    f"{scenario.max_latency_ms:.4f}",
                ]
            )

    with mean_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "tx_count",
                "batch_size",
                "repetitions",
                "mean_throughput_tps",
                "std_throughput_tps",
                "mean_avg_latency_ms",
                "std_avg_latency_ms",
            ]
        )
        for summary in combo_summaries:
            writer.writerow(
                [
                    summary.tx_count,
                    summary.batch_size,
                    summary.repetitions,
                    f"{summary.mean_throughput_tps:.4f}",
                    f"{summary.std_throughput_tps:.4f}",
                    f"{summary.mean_avg_latency_ms:.4f}",
                    f"{summary.std_avg_latency_ms:.4f}",
                ]
            )

    return detail_path, runs_path, mean_path


def plot_metric_vs_load(
    mean_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    filename_prefix: str,
    output_slug: str,
    plots_dir: Path,
) -> Path:
    ensure_dir(plots_dir)
    if mean_df.empty:
        raise ValueError("Mean summary dataframe is empty; nothing to plot.")
    fig, ax = plt.subplots(figsize=(8, 5))
    for batch_size, group in sorted(mean_df.groupby("batch_size"), key=lambda kv: kv[0]):
        group = group.sort_values("tx_count")
        ax.plot(group["tx_count"], group[metric_col], marker="o", label=f"batch {batch_size}")
    ax.set_xlabel("Transaction count")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Batch size")
    path = plots_dir / f"{filename_prefix}_{output_slug}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_throughput_vs_load(mean_df: pd.DataFrame, output_slug: str, plots_dir: Path) -> Path:
    return plot_metric_vs_load(
        mean_df,
        metric_col="mean_throughput_tps",
        ylabel="Throughput (TPS)",
        title="Throughput vs Transaction Count",
        filename_prefix="throughput_vs_load",
        output_slug=output_slug,
        plots_dir=plots_dir,
    )


def plot_avg_latency_vs_load(mean_df: pd.DataFrame, output_slug: str, plots_dir: Path) -> Path:
    return plot_metric_vs_load(
        mean_df,
        metric_col="mean_avg_latency_ms",
        ylabel="Average latency (ms)",
        title="Average Latency vs Transaction Count",
        filename_prefix="latency_vs_load",
        output_slug=output_slug,
        plots_dir=plots_dir,
    )


def plot_latency_hist(records_df: pd.DataFrame, output_slug: str, plots_dir: Path) -> Path:
    ensure_dir(plots_dir)
    if records_df.empty:
        raise ValueError("No transaction records available for histogram plot.")
    max_tx = records_df["tx_count"].max()
    subset = records_df[records_df["tx_count"] == max_tx]
    max_batch = subset["batch_size"].max()
    subset = subset[subset["batch_size"] == max_batch]
    title_suffix = f"tx={max_tx}, batch={max_batch}"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(subset["latency_ms"], bins=40, color="#3182bd", alpha=0.85)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Latency Distribution ({title_suffix})")
    ax.grid(True, alpha=0.2)
    path = plots_dir / f"latency_histogram_tx{max_tx}_batch{max_batch}_{output_slug}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be >= 1")
    ensure_dir(args.results_root)
    ensure_dir(args.plots_root)

    account = load_account(args)
    recipient = load_recipient(args, account.address)

    w3 = Web3(Web3.HTTPProvider(args.rpc_url))
    if not w3.is_connected():
        raise ConnectionError(f"Unable to reach RPC at {args.rpc_url}")

    scenario_results: List[ScenarioResult] = []
    records: List[TxRecord] = []

    for tx_total in args.transaction_counts:
        for batch_size in args.batch_sizes:
            for repetition in range(1, args.repetitions + 1):
                scenario, tx_records = run_scenario(
                    w3, account, recipient, args, tx_total, batch_size, repetition
                )
                scenario_results.append(scenario)
                records.extend(tx_records)
                print(
                    f"[{scenario.scenario_id}] throughput={scenario.throughput_tps:.2f} TPS, "
                    f"avg latency={scenario.avg_latency_ms:.2f} ms"
                )

    slug_parts = ["hardhat", str(args.chain_id), timestamp_slug()]
    if args.tag:
        slug_parts.insert(1, args.tag)
    output_slug = "_".join(slug_parts)

    combo_summaries = summarize_combos(scenario_results)
    detail_path, runs_path, mean_path = save_csvs(
        records, scenario_results, combo_summaries, output_slug, args.results_root
    )

    records_df = pd.DataFrame([r.__dict__ for r in records])
    combo_df = pd.DataFrame([s.__dict__ for s in combo_summaries])

    throughput_plot = plot_throughput_vs_load(combo_df, output_slug, args.plots_root)
    latency_vs_load_plot = plot_avg_latency_vs_load(combo_df, output_slug, args.plots_root)
    histogram_plot = plot_latency_hist(records_df, output_slug, args.plots_root)

    print("Benchmark complete.")
    print(f"Detail CSV: {detail_path}")
    print(f"Runs summary CSV: {runs_path}")
    print(f"Mean summary CSV: {mean_path}")
    print(f"Throughput plot: {throughput_plot}")
    print(f"Latency-vs-load plot: {latency_vs_load_plot}")
    print(f"Latency histogram: {histogram_plot}")


if __name__ == "__main__":
    main()

