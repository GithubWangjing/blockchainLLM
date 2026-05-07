import argparse
import csv
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure end-to-end text generation latency for small causal LMs."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Hugging Face model id to benchmark.",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=[32, 128, 512, 1024],
        help="Prompt token counts to benchmark.",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        nargs="+",
        default=[8, 32, 64],
        help="Number of tokens to generate.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=20,
        help="Number of timed runs per configuration.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("prototypes/llm/results"),
        help="Directory to store CSV outputs.",
    )
    parser.add_argument(
        "--plots-root",
        type=Path,
        default=Path("prototypes/llm/plots"),
        help="Directory to store plots.",
    )
    return parser.parse_args()


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synthesize_prompt_tokens(tokenizer, target_length: int) -> List[int]:
    base_tokens = tokenizer.encode(
        "Patient vitals remain steady. ", add_special_tokens=False
    )
    if not base_tokens:
        raise ValueError("Tokenizer returned empty base token list.")
    repeats = (target_length + len(base_tokens) - 1) // len(base_tokens)
    tokens = (base_tokens * repeats)[:target_length]
    return tokens


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def time_generation(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_tokens: int,
) -> float:
    start = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_tokens,
            do_sample=False,
        )
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_latency_vs_prompt(
    summary_rows: Sequence[Dict], path: Path, title_suffix: str
) -> None:
    plt.figure()
    grouped: Dict[int, List[Tuple[int, float]]] = {}
    for row in summary_rows:
        grouped.setdefault(row["gen_tokens"], []).append(
            (row["prompt_length"], row["mean_latency_ms"])
        )
    for gen_tokens, points in grouped.items():
        points.sort(key=lambda item: item[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=f"{gen_tokens} gen tokens")
    plt.xlabel("Prompt length (tokens)")
    plt.ylabel("Mean latency (ms)")
    plt.title(f"Latency vs Prompt Length {title_suffix}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_latency_vs_gen_tokens(
    summary_rows: Sequence[Dict], path: Path, title_suffix: str
) -> None:
    plt.figure()
    grouped: Dict[int, List[Tuple[int, float]]] = {}
    for row in summary_rows:
        grouped.setdefault(row["prompt_length"], []).append(
            (row["gen_tokens"], row["mean_latency_ms"])
        )
    for prompt_length, points in grouped.items():
        points.sort(key=lambda item: item[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=f"{prompt_length} prompt tokens")
    plt.xlabel("Generated tokens")
    plt.ylabel("Mean latency (ms)")
    plt.title(f"Latency vs Generated Tokens {title_suffix}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()
    device = resolve_device()

    print(f"Loading model {args.model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = args.results_root
    plots_root = args.plots_root
    ensure_dirs(results_root, plots_root)

    detail_rows = []
    summary_rows = []

    for prompt_length in args.prompt_lengths:
        prompt_tokens = synthesize_prompt_tokens(tokenizer, prompt_length)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        for gen_tokens in args.gen_tokens:
            latencies = []
            print(
                f"Benchmarking prompt={prompt_length} tokens, gen={gen_tokens} tokens..."
            )
            for run_idx in range(args.repetitions):
                latency_ms = time_generation(
                    model,
                    input_ids,
                    attention_mask,
                    gen_tokens,
                )
                latencies.append(latency_ms)
                detail_rows.append(
                    {
                        "model": args.model_name,
                        "device": str(device),
                        "prompt_length": prompt_length,
                        "gen_tokens": gen_tokens,
                        "run_index": run_idx,
                        "latency_ms": latency_ms,
                    }
                )

            mean_latency = statistics.mean(latencies)
            std_latency = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
            summary_rows.append(
                {
                    "model": args.model_name,
                    "device": str(device),
                    "prompt_length": prompt_length,
                    "gen_tokens": gen_tokens,
                    "mean_latency_ms": mean_latency,
                    "std_latency_ms": std_latency,
                    "repetitions": args.repetitions,
                }
            )

    detail_path = results_root / f"llm_latency_detail_{timestamp}.csv"
    summary_path = results_root / f"llm_latency_summary_{timestamp}.csv"
    write_csv(
        detail_path,
        [
            "model",
            "device",
            "prompt_length",
            "gen_tokens",
            "run_index",
            "latency_ms",
        ],
        detail_rows,
    )
    write_csv(
        summary_path,
        [
            "model",
            "device",
            "prompt_length",
            "gen_tokens",
            "mean_latency_ms",
            "std_latency_ms",
            "repetitions",
        ],
        summary_rows,
    )

    prompt_plot = plots_root / f"latency_vs_prompt_length_{timestamp}.png"
    gen_plot = plots_root / f"latency_vs_gen_tokens_{timestamp}.png"
    title_suffix = f"({args.model_name} on {device})"
    plot_latency_vs_prompt(summary_rows, prompt_plot, title_suffix)
    plot_latency_vs_gen_tokens(summary_rows, gen_plot, title_suffix)

    print(f"Wrote detail CSV to {detail_path}")
    print(f"Wrote summary CSV to {summary_path}")
    print(f"Wrote plots to {prompt_plot} and {gen_plot}")


if __name__ == "__main__":
    main()

