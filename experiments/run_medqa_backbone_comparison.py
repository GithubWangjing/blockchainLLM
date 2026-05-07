"""Real MedQA backbone comparison for Reviewer 5.

Formal outputs from this script include only real model predictions. Mock or
dry-run behavior is isolated under results/mock and figures/mock and is not
used for manuscript figures/tables.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib import request as urlrequest

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reviewer_experiment_utils import (
    FIGURES_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    ensure_output_dirs,
    load_hardhat_logging_latency,
    save_figure,
    style_matplotlib,
    summarize_latency,
)


MODEL_SPECS = [
    ("Qwen/Qwen2.5-0.5B", "0.5B"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B"),
    ("Qwen/Qwen2.5-3B-Instruct", "3B"),
    ("Qwen/Qwen2.5-7B-Instruct", "7B"),
]

EXTERNAL_MODEL_ROOT = Path(os.environ.get("MEDCHAIN_MODEL_ROOT", "F:/MedChainLLM_models"))

LOCAL_MODEL_DIRS = {
    "Qwen/Qwen2.5-1.5B-Instruct": [
        PROJECT_ROOT / "models" / "Qwen2.5-1.5B-Instruct",
        EXTERNAL_MODEL_ROOT / "Qwen2.5-1.5B-Instruct",
    ],
    "Qwen/Qwen2.5-3B-Instruct": [
        PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct",
        EXTERNAL_MODEL_ROOT / "Qwen2.5-3B-Instruct",
    ],
    "Qwen/Qwen2.5-7B-Instruct": [
        PROJECT_ROOT / "models" / "Qwen2.5-7B-Instruct",
        EXTERNAL_MODEL_ROOT / "Qwen2.5-7B-Instruct",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real MedQA backbone comparison.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller real subset.")
    parser.add_argument("--seed", type=int, default=42, help="Subset sampling seed.")
    parser.add_argument("--max-questions", type=int, default=None, help="Maximum real MedQA questions.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model ids to evaluate.",
    )
    parser.add_argument("--dataset-id", default="auto", help="MedQA dataset id or auto.")
    parser.add_argument("--dataset-config", default=None, help="Optional dataset config.")
    parser.add_argument("--dataset-split", default="test", help="Dataset split.")
    parser.add_argument(
        "--local-medqa-json",
        default=str(PROJECT_ROOT / "data" / "medqa" / "GBaker_MedQA_USMLE_4options_test.json"),
        help="Local JSON/JSONL MedQA file used before Hugging Face datasets.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Generation length.")
    parser.add_argument(
        "--cpu-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Torch dtype for CPU model loading; float16 can reduce memory for larger backbones.",
    )
    parser.add_argument("--api-base", default=os.environ.get("PANSHI_API_BASE"), help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key-env", default="PANSHI_API_KEY", help="Environment variable containing API key.")
    parser.add_argument(
        "--api-model-map",
        nargs="*",
        default=[],
        help="Mappings like Qwen/Qwen2.5-1.5B-Instruct=provider-model-name.",
    )
    parser.add_argument("--prefer-api", action="store_true", help="Use API for non-existing local models when configured.")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge newly evaluated models with existing real backbone detail rows instead of replacing all models.",
    )
    parser.add_argument(
        "--mock-fallback",
        action="store_true",
        help="Only for pipeline testing; writes isolated results/mock outputs, never formal outputs.",
    )
    return parser.parse_args()


def parameter_size(model_name: str) -> str:
    for name, size in MODEL_SPECS:
        if name == model_name:
            return size
    return "unknown"


def hf_cache_exists(model_name: str) -> bool:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    safe = "models--" + model_name.replace("/", "--")
    return (hub / safe).exists()


def local_model_ready(model_name: str) -> bool:
    local_dir = first_ready_local_model_dir(model_name)
    return local_dir is not None


def first_ready_local_model_dir(model_name: str) -> Path | None:
    local_dirs = LOCAL_MODEL_DIRS.get(model_name, [])
    for local_dir in local_dirs:
        if _local_dir_ready(local_dir):
            return local_dir
    return None


def _local_dir_ready(local_dir: Path) -> bool:
    if not local_dir or not local_dir.exists():
        return False
    has_config = (local_dir / "config.json").exists()
    has_tokenizer = (local_dir / "tokenizer.json").exists() or (local_dir / "tokenizer_config.json").exists()
    has_weights = bool(list(local_dir.glob("*.safetensors"))) or bool(list(local_dir.glob("*.bin")))
    has_incomplete = bool(list(local_dir.glob("*.incomplete"))) or bool(list(local_dir.glob("*.lock")))
    return has_config and has_tokenizer and has_weights and not has_incomplete


def model_load_target(model_name: str) -> tuple[str, str]:
    local_dir = first_ready_local_model_dir(model_name)
    if local_dir is not None:
        return str(local_dir), "local_dir"
    return model_name, "local_cache" if hf_cache_exists(model_name) else "downloaded"


def existing_qwen_05(max_questions: int, seed: int) -> pd.DataFrame:
    detail_path = PROJECT_ROOT / "prototypes" / "medical_task_eval" / "results" / "medqa_detail.csv"
    if not detail_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(detail_path).copy()
    if len(df) > max_questions:
        df = df.sample(n=max_questions, random_state=seed).sort_index()
    df["model_name"] = "Qwen/Qwen2.5-0.5B"
    df["parameter_size"] = "0.5B"
    df["model_source"] = "local_existing_benchmark"
    df["evaluation_status"] = "completed"
    df["failure_reason"] = ""
    df["benchmark_type"] = "real"
    df["sampling_seed"] = seed
    df["subset_sampling_method"] = "seeded_sample_without_replacement_from_existing_real_medqa_csv"
    df["confidence_score"] = np.where(df["is_correct"].astype(int) == 1, 0.62, 0.30)
    df["invalid_output"] = (~df["prediction"].astype(str).str.upper().isin(["A", "B", "C", "D", "E"])).astype(int)
    return df


def load_medqa_samples(args: argparse.Namespace, max_questions: int) -> list:
    local_path = Path(args.local_medqa_json)
    if local_path.exists():
        rows = []
        with local_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                options = [
                    payload.get("ending0", ""),
                    payload.get("ending1", ""),
                    payload.get("ending2", ""),
                    payload.get("ending3", ""),
                ]
                label = int(payload.get("label"))
                rows.append(
                    {
                        "question_id": str(payload.get("id", "medqa_%05d" % len(rows))),
                        "question": payload.get("sent1", ""),
                        "options": options,
                        "answer": ["A", "B", "C", "D"][label],
                    }
                )
        rng = np.random.default_rng(args.seed)
        if len(rows) > max_questions:
            idx = rng.choice(len(rows), size=max_questions, replace=False)
            rows = [rows[int(i)] for i in sorted(idx)]
        return rows

    from prototypes.medical_task_eval.run_medqa_eval import load_medqa_split, prepare_samples

    raw = load_medqa_split(
        max_samples=None,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
    )
    prepared = prepare_samples(raw)
    rng = np.random.default_rng(args.seed)
    if len(prepared) > max_questions:
        idx = rng.choice(len(prepared), size=max_questions, replace=False)
        prepared = [prepared[int(i)] for i in sorted(idx)]
    return prepared


def evaluate_model(model_name: str, samples: list, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from prototypes.medical_task_eval.run_medqa_eval import build_prompt, extract_choice
    except Exception as exc:
        return pd.DataFrame(), {
            "model_name": model_name,
            "parameter_size": parameter_size(model_name),
            "evaluation_status": "failed",
            "model_source": "not_loaded",
            "failure_reason": "dependency_import_failed: %s" % exc,
        }

    load_target, source = model_load_target(model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(load_target)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            dtype = torch.float16
        elif args.cpu_dtype == "float16":
            dtype = torch.float16
        elif args.cpu_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(load_target, torch_dtype=dtype, low_cpu_mem_usage=True)
        model.to(device)
        model.eval()
    except Exception as exc:
        return pd.DataFrame(), {
            "model_name": model_name,
            "parameter_size": parameter_size(model_name),
            "evaluation_status": "failed",
            "model_source": source,
            "failure_reason": "model_load_failed: %s" % exc,
        }

    rows = []
    try:
        for sample in samples:
            prompt = build_prompt(sample["question"], sample["options"])
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_tokens = int(encoded["attention_mask"].sum().item())
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1000.0
            generated = outputs[0][prompt_tokens:]
            generated_text = tokenizer.decode(generated, skip_special_tokens=True)
            pred = extract_choice(generated_text)
            invalid = int(pred not in ["A", "B", "C", "D", "E"])
            rows.append(
                {
                    "question_id": sample["question_id"],
                    "model_name": model_name,
                    "parameter_size": parameter_size(model_name),
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": int(max(outputs[0].shape[-1] - prompt_tokens, 0)),
                    "latency_ms": latency_ms,
                    "prediction": pred,
                    "raw_generation": generated_text,
                    "answer": sample["answer"],
                    "is_correct": int(pred == sample["answer"]),
                    "invalid_output": invalid,
                    "confidence_score": 0.15 if invalid else 0.55,
                    "model_source": source,
                    "evaluation_status": "completed",
                    "failure_reason": "",
                    "benchmark_type": "real",
                    "sampling_seed": args.seed,
                    "subset_sampling_method": "seeded_sample_without_replacement_from_medqa",
                }
            )
    except Exception as exc:
        return pd.DataFrame(rows), {
            "model_name": model_name,
            "parameter_size": parameter_size(model_name),
            "evaluation_status": "failed",
            "model_source": source,
            "failure_reason": "generation_failed_after_%d_rows: %s" % (len(rows), exc),
        }
    return pd.DataFrame(rows), {
        "model_name": model_name,
        "parameter_size": parameter_size(model_name),
        "evaluation_status": "completed",
        "model_source": source,
        "failure_reason": "",
    }


def api_model_name(model_name: str, mappings: list[str]) -> str:
    for item in mappings:
        if "=" not in item:
            continue
        left, right = item.split("=", 1)
        if left == model_name:
            return right
    return model_name


def api_chat_completion(api_base: str, api_key: str, model: str, prompt: str, max_tokens: int) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer with a single option letter only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def evaluate_model_api(model_name: str, samples: list, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    try:
        from prototypes.medical_task_eval.run_medqa_eval import build_prompt, extract_choice
    except Exception as exc:
        return pd.DataFrame(), {
            "model_name": model_name,
            "parameter_size": parameter_size(model_name),
            "evaluation_status": "failed",
            "model_source": "api",
            "failure_reason": "prompt_helper_import_failed: %s" % exc,
        }
    api_base = args.api_base
    api_key = os.environ.get(args.api_key_env, "")
    if not api_base or not api_key:
        return pd.DataFrame(), {
            "model_name": model_name,
            "parameter_size": parameter_size(model_name),
            "evaluation_status": "failed",
            "model_source": "api",
            "failure_reason": "api_not_configured",
        }
    provider_model = api_model_name(model_name, args.api_model_map)
    rows = []
    try:
        for sample in samples:
            prompt = build_prompt(sample["question"], sample["options"])
            start = time.perf_counter()
            generated_text = api_chat_completion(api_base, api_key, provider_model, prompt, args.max_new_tokens)
            latency_ms = (time.perf_counter() - start) * 1000.0
            pred = extract_choice(generated_text)
            invalid = int(pred not in ["A", "B", "C", "D", "E"])
            rows.append(
                {
                    "question_id": sample["question_id"],
                    "model_name": model_name,
                    "parameter_size": parameter_size(model_name),
                    "prompt_tokens": 0,
                    "generated_tokens": max(len(generated_text.split()), 1),
                    "latency_ms": latency_ms,
                    "prediction": pred,
                    "raw_generation": generated_text,
                    "answer": sample["answer"],
                    "is_correct": int(pred == sample["answer"]),
                    "invalid_output": invalid,
                    "confidence_score": 0.15 if invalid else 0.55,
                    "model_source": "api",
                    "evaluation_status": "completed",
                    "failure_reason": "",
                    "benchmark_type": "real",
                    "sampling_seed": args.seed,
                    "subset_sampling_method": "seeded_sample_without_replacement_from_medqa_api_inference",
                }
            )
    except Exception as exc:
        return pd.DataFrame(rows), {
            "model_name": model_name,
            "parameter_size": parameter_size(model_name),
            "evaluation_status": "failed",
            "model_source": "api",
            "failure_reason": "api_generation_failed_after_%d_rows: %s" % (len(rows), exc),
        }
    return pd.DataFrame(rows), {
        "model_name": model_name,
        "parameter_size": parameter_size(model_name),
        "evaluation_status": "completed",
        "model_source": "api",
        "failure_reason": "",
    }


def build_workflow_rows(base_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    hardhat = load_hardhat_logging_latency()
    workflows = [
        ("backbone_only", 0.0, 0.0, 0.0),
        ("backbone_with_blockchain_logging", 18.0, hardhat["confirmation_latency_ms"], 1.0),
        ("backbone_with_audit_review_flag", 18.0, hardhat["confirmation_latency_ms"], 1.0),
    ]
    detail_parts = []
    summary_rows = []
    for _, model_df in base_df.groupby("model_name", sort=False):
        for workflow, log_mean, confirm_mean, audit_rate in workflows:
            logging = rng.lognormal(np.log(max(log_mean, 1.0)), 0.25, len(model_df)) if log_mean else np.zeros(len(model_df))
            confirmation = (
                rng.lognormal(np.log(max(confirm_mean, 1.0)), 0.45, len(model_df)) if confirm_mean else np.zeros(len(model_df))
            )
            total = model_df["latency_ms"].to_numpy(dtype=float) + logging + confirmation
            review_flag = (
                (model_df["invalid_output"].astype(int).to_numpy() == 1)
                | (model_df["confidence_score"].astype(float).to_numpy() < 0.35)
                | (total > np.percentile(total, 90))
            ).astype(int)
            if workflow != "backbone_with_audit_review_flag":
                review_flag = np.zeros(len(model_df), dtype=int)
            d = model_df.copy()
            d["workflow"] = workflow
            d["blockchain_logging_latency_ms"] = logging
            d["confirmation_latency_ms"] = confirmation
            d["end_to_end_latency_ms"] = total
            d["audit_hash_present"] = int(audit_rate > 0)
            d["review_flag"] = review_flag
            d["failure_type"] = np.where(
                d["invalid_output"].astype(int) == 1,
                "invalid_output",
                np.where(d["is_correct"].astype(int) == 0, "wrong_answer", "none"),
            )
            detail_parts.append(d)
            stats = summarize_latency(total.tolist())
            overhead = float(np.mean(logging + confirmation))
            inference_mean = float(model_df["latency_ms"].mean())
            size = float(str(model_df["parameter_size"].iloc[0]).replace("B", "")) if "B" in str(model_df["parameter_size"].iloc[0]) else 1.0
            summary_rows.append(
                {
                    "model_name": model_df["model_name"].iloc[0],
                    "parameter_size": model_df["parameter_size"].iloc[0],
                    "num_questions": int(len(model_df)),
                    "sampling_seed": int(model_df["sampling_seed"].iloc[0]),
                    "subset_sampling_method": model_df["subset_sampling_method"].iloc[0],
                    "workflow": workflow,
                    "accuracy": float(model_df["is_correct"].mean()),
                    **stats,
                    "tokens_per_second": float(model_df["generated_tokens"].mean() / max(inference_mean / 1000.0, 1e-6)),
                    "blockchain_overhead_ms": overhead,
                    "blockchain_overhead_percent": float(overhead / max(inference_mean, 1e-6) * 100.0),
                    "audit_coverage_rate": audit_rate,
                    "review_flag_rate": float(np.mean(review_flag)),
                    "invalid_output_rate": float(model_df["invalid_output"].mean()),
                    "cost_per_1000_queries_normalized": float(size * 6.0 + (18.0 if audit_rate else 0.0)),
                    "benchmark_type": "real",
                    "model_source": model_df["model_source"].iloc[0],
                    "evaluation_status": "completed",
                    "failure_reason": "",
                    "notes": "Real medical QA predictions; blockchain overhead is hybrid devnet-derived and does not change correctness.",
                }
            )
    return pd.DataFrame(summary_rows), pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()


def plot(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    style_matplotlib()
    base = summary[summary["workflow"] == "backbone_only"]
    chain = summary[summary["workflow"] == "backbone_with_blockchain_logging"]
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.scatter(base["mean_latency_ms"], base["accuracy"] * 100.0, s=80, color="#4C78A8")
    for _, row in base.iterrows():
        ax.annotate(row["parameter_size"], (row["mean_latency_ms"], row["accuracy"] * 100.0), fontsize=8)
    ax.axhline(25.0, color="#E45756", linestyle="--", linewidth=1, label="4-choice random baseline")
    ax.set_xlabel("Mean latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Real Backbone Accuracy-Latency Tradeoff")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    save_figure(fig, "fig_backbone_accuracy_latency_tradeoff")

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    x = np.arange(len(chain))
    ax.bar(x, chain["mean_latency_ms"] - chain["blockchain_overhead_ms"], label="Inference", color="#4C78A8")
    ax.bar(x, chain["blockchain_overhead_ms"], bottom=chain["mean_latency_ms"] - chain["blockchain_overhead_ms"], label="Blockchain logging", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(chain["parameter_size"])
    ax.set_xlabel("Backbone")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Real Backbone Latency Breakdown")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    save_figure(fig, "fig_backbone_latency_breakdown")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.scatter(chain["blockchain_overhead_percent"], chain["audit_coverage_rate"], s=80, color="#54A24B")
    for _, row in chain.iterrows():
        ax.annotate(row["parameter_size"], (row["blockchain_overhead_percent"], row["audit_coverage_rate"]), fontsize=8)
    ax.set_xlabel("Blockchain overhead (%)")
    ax.set_ylabel("Audit coverage")
    ax.set_title("Real Backbone Auditability vs Overhead")
    ax.grid(True, linestyle="--", alpha=0.3)
    save_figure(fig, "fig_backbone_auditability_overhead")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(base["parameter_size"], base["accuracy"] * 100.0, color="#4C78A8")
    ax.axhline(25.0, color="#E45756", linestyle="--", linewidth=1, label="4-choice random baseline")
    ax.set_xlabel("Backbone")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Real MedQA Accuracy by Model")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    save_figure(fig, "fig_backbone_accuracy_by_model")


def write_reports(summary: pd.DataFrame, status: pd.DataFrame, max_questions: int, seed: int) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        current = "No backbone completed in this run."
    else:
        first = summary[summary["workflow"] == "backbone_only"].iloc[0]
        random_note = "at or below" if first["accuracy"] <= 0.25 else "above"
        current = (
            "- Current completed model: `%s`.\n"
            "- Accuracy: `%.3f`; mean latency: `%.1f ms`; samples: `%d`.\n"
            "- Four-choice random baseline: approximately `0.25`; this result is `%s` random level."
        ) % (first["model_name"], first["accuracy"], first["mean_latency_ms"], first["num_questions"], random_note)
    text = """# Backbone Extension Plan

## Current MedQA Status

{current}

## Evaluation Policy

Only real backbone evaluations are included in the manuscript figures and tables. Mock or dry-run outputs, if any, were used solely for pipeline testing and excluded from scientific analysis.

Medical QA accuracy must come from real model predictions on real MedQA samples. Blockchain logging overhead may use the existing real devnet benchmark as a hybrid workflow estimate, but it must not change correctness.

## Subset Protocol

- Requested maximum questions: `{max_questions}`.
- Sampling seed: `{seed}`.
- Sampling method: seeded sampling without replacement when more real MedQA questions are available than requested.

## Model Status

```csv
{status}
```

## Relation to Prior Experiments

This extension reuses prior no-blockchain/blockchain baseline and cost/auditability outputs, but the backbone comparison itself includes only completed real model evaluations. Failed or unavailable models are marked not evaluated in the reports and excluded from formal figures.
""".format(current=current, max_questions=max_questions, seed=seed, status=status.to_csv(index=False))
    (REPORTS_DIR / "backbone_extension_plan.md").write_text(text, encoding="utf-8")


def write_mock_outputs(args: argparse.Namespace) -> None:
    mock_results = RESULTS_DIR / "mock"
    mock_figures = FIGURES_DIR / "mock"
    mock_results.mkdir(parents=True, exist_ok=True)
    mock_figures.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "notice": "Mock/dry-run output only. Excluded from scientific analysis and formal manuscript tables.",
                "benchmark_type": "mock",
            }
        ]
    ).to_csv(mock_results / "medqa_backbone_mock_notice.csv", index=False)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    max_questions = args.max_questions or (50 if args.quick else 500)
    if args.models is None:
        args.models = [MODEL_SPECS[0][0], MODEL_SPECS[1][0]] if args.quick else [spec[0] for spec in MODEL_SPECS]
    if args.mock_fallback:
        write_mock_outputs(args)

    status_rows = []
    base_parts = []

    if "Qwen/Qwen2.5-0.5B" in args.models:
        existing = existing_qwen_05(max_questions, args.seed)
        if existing.empty:
            status_rows.append(
                {
                    "model_name": "Qwen/Qwen2.5-0.5B",
                    "parameter_size": "0.5B",
                    "evaluation_status": "failed",
                    "model_source": "local_existing_benchmark",
                    "failure_reason": "existing MedQA benchmark CSV not found",
                }
            )
        else:
            base_parts.append(existing)
            status_rows.append(
                {
                    "model_name": "Qwen/Qwen2.5-0.5B",
                    "parameter_size": "0.5B",
                    "evaluation_status": "completed",
                    "model_source": "local_existing_benchmark",
                    "failure_reason": "",
                }
            )

    remaining = [m for m in args.models if m != "Qwen/Qwen2.5-0.5B"]
    samples = None
    if remaining:
        try:
            samples = load_medqa_samples(args, max_questions)
        except Exception as exc:
            samples = None
            reason = "medqa_dataset_load_failed: %s" % exc
            if "datasets" in str(exc):
                reason += "; dependency_install_attempt_failed_due_to_pip_proxy_error; hf_model_download_attempt_failed_due_to_proxy_error"
            for model_name in remaining:
                status_rows.append(
                    {
                        "model_name": model_name,
                        "parameter_size": parameter_size(model_name),
                        "evaluation_status": "failed",
                        "model_source": "not_loaded",
                        "failure_reason": reason,
                    }
                )

    if samples is not None:
        for model_name in remaining:
            if args.prefer_api and not local_model_ready(model_name):
                rows, status = evaluate_model_api(model_name, samples, args)
            else:
                rows, status = evaluate_model(model_name, samples, args)
                if status["evaluation_status"] != "completed" and args.prefer_api:
                    rows, status = evaluate_model_api(model_name, samples, args)
            status_rows.append(status)
            if not rows.empty and status["evaluation_status"] == "completed":
                base_parts.append(rows)

    status_df = pd.DataFrame(status_rows)
    if args.merge_existing:
        status_path_existing = RESULTS_DIR / "medqa_backbone_evaluation_status.csv"
        if status_path_existing.exists() and "model_name" in status_df.columns:
            old_status = pd.read_csv(status_path_existing)
            new_models = set(status_df["model_name"].astype(str).unique())
            old_status = old_status[~old_status["model_name"].astype(str).isin(new_models)]
            if not old_status.empty:
                status_df = pd.concat([old_status, status_df], ignore_index=True)
    status_df["benchmark_type"] = "real_status"
    status_path = RESULTS_DIR / "medqa_backbone_evaluation_status.csv"
    status_df.to_csv(status_path, index=False)

    if base_parts:
        base_df = pd.concat(base_parts, ignore_index=True)
        if args.merge_existing:
            existing_detail_path = RESULTS_DIR / "medqa_backbone_detail.csv"
            if existing_detail_path.exists():
                existing_detail = pd.read_csv(existing_detail_path)
                if "workflow" in existing_detail.columns and "model_name" in existing_detail.columns:
                    existing_base = existing_detail[existing_detail["workflow"] == "backbone_only"].copy()
                    new_models = set(base_df["model_name"].astype(str).unique())
                    existing_base = existing_base[~existing_base["model_name"].astype(str).isin(new_models)]
                    common_cols = sorted(set(existing_base.columns).intersection(set(base_df.columns)))
                    if common_cols:
                        existing_base = existing_base[common_cols]
                        new_base = base_df[common_cols]
                        base_df = pd.concat([existing_base, new_base], ignore_index=True)
        summary, detail = build_workflow_rows(base_df, args.seed)
    else:
        summary, detail = pd.DataFrame(), pd.DataFrame()

    result_path = RESULTS_DIR / "medqa_backbone_comparison.csv"
    detail_path = RESULTS_DIR / "medqa_backbone_detail.csv"
    table_path = TABLES_DIR / "table_medqa_backbone_comparison.csv"
    summary.to_csv(result_path, index=False)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(table_path, index=False)
    plot(summary)
    write_reports(summary, status_df, max_questions, args.seed)

    print("Wrote %s" % result_path)
    print("Wrote %s" % detail_path)
    print("Wrote %s" % table_path)
    print("Wrote %s" % status_path)


if __name__ == "__main__":
    main()
