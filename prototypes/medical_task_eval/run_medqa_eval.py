import argparse
import csv
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CHOICE_LABELS = ["A", "B", "C", "D", "E"]
DEFAULT_DATASET_CANDIDATES: Tuple[Tuple[str, Optional[str]], ...] = (
    ("medqa", "usmle"),
    ("openlifescienceai/medqa-usmle", None),
    ("qiaojin/medqa", "usmle"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen on MedQA-USMLE accuracy and latency."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Hugging Face model id to benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of questions loaded per evaluation chunk.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of MedQA test questions (default uses all).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repeat generations per question for additional latency samples.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="auto",
        help=(
            "Hugging Face dataset id containing MedQA. "
            "Use 'auto' to try known public mirrors."
        ),
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional dataset configuration/name (when not using auto).",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("prototypes/medical_task_eval/results"),
        help="Directory to store CSV outputs.",
    )
    parser.add_argument(
        "--plots-root",
        type=Path,
        default=Path("prototypes/medical_task_eval/plots"),
        help="Directory to store plots.",
    )
    return parser.parse_args()


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def clean_text(value: Optional[Any]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def get_nested_value(payload: Dict, path: Sequence[str]) -> Optional[Any]:
    current: Any = payload
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def normalize_option_container(container: Any) -> List[str]:
    if container is None:
        return []
    if isinstance(container, list):
        normalized = [clean_text(opt) for opt in container]
        return [opt for opt in normalized if opt]
    if isinstance(container, dict):
        normalized = []
        for label in CHOICE_LABELS:
            text = clean_text(
                container.get(label)
                or container.get(label.lower())
                or container.get(label.upper())
            )
            if text:
                normalized.append(text)
        if normalized:
            return normalized
        normalized = [clean_text(val) for val in container.values()]
        return [opt for opt in normalized if opt]
    if isinstance(container, str):
        parts = [clean_text(line) for line in container.split("\n")]
        return [part for part in parts if part]
    return []


def extract_question(example: Dict) -> str:
    candidates = [
        get_nested_value(example, (field,))
        for field in ("question", "Question", "sent1", "prompt")
    ]
    data = example.get("data")
    if isinstance(data, dict):
        candidates.append(data.get("Question"))
        candidates.append(data.get("question"))
    for candidate in candidates:
        text = clean_text(candidate)
        if text:
            return text
    return ""


def extract_options(example: Dict) -> List[str]:
    option_paths = [
        ("options",),
        ("Options",),
        ("choices",),
        ("data", "Options"),
        ("data", "options"),
    ]
    for path in option_paths:
        container = get_nested_value(example, path)
        normalized = normalize_option_container(container)
        if normalized:
            return normalized

    collected = []
    for label in CHOICE_LABELS:
        text = clean_text(example.get(label))
        if text:
            collected.append(text)

    if not collected:
        data = example.get("data")
        if isinstance(data, dict):
            for label in CHOICE_LABELS:
                text = clean_text(data.get(label))
                if text:
                    collected.append(text)
    return collected


def extract_answer_label(example: Dict, options: Sequence[str]) -> str:
    answer_paths = [
        ("answer",),
        ("Answer",),
        ("correct_option",),
        ("Correct Option",),
        ("label",),
    ]
    for path in answer_paths:
        candidate = get_nested_value(example, path)
        normalized = normalize_answer(candidate, options)
        if normalized:
            return normalized

    data = example.get("data")
    if isinstance(data, dict):
        for key in [
            "Correct Option",
            "correct_option",
            "correct answer",
            "Correct option",
        ]:
            normalized = normalize_answer(data.get(key), options)
            if normalized:
                return normalized

        for key in ["Correct Answer", "correct_answer"]:
            normalized = normalize_answer(data.get(key), options)
            if normalized:
                return normalized

    for key in ["Correct Answer", "correctAnswer"]:
        normalized = normalize_answer(get_nested_value(example, (key,)), options)
        if normalized:
            return normalized

    return ""


def prepare_samples(dataset) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for idx in range(len(dataset)):
        example = dataset[idx]
        question = extract_question(example)
        options = [opt for opt in extract_options(example) if opt]
        if not question or len(options) < 2:
            continue
        if len(options) > len(CHOICE_LABELS):
            options = options[: len(CHOICE_LABELS)]
        labels_in_use = CHOICE_LABELS[: len(options)]
        answer_label = extract_answer_label(example, options)
        if not answer_label or answer_label not in labels_in_use:
            continue
        question_id = example.get("id") or example.get("question_id")
        if question_id is None:
            question_id = f"medqa_{idx:05d}"
        prepared.append(
            {
                "question_id": str(question_id),
                "question": question,
                "options": options,
                "answer": answer_label,
            }
        )
    return prepared


def iter_dataset_candidates(
    dataset_id: str, dataset_config: Optional[str], dataset_split: str
) -> List[Tuple[str, Optional[str], str]]:
    if dataset_id != "auto":
        return [(dataset_id, dataset_config, dataset_split)]
    return [
        (candidate_id, candidate_cfg, dataset_split)
        for candidate_id, candidate_cfg in DEFAULT_DATASET_CANDIDATES
    ]


def load_medqa_split(
    max_samples: Optional[int],
    dataset_id: str,
    dataset_config: Optional[str],
    dataset_split: str,
) -> List[Dict]:
    last_error: Optional[Exception] = None
    for cand_id, cand_cfg, split in iter_dataset_candidates(
        dataset_id, dataset_config, dataset_split
    ):
        try:
            kwargs = {"path": cand_id, "split": split}
            if cand_cfg:
                kwargs["name"] = cand_cfg
            dataset = load_dataset(**kwargs)
            if max_samples is not None:
                max_idx = min(max_samples, len(dataset))
                dataset = dataset.select(range(max_idx))
            print(f"Loaded dataset '{cand_id}' (config={cand_cfg or 'default'}).")
            return dataset
        except Exception as err:
            last_error = err
            print(
                f"Warning: failed to load dataset '{cand_id}' "
                f"(config={cand_cfg or 'default'}): {err}"
            )
    raise RuntimeError(
        "Unable to load the MedQA-USMLE test split from the available dataset "
        "mirrors. Provide --dataset-id/--dataset-config pointing to a local or "
        "custom dataset."
    ) from last_error


def normalize_answer(
    answer: Optional[Union[str, int]], options: Sequence[str]
) -> str:
    if answer is None:
        return ""

    ans_str = str(answer).strip()
    upper = ans_str.upper()
    if upper in CHOICE_LABELS:
        return upper

    if ans_str.isdigit():
        idx = int(ans_str)
        if 0 <= idx < len(options):
            return CHOICE_LABELS[idx]

    lowered_opts = [opt.strip().lower() for opt in options]
    if ans_str.strip().lower() in lowered_opts:
        idx = lowered_opts.index(ans_str.strip().lower())
        return CHOICE_LABELS[idx]

    return ""


def build_prompt(question: str, options: Sequence[str]) -> str:
    lines = [
        "You are a medical expert. Answer the USMLE-style question by choosing the best option.",
        f"Question: {question.strip()}",
        "Options:",
    ]
    active_labels = CHOICE_LABELS[: len(options)]
    for label, option in zip(active_labels, options):
        lines.append(f"{label}. {option.strip()}")
    lines.append(
        f"Respond with the single letter {', '.join(active_labels)}."
    )
    lines.append("Final answer:")
    return "\n".join(lines)


def extract_choice(text: str) -> str:
    for char in text.strip().upper():
        if char in {"A", "B", "C", "D"}:
            return char
    return ""


def percentile(data: Sequence[float], pct: float) -> float:
    if not data:
        return float("nan")
    clipped_pct = max(0.0, min(100.0, pct))
    sorted_vals = sorted(data)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * (clipped_pct / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_vals[int(rank)]
    weight = rank - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def time_single_question(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    repetitions: int,
    max_new_tokens: int = 8,
) -> Tuple[int, int, float, str]:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    prompt_tokens = int(attention_mask.sum().item())

    latencies: List[float] = []
    generated_tokens = 0
    prediction = ""

    for rep in range(repetitions):
        start = time.perf_counter()
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(latency_ms)

        if rep == 0:
            sequence = outputs[0]
            total_tokens = sequence.shape[-1]
            generated_tokens = max(total_tokens - prompt_tokens, 0)
            generated_text = tokenizer.decode(
                sequence[prompt_tokens:], skip_special_tokens=True
            )
            prediction = extract_choice(generated_text)

    avg_latency = statistics.mean(latencies)
    return prompt_tokens, generated_tokens, avg_latency, prediction


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_accuracy(accuracy: float, path: Path) -> None:
    plt.figure(figsize=(4, 4))
    plt.bar(["MedQA"], [accuracy * 100.0], color="#4e79a7")
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("MedQA Accuracy")
    for idx, value in enumerate([accuracy * 100.0]):
        plt.text(idx, value + 1, f"{value:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_latency(latencies: Sequence[float], path: Path) -> None:
    if not latencies:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(latencies, bins=20, color="#59a14f", alpha=0.85)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Samples")
    plt.title("MedQA Latency Distribution")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()
    device = resolve_device()

    print("Loading MedQA-USMLE test split...")
    medqa = load_medqa_split(
        max_samples=args.max_samples,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
    )
    raw_count = len(medqa)
    print(f"Loaded {raw_count} raw questions (after max-samples).")

    prepared_samples = prepare_samples(medqa)
    prepared_count = len(prepared_samples)
    print(
        f"{prepared_count} questions remain after filtering for "
        "multiple-choice format with labeled answers."
    )
    if prepared_count == 0:
        raise RuntimeError(
            "No MedQA questions are usable after filtering. "
            "Ensure the dataset contains multiple-choice entries with answer keys."
        )

    print(f"Loading model {args.model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    ensure_dirs(args.results_root, args.plots_root)

    detail_rows: List[Dict] = []
    latencies: List[float] = []
    correct = 0
    total = 0

    for start in range(0, prepared_count, args.batch_size):
        batch = prepared_samples[start : start + args.batch_size]
        for sample in batch:
            prompt = build_prompt(sample["question"], sample["options"])
            prompt_tokens, generated_tokens, latency_ms, prediction = (
                time_single_question(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    repetitions=max(1, args.repetitions),
                )
            )

            is_correct = prediction == sample["answer"]
            correct += int(is_correct)
            total += 1
            latencies.append(latency_ms)

            detail_rows.append(
                {
                    "question_id": sample["question_id"],
                    "model": args.model_name,
                    "device": str(device),
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "latency_ms": latency_ms,
                    "prediction": prediction,
                    "answer": sample["answer"],
                    "is_correct": int(is_correct),
                }
            )

    accuracy = (correct / total) if total else 0.0
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = percentile(latencies, 95.0) if latencies else 0.0

    detail_path = args.results_root / "medqa_detail.csv"
    summary_path = args.results_root / "medqa_summary.csv"
    accuracy_plot = args.plots_root / "medqa_accuracy.png"
    latency_plot = args.plots_root / "medqa_latency.png"

    write_csv(
        detail_path,
        [
            "question_id",
            "model",
            "device",
            "prompt_tokens",
            "generated_tokens",
            "latency_ms",
            "prediction",
            "answer",
            "is_correct",
        ],
        detail_rows,
    )
    write_csv(
        summary_path,
        [
            "model",
            "device",
            "num_questions",
            "accuracy",
            "avg_latency_ms",
            "p95_latency_ms",
        ],
        [
            {
                "model": args.model_name,
                "device": str(device),
                "num_questions": total,
                "accuracy": accuracy,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
            }
        ],
    )

    plot_accuracy(accuracy, accuracy_plot)
    plot_latency(latencies, latency_plot)

    print(f"Evaluated {total} questions.")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Average latency: {avg_latency:.2f} ms (p95 {p95_latency:.2f} ms)")
    print(f"Detail CSV: {detail_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Plots: {accuracy_plot}, {latency_plot}")


if __name__ == "__main__":
    main()

