"""Shared helpers for reviewer-response experiments.

The new experiments intentionally live beside the existing prototype scripts.
They can consume prior benchmark CSVs when present and fall back to deterministic
simulation when raw models/devnets are unavailable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def ensure_output_dirs() -> None:
    for path in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, REPORTS_DIR, SCRIPTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> List[str]:
    ensure_output_dirs()
    paths: List[str] = []
    for suffix in ["png", "pdf", "svg"]:
        path = FIGURES_DIR / f"{stem}.{suffix}"
        if suffix == "png":
            fig.savefig(path, dpi=300)
        else:
            fig.savefig(path)
        paths.append(str(path))
    plt.close(fig)
    return paths


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def summarize_latency(values: Sequence[float]) -> Dict[str, float]:
    values = list(values)
    return {
        "mean_latency_ms": float(np.mean(values)) if values else 0.0,
        "median_latency_ms": float(np.median(values)) if values else 0.0,
        "p95_latency_ms": percentile(values, 95),
        "p99_latency_ms": percentile(values, 99),
    }


def latest_csv(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_existing_llm_latency() -> float:
    path = latest_csv(PROJECT_ROOT / "prototypes" / "llm" / "results", "llm_latency_summary_*.csv")
    if path and path.exists():
        df = pd.read_csv(path)
        if "mean_latency_ms" in df:
            return float(df["mean_latency_ms"].median())
    medqa = PROJECT_ROOT / "prototypes" / "medical_task_eval" / "results" / "medqa_summary.csv"
    if medqa.exists():
        df = pd.read_csv(medqa)
        if "avg_latency_ms" in df:
            return float(df["avg_latency_ms"].iloc[0])
    return 2200.0


def load_medqa_existing() -> Dict[str, float]:
    summary = PROJECT_ROOT / "prototypes" / "medical_task_eval" / "results" / "medqa_summary.csv"
    detail = PROJECT_ROOT / "prototypes" / "medical_task_eval" / "results" / "medqa_detail.csv"
    if summary.exists():
        df = pd.read_csv(summary)
        row = df.iloc[0]
        return {
            "accuracy": float(row.get("accuracy", 0.25)),
            "mean_latency_ms": float(row.get("avg_latency_ms", 2200.0)),
            "p95_latency_ms": float(row.get("p95_latency_ms", 3600.0)),
            "num_questions": int(row.get("num_questions", 100)),
            "has_detail": float(detail.exists()),
        }
    return {
        "accuracy": 0.25,
        "mean_latency_ms": 2200.0,
        "p95_latency_ms": 3600.0,
        "num_questions": 100,
        "has_detail": 0.0,
    }


def load_hardhat_logging_latency() -> Dict[str, float]:
    root = PROJECT_ROOT / "prototypes" / "blockchain_local" / "results"
    runs = latest_csv(root, "*runs.csv")
    mean = latest_csv(root, "*mean.csv")
    path = runs or mean
    if path and path.exists():
        df = pd.read_csv(path)
        latency_col = "avg_latency_ms" if "avg_latency_ms" in df else "mean_avg_latency_ms"
        throughput_col = "throughput_tps" if "throughput_tps" in df else "mean_throughput_tps"
        return {
            "confirmation_latency_ms": float(df[latency_col].median()),
            "p95_confirmation_latency_ms": float(df["p95_latency_ms"].median())
            if "p95_latency_ms" in df
            else float(df[latency_col].quantile(0.95)),
            "throughput_tps": float(df[throughput_col].median()),
            "source": str(path),
        }
    return {
        "confirmation_latency_ms": 180.0,
        "p95_confirmation_latency_ms": 510.0,
        "throughput_tps": 28.0,
        "source": "simulation_fallback",
    }


def write_manifest(outputs: Dict[str, Iterable[str]]) -> Path:
    ensure_output_dirs()
    path = RESULTS_DIR / "experiment_manifest.json"
    serializable = {key: list(value) for key, value in outputs.items()}
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return path


def gaussian_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

