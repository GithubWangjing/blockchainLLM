"""Analytic simulation for E2: decentralized medical LLM inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

try:  # Python 3.7 compatibility
    from typing import Literal
except ImportError:  # pragma: no cover
    try:
        from typing_extensions import Literal  # type: ignore
    except ImportError:  # pragma: no cover
        Literal = str  # type: ignore[misc,assignment]

import numpy as np

Strategy = Literal["baseline_full", "baseline_compressed", "decentralized"]


def _pruning_penalty(pruning: float) -> float:
    """Return latency multiplier delta induced by structural pruning."""
    if pruning >= 0.5:
        return -0.60
    if pruning >= 0.3:
        return -0.40
    return 0.0


def _quantization_penalty(quantization: str) -> float:
    """Return latency multiplier delta induced by quantization."""
    mode = quantization.lower()
    if mode == "int8":
        return -0.30
    if mode == "fp16":
        return -0.15
    return 0.0


def _pruning_accuracy_drop(pruning: float) -> float:
    """Absolute accuracy drop (fraction) attributed to pruning."""
    if pruning >= 0.5:
        return 0.010
    if pruning >= 0.3:
        return 0.005
    return 0.0


def _quantization_accuracy_drop(quantization: str) -> float:
    """Absolute accuracy drop (fraction) attributed to quantization."""
    mode = quantization.lower()
    if mode == "int8":
        return 0.010
    if mode == "fp16":
        return 0.005
    return 0.0


@dataclass
class InferenceParameters:
    """Knobs controlling timing/accuracy models."""

    base_latency_ms: float
    base_accuracy: float
    verification_overhead_ms: float
    jitter_std_fraction: float
    requests_per_config: int


@dataclass
class InferenceResult:
    """Aggregated metrics for one deployment strategy."""

    strategy: Strategy
    pruning: float
    quantization: str
    num_edge_nodes: int
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    accuracy: float

    def to_record(self) -> Dict[str, float]:
        return {
            "strategy": self.strategy,
            "pruning": self.pruning,
            "quantization": self.quantization,
            "num_edge_nodes": self.num_edge_nodes,
            "mean_latency_ms": self.mean_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput_rps": self.throughput_rps,
            "accuracy": self.accuracy,
        }


class InferenceSimulation:
    """Runs the E2 deployment sweep with reproducible randomness."""

    def __init__(self, params: InferenceParameters, seed: int) -> None:
        self.params = params
        self.rng = np.random.default_rng(seed)

    def run_configuration(
        self, pruning: float, quantization: str, num_edge_nodes: int
    ) -> List[InferenceResult]:
        """Evaluate all deployment strategies for a given compression config."""
        results: List[InferenceResult] = []
        compression_factor = self._compression_factor(pruning, quantization)
        compressed_latency = self.params.base_latency_ms * compression_factor

        for strategy in ("baseline_full", "baseline_compressed", "decentralized"):
            mean_latency = self._mean_latency_ms(
                strategy=strategy,
                compressed_latency_ms=compressed_latency,
                num_edge_nodes=num_edge_nodes,
            )
            latencies = self._sample_latencies(mean_latency)
            throughput = 1000.0 / mean_latency if mean_latency > 0 else 0.0
            accuracy = self._accuracy(strategy, pruning, quantization)
            results.append(
                InferenceResult(
                    strategy=strategy,
                    pruning=pruning,
                    quantization=quantization,
                    num_edge_nodes=num_edge_nodes,
                    mean_latency_ms=float(np.mean(latencies)),
                    p95_latency_ms=float(np.percentile(latencies, 95)),
                    p99_latency_ms=float(np.percentile(latencies, 99)),
                    throughput_rps=throughput,
                    accuracy=accuracy,
                )
            )

        return results

    def _compression_factor(self, pruning: float, quantization: str) -> float:
        """Multiplicative latency factor (<1 speeds up inference)."""
        return max(0.2, 1.0 + _pruning_penalty(pruning) + _quantization_penalty(quantization))

    def _mean_latency_ms(
        self,
        strategy: Strategy,
        compressed_latency_ms: float,
        num_edge_nodes: int,
    ) -> float:
        """Apply the latency equations described in the instructions."""
        if strategy == "baseline_full":
            return self.params.base_latency_ms
        if strategy == "baseline_compressed":
            return compressed_latency_ms

        distributed = compressed_latency_ms / max(num_edge_nodes, 1)
        return distributed + self.params.verification_overhead_ms

    def _accuracy(self, strategy: Strategy, pruning: float, quantization: str) -> float:
        """Accuracy drops only when compression is active."""
        if strategy == "baseline_full":
            return self.params.base_accuracy

        drop = _pruning_accuracy_drop(pruning) + _quantization_accuracy_drop(quantization)
        return max(self.params.base_accuracy - drop, 0.0)

    def _sample_latencies(self, mean_latency_ms: float) -> np.ndarray:
        """Introduce small Gaussian jitter to mimic system noise."""
        std = mean_latency_ms * self.params.jitter_std_fraction
        samples = self.rng.normal(loc=mean_latency_ms, scale=max(std, 1e-3), size=self.params.requests_per_config)
        return np.clip(samples, 0.1, None)

