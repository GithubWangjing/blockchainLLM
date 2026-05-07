"""Discrete-time simulation models for the dynamic sharding experiment (E1)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

try:  # Python 3.7 compatibility
    from typing import Literal
except ImportError:  # pragma: no cover
    try:
        from typing_extensions import Literal  # type: ignore
    except ImportError:  # pragma: no cover
        Literal = str  # type: ignore[misc,assignment]

import numpy as np

Strategy = Literal["none", "static", "dynamic"]


@dataclass
class DynamicShardParams:
    """Control parameters for the adaptive MedChainLLM sharding policy."""

    min_shards: int
    max_shards: int
    scale_up_threshold: float  # average queue length (> => add shard)
    scale_down_threshold: float  # average queue length (< => remove shard)
    grace_period: int  # number of steps below threshold before merging


@dataclass
class SimulationParameters:
    """High-level knobs shared by all strategies."""

    duration: float  # seconds
    time_step: float  # seconds
    block_time: float  # seconds to assemble a block
    block_size: int  # tx per block
    consensus_delay: float  # base consensus / ordering cost (seconds)
    base_latency_overhead: float  # networking + signature overhead (seconds)
    static_shards: int
    dynamic: DynamicShardParams


@dataclass
class SimulationResult:
    """Aggregated statistics for a single run."""

    strategy: Strategy
    arrival_rate: float
    node_count: int
    throughput_tps: float
    avg_latency: float
    p95_latency: float
    p99_latency: float
    mean_load_variance: float
    load_variance_series: List[float]
    shard_count_series: List[int]

    def to_record(self) -> Dict[str, float]:
        """Return a flat dict suitable for CSV logging."""
        return {
            "strategy": self.strategy,
            "arrival_rate": self.arrival_rate,
            "node_count": self.node_count,
            "throughput_tps": self.throughput_tps,
            "avg_latency": self.avg_latency,
            "p95_latency": self.p95_latency,
            "p99_latency": self.p99_latency,
            "mean_load_variance": self.mean_load_variance,
        }


class ShardManager:
    """Tracks shard queues and executes scale operations."""

    def __init__(self, shard_count: int, dynamic_params: DynamicShardParams | None = None) -> None:
        self.queues: List[Deque[float]] = [deque() for _ in range(shard_count)]
        self.dynamic_params = dynamic_params
        self.steps_underutilized = 0

    def shard_count(self) -> int:
        return len(self.queues)

    def assign_transactions(self, arrival_times: List[float]) -> None:
        """Distribute incoming transactions to the least loaded shard."""
        for arrival_time in arrival_times:
            target = min(range(len(self.queues)), key=lambda idx: len(self.queues[idx]))
            self.queues[target].append(arrival_time)

    def process_step(
        self,
        current_time: float,
        params: SimulationParameters,
        node_count: int,
        strategy: Strategy,
    ) -> Tuple[int, List[float]]:
        """Process up to the per-step capacity for each shard."""
        latencies: List[float] = []
        per_shard_capacity = self._per_shard_capacity(params, node_count, strategy)
        processing_delay = self._processing_delay(params, node_count, strategy)
        processed = 0

        for queue in self.queues:
            shard_processed = min(per_shard_capacity, len(queue))
            for _ in range(shard_processed):
                arrival = queue.popleft()
                latencies.append(max(current_time + processing_delay - arrival, params.time_step))
            processed += shard_processed

        return processed, latencies

    def maybe_scale(self) -> bool:
        """Adjust shard count if the dynamic policy allows it."""
        if not self.dynamic_params:
            return False

        avg_queue = (
            sum(len(queue) for queue in self.queues) / len(self.queues) if self.queues else 0.0
        )
        scaled = False

        if avg_queue > self.dynamic_params.scale_up_threshold and len(self.queues) < self.dynamic_params.max_shards:
            self.queues.append(deque())
            self.steps_underutilized = 0
            scaled = True
        elif avg_queue < self.dynamic_params.scale_down_threshold and len(self.queues) > self.dynamic_params.min_shards:
            self.steps_underutilized += 1
            if self.steps_underutilized >= self.dynamic_params.grace_period:
                # Merge the smallest shard back into the most lightly loaded partner.
                victim_idx = min(range(len(self.queues)), key=lambda idx: len(self.queues[idx]))
                victim_queue = self.queues.pop(victim_idx)
                target_idx = min(range(len(self.queues)), key=lambda idx: len(self.queues[idx]))
                self.queues[target_idx].extend(victim_queue)
                self.steps_underutilized = 0
                scaled = True
        else:
            self.steps_underutilized = 0

        return scaled

    def load_variance(self) -> float:
        """Return the variance of queue lengths."""
        if len(self.queues) <= 1:
            return 0.0
        lengths = np.array([len(q) for q in self.queues], dtype=float)
        return float(np.var(lengths))

    def _per_shard_capacity(
        self, params: SimulationParameters, node_count: int, strategy: Strategy
    ) -> int:
        """Capacity (tx) each shard can confirm during the current time step."""
        base_blocks_per_step = max(params.time_step / params.block_time, 1e-3)
        base_capacity = params.block_size * base_blocks_per_step

        # Model consensus slowing down with the log of node count.
        node_penalty = 1.0 + 0.015 * np.log2(max(node_count, 2) / 20)
        if strategy == "none":
            node_penalty *= 1.1  # Single chain suffers more as membership grows.
        elif strategy == "dynamic":
            node_penalty *= 1.05  # Coordination overhead for re-sharding.

        effective_capacity = base_capacity / node_penalty
        return max(int(effective_capacity), 1)

    def _processing_delay(
        self, params: SimulationParameters, node_count: int, strategy: Strategy
    ) -> float:
        """Confirmation delay applied to processed transactions."""
        node_penalty = 1.0 + 0.008 * max(node_count - 20, 0)
        if strategy == "dynamic":
            node_penalty *= 1.05
        return params.block_time + params.consensus_delay * node_penalty + params.base_latency_overhead


class ShardingSimulation:
    """Runs the E1 simulation for a given configuration."""

    def __init__(self, params: SimulationParameters, seed: int) -> None:
        self.params = params
        self.rng = np.random.default_rng(seed)

    def run(self, arrival_rate: float, node_count: int, strategy: Strategy) -> SimulationResult:
        steps = int(self.params.duration / self.params.time_step)
        shard_manager = self._build_manager(strategy)

        total_processed = 0
        latencies: List[float] = []
        load_variances: List[float] = []
        shard_counts: List[int] = []

        for step in range(steps):
            current_time = step * self.params.time_step
            arrivals = self._sample_arrivals(arrival_rate)
            shard_manager.assign_transactions([current_time] * arrivals)
            processed, step_latencies = shard_manager.process_step(
                current_time=current_time,
                params=self.params,
                node_count=node_count,
                strategy=strategy,
            )
            total_processed += processed
            latencies.extend(step_latencies)
            load_variances.append(shard_manager.load_variance())
            shard_counts.append(shard_manager.shard_count())

            if strategy == "dynamic":
                shard_manager.maybe_scale()

        throughput = total_processed / self.params.duration
        return SimulationResult(
            strategy=strategy,
            arrival_rate=arrival_rate,
            node_count=node_count,
            throughput_tps=throughput,
            avg_latency=float(np.mean(latencies)) if latencies else 0.0,
            p95_latency=float(np.percentile(latencies, 95)) if latencies else 0.0,
            p99_latency=float(np.percentile(latencies, 99)) if latencies else 0.0,
            mean_load_variance=float(np.mean(load_variances)) if load_variances else 0.0,
            load_variance_series=load_variances,
            shard_count_series=shard_counts,
        )

    def _sample_arrivals(self, arrival_rate: float) -> int:
        """Sample the Poisson arrivals for one time step."""
        lam = max(arrival_rate * self.params.time_step, 0.0)
        return int(self.rng.poisson(lam=lam))

    def _build_manager(self, strategy: Strategy) -> ShardManager:
        if strategy == "none":
            return ShardManager(shard_count=1)
        if strategy == "static":
            return ShardManager(shard_count=self.params.static_shards)
        return ShardManager(
            shard_count=self.params.dynamic.min_shards,
            dynamic_params=self.params.dynamic,
        )

