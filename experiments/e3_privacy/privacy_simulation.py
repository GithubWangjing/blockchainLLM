"""Privacy-utility simulation (E3) for MedChainLLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

SENSITIVITY_SCORES: Dict[str, float] = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
}

ENCRYPTION_COST: Dict[str, float] = {
    "aes128": 1.10,
    "hybrid": 1.50,
    "fhe": 3.00,
}


@dataclass
class PrivacyParameters:
    """Global knobs for the privacy-performance tradeoff."""

    base_accuracy: float
    base_training_time: float
    alpha_scale: float
    noise_penalty_scale: float
    training_time_jitter: float


@dataclass
class PrivacyResult:
    """Captures the metrics required by the experiment."""

    sensitivity: str
    encryption: str
    epsilon: float
    training_time: float
    accuracy: float
    privacy_risk_score: float
    combined_cost: float

    def to_record(self) -> Dict[str, float]:
        return {
            "sensitivity": self.sensitivity,
            "encryption": self.encryption,
            "epsilon": self.epsilon,
            "training_time": self.training_time,
            "accuracy": self.accuracy,
            "privacy_risk_score": self.privacy_risk_score,
            "combined_cost": self.combined_cost,
        }


class PrivacySimulation:
    """Runs deterministic-yet-jittered privacy sweeps."""

    def __init__(self, params: PrivacyParameters, seed: int) -> None:
        self.params = params
        self.rng = np.random.default_rng(seed)

    def run_configuration(self, sensitivity: str, encryption: str, epsilon: float) -> PrivacyResult:
        """Evaluate a single (sensitivity, encryption, epsilon) tuple."""
        sensitivity_key = sensitivity.lower()
        encryption_key = encryption.lower()
        s_score = SENSITIVITY_SCORES[sensitivity_key]
        encryption_cost = ENCRYPTION_COST[encryption_key]

        alpha = s_score * self.params.alpha_scale
        accuracy = max(self.params.base_accuracy - alpha / epsilon, 0.0)

        noise_penalty = (1.0 / epsilon) * self.params.noise_penalty_scale
        training_time = self.params.base_training_time * encryption_cost * (1.0 + noise_penalty)
        if self.params.training_time_jitter > 0:
            jitter = self.rng.normal(loc=1.0, scale=self.params.training_time_jitter)
            training_time *= max(jitter, 0.5)

        privacy_risk = 1.0 / epsilon
        combined_cost = training_time / max(accuracy, 1e-3)

        return PrivacyResult(
            sensitivity=sensitivity_key,
            encryption=encryption_key,
            epsilon=epsilon,
            training_time=training_time,
            accuracy=accuracy,
            privacy_risk_score=privacy_risk,
            combined_cost=combined_cost,
        )

