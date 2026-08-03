"""Closed-loop datasets for adaptive flight-control identification."""

from .config import IdentificationExperimentConfig, load_experiment_config
from .experiment import generate_dataset

__all__ = [
    "IdentificationExperimentConfig",
    "generate_dataset",
    "load_experiment_config",
]
