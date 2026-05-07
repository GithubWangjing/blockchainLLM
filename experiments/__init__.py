"""Experiment package for MedChainLLM simulations."""

from importlib import resources


def get_data_path(relative: str) -> str:
    """Return an absolute path inside the package directory."""
    return str(resources.files(__package__) / relative)

