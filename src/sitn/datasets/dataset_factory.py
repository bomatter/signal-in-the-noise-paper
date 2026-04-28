import argparse
import importlib
import inspect
from pathlib import Path

from sitn.user_config import user_config
from sitn.utils import parse_nullable, resolve_scratch


def create_dataset(
    dataset_name: str,
    rawdata_root: str | Path | None = None,  # falls back to user_config["data"][dataset_name]["rawdata_root"]
    data_split: str | None = "default",
    pick: str | list[str] | None = None,
    image_size: tuple[int, int, int] | None = None,  # if None, the dataset module's NATURAL_SIZE is used
    normalize_mean: tuple | None = None,  # if None, the dataset module's MEAN is used
    normalize_std: tuple | None = None,  # if None, the dataset module's STD is used
    transform=None,
    scratch_root=None,  # falls back to user_config["scratch_root"] if None
    corruptions: list[str] | None = None,  # dataset-specific: subset of corruption types
    severities: list[int] | None = None,  # dataset-specific: subset of severity levels
):
    """Factory function to create dataset instances.

    Args:
        dataset_name: Name of the dataset (e.g. ``'cifar10'``). Must match a module
            in ``sitn/datasets/``.
        rawdata_root: Root directory where the dataset is stored / downloaded to.
            Falls back to ``user_config["data"][dataset_name]["rawdata_root"]``.
        data_split: ``"default"`` returns a ``{train, val, test}`` dict; ``None``
            returns an unsplit dataset.
        pick: Select a specific split or list of splits from the returned dict.
        image_size: Target ``(C, H, W)`` shape passed through to the dataset module.
            When ``None`` the module's own default (``NATURAL_SIZE``) is used.
        normalize_mean: Per-channel normalisation mean. When ``None`` the dataset
            module's own ``MEAN`` is used.
        normalize_std: Per-channel normalisation std. When ``None`` the dataset
            module's own ``STD`` is used.
        transform: Additional transform appended after the built-in pipeline.
        scratch_root: Local scratch directory for fast data access on SLURM nodes.
            Falls back to ``user_config["scratch_root"]`` if ``None``.
        corruptions: Subset of corruption types to evaluate.
            Ignored by datasets that don't support this parameter.
        severities: Subset of corruption severity levels to evaluate.
            Ignored by datasets that don't support this parameter.
    """
    if rawdata_root is None:
        rawdata_root = user_config["data"][dataset_name]["rawdata_root"]

    scratch_root = resolve_scratch(scratch_root)

    try:
        dataset_module = importlib.import_module(f"sitn.datasets.{dataset_name.lower()}")
    except ModuleNotFoundError:
        raise ValueError(f"Dataset '{dataset_name}' is not available. Add a module in sitn/datasets/")

    # Build kwargs: only pass corruptions/severities if the module's create_dataset accepts them
    module_params = inspect.signature(dataset_module.create_dataset).parameters
    extra_kwargs = {}
    if corruptions is not None and "corruptions" in module_params:
        extra_kwargs["corruptions"] = corruptions
    if severities is not None and "severities" in module_params:
        extra_kwargs["severities"] = severities

    return dataset_module.create_dataset(
        rawdata_root=rawdata_root,
        data_split=data_split,
        pick=pick,
        image_size=image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        transform=transform,
        scratch_root=scratch_root,
        **extra_kwargs,
    )


def get_dataset_info(name: str) -> dict:
    """Return dataset-specific metadata (natural size, normalization stats, etc.).

    Args:
        name: Dataset name (e.g. ``'cifar10'``).

    Returns:
        Dictionary with keys ``natural_size``, ``mean``, ``std``.
    """
    try:
        dataset_module = importlib.import_module(f"sitn.datasets.{name.lower()}")
    except ModuleNotFoundError:
        raise ValueError(f"Dataset '{name}' is not available. Add a module in sitn/datasets/")

    return {
        "natural_size": getattr(dataset_module, "NATURAL_SIZE", None),
        "mean": getattr(dataset_module, "MEAN", None),
        "std": getattr(dataset_module, "STD", None),
    }


def main():
    """Download / verify a dataset. Example usage:

    uv run sitn-preprocess --dataset_name cifar10
    python -m sitn.datasets.dataset_factory --dataset_name cifar10
    """
    parser = argparse.ArgumentParser(description="Download or verify a dataset.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name (e.g. 'cifar10')")
    parser.add_argument("--rawdata_root", type=parse_nullable(str), default=None, help="Root directory for raw data")
    args = parser.parse_args()
    create_dataset(**vars(args), scratch_root=False)


if __name__ == "__main__":
    main()
