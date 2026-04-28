from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import random_split

from sitn.datasets.dataset import ScratchManager, build_default_transform

# Dataset constants
NATURAL_SIZE = (3, 32, 32)  # (C, H, W)

# Per-channel mean and std computed over the training set
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def create_dataset(
    rawdata_root: str | Path,
    data_split: str | None = "default",
    val_fraction: float = 0.1,
    pick: str | list[str] | None = None,
    image_size: tuple[int, int, int] | None = None,
    normalize_mean: tuple | None = None,
    normalize_std: tuple | None = None,
    transform=None,
    scratch_root: str | Path | None = None,
):
    """Create cifar10 dataset split(s).

    Args:
        rawdata_root: Directory where torchvision will download / find the dataset.
        data_split: ``"default"`` returns a ``{train, val, test}`` dict; ``None``
            returns a single concatenated dataset covering all data.
        pick: If given, return only the named split(s) instead of the full dict.
        image_size: Target ``(C, H, W)`` shape. Resize and channel conversion
            transforms are inserted automatically when this differs from
            ``NATURAL_SIZE``. Defaults to ``NATURAL_SIZE`` (no transformation).
        val_fraction: Fraction of the training set to use for validation.
            Only used when ``data_split="default"``.
        normalize_mean: Per-channel normalisation mean. Defaults to the
            dataset-specific ``MEAN`` constant.
        normalize_std: Per-channel normalisation std. Defaults to the
            dataset-specific ``STD`` constant.
        transform: Additional transform appended after the built-in pipeline.
        scratch_root: If provided, the dataset files are copied here for faster
            access on SLURM compute nodes (see ``ScratchManager``).
    """
    rawdata_root = Path(rawdata_root)
    image_size = image_size if image_size is not None else NATURAL_SIZE
    mean = normalize_mean if normalize_mean is not None else MEAN
    std = normalize_std if normalize_std is not None else STD

    # Ensure both splits are downloaded to rawdata_root before any scratch copy
    torchvision.datasets.CIFAR10(root=rawdata_root, train=True, download=True, transform=None)
    torchvision.datasets.CIFAR10(root=rawdata_root, train=False, download=True, transform=None)

    # Optionally copy dataset files to scratch for faster access
    if scratch_root:
        scratch = ScratchManager(scratch_root)
        src_dir = rawdata_root / torchvision.datasets.CIFAR10.base_folder
        effective_root = scratch.prepare(src_dir).parent
    else:
        effective_root = rawdata_root

    default_transform = build_default_transform(NATURAL_SIZE, image_size, mean, std)
    transform = T.Compose([default_transform, transform]) if transform is not None else default_transform

    train_full = torchvision.datasets.CIFAR10(root=effective_root, train=True, download=False, transform=transform)
    test_ds = torchvision.datasets.CIFAR10(root=effective_root, train=False, download=False, transform=transform)

    if data_split is None:
        return torch.utils.data.ConcatDataset([train_full, test_ds])

    if data_split == "default":
        train_ds, val_ds = random_split(
            train_full,
            [1 - val_fraction, val_fraction],
            generator=torch.Generator().manual_seed(42),
        )
        datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    else:
        raise ValueError(f"Unsupported data_split: {data_split!r}")

    if pick is None:
        return datasets
    elif isinstance(pick, str):
        return datasets[pick]
    elif isinstance(pick, list):
        return datasets[pick[0]] if len(pick) == 1 else tuple(datasets[p] for p in pick)
    else:
        raise TypeError("pick must be a string or list of strings.")
