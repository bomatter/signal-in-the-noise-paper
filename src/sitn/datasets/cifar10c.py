"""CIFAR-10-C OOD corruption benchmark dataset.

Reference:
    Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness
    to common corruptions and perturbations. ICLR 2019.
    https://github.com/hendrycks/robustness
"""

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from sitn.datasets.cifar10 import MEAN, NATURAL_SIZE, STD
from sitn.datasets.dataset import ScratchManager, build_default_transform

# CIFAR-10-C has the same image distribution as CIFAR-10 before corruption
# so we reuse CIFAR-10's normalisation constants (imported from cifar10.py).

# Zenodo download URL for the full CIFAR-10-C archive
_ZENODO_URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
_ARCHIVE_NAME = "CIFAR-10-C.tar"
_EXTRACTED_DIR = "CIFAR-10-C"

# All 19 corruption types in the dataset
CORRUPTIONS = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur",
]

# Number of test images per severity level
_N_PER_SEVERITY = 10_000
_N_SEVERITIES = 5


class CIFAR10C(Dataset):
    """CIFAR-10-C corruption benchmark dataset.

    Each sample is a corrupted CIFAR-10 test image. The dataset exposes three
    metadata arrays that are parallel to the samples and allow downstream
    analysis by corruption type, severity level, and pairwise comparison with
    the original CIFAR-10 test set:

    Attributes:
        corruption_labels (list[str]): Corruption name for each sample.
        severity_labels (list[int]): Severity level (1–5) for each sample.
        original_index_labels (list[int]): CIFAR-10 test-set index (0–9999)
            for each sample. Images with the same ``original_index_labels``
            value are corrupted versions of the same underlying image.

    Args:
        root: Directory where CIFAR-10-C is downloaded / found.
        corruptions: Subset of corruption types to include. Defaults to all 19.
        severities: Subset of severity levels (1–5) to include. Defaults to all 5.
        transform: Transform applied to each PIL image.
    """

    def __init__(
        self,
        root: str | Path,
        corruptions: list[str] | None = None,
        severities: list[int] | None = None,
        transform=None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.corruptions = corruptions if corruptions is not None else CORRUPTIONS
        self.severities = sorted(severities if severities is not None else list(range(1, _N_SEVERITIES + 1)))

        for c in self.corruptions:
            if c not in CORRUPTIONS:
                raise ValueError(f"Unknown corruption {c!r}. Must be one of: {CORRUPTIONS}")
        for s in self.severities:
            if s not in range(1, _N_SEVERITIES + 1):
                raise ValueError(f"Severity {s} out of range. Must be 1–{_N_SEVERITIES}.")

        data_dir = self.root / _EXTRACTED_DIR
        labels_path = data_dir / "labels.npy"
        cifar_labels = np.load(labels_path)  # shape (50000,), same for every corruption

        images_list: list[np.ndarray] = []
        labels_list: list[int] = []
        corruption_list: list[str] = []
        severity_list: list[int] = []
        original_index_list: list[int] = []

        for corruption in self.corruptions:
            npy_path = data_dir / f"{corruption}.npy"
            corruption_data = np.load(npy_path)  # (50000, 32, 32, 3), uint8

            for severity in self.severities:
                start = (severity - 1) * _N_PER_SEVERITY
                end = severity * _N_PER_SEVERITY
                block = corruption_data[start:end]  # (10000, 32, 32, 3)
                block_labels = cifar_labels[start:end]  # (10000,)

                images_list.append(block)
                labels_list.extend(block_labels.tolist())
                corruption_list.extend([corruption] * _N_PER_SEVERITY)
                severity_list.extend([severity] * _N_PER_SEVERITY)
                original_index_list.extend(range(_N_PER_SEVERITY))

        self._images = np.concatenate(images_list, axis=0)  # (N, 32, 32, 3), uint8
        self._labels = labels_list

        # Metadata arrays
        self.corruption_labels = corruption_list
        self.severity_labels = severity_list
        self.original_index_labels = original_index_list

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int):
        img = Image.fromarray(self._images[index])
        label = self._labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _is_downloaded(rawdata_root: Path) -> bool:
    """Return True if all corruption .npy files are present."""
    data_dir = rawdata_root / _EXTRACTED_DIR
    if not (data_dir / "labels.npy").exists():
        return False
    return all((data_dir / f"{c}.npy").exists() for c in CORRUPTIONS)


def _download(rawdata_root: Path) -> None:
    """Download and extract CIFAR-10-C from Zenodo."""
    rawdata_root.mkdir(parents=True, exist_ok=True)
    archive_path = rawdata_root / _ARCHIVE_NAME

    if not archive_path.exists():
        print(f"[cifar10c] Downloading {_ZENODO_URL}  (this may take a while — ~2.8 GB)")
        urllib.request.urlretrieve(_ZENODO_URL, archive_path)
        print("[cifar10c] Download complete.")

    print(f"[cifar10c] Extracting {archive_path} …")
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(rawdata_root)
    print("[cifar10c] Extraction complete.")


def create_dataset(
    rawdata_root: str | Path,
    data_split: str | None = "default",
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
    pick: str | list[str] | None = None,
    image_size: tuple[int, int, int] | None = None,
    normalize_mean: tuple | None = None,
    normalize_std: tuple | None = None,
    transform=None,
    scratch_root: str | Path | None = None,
):
    """Create CIFAR-10-C dataset (OOD corruption benchmark).

    CIFAR-10-C is a test-only dataset. When ``data_split="default"``, a dict
    ``{"test": dataset}`` is returned (no train/val splits exist).

    Args:
        rawdata_root: Directory where CIFAR-10-C is downloaded / found.
        data_split: ``"default"`` returns ``{"test": dataset}``; ``None``
            returns the dataset directly.
        corruptions: Corruption types to include. Defaults to all 19.
        severities: Severity levels (1–5) to include. Defaults to all 5.
        pick: Return only the named split(s) from the returned dict.
        image_size: Target ``(C, H, W)`` shape. Defaults to ``NATURAL_SIZE``.
        normalize_mean: Per-channel normalisation mean. Defaults to CIFAR-10's.
        normalize_std: Per-channel normalisation std. Defaults to CIFAR-10's.
        transform: Additional transform appended after the built-in pipeline.
        scratch_root: If provided, dataset files are copied here for faster
            access on SLURM compute nodes (see ``ScratchManager``).
    """
    rawdata_root = Path(rawdata_root)
    image_size = image_size if image_size is not None else NATURAL_SIZE
    mean = normalize_mean if normalize_mean is not None else MEAN
    std = normalize_std if normalize_std is not None else STD

    # Download if needed
    if not _is_downloaded(rawdata_root):
        _download(rawdata_root)

    # Optionally copy to scratch
    if scratch_root:
        scratch = ScratchManager(scratch_root)
        src_dir = rawdata_root / _EXTRACTED_DIR
        effective_root = scratch.prepare(src_dir).parent
    else:
        effective_root = rawdata_root

    default_transform = build_default_transform(NATURAL_SIZE, image_size, mean, std)
    full_transform = T.Compose([default_transform, transform]) if transform is not None else default_transform

    dataset = CIFAR10C(
        root=effective_root,
        corruptions=corruptions,
        severities=severities,
        transform=full_transform,
    )

    if data_split is None:
        return dataset

    if data_split == "default":
        datasets = {"test": dataset}
    else:
        raise ValueError(f"Unsupported data_split: {data_split!r}. CIFAR-10-C is test-only.")

    if pick is None:
        return datasets
    elif isinstance(pick, str):
        return datasets[pick]
    elif isinstance(pick, list):
        return datasets[pick[0]] if len(pick) == 1 else tuple(datasets[p] for p in pick)
    else:
        raise TypeError("pick must be a string or list of strings.")
