import csv
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torchvision
import torchvision.transforms as T

from sitn.datasets.dataset import ScratchManager, build_default_transform

# Dataset constants — images are stored as aligned+cropped JPEGs at 178×218 px
NATURAL_SIZE = (3, 218, 178)  # (C, H, W)

# Per-channel mean and std computed over the training set at natural size
MEAN = (0.5063, 0.4258, 0.3832)
STD = (0.3107, 0.2904, 0.2897)

# Files that torchvision.datasets.CelebA expects inside {root}/celeba/
_REQUIRED_TXT_FILES = [
    "list_attr_celeba.txt",
    "identity_CelebA.txt",
    "list_bbox_celeba.txt",
    "list_landmarks_align_celeba.txt",
    "list_eval_partition.txt",
]

# Mapping from Kaggle CSV filenames to torchvision TXT filenames.
# Files with ``header_lines == 2`` use the torchvision convention where
# line 1 = row-count and line 2 = column names (space-separated).
_KAGGLE_CSV_TO_TXT = {
    # csv_name → (txt_name, header_lines)
    "list_attr_celeba.csv": ("list_attr_celeba.txt", 2),
    "list_eval_partition.csv": ("list_eval_partition.txt", 0),
    "list_bbox_celeba.csv": ("list_bbox_celeba.txt", 2),
    "list_landmarks_align_celeba.csv": ("list_landmarks_align_celeba.txt", 2),
}


def _celeba_is_ready(rawdata_root: Path) -> bool:
    """Return True if all torchvision-expected CelebA files are present."""
    celeba_dir = rawdata_root / "celeba"
    if not celeba_dir.is_dir():
        return False
    for fname in _REQUIRED_TXT_FILES:
        if not (celeba_dir / fname).exists():
            return False
    if not (celeba_dir / "img_align_celeba").is_dir():
        return False
    return True


@contextmanager
def _skip_md5_check():
    """Temporarily patch torchvision's ``check_integrity`` to skip MD5 verification.

    Kaggle-converted .txt files have correct content but different byte-level
    formatting (e.g. spacing), so MD5 hashes won't match the originals.
    We only need to verify that the files exist.
    """

    def _exists_only(fpath, md5=None):
        return os.path.isfile(fpath)

    with patch("torchvision.datasets.celeba.check_integrity", _exists_only):
        yield


def _convert_kaggle_to_torchvision(kaggle_dir: Path, target_dir: Path) -> None:
    """Convert a Kaggle CelebA download into the torchvision-expected layout.

    Args:
        kaggle_dir: Path to the Kaggle download (the directory that contains
            ``img_align_celeba/``, ``list_attr_celeba.csv``, etc.).
        target_dir: Destination directory (typically ``rawdata_root / "celeba"``).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[celeba] Converting Kaggle data: {kaggle_dir} → {target_dir}")

    # Convert CSV files to TXT
    for csv_name, (txt_name, header_lines) in _KAGGLE_CSV_TO_TXT.items():
        csv_path = kaggle_dir / csv_name
        txt_path = target_dir / txt_name
        if txt_path.exists():
            print(f"  {txt_name} already exists, skipping.")
            continue
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Expected Kaggle file {csv_path} not found. Make sure your Kaggle download is complete."
            )
        print(f"  {csv_name} → {txt_name}")
        _csv_to_txt(csv_path, txt_path, header_lines)

    # Dummy identity file (not provided by Kaggle)
    identity_path = target_dir / "identity_CelebA.txt"
    if not identity_path.exists():
        print("  Creating dummy identity_CelebA.txt")
        # Read image filenames from the partition file (already converted)
        partition_path = target_dir / "list_eval_partition.txt"
        with open(partition_path) as f:
            lines = f.readlines()
        with open(identity_path, "w") as f:
            for idx, line in enumerate(lines, start=1):
                filename = line.strip().split()[0]
                f.write(f"{filename} {idx}\n")

    # Image directory
    img_dest = target_dir / "img_align_celeba"
    if not img_dest.exists():
        # Kaggle nests images: img_align_celeba/img_align_celeba/*.jpg
        kaggle_img_inner = kaggle_dir / "img_align_celeba" / "img_align_celeba"
        kaggle_img_flat = kaggle_dir / "img_align_celeba"

        if kaggle_img_inner.is_dir():
            src = kaggle_img_inner
        elif kaggle_img_flat.is_dir():
            src = kaggle_img_flat
        else:
            raise FileNotFoundError(
                f"Cannot find image directory in {kaggle_dir}. Expected img_align_celeba/ (possibly nested)."
            )

        # Verify it actually contains images
        sample = next(src.iterdir(), None)
        if sample is None or sample.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            raise FileNotFoundError(f"Image directory {src} appears empty or does not contain images.")

        print(f"  Copying images {src} → {img_dest}  (this may take a few minutes)")
        shutil.copytree(src, img_dest)

    print("[celeba] Conversion complete.")


def _csv_to_txt(csv_path: Path, txt_path: Path, header_lines: int) -> None:
    """Convert a Kaggle-style CSV to torchvision's space-delimited TXT format.

    Args:
        csv_path: Source CSV (comma-separated, single header row).
        txt_path: Destination TXT.
        header_lines: Number of header lines torchvision expects.
            - 0: no header, just ``filename  value`` per line.
            - 2: line 1 = row count, line 2 = column names (space-separated),
              followed by data rows.
    """
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        csv_header = next(reader)  # first row is always columns in Kaggle CSVs
        rows = list(reader)

    # Column names (excluding the image_id / filename column)
    col_names = csv_header[1:]

    with open(txt_path, "w") as f:
        if header_lines == 2:
            f.write(f"{len(rows)}\n")
            f.write("  ".join(col_names) + "\n")

        for row in rows:
            filename = row[0]
            values = row[1:]
            f.write(f"{filename}  {'  '.join(values)}\n")


def _ensure_celeba_available(rawdata_root: Path) -> None:
    """Make sure CelebA data is available under *rawdata_root*.

    Fallback chain:
        1. Data already present    → return
        2. Try torchvision download → catch gdown rate-limit errors
        3. Kaggle folder present   → convert automatically
        4. Nothing works           → raise with instructions
    """
    # 1. Already ready
    if _celeba_is_ready(rawdata_root):
        return

    # 2. Try the default torchvision (gdown) download
    try:
        print("[celeba] Attempting torchvision download (gdown) …")
        for split in ("train", "valid", "test"):
            torchvision.datasets.CelebA(
                root=rawdata_root,
                split=split,
                download=True,
                transform=None,
            )
        if _celeba_is_ready(rawdata_root):
            return
    except Exception as e:
        err_msg = str(e).lower()
        is_rate_limit = (
            "too many users" in err_msg
            or "failed to retrieve file url" in err_msg
            or "fileurlretrievalerror" in err_msg
        )
        if not is_rate_limit:
            raise  # re-raise unexpected errors
        print(f"[celeba] gdown download failed (rate-limited): {e}")

    # 3. Check for a kaggle/ subfolder in rawdata_root
    kaggle_dir = rawdata_root / "kaggle"
    if kaggle_dir.is_dir():
        target_dir = rawdata_root / "celeba"
        _convert_kaggle_to_torchvision(kaggle_dir, target_dir)
        if _celeba_is_ready(rawdata_root):
            return

    # 4. Nothing worked — inform the user
    raise FileNotFoundError(
        "\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        "  CelebA download failed (Google Drive rate limit).\n"
        "═══════════════════════════════════════════════════════════════\n"
        "\n"
        "  You can work around this by downloading CelebA manually\n"
        "  from Kaggle and placing the files in:\n"
        f"\n"
        f"      {kaggle_dir}/\n"
        f"\n"
        "  Steps:\n"
        "    1. Download from https://www.kaggle.com/datasets/jessicali9530/celeba-dataset\n"
        "    2. Extract the archive so that the directory contains:\n"
        "         kaggle/img_align_celeba/   (image folder)\n"
        "         kaggle/list_attr_celeba.csv\n"
        "         kaggle/list_eval_partition.csv\n"
        "         kaggle/list_bbox_celeba.csv\n"
        "         kaggle/list_landmarks_align_celeba.csv\n"
        "    3. Re-run this command.\n"
        "\n"
        "═══════════════════════════════════════════════════════════════\n"
    )


def create_dataset(
    rawdata_root: str | Path,
    data_split: str | None = "default",
    pick: str | list[str] | None = None,
    image_size: tuple[int, int, int] | None = None,
    normalize_mean: tuple | None = None,
    normalize_std: tuple | None = None,
    transform=None,
    scratch_root: str | Path | None = None,
):
    """Create CelebA dataset split(s).

    CelebA provides predefined train / valid / test splits, so no ``val_fraction``
    parameter is needed — validation data comes from the official ``"valid"`` split.

    Args:
        rawdata_root: Directory where torchvision will download / find the dataset.
        data_split: ``"default"`` returns a ``{train, val, test}`` dict using the
            official splits; ``None`` returns a single dataset covering all data.
        pick: If given, return only the named split(s) instead of the full dict.
        image_size: Target ``(C, H, W)`` shape. Resize and channel conversion
            transforms are inserted automatically when this differs from
            ``NATURAL_SIZE``. Defaults to ``NATURAL_SIZE`` (no transformation).
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

    # Ensure data is available (download or convert from Kaggle)
    _ensure_celeba_available(rawdata_root)

    # Optionally copy dataset files to scratch for faster access.
    # CelebA stores files in rawdata_root/celeba/
    if scratch_root:
        scratch = ScratchManager(scratch_root)
        src_dir = rawdata_root / "celeba"
        effective_root = scratch.prepare(src_dir).parent
    else:
        effective_root = rawdata_root

    default_transform = build_default_transform(NATURAL_SIZE, image_size, mean, std)
    transform = T.Compose([default_transform, transform]) if transform is not None else default_transform

    # Use _skip_md5_check because Kaggle-converted files have correct content
    # but different formatting, so torchvision's MD5 verification would fail.
    with _skip_md5_check():
        if data_split is None:
            all_ds = torchvision.datasets.CelebA(root=effective_root, split="all", download=False, transform=transform)
            return all_ds

        if data_split == "default":
            train_ds = torchvision.datasets.CelebA(
                root=effective_root, split="train", download=False, transform=transform
            )
            val_ds = torchvision.datasets.CelebA(
                root=effective_root, split="valid", download=False, transform=transform
            )
            test_ds = torchvision.datasets.CelebA(
                root=effective_root, split="test", download=False, transform=transform
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
