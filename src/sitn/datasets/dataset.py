import atexit
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import torchvision.transforms as T
from filelock import FileLock


class ScratchManager:
    """Manages a scratch directory for fast local data access on SLURM compute nodes.

    Multiple jobs can safely share the same scratch path: a ref file is registered
    per job/process, and the directory is only removed when no live ref files remain.

    Cleanup is triggered both via ``atexit`` (reliable for normal exits and SIGTERM)
    and ``__del__``. Ref files older than 31 days are considered stale and ignored
    during the liveness check, providing a safety net for SIGKILL.
    """

    _STALE_REF_SECONDS = 31 * 24 * 3600  # 31 days

    def __init__(self, scratch_root: str | Path):
        self._scratch_root = Path(scratch_root)
        self._scratch_root.mkdir(parents=True, exist_ok=True)
        self._cleaned_up = False

        # Register a unique ref file for this job/process
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        unique = f"{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self._ref_file = self._scratch_root / "refcounts" / f"{job_id}_{unique}.ref"
        self._ref_file.parent.mkdir(parents=True, exist_ok=True)
        self._ref_file.touch(exist_ok=False)

        atexit.register(self._cleanup)
        print(f"Using scratch directory: {self._scratch_root}", flush=True)

    def prepare(self, src_dir: str | Path) -> Path:
        """Copy *src_dir* to scratch (if not already present) and return the scratch path.

        Uses a file lock so concurrent jobs sharing the same scratch root only copy
        once. Safe to call from multiple processes simultaneously.

        Args:
            src_dir: Source directory to copy to scratch.

        Returns:
            Path to the copied directory inside the scratch root.
        """
        src_dir = Path(src_dir)
        dest_dir = self._scratch_root / src_dir.name
        lock_path = self._scratch_root / f"{src_dir.name}.lock"

        with FileLock(lock_path):
            if not dest_dir.exists():
                print(f"Copying {src_dir} -> {dest_dir}", flush=True)
                shutil.copytree(src_dir, dest_dir)

        return dest_dir

    def _cleanup(self):
        """Remove ref file and delete scratch directory if no live refs remain."""
        if self._cleaned_up:
            return

        if self._scratch_root.exists():
            self._ref_file.unlink(missing_ok=True)
            try:
                now = datetime.now().timestamp()
                keep = False
                for ref_file in (self._scratch_root / "refcounts").glob("*.ref"):
                    if now - ref_file.stat().st_mtime < self._STALE_REF_SECONDS:
                        keep = True
                        break
                if not keep:
                    shutil.rmtree(self._scratch_root)
            except Exception as e:
                print(f"Error during scratch cleanup: {e}")

        self._cleaned_up = True

    def __del__(self):
        self._cleanup()


def build_default_transform(
    natural_size: tuple[int, int, int],
    image_size: tuple[int, int, int],
    mean: tuple,
    std: tuple,
) -> T.Compose:
    """Build the standard transform pipeline for the given target `image_size`.

    Inserts a spatial resize and/or channel conversion when the requested size
    differs from `natural_size`.

    Args:
        natural_size: Native ``(C, H, W)`` shape of the dataset.
        image_size: Target ``(C, H, W)`` shape.
        mean: Per-channel normalisation mean (matched to `image_size` channels).
        std: Per-channel normalisation std (matched to `image_size` channels).
    """
    C, H, W = image_size
    nat_C, nat_H, nat_W = natural_size

    if C not in (1, 3):
        raise ValueError(f"Unsupported number of channels: {C}. Only 1 and 3 are supported.")
    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid spatial dimensions: ({H}, {W}). Must be positive integers.")

    transforms = []

    # Spatial resize (operates on PIL image, before ToTensor)
    if (H, W) != (nat_H, nat_W):
        transforms.append(T.Resize((H, W)))

    # Channel reduction: 3 -> 1 (operates on PIL image, before ToTensor)
    if nat_C == 3 and C == 1:
        transforms.append(T.Grayscale(num_output_channels=1))

    transforms.append(T.ToTensor())  # -> float32 tensor in [0, 1]

    # Channel expansion: 1 -> 3 (operates on tensor, after ToTensor)
    if nat_C == 1 and C == 3:
        transforms.append(T.Lambda(lambda x: x.repeat(3, 1, 1)))

    # Normalisation — adapt stats if channel count changed
    if C == 1 and nat_C == 3:
        norm_mean = (sum(mean) / len(mean),)
        norm_std = (sum(std) / len(std),)
    elif C == 3 and nat_C == 1:
        expanded = tuple(mean) * 3 if len(mean) == 1 else tuple(mean)
        norm_mean = expanded[:3]
        expanded_std = tuple(std) * 3 if len(std) == 1 else tuple(std)
        norm_std = expanded_std[:3]
    else:
        norm_mean = mean
        norm_std = std

    transforms.append(T.Normalize(mean=norm_mean, std=norm_std))

    return T.Compose(transforms)
