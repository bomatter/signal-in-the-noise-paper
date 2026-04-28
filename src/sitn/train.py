import argparse
import random
import sys
import time
from copy import deepcopy
from math import ceil
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from sitn.datasets import create_dataset
from sitn.evaluate import evaluate_likelihoods
from sitn.metrics import Tracker
from sitn.models import create_model
from sitn.utils import construct_output_dir, construct_results_path, parse_nullable, set_seed


def save_results(
    output_dir,
    config,
    metrics_dict,
    train_predictions=None,
    val_predictions=None,
    test_predictions=None,
    eval_params=None,
    model=None,
):
    """Save config and metrics, and optionally predictions and model checkpoint."""
    if eval_params is None:
        eval_params = {}

    # Save config
    with open(output_dir / "config.yml", "w") as f:
        yaml.dump(config, f)

    # Save metrics
    metrics_df = pd.DataFrame([metrics_dict]).assign(
        **{k: ",".join(str(d) for d in v) if isinstance(v, (list, tuple)) else v for k, v in config.items()}
    )
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    # Save predictions if provided
    if train_predictions is not None:
        train_predictions.to_csv(
            construct_results_path(
                config=config, output_dir=output_dir, split_pick="train", result_type="predictions", **eval_params
            ),
            index=False,
        )
    if val_predictions is not None:
        val_predictions.to_csv(
            construct_results_path(
                config=config, output_dir=output_dir, split_pick="val", result_type="predictions", **eval_params
            ),
            index=False,
        )
    if test_predictions is not None:
        test_predictions.to_csv(
            construct_results_path(
                config=config, output_dir=output_dir, split_pick="test", result_type="predictions", **eval_params
            ),
            index=False,
        )

    # Save model checkpoint if provided
    if model is not None:
        torch.save(getattr(model, "_orig_mod", model).state_dict(), output_dir / "model_checkpoint.pth")


def save_training_checkpoint(
    output_dir,
    state,
    model,
    optimizer,
    tracker,
):
    """Save an intermediate training checkpoint for resumption.

    Args:
        output_dir: Directory to save the checkpoint to.
        state: Mutable dict of training progress (global_batch, epoch, etc.).
        model: The training model.
        optimizer: The optimizer.
        tracker: The metric tracker.
    """
    checkpoint = {
        "state": state,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tracker_state": tracker.state_dict(),
        "rng_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    # Write to a temp file then rename for atomicity
    tmp_path = output_dir / "training_checkpoint.pth.tmp"
    torch.save(checkpoint, tmp_path)
    tmp_path.rename(output_dir / "training_checkpoint.pth")
    print(f"Saved training checkpoint at batch {state['global_batch']}.")


def load_training_checkpoint(
    output_dir,
    state,
    model,
    optimizer,
    tracker,
    device,
):
    """Load a training checkpoint.

    Modifies state, model, optimizer, and tracker in-place.

    Args:
        output_dir: Directory containing the checkpoint.
        state: Mutable dict to update with saved training progress.
        model: The training model.
        optimizer: The optimizer.
        tracker: The metric tracker.
        device: Device to map tensors to.

    Returns:
        True if a checkpoint was loaded, False otherwise.
    """
    ckpt_path = output_dir / "training_checkpoint.pth"
    if not ckpt_path.exists():
        return False

    print(f"Resuming from checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Restore training progress
    state.update(checkpoint["state"])

    # Restore model, optimizer, tracker
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    tracker.load_state_dict(checkpoint["tracker_state"])

    # Restore RNG states
    rng = checkpoint["rng_states"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch"].cpu())
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([b.cpu() for b in rng["cuda"]])

    print(
        f"Resumed from batch {state['global_batch']} "
        f"(epoch {state['epoch']}, batch-in-epoch {state['batch_in_epoch']})."
    )

    return True


def evaluate_on_val(model, dataloader, loss_function, tracker, device):
    """Evaluate model on validation set."""
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating on validation set", leave=True):
            x_1 = batch[0].to(device, non_blocking=True)

            x_0 = torch.randn_like(x_1, device=device)
            t = torch.rand(len(x_1), device=device)
            x_t = (1 - t.view(-1, 1, 1, 1)) * x_0 + t.view(-1, 1, 1, 1) * x_1
            dx_t = x_1 - x_0

            out = model(timesteps=t, x=x_t, y=None)  # y=None for unconditional flow matching
            loss = loss_function(out, dx_t)

            tracker.report_val_step(loss=loss.item())

    tracker.report_val_done()
    model.train()


def train(
    # Dataset parameters
    dataset_name: str = "cifar10",
    image_size: list[int] = [3, 32, 32],
    # Model parameters
    model_name: str = "UNet",
    model_channels: int = 128,
    num_res_blocks: int = 2,
    attention_resolutions: list[int] = [16],
    channel_mult: list[int] = [1, 2, 2, 2],
    num_heads: int = 1,
    num_head_channels: int = -1,
    # Training parameters
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    batch_size: int = 128,
    max_batches: int = 500000,
    evaluation_interval: int = 500,  # in batches
    early_stopping_patience: int | None = None,  # in number of evaluations; set to None to disable early stopping
    early_stopping_metric: str = "Loss",  # metric to use for early stopping
    early_stopping_mode: str = "min",  # whether to maximize or minimize the early stopping metric
    restore_best: bool = True,  # restore best model weights (by early_stopping_metric) at end of training, regardless of whether early stopping is enabled
    # Output parameters
    output_root: str | None = None,  # will use user_config["output_root"] if None
    save_checkpoint: bool = True,  # save model checkpoint after training
    checkpoint_interval: int | None = None,  # save a model snapshot every N evaluations; None to disable
    use_wandb: bool = False,
    overwrite: bool = False,  # delete existing intermediate checkpoint and start fresh
    # Other parameters
    scratch_root: str = None,  # will use user_config["scratch_root"] if None; set to "" to explicitly disable scratch usage
    num_workers: int = 4,  # number of workers for DataLoader
    seed: int = 42,
    deterministic: bool = False,  # deterministic cuda operations (may slow down training)
    debug: bool = False,  # cap training and evaluation to a few batches for quick checks
):

    # Save configuration
    config = {
        k: v
        for k, v in locals().items()
        if k
        not in {
            "output_root",
            "save_checkpoint",
            "checkpoint_interval",
            "use_wandb",
            "overwrite",
            "scratch_root",
            "num_workers",
            "debug",
        }
    }

    print("Running training with the following configurations:")
    pprint(config)
    sys.stdout.flush()

    # Construct and create output directory
    output_dir = construct_output_dir(config=config, output_root=output_root, debug=debug)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if training has already completed (final checkpoint exists)
    if (output_dir / "model_checkpoint.pth").exists() and not overwrite:
        print("Training already completed (found model_checkpoint.pth). Skipping. Use --overwrite to retrain.")
        return None

    # Handle overwrite: remove existing intermediate checkpoint
    if overwrite and (output_dir / "training_checkpoint.pth").exists():
        print("Overwrite requested — removing existing training checkpoint.")
        (output_dir / "training_checkpoint.pth").unlink()

    # Record start time
    start_time = time.time()

    # Configure random seed and deterministic behavior
    if seed is not None:
        set_seed(seed=seed, deterministic=deterministic)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset
    dataset_train, dataset_val, dataset_test = create_dataset(
        dataset_name=dataset_name,
        pick=["train", "val", "test"],
        image_size=image_size,
        scratch_root=scratch_root,
    )

    if debug:
        max_batches = 10
        evaluation_interval = 10
        n = 2 * batch_size
        dataset_train = torch.utils.data.Subset(dataset_train, range(min(n, len(dataset_train))))
        dataset_val = torch.utils.data.Subset(dataset_val, range(min(n, len(dataset_val))))
        dataset_test = torch.utils.data.Subset(dataset_test, range(min(n, len(dataset_test))))

    # Create data loaders
    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    dataloader_test = DataLoader(
        dataset_test, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    # Create model
    model = create_model(
        model_name=model_name,
        image_size=image_size,
        device=device,
        compile=False,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        channel_mult=channel_mult,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Create loss function
    loss_function = torch.nn.MSELoss()

    # Initialize metric tracker (wandb run is set after checkpoint restore)
    tracker = Tracker(
        n_batches_per_epoch=len(dataloader_train),
        metrics=None,
        early_stopping_metric=(early_stopping_metric, early_stopping_mode),
        device=device,
        wandb=None,
    )

    # Mutable training state — modified in-place by load_training_checkpoint
    state = {
        "global_batch": 0,
        "epoch": 0,
        "batch_in_epoch": 0,
        "evaluations": 0,
        "training_time_offset": 0.0,
        "best_model_state": None,
    }

    # Attempt to resume from intermediate checkpoint
    resumed = load_training_checkpoint(
        output_dir=output_dir,
        state=state,
        model=model,
        optimizer=optimizer,
        tracker=tracker,
        device=device,
    )

    # Create wandb run (if enabled) — done after checkpoint restore so we can
    # retrieve the run ID from the tracker for proper wandb resumption.
    wandb_run = None
    if use_wandb:
        wandb_kwargs = {}
        if resumed and tracker.wandb_run_id is not None:
            wandb_kwargs = {"id": tracker.wandb_run_id, "resume": "must"}
        wandb_run = wandb.init(
            config=config,
            project="SITN",
            group="-".join([dataset_name, model_name]),
            dir=output_dir,
            mode="offline",
            **wandb_kwargs,
        )
        wandb_run.watch(model)
        tracker.wandb = wandb_run
        tracker.wandb_run_id = wandb_run.id

    # Train
    global_batch = state["global_batch"]
    evaluations = state["evaluations"]
    best_model_state = state["best_model_state"]
    resume_epoch = state["epoch"]
    resume_batch_in_epoch = state["batch_in_epoch"]
    training_time_offset = state["training_time_offset"]
    stop_training = False

    model.train()
    max_epochs = ceil(max_batches / len(dataloader_train))
    for epoch in tqdm(range(resume_epoch, max_epochs), position=0, desc="Epochs ", leave=True):
        if stop_training:
            break

        for batch_idx, batch in enumerate(tqdm(dataloader_train, position=1, desc="Batches", leave=True)):
            # Skip batches already processed in a resumed epoch
            if epoch == resume_epoch and batch_idx < resume_batch_in_epoch:
                continue

            if global_batch >= max_batches:
                break

            x_1 = batch[0].to(device, non_blocking=True)

            x_0 = torch.randn_like(x_1, device=device)
            t = torch.rand(len(x_1), device=device)
            x_t = (1 - t.view(-1, 1, 1, 1)) * x_0 + t.view(-1, 1, 1, 1) * x_1
            dx_t = x_1 - x_0

            out = model(timesteps=t, x=x_t, y=None)  # y=None for unconditional flow matching
            loss = loss_function(out, dx_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_batch += 1
            tracker.report_train_step(loss=loss.item(), epoch=epoch, batch=batch_idx)

            # Evaluate model
            if global_batch % evaluation_interval == 0:
                evaluate_on_val(model, dataloader_val, loss_function, tracker, device)
                current_metrics = tracker.get_metrics()
                pprint(current_metrics)
                evaluations += 1

                # Track best model state (for restore_best)
                if restore_best:
                    if tracker.evaluations_since_improvement == 0:
                        best_model_state = deepcopy(model.state_dict())

                # Save periodic model snapshot + metrics
                if checkpoint_interval is not None and evaluations % checkpoint_interval == 0:
                    ckpt_dir = output_dir / "checkpoints"
                    ckpt_dir.mkdir(exist_ok=True)
                    torch.save(
                        getattr(model, "_orig_mod", model).state_dict(),
                        ckpt_dir / f"checkpoint_{global_batch}.pth",
                    )
                    metrics_df = pd.DataFrame([current_metrics])
                    metrics_df.to_csv(ckpt_dir / f"metrics_{global_batch}.csv", index=False)
                    print(f"Saved checkpoint and metrics at batch {global_batch}.")

                # Early stopping
                if early_stopping_patience is not None:
                    if tracker.evaluations_since_improvement >= early_stopping_patience:
                        print(f"Early stopping triggered after {global_batch} batches.")
                        stop_training = True
                        break

                # Save intermediate training checkpoint
                state.update(
                    {
                        "global_batch": global_batch,
                        "epoch": epoch,
                        "batch_in_epoch": batch_idx + 1,
                        "evaluations": evaluations,
                        "training_time_offset": training_time_offset + (time.time() - start_time),
                        "best_model_state": best_model_state,
                    }
                )
                save_training_checkpoint(
                    output_dir=output_dir,
                    state=state,
                    model=model,
                    optimizer=optimizer,
                    tracker=tracker,
                )

    training_time = training_time_offset + (time.time() - start_time)

    if use_wandb:
        wandb_run.unwatch(model)
        wandb_run.finish()

    # Restore best model if requested
    if restore_best and best_model_state is not None:
        print("Restoring best model weights.")
        model.load_state_dict(best_model_state)

    # Run evaluations to compute sample-wise metrics
    eval_params = {
        "method": "dopri5",
        "step_size": None,
    }

    print("Computing likelihoods on training set...")
    train_predictions = evaluate_likelihoods(
        model=model,
        data_loader=dataloader_train,
        device=str(device),
        prefix="train",
        **eval_params,
    )

    print("Computing likelihoods on validation set...")
    val_predictions = evaluate_likelihoods(
        model=model,
        data_loader=dataloader_val,
        device=str(device),
        prefix="val",
        **eval_params,
    )

    print("Computing likelihoods on test set...")
    test_predictions = evaluate_likelihoods(
        model=model,
        data_loader=dataloader_test,
        device=str(device),
        prefix="test",
        **eval_params,
    )

    total_time = time.time() - start_time

    print("Final evaluation metrics:")
    metrics_dict = {
        **(tracker.get_best_metrics() if restore_best else tracker.get_metrics()),
        "training_time": training_time,
        "total_time": total_time,
    }
    pprint(metrics_dict)

    # Save checkpoint and config
    save_results(
        output_dir=output_dir,
        config=config,
        metrics_dict=metrics_dict,
        train_predictions=train_predictions,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        eval_params=eval_params,
        model=model if save_checkpoint else None,
    )

    # Clean up intermediate checkpoint
    ckpt_path = output_dir / "training_checkpoint.pth"
    if ckpt_path.exists():
        ckpt_path.unlink()
        print("Removed intermediate training checkpoint.")

    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description="Run training with specified configurations.")

    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, default="cifar10", help="Name of the dataset to use.")
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=3,
        default=[3, 32, 32],
        metavar=("C", "H", "W"),
        help="Target image size as C H W (e.g. 3 32 32).",
    )

    # Model parameters
    parser.add_argument("--model_name", type=str, default="UNet", help="Name of the model to use.")
    parser.add_argument("--model_channels", type=int, default=64, help="Base channel count for the model.")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks per resolution level.")
    parser.add_argument(
        "--attention_resolutions",
        type=int,
        nargs="+",
        default=[16, 8],
        help="Spatial resolutions at which to apply attention.",
    )
    parser.add_argument(
        "--channel_mult",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Channel multipliers for each resolution level.",
    )
    parser.add_argument("--num_heads", type=int, default=1, help="Number of attention heads.")
    parser.add_argument("--num_head_channels", type=int, default=-1, help="Number of channels per attention head.")

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimization.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--seed", type=parse_nullable(int), default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--deterministic", action="store_true", help="Enable deterministic CUDA operations (may slow training)."
    )

    # Training duration and early stopping
    parser.add_argument("--max_batches", type=int, default=50000, help="Maximum number of training batches.")
    parser.add_argument("--evaluation_interval", type=int, default=500, help="Evaluation interval in batches.")
    parser.add_argument(
        "--early_stopping_patience",
        type=parse_nullable(int),
        default=10,
        help="Early stopping patience in number of evaluations (set to 'None' to disable).",
    )
    parser.add_argument("--early_stopping_metric", type=str, default="Loss", help="Metric to use for early stopping.")
    parser.add_argument(
        "--early_stopping_mode",
        type=str,
        default="min",
        help="Whether to maximize ('max') or minimize ('min') the early stopping metric.",
    )
    parser.add_argument(
        "--restore_best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore best model weights at end of training, regardless of early stopping.",
    )

    # Output parameters
    parser.add_argument("--output_root", type=parse_nullable(str), default=None, help="Root directory for outputs.")
    parser.add_argument("--save_checkpoint", action="store_true", help="Save model checkpoint after training.")
    parser.add_argument(
        "--checkpoint_interval",
        type=parse_nullable(int),
        default=None,
        help="Save a model snapshot and metrics every N evaluations into checkpoints/ (set to 'None' to disable).",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing intermediate checkpoint and start training from scratch.",
    )

    # Other parameters
    parser.add_argument(
        "--scratch_root",
        type=parse_nullable(str),
        default=None,
        help="Path to scratch directory for storing data files.",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument(
        "--debug", action="store_true", help="Cap training and evaluation to a few batches for quick checks."
    )

    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
