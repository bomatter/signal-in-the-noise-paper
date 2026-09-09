import argparse
import math
from pathlib import Path
from pprint import pprint

import pandas as pd
import torch
import yaml
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

from sitn.datasets import create_dataset, get_dataset_info
from sitn.models import create_model
from sitn.utils import construct_output_dir, construct_results_path, parse_nullable


class WrappedModel(ModelWrapper):
    def __init__(self, model):
        super().__init__(None)
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        return self.model(timesteps=t, x=x, **extras)


def log_p0(x: torch.Tensor) -> torch.Tensor:
    """Compute log probability of standard Gaussian prior N(0, I)."""
    flat = x.flatten(start_dim=1)
    D = flat.shape[1]
    log_2pi = math.log(2 * math.pi)
    return -0.5 * ((flat**2).sum(dim=1) + D * log_2pi)


def compute_power_spectrum(x: torch.Tensor) -> torch.Tensor:
    """Compute the power spectrum, flattening all non-batch dims into a single signal.

    Args:
        x (torch.Tensor): Input tensor, batch on dim 0.

    Returns:
        torch.Tensor: Power spectrum of shape [batch_size, N//2 + 1] where N is the
            flattened signal length.
    """
    return torch.abs(torch.fft.rfft(x.flatten(start_dim=1), dim=-1)) ** 2



def convert_velocity_to_noise(v_theta: torch.Tensor, x_t: torch.Tensor, t: float) -> torch.Tensor:
    """
    Converts velocity v_theta(x_t, t) to predicted noise Z (epsilon_theta)
    under the linear interpolant X_t = (1-t)Z + tX.
    """
    return x_t - t * v_theta


def compute_12d_trajectory_score(field_history: list[torch.Tensor], dt: float) -> dict[str, torch.Tensor]:
    """
    Computes the 12 canonical DiffPath features from a sequence of field evaluation tensors
    recorded along the ODE path.
    """
    # Stack into shape (T, B, -1)
    U = torch.stack([u.flatten(start_dim=1) for u in field_history], dim=0)

    U_sum = U.sum(dim=(0, 2))
    U_sum_abs = U.abs().sum(dim=(0, 2))
    
    U_sq = (U**2).sum(dim=(0, 2))
    U_sum_sq_sqrt = U_sq.sqrt()
    
    U_cb = (U**3).sum(dim=(0, 2))
    U_sum_cb_cbrt = U_cb.sign() * U_cb.abs().pow(1.0 / 3.0)
    
    dU = (U[:-1] - U[1:]) / dt
    
    dU_sum = dU.sum(dim=(0, 2))
    dU_sum_abs = dU.abs().sum(dim=(0, 2))
    
    dU_sq = (dU**2).sum(dim=(0, 2))
    dU_sum_sq_sqrt = dU_sq.sqrt()
    
    dU_cb = (dU**3).sum(dim=(0, 2))
    dU_sum_cb_cbrt = dU_cb.sign() * dU_cb.abs().pow(1.0 / 3.0)

    return {
        "sum": U_sum,
        "sum_abs": U_sum_abs,
        "sum_sq": U_sq,
        "sum_sq_sqrt": U_sum_sq_sqrt,
        "sum_cb": U_cb,
        "sum_cb_cbrt": U_sum_cb_cbrt,
        "d_dt": dU_sum,
        "d_dt_abs": dU_sum_abs,
        "d_dt_sq": dU_sq,
        "d_dt_sq_sqrt": dU_sum_sq_sqrt,
        "d_dt_cb": dU_cb,
        "d_dt_cb_cbrt": dU_sum_cb_cbrt,
    }


def evaluate_diffpath(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str = None,
    t_min=0.0,
    steps=10,
    prefix: str | None = None,
):
    """
    Evaluates 12D DiffPath metrics.
    Integrates ODE using Euler over the specified steps.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WrappedModel(model).to(device)
    model.eval()

    batch_size = data_loader.batch_size
    dt_forward = (1.0 - t_min) / steps

    predictions = []
    sample_index = 0

    for batch in tqdm(data_loader, desc=f"Computing DiffPath ({prefix or 'eval'}):"):
        x_1 = batch[0].to(device)
        labels = batch[1]
        
        if x_1.dim() == 2:
            x_1 = x_1.unsqueeze(0)
            
        B = x_1.shape[0]
        t_seq = torch.linspace(1.0, t_min, steps + 1, device=device)
        
        x_t = x_1.clone()
        score_history = []
        vel_history = []
        
        with torch.no_grad():
            for i in range(steps):
                t_curr = t_seq[i]
                t_tensor = torch.full((B,), t_curr.item(), device=device)
                
                v_theta = model(x_t, t_tensor)
                eps_theta = convert_velocity_to_noise(v_theta, x_t, t_curr.item())
                
                score_history.append(eps_theta)
                vel_history.append(v_theta)
                
                x_t = x_t - v_theta * dt_forward
            
        dp_score = compute_12d_trajectory_score(score_history, dt_forward)
        for i in range(B):
            sample_info = {
                "sample_index": sample_index,
                "label": labels[i].item() if labels[i].numel() == 1 else None,
            }
            if hasattr(data_loader.dataset, "corruption_labels"):
                sample_info["corruption"] = data_loader.dataset.corruption_labels[sample_index]
                sample_info["severity"] = data_loader.dataset.severity_labels[sample_index]
                sample_info["original_test_index"] = data_loader.dataset.original_index_labels[sample_index]

            for key, val in dp_score.items():
                sample_info[f"dp_score_{key}"] = val[i].item()

            predictions.append(sample_info)
            sample_index += 1
            
    return pd.DataFrame(predictions)


def evaluate_likelihoods(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str = None,  # "cuda" or "cpu"; auto-detect if None
    # Solver parameters
    step_size=None,
    method="dopri5",
    exact_divergence=False,
    # Optional prefix for metric names
    prefix: str | None = None,  # e.g. "val" or "test"
):
    """Evaluate unconditional log-likelihoods for samples.

    Args:
        model (torch.nn.Module): The unconditional flow matching model.
        data_loader (DataLoader): The data loader for evaluation data.
        device (str): The device to run the evaluation on.
        step_size: Step size for ODE solver.
        method: ODE solver method.
        exact_divergence: Whether to compute exact divergence.
        prefix: Optional prefix for output naming.

    Returns:
        pd.DataFrame: DataFrame containing log-likelihoods and sample metadata.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WrappedModel(model).to(device)
    model.eval()

    solver = ODESolver(velocity_model=model)

    batch_size = data_loader.batch_size

    predictions = []
    sample_index = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Computing likelihoods ({prefix or 'eval'}):"):
            x_1 = batch[0].to(device)
            labels = batch[1]  # class label (int tensor)

            if x_1.dim() == 2:
                x_1 = x_1.unsqueeze(0)

            batch_size = x_1.shape[0]

            # Compute unconditional log-likelihood (no class conditioning)
            x_source, log_likelihood = solver.compute_likelihood(
                x_1=x_1,
                log_p0=log_p0,
                method=method,
                step_size=step_size,
                exact_divergence=exact_divergence,
                y=None,  # No class conditioning
            )

            source_log_likelihood = log_p0(x_source)
            log_determinant = log_likelihood - source_log_likelihood

            power_spectrum = compute_power_spectrum(x_source)
            p_cv = power_spectrum.std(dim=-1) / (power_spectrum.mean(dim=-1) + 1e-8)

            # Store predictions for DataFrame (iterate over batch)
            for i in range(batch_size):
                x_source_i = x_source[i].flatten().cpu().numpy()
                ad_stat = stats.anderson(x_source_i, dist="norm", method="interpolate").statistic

                sample_info = {
                    "sample_index": sample_index,
                    "label": labels[i].item() if labels[i].numel() == 1 else None,
                }

                # Add corruption metadata if available for the dataset
                if hasattr(data_loader.dataset, "corruption_labels"):
                    sample_info["corruption"] = data_loader.dataset.corruption_labels[sample_index]
                    sample_info["severity"] = data_loader.dataset.severity_labels[sample_index]
                    sample_info["original_test_index"] = data_loader.dataset.original_index_labels[sample_index]

                predictions.append(
                    {
                        **sample_info,
                        "log_likelihood": log_likelihood[i].item(),
                        "source_log_likelihood": source_log_likelihood[i].item(),
                        "log_determinant": log_determinant[i].item(),
                        "anderson_darling_statistic": ad_stat,
                        "ps_cv": p_cv[i].item(),
                    }
                )
                sample_index += 1

    df_predictions = pd.DataFrame(predictions)

    return df_predictions


def evaluate_from_config(
    config: dict | str | Path,  # config can be a dict or a path to a config file
    split_pick: str = "val",  # split method is selected from config, but this specifies which split to use e.g. "val" or "test"
    eval_dataset_name: str | None = None,  # if set, evaluate on this dataset instead of the training dataset
    corruptions: list[str] | None = None,  # subset of corruption types to evaluate; None = all
    severities: list[int] | None = None,  # subset of corruption severity levels; None = all
    device: str | None = None,
    batch_size: int = 128,
    num_workers: int = 4,
    # Solver parameters
    step_size=None,
    method="dopri5",
    exact_divergence=False,
    diffpath: bool = False,
    checkpoint_batch: int | None = None,
):
    """Evaluate a model based on the provided configuration.

    Args:
        config: Training configuration dict or path to a ``config.yml`` file.
        split_pick: Which split to evaluate (e.g. ``"val"`` or ``"test"``).
        eval_dataset_name: If provided, evaluate on this dataset instead of the
            one the model was trained on.  The training dataset's normalisation
            parameters (mean/std) and the ``image_size`` from the training config
            are used so that inputs are compatible with the model.
        corruptions: A subset of corruption types to evaluate.
            ``None`` evaluates all corruptions.
        severities: A subset of corruption severity levels.
            ``None`` evaluates all severities.
        device: Device string (``"cuda"`` / ``"cpu"``); auto-detected when ``None``.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.
        step_size: Step size for the ODE solver.
        method: ODE solver method.
        exact_divergence: Whether to use exact divergence estimation.
        checkpoint_batch: Specific training batch to evaluate (uses checkpoints/checkpoint_{batch}.pth).
    """

    # Capture evaluation parameters for later inclusion in metrics
    eval_params = {k: v for k, v in locals().items() if k not in ("config", "device", "batch_size", "num_workers")}

    if isinstance(config, (str, Path)):
        if Path(config).is_file():
            with open(config) as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Config at path {config} is not a valid file.")

    print("Running evaluation with the following configurations:")
    pprint(eval_params)

    train_dataset_name = config["dataset_name"]

    evaluating_ood = eval_dataset_name is not None and eval_dataset_name != train_dataset_name

    # When evaluating on a different dataset, re-use the training dataset's
    # normalisation parameters and the image_size from the training config
    if evaluating_ood:
        train_info = get_dataset_info(train_dataset_name)
        normalize_mean = train_info["mean"]
        normalize_std = train_info["std"]
    else:
        normalize_mean = None  # let the eval dataset use its own defaults
        normalize_std = None

    dataset = create_dataset(
        dataset_name=eval_dataset_name if eval_dataset_name is not None else train_dataset_name,
        pick=split_pick,
        image_size=config["image_size"],
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        transform=None,
        scratch_root=False,
        corruptions=corruptions,
        severities=severities,
    )
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    model = create_model(
        model_name=config["model_name"],
        image_size=config["image_size"],
        model_channels=config.get("model_channels", 128),
        num_res_blocks=config.get("num_res_blocks", 2),
        attention_resolutions=config.get("attention_resolutions", [16]),
        channel_mult=config.get("channel_mult", [1, 2, 2, 2]),
        num_heads=config.get("num_heads", 1),
        num_head_channels=config.get("num_head_channels", -1),
        checkpoint=construct_output_dir(config).joinpath(
            f"checkpoints/checkpoint_{checkpoint_batch}.pth" if checkpoint_batch is not None else "model_checkpoint.pth"
        ),
        device=device,
    )

    if diffpath:
        df_predictions = evaluate_diffpath(
            model=model,
            data_loader=data_loader,
            device=device,
            t_min=0.0,
            steps=int(1.0 / step_size) if step_size is not None else 10,
            prefix=split_pick,
        )
    else:
        df_predictions = evaluate_likelihoods(
            model=model,
            data_loader=data_loader,
            device=device,
            step_size=step_size,
            method=method,
            exact_divergence=exact_divergence,
            prefix=split_pick,
        )

    df_predictions["eval_dataset"] = eval_dataset_name if eval_dataset_name is not None else train_dataset_name

    return df_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate model from config.")
    parser.add_argument("config", type=str, help="Path to the configuration YAML file.")
    parser.add_argument("--split_pick", type=str, default="val", help="Data split to evaluate on (default: 'val').")
    parser.add_argument(
        "--eval_dataset_name",
        type=parse_nullable(str),
        default=None,
        help="Evaluate on this dataset instead of the training dataset. "
        "The training dataset's normalisation stats and image_size are used.",
    )
    parser.add_argument(
        "--corruptions", type=str, nargs="+", default=None, help="Corruption type(s) to evaluate (default: all)."
    )
    parser.add_argument(
        "--severities",
        type=int,
        nargs="+",
        default=None,
        help="Corruption severity level(s) to evaluate (default: all).",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation (default: 128).")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader (default: 4).")
    parser.add_argument(
        "--device", type=parse_nullable(str), default=None, help="Device to use for evaluation (default: auto-detect)."
    )
    # Solver parameters
    parser.add_argument("--method", type=str, default="dopri5", help="ODE solver method.")
    parser.add_argument(
        "--step_size",
        type=parse_nullable(float),
        default=None,
        help="Step size for ODE solver. Pass 'None' for adaptive solvers like dopri5.",
    )
    parser.add_argument("--exact_divergence", action="store_true", help="Whether to compute exact divergence.")
    parser.add_argument(
        "--diffpath", action=argparse.BooleanOptionalAction, help="Whether to evaluate DiffPath metrics instead of likelihoods."
    )
    parser.add_argument(
        "--checkpoint_batch",
        type=parse_nullable(int),
        default=None,
        help="Specific training batch to evaluate. Will use checkpoints/checkpoint_{batch}.pth.",
    )
    args = parser.parse_args()

    df_predictions = evaluate_from_config(**vars(args))

    # Filter out args that are not used in construct_results_path
    results_path_args = {k: v for k, v in vars(args).items() if k not in ("batch_size", "num_workers", "device")}

    predictions_path = construct_results_path(result_type="predictions", **results_path_args)
    df_predictions.to_csv(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()
