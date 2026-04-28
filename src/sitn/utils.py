import inspect
import os
import random
import subprocess
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml

from sitn.user_config import user_config


def set_seed(seed: int, deterministic: bool = False):
    """
    Set random seed for better reproducibility and to reduce randomness in experiments.
    Additionally, `deterministic` can be set to True to enforce deterministic behavior
    in CUDA operations (may however slow training significantly).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def resolve_scratch(scratch_root=None):
    """
    Resolve the scratch root directory based on user_config and provided value.

    If scratch_root is None, it will check user_config for 'scratch_root'.
    If 'scratch_root' is a list, it will check each path in the list and
    return the first accessible directory. If no accessible directory is found,
    it will return None.
    """
    if scratch_root is None:
        # Check user_config for scratch_root configuration
        scratch_root = user_config.get("scratch_root", None)

    if not scratch_root:
        print("No scratch root provided or found in user_config['scratch_root'].")
        return None

    elif isinstance(scratch_root, (str, Path)):
        scratch_root = [scratch_root]

    # Attempt to create or access scratch_root
    for root in scratch_root:
        root = Path(root)
        try:
            root.mkdir(parents=True, exist_ok=True)
            print(f"Accessible scratch_root found (or created): {root}")
            return root  # Return root if it was successfully created or already exists
        except Exception as e:
            print(f"Could not create or access scratch root '{root}': {e}")
            continue
    print("No accessible scratch root found in user_config['scratch_root']. Using None.")
    return None


def parse_nullable(type_func):
    """Type converter that handles 'None' strings.

    Intended for use with argparse to convert command line arguments.
    Converts 'None' string to None and applies the specified type function
    to other values.

    Example:
        parser.add_argument('--learning_rate', type=parse_nullable(float), default='None')
    """

    def converter(value):
        if value == "None":
            return None
        return type_func(value)

    return converter


def construct_output_dir(config: dict, output_root: str | None = None, debug: bool = False) -> Path:
    """
    Construct the output directory path based on the provided configuration
    and output root directory. If output_root is None, it will use the
    'output_root' configured in user_config.

    Args:
        config (dict): Configuration dictionary containing parameters.
        output_root (str): Root directory relative to which the output_dir path will be constructed.
        debug (bool): If True, prefix the output path relative to output_root with 'debug/'.

    Returns:
        Path: The constructed output directory path.
    """

    config = config.copy()

    # Use user_config output_root if none provided
    if output_root is None:
        output_root = user_config["output_root"]

    # Check config
    if not isinstance(config, dict):
        raise ValueError("Each config must be a dictionary.")

    from sitn.train import train

    if invalid_args := set(config.keys()) - set(inspect.signature(train).parameters.keys()):
        raise ValueError(f"The following arguments are not accepted by train: {invalid_args}")

    # Fill in missing cfg values with train's defaults so we can use them below
    train_signature = inspect.signature(train)
    for param in train_signature.parameters.values():
        if param.name not in config and param.default is not inspect.Parameter.empty:
            config[param.name] = param.default

    # Definition of the output schema
    # List of (key, always include, default value, format string)
    # "/" indicates a directory separator
    schema = [
        ("dataset_name", True, None, "{dataset_name}"),
        ("image_size", False, [3, 32, 32], "imgsize-{image_size_str}"),
        "/",
        ("model_name", True, None, "{model_name}"),
        ("model_channels", False, 128, "mc-{model_channels}"),
        ("num_res_blocks", False, 2, "nrb-{num_res_blocks}"),
        ("attention_resolutions", False, [16], "attnres-{attention_resolutions_str}"),
        ("channel_mult", False, [1, 2, 2, 2], "chm-{channel_mult_str}"),
        ("num_heads", False, 1, "nh-{num_heads}"),
        ("num_head_channels", False, -1, "nhc-{num_head_channels}"),
        "/",
        ("learning_rate", True, 1e-4, "lr-{learning_rate}"),
        ("weight_decay", False, 0.0, "wd-{weight_decay}"),
        "/",
        ("batch_size", True, 128, "bs-{batch_size}"),
        ("max_batches", True, 500000, "maxb-{max_batches}"),
        ("evaluation_interval", True, 500, "evalint-{evaluation_interval}"),
        ("early_stopping_patience", False, None, "earlystoppatience-{early_stopping_patience}"),
        ("early_stopping_metric", False, "Loss", "earlystopmetric-{early_stopping_metric}"),
        ("early_stopping_mode", False, "min", "earlystopmode-{early_stopping_mode}"),
        ("restore_best", False, True, "no-restorebest"),
        "/",
        ("seed", False, None, "seed-{seed}"),
        ("deterministic", False, False, "deterministic"),
    ]

    # Parameters that are explicitly ignored in output directory naming
    ignored_params = {
        "output_root",
        "save_checkpoint",
        "checkpoint_interval",
        "use_wandb",
        "overwrite",
        "scratch_root",
        "num_workers",
        "debug",
    }

    # Check that all config parameters are accounted for
    all_params = set(train_signature.parameters.keys())
    schema_keys = {item[0] for item in schema if isinstance(item, tuple)}
    unaccounted = all_params - schema_keys - ignored_params
    if unaccounted:
        raise ValueError(
            f"The following train parameters are not in the path schema or ignored_params: {unaccounted}. "
            "Please update construct_output_dir to include them in the schema or add them to ignored_params."
        )

    # Build the path based on schema
    path_parts = []  # List of directory levels
    current_part = []  # Current directory name components
    # Replace None with "None" for formatting
    format_config = {k: "None" if v is None else v for k, v in config.items()}
    if "image_size" in format_config:
        format_config["image_size_str"] = "x".join(str(d) for d in format_config["image_size"])
    if "attention_resolutions" in format_config:
        format_config["attention_resolutions_str"] = ".".join(str(r) for r in format_config["attention_resolutions"])
    if "channel_mult" in format_config:
        format_config["channel_mult_str"] = ".".join(str(m) for m in format_config["channel_mult"])
    for item in schema:
        if item == "/":  # Add current part to path_parts if not empty
            if current_part:
                path_parts.append("_".join(current_part))
                current_part = []
        else:
            key, always_include, default_value, format_string = item
            # Include if always_include or value differs from default
            if always_include or config[key] != default_value:
                current_part.append(format_string.format(**format_config))

    # Add any remaining components
    if current_part:
        path_parts.append("_".join(current_part))

    output_dir = Path(output_root)
    if debug:
        output_dir /= "debug"
    output_dir /= "/".join(path_parts)

    return output_dir


def construct_results_path(
    config: dict | str | Path,
    split_pick: str = "val",
    result_type: str = "predictions",
    method: str = "euler",
    step_size: float | None = 0.01,
    exact_divergence: bool = False,
    eval_dataset_name: str | None = None,
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
    output_dir: Path | str | None = None,
    checkpoint_batch: int | None = None,
) -> Path:
    """
    Construct the path to a results file based on the provided configuration
    and evaluation parameters.

    Args:
        config: Training configuration dictionary or path to config.yml.
        result_type: Type of results file ("predictions" or "metrics").
        eval_dataset_name: If set, appended to the filename suffix to distinguish.
        corruptions: If set, appended to the filename suffix.
        severities: If set, appended to the filename suffix.
        output_dir: If set, used as the output directory. If not set, the output directory is constructed from the config.
        checkpoint_batch: If set, redirects output to checkpoints/ and appends batch to the filename suffix.
        The rest of the args correspond to the parameters of the evaluation function.

    Returns:
        Path: The constructed path to the results file.
    """
    if result_type not in ("predictions", "metrics"):
        raise ValueError(f"result_type must be 'predictions' or 'metrics', got '{result_type}'")

    # Get output directory from config if not explicitly provided
    if output_dir is not None:
        output_dir = Path(output_dir)
    elif isinstance(config, (str, Path)):
        config_path = Path(config)
        output_dir = config_path.parent
    else:
        output_dir = construct_output_dir(config)

    # Build suffix (must match evaluate.py)
    suffix = f"_solver-{method}"
    if step_size is not None:
        suffix += f"_stepsize-{step_size}"
    if exact_divergence:
        suffix += "_exactdiv"
    if eval_dataset_name is not None:
        suffix += f"_evaldata-{eval_dataset_name}"
    if corruptions is not None:
        suffix += "_corrupt-" + ".".join(sorted(corruptions))
    if severities is not None:
        suffix += "_sev-" + ".".join(str(s) for s in sorted(severities))

    if checkpoint_batch is not None:
        output_dir /= "checkpoints"
        suffix += f"_batch-{checkpoint_batch}"

    if result_type == "metrics":
        if checkpoint_batch is not None:
            return output_dir / f"metrics_{checkpoint_batch}.csv"
        else:
            return output_dir / "metrics.csv"
    else:  # predictions
        return output_dir / f"{split_pick}_{result_type}{suffix}.csv"


def submit_train_jobs(
    configs,
    cpus=4,
    mem="4G",
    gpus=1,
    gpu_type=None,
    time_limit="48:00:00",
    partition=None,
    exclude=None,  # e.g. "node02,node04" prevent jobs from running on these nodes
    log_dir="logs",
    dry_run=False,  # set true to only print the commands without executing them
):
    """Utility function to submit training jobs to a slurm cluster.

    Args:
        configs (list of dict): List of configurations for each job.
            Each config should be a dictionary with parameters accepted by the `train` function.
            Configs will be validated against the `train` function's signature and defaults from
            it will be used to fill in missing values.

            Example:
                configs = [
                    {"dataset_name": "cifar10", "learning_rate": 1e-3},
                    {"dataset_name": "svhn", "learning_rate": 1e-4},
                ]
        cpus (int): Number of CPUs per task. Default: 4
        mem (str): Memory per job. Default: "4G"
        gpus (int): Number of GPUs per job. Default: 1
        gpu_type (str, optional): GPU type specification. Default: None
        time_limit (str): Time limit for jobs (HH:MM:SS format). Default: "24:00:00"
        partition (str): Slurm partition to use.
        exclude (str, optional): Nodes to exclude (e.g., "node02,node04"). Default: None
        log_dir (str): Directory where job logs will be saved. Default: "logs"
        dry_run (bool): If True, only prints the commands without executing them. Default: False
    """

    # Execute jobs from repo root directory
    original_cwd = os.getcwd()  # Save the current working directory to restore it later
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = Path(script_dir).parent.parent
    os.chdir(repo_root)

    try:
        from sitn.train import train

        train_signature = inspect.signature(train)

        for cfg in deepcopy(configs):
            # Check config
            if not isinstance(cfg, dict):
                raise ValueError("Each config must be a dictionary.")

            if invalid_args := set(cfg.keys()) - set(train_signature.parameters.keys()):
                raise ValueError(f"The following arguments are not accepted by train: {invalid_args}")

            # Fill in missing cfg values with train's defaults so we can use them below
            for param in train_signature.parameters.values():
                if param.name not in cfg and param.default is not inspect.Parameter.empty:
                    cfg[param.name] = param.default

            # Unpack config dictionary into CLI args.
            # Booleans are handled based on train defaults:
            #   default=False -> True => --flag, False => omit
            #   default=True  -> False => --no-flag, True => --flag
            bool_defaults = {
                param.name: param.default
                for param in train_signature.parameters.values()
                if isinstance(param.default, bool)
            }

            args_list = []
            for key, value in cfg.items():
                if isinstance(value, bool):
                    if value:
                        args_list.append(f"--{key}")
                    elif bool_defaults.get(key) is True:
                        args_list.append(f"--no-{key}")
                    # else: False with False default -> omit
                elif isinstance(value, (list, tuple)):
                    args_list.append(f"--{key}")
                    args_list.extend(str(v) for v in value)
                else:
                    args_list.extend([f"--{key}", str(value) if value is not None else "None"])

            cmd_args = " ".join(args_list)

            command = f"uv run sitn-train {cmd_args}"
            job_name = "train_" + str(construct_output_dir(config=cfg, output_root="")) + f"_run-{str(uuid4())[:4]}"
            sbatch_command = [
                "sbatch",
                f"--time={time_limit}",
                f"--cpus-per-task={cpus}",
                f"--mem={mem}",
                f"--gres=gpu:{gpu_type}:{gpus}" if gpu_type else f"--gres=gpu:{gpus}",
                f"--job-name={job_name}",
                f"--output={log_dir}/{job_name}.log",
            ]

            if partition:
                sbatch_command.append(f"--partition={partition}")

            if exclude:
                sbatch_command.append(f"--exclude={exclude}")

            sbatch_command.extend(["--wrap", command])

            if dry_run:
                print("Dry run - the following command would be executed:")
                print(" ".join(sbatch_command))
            else:
                result = subprocess.run(sbatch_command, capture_output=True, text=True)
                if result.returncode != 0:
                    error_message = result.stderr.strip()
                    raise RuntimeError(f"Error submitting job {job_name}: {error_message}")

    finally:
        # Restore the original working directory
        os.chdir(original_cwd)


def submit_eval_jobs(
    configs,
    cpus=4,
    mem="4G",
    gpus=1,
    gpu_type=None,
    time_limit="24:00:00",
    partition=None,
    exclude=None,
    log_dir="logs/eval",
    dry_run=False,
):
    """Utility function to submit evaluation jobs to a slurm cluster.

    Args:
        configs (list of dict): List of evaluation configurations for each job.
            Each config should be a dictionary with a "config" key specifying the training
            config (as a dict) or path to config.yml, plus any additional parameters
            accepted by `evaluate_from_config`. Configs will be validated against the
            `evaluate_from_config` function's signature and defaults from it will be used
            to fill in missing values.

            Example:
                configs = [
                    {"config": train_cfg1, "split_pick": "test", "eval_dataset_name": "cifar10"},
                    {"config": "/path/to/config.yml", "split_pick": "val"},
                ]
        cpus (int): Number of CPUs per task. Default: 4
        mem (str): Memory per job. Default: "4G"
        gpus (int): Number of GPUs per job. Default: 1
        gpu_type (str, optional): GPU type specification. Default: None
        time_limit (str): Time limit for jobs (HH:MM:SS format). Default: "24:00:00"
        partition (str): Slurm partition to use.
        exclude (str, optional): Nodes to exclude (e.g., "node02,node04"). Default: None
        log_dir (str): Directory where job logs will be saved. Default: "logs/eval"
        dry_run (bool): If True, only prints the commands without executing them. Default: False
    """

    # Execute jobs from repo root directory
    original_cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = Path(script_dir).parent.parent
    os.chdir(repo_root)

    try:
        from sitn.evaluate import evaluate_from_config

        for cfg in deepcopy(configs):
            # Check config
            if not isinstance(cfg, dict):
                raise ValueError("Each config must be a dictionary.")

            if "config" not in cfg:
                raise ValueError("Each config must contain a 'config' key specifying the training config or path.")

            # Validate config keys against evaluate_from_config signature
            eval_params = set(inspect.signature(evaluate_from_config).parameters.keys())
            if invalid_args := set(cfg.keys()) - eval_params:
                raise ValueError(f"The following arguments are not accepted by evaluate_from_config: {invalid_args}")

            # Make sure we have both the train_cfg dict (for job naming)
            # and the config path (required by the evaluation script)
            train_cfg = cfg["config"]
            if isinstance(train_cfg, dict):
                # Replace config entry with path
                config_path = construct_output_dir(train_cfg) / "config.yml"
                cfg["config"] = str(config_path)
            elif isinstance(train_cfg, (str, Path)):
                # Load config to construct job name
                config_path = Path(train_cfg)
                with open(config_path) as f:
                    train_cfg = yaml.safe_load(f)
            else:
                raise ValueError("'config' must be a dictionary or a path string.")

            # Fill in missing cfg values with evaluate_from_config's defaults
            eval_signature = inspect.signature(evaluate_from_config)
            for param in eval_signature.parameters.values():
                if param.name not in cfg and param.default is not inspect.Parameter.empty:
                    cfg[param.name] = param.default

            # Unpack the config dictionary into a string of command line arguments.
            # The "config" argument is positional, so handle it separately.
            # Booleans are handled based on their default:
            #   store_true style (default=False): True → --flag, False → omit
            #   BooleanOptionalAction style (default=True): False → --no-flag, True → --flag
            bool_defaults = {
                param.name: param.default
                for param in eval_signature.parameters.values()
                if isinstance(param.default, bool)
            }
            param_defaults = {
                param.name: param.default
                for param in eval_signature.parameters.values()
                if param.default is not inspect.Parameter.empty
            }
            config_path_arg = cfg.pop("config")
            args_list = []
            for key, value in cfg.items():
                # Skip None values that match the default (e.g. corruptions=None, severities=None).
                # But emit --key None for intentional None overrides (e.g. step_size=None vs default 0.01).
                if value is None and param_defaults.get(key) is None:
                    continue
                if isinstance(value, bool):
                    if value:
                        args_list.append(f"--{key}")
                    elif bool_defaults.get(key) is True:
                        args_list.append(f"--no-{key}")
                    # else: False with False default (store_true style) → omit
                elif isinstance(value, (list, tuple)):
                    args_list.append(f"--{key} {' '.join(str(v) for v in value)}")
                else:
                    args_list.append(f"--{key} {value if value is not None else 'None'}")
            cmd_args = config_path_arg + " " + " ".join(args_list)

            command = f"uv run sitn-eval {cmd_args}"
            job_name = (
                "eval_" + str(construct_output_dir(config=train_cfg, output_root="")) + f"_run-{str(uuid4())[:4]}"
            )
            sbatch_command = [
                "sbatch",
                f"--time={time_limit}",
                f"--cpus-per-task={cpus}",
                f"--mem={mem}",
                f"--gres=gpu:{gpu_type}:{gpus}" if gpu_type else f"--gres=gpu:{gpus}",
                f"--job-name={job_name}",
                f"--output={log_dir}/{job_name}.log",
            ]

            if partition:
                sbatch_command.append(f"--partition={partition}")

            if exclude:
                sbatch_command.append(f"--exclude={exclude}")

            sbatch_command.extend(["--wrap", command])

            if dry_run:
                print("Dry run - the following command would be executed:")
                print(" ".join(sbatch_command))
            else:
                result = subprocess.run(sbatch_command, capture_output=True, text=True)
                if result.returncode != 0:
                    error_message = result.stderr.strip()
                    raise RuntimeError(f"Error submitting job {job_name}: {error_message}")

    finally:
        # Restore the original working directory
        os.chdir(original_cwd)
