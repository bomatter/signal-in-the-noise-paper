# Signal in the Noise



## Setup

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Copy the example config and fill in the paths:

```bash
cp user_config.example.yaml user_config.yaml
```

`user_config.yaml` fields:

| Field | Description |
|---|---|
| `data.<dataset>.rawdata_root` | Directory where the dataset is downloaded to / read from |
| `output_root` | Root directory for model checkpoints and results |
| `scratch_root` | (Optional) Fast local storage for SLURM compute nodes |



## Data preparation

Download a dataset:

```bash
uv run sitn-preprocess --dataset_name cifar10
```

The raw data will be placed under the `rawdata_root` specified in `user_config.yaml`. 



## Training

```bash
uv run sitn-train --dataset_name cifar10 --save_checkpoint
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--dataset_name` | `cifar10` | Dataset to train on |
| `--image_size C H W` | `3 32 32` | Target image size |
| `--batch_size` | `128` | Batch size |
| `--learning_rate` | `1e-4` | Learning rate |
| `--max_batches` | `50000` | Total training batches |
| `--evaluation_interval` | `500` | Batches between validation runs |
| `--early_stopping_patience` | `10` | Evaluations without improvement before stopping |
| `--save_checkpoint` | off | Save model checkpoint on completion |
| `--use_wandb` | off | Log metrics to Weights & Biases |
| `--scratch_root` | `None` | Override scratch directory for data |

Run `uv run sitn-train --help` for the full list of options.
