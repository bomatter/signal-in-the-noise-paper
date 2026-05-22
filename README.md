# The Signal in the Noise

This repository contains the research code for the experiments in our paper *"[The Signal in the Noise: OOD Detection Through Goodness-of-Fit Testing in Factorised Latent Spaces](https://arxiv.org/abs/2605.22496)"*.



## About

Signal in the Noise (SITN) is a method for out-of-distribution (OOD) detection. It utilises an unconditional flow matching model that was trained on the training distribution and performs OOD detection by probing noise samples obtained through backwards integration along the probability flow ODE against the factorised noise prior.



## Setup

1. Install dependencies with [uv](https://docs.astral.sh/uv/):

   ```
   uv sync
   ```

2. Create a copy of `user_config.example.yaml` and rename it to `user_config.yaml`, then open it and configure the paths for where you want datasets to be downloaded to and where training and evaluation outputs (model checkpoints, predictions, metrics) should be saved.

   ```
   cp user_config.example.yaml user_config.yaml
   ```

   Optionally, you can also configure scratch_roots in this file. If specified, data is automatically copied from the data directories to the scratch directory during training, which can speed up training if the compute node has access to faster disks than the data is stored on.



## Data preparation

Datasets for the reproduction of the results in the paper can conveniently be downloaded using the following command. Use the following names: "cifar10", "svhn", "celeba", "cifar10c". For example:

```bash
uv run sitn-preprocess --dataset_name cifar10
```

The raw data will be placed under the `rawdata_root` specified in `user_config.yaml`. 



## Training

To train a model, the following command can be used:

```bash
uv run sitn-train --dataset_name cifar10
```

Note that the default options of the train function match the configurations used in the paper, such that only the desired `dataset_name` for the training data needs to be specified to reproduce our model training.

If you want to experiment with different configurations, run `uv run sitn-train --help` for the full list of options.



## Evaluation

To evaluate a trained model, which includes the computation of likelihoods through backwards integration along the probability flow ODE as well as the computation of derived metrics used by SITN and baseline methods (e.g. the anderson darling statistic), the following command can be used:

```
uv run sitn-eval /path/to/training_cfg --eval_dataset_name svhn --split_pick test
```

Note that during training, a config file is saved along other outputs in a path relative to the `output_dir` configured in your `user_config.yaml` file. The path to this config file is required as the first argument for the evaluation command.

Moreover, a training run will automatically include evaluations on the train, val, and test splits of the training dataset for the final model. Therefore, you will usually only have to run separate evaluations for cross-dataset experiments (e.g. to evaluate the CIFAR-10-trained model on SVHN).

The final OOD scores for SITN and various baseline methods are computed based on the outputs of these evaluations. See the section [Reproducing OOD Scores and Paper Results](#reproducing-ood-scores-and-paper-results) for details.



## Reproducing OOD Scores and Paper Results

We provide a collection of notebooks in the `experiments/` folder with guides to reproduce the results in our paper. Training and evaluation jobs can be conveniently submitted as slurm jobs from these notebooks or alternatively executed manually via the CLI as described above and in the notebooks.

The notebooks also illustrate how the final OOD scores for SITN and various baseline methods are computed.



## Citation
```
@misc{bomatter2026signalnoiseooddetection,
      title={The Signal in the Noise: OOD Detection Through Goodness-of-Fit Testing in Factorised Latent Spaces}, 
      author={Philipp Bomatter and Jack Geary and Henry Gouk},
      year={2026},
      eprint={2605.22496},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.22496}, 
}
```
