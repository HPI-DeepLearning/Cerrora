# Cerrora

This repository contains the code to train Cerrora, a weather model based on 
Microsoft's Aurora foundation model.
You can find the trained model weights on [Hugging Face](https://huggingface.co/HPI-MML/cerrora).
It is a fork of the [Microsoft Aurora repository](https://github.com/microsoft/aurora).
The Aurora model is a foundation model for atmospheric forecasting, which we finetune 
on the CERRA regional reanalysis dataset to predict weather in the European region. 
The original repo has a [documentation website](https://microsoft.github.io/aurora)
, which contains detailed information on how to use the model.

## Getting Started

We use `conda` / `mamba` for development. To install the dependencies, navigate to the repository folder and run:

```bash
mamba env create -f environment.yml
```

Then activate the environment with:

```bash
conda activate cerrora
```

## Setting up the CERRA dataset

If you just want to quickly test the pretrained Cerrora models, you can use our provided test data 
that already contains the necessary preprocessing. They are available on Hugging Face:
`hf://HPI-MML/cerrora/cerra_excerpt.zarr` for the CERRA data and 
`hf://HPI-MML/cerrora/hres_forecasts_excerpt.zarr` for the IFS forecasts.
These datasets contain data for the first 7 days of January 2022.
They can be used by adjusting the `dataset.common.data_path` and 
`dataset.common.boundary_path` arguments in the training and inference scripts.
For the provided `forecast_hf_{$MODEL}.sh` scripts, these paths are already set to 
the Hugging Face datasets.

For full training or inference runs, you need to set up the full CERRA dataset
and the boundary conditions based on IFS forecasts.
For this, please follow the [full instructions](data_setup.md). 

## Training Cerrora

To train the 6-hour lead time Cerrora model that is used as a base for the rollout training, adjust
the dataset and checkpoint paths in `train_6h.sh`, then run the script.
In case you need to resume training from a checkpoint, you can run `train_resume_6h.sh`, after
adjusting the paths accordingly.

Afterwards, to train the rollout Cerrora model, first create the checkpoint path for the training, 
and place the 6-hour model checkpoint in that folder.
Then, adjust the dataset and checkpoint paths in `train_rollout_long_w_boundaries.sh`, and run the script.

To reduce VRAM usage, we can use `task.use_activation_checkpointing=True`, which will trade off
some speed for memory usage, by checkpointing the backbone activations.
This setting is active by default in the provided training scripts.
In addition, we can use `task.checkpoint_encoder_decoder=True` to enable checkpointing for the
encoder and decoder parts of the model as well.
This setting is active by default only for the rollout training scripts.
As a result, the training can be run on GPUs with 80GB of VRAM (e.g. H100 80GB).
We trained on one node with 8 GPUs, this can be changed by adjusting the torchrun `--nnodes` and 
`--nproc_per_node` parameters.

### Logging

We use Weights and Biases for logging metrics and settings, adjust  ```logging.use_wandb``` to deactivate 
the logging if you wish to disable it.
Before starting a run with W&B logging, you need to run ```wandb login``` and enter your access key.
You find the key here: [Authorize page](https://wandb.ai/authorize)

## Inferencing

To just run inference with the pretrained Cerrora models available on Hugging Face, you can 
run `forecast_hf_{$MODEL}.sh {$OUTPUT_PATH}`. This will download the model weights from Hugging Face
and use the provided test data to create forecasts, which will be saved to the specified output path.
If you want to use your own data, adjust the dataset and boundary condition paths, as well as the 
start time and end time in the script accordingly.
The available models are:
- `base`: The 6-hour lead time Cerrora model
- `rollout`: The rollout-trained Cerrora model with boundary conditions
- `rollout_quantized`: The rollout model quantized to 4-bit weights

To create forecasts based on a trained local model and custom data, adjust the dataset and checkpoint paths in 
`forecast_local.sh`, then run the script.
For evaluating the forecasts, use our forked WeatherBench2 repository: [MISSING LINK]

## License

See [`LICENSE.txt`](LICENSE.txt).
