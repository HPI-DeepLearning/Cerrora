# Cerrora

This repository contains the code to train Cerrora, a weather model based on 
Microsoft's Aurora foundation model.
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

The CERRA dataset is available on the ECMWF's Climate Data Store (CDS). 
To access the data, you need to create an account on the [CDS website](https://cds.climate.copernicus.eu/).
Then, you can use our provided `download_cerra.py` script to download the data.
For this, navigate to the scripts folder and run:

```bash
python download_cerra.py \
--request_template cerra_full.yaml \
--start_year 2022 \
--start_month 1 \
--end_year 2022 \
--end_month 6 \
--outdir /path/to/cerra/data
```

This will download the CERRA data for the specified time range and save it in the specified output directory.
After downloading, we need to compute a few derived variables to fit the settings of the Aurora model.
This for example includes converting 10m wind speed and direction to u and v components.
To precompute the derived variables, run:

```bash
python precompute_derived_variables.py \
--src_dir /path/to/cerra/data \
--dst_dir /path/to/updated/data \
--compute_10m_wind \
--compute_relative_humidity \
--compute_specific_humidity \
--compute_geopotential_at_surface
```

This will create a new dataset in the specified destination directory with the derived variables added.
Finally, we need to add the soil type information to the dataset, which is not included in the CERRA data.
We provide a script to interpolate the ERA5 soil type data to the CERRA grid.
To do this, run:

```bash
python regrid_slt.py --cerra_zarr_path /path/to/updated/data
```

## Setting up the boundary conditions

The rollout-trained Cerrora model requires lateral boundary conditions to maintain the forecast quality 
over longer lead times.
For this, we use the IFS forecasts provided in WeatherBench2.
To download the forecasts, navigate to the scripts/rollout folder and run:

```bash
python download_ifs_forecasts.py \
--start_year 2022 \
--start_month 1 \
--end_year 2022 \
--end_month 6 \
--outdir /path/to/ifs/forecasts
```

After downloading, we need to regrid the forecasts to the CERRA grid.
To do this, run:
```bash
python create_rollout_boundaries.py \
--global_path /path/to/ifs/forecasts \
--boundary_size 250 \
--num_cores 10 \
--start_time "2022-01-01T00:00:00" \
--end_time "2022-06-30T21:00:00" \
--output_path /path/to/regridded/forecasts
```

Finally, we need to add the static variables to the regridded forecasts.
To do this, run:
```bash
python add_static_vars_to_rollout_boundaries.py \
--global_path /path/to/ifs/forecasts \
--boundary_size 250 \
--num_cores 10 \
--output_path /path/to/regridded/forecasts
```

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

To create forecasts, adjust the dataset and checkpoint paths in `forecast.sh`, then run the script.
For evaluating the forecasts, use our forked WeatherBench2 repository: [MISSING LINK]

## License

See [`LICENSE.txt`](LICENSE.txt).
