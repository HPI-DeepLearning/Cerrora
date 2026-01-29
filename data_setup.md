# Dataset Setup Instructions

Use these instructions to set up the CERRA dataset and the required boundary conditions 
for the rollout-trained Cerrora model. If you just want to test the pretrained model, you can 
instead use our provided test data that already contains the necessary preprocessing.

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
