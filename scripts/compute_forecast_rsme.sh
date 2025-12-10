#!/bin/bash

FORECAST_PATH="/mnt/ssd/datasets/graphcast-prediction-2021.zarr/"
GT_PATH="/mnt/ssd/datasets/cerra_full_derived.zarr/"

# This script computes the RSME of the forecast against the ground truth data.
# Adjust the dataset.val.start_time and dataset.val.end_time as needed to set the evaluation period.
# Adjust dataset.common.lead_time_hours to set the lead time of the forecast.
# Unlike WeatherBench2, this script can only evaluate one lead time at a time,
# so if you want to evaluate multiple lead times, you need to run this script
# multiple times with different lead times.
# The results will be saved in a CSV file specified by task.csv_save_path.
python compute_forecast_rsme.py \
task.forecast_data_path="$FORECAST_PATH" \
dataset.common.data_path="$GT_PATH" \
dataset.val.start_time="2021-01-01T00:00:00" \
dataset.val.end_time="2021-12-31T18:00:00" \
dataset.common.lead_time_hours=6 \
task.save_csv=True \
task.csv_save_path="forecast_rsme_results.csv"
