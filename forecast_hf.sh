#!/bin/bash

# Initialize conda:
export OMP_NUM_THREADS=4

# Config options - should remain fixed for now
MODEL="standard_Aurora"
DATASET="cerra_with_boundaries"  # Use "cerra" if not using boundaries

# Paths - change these to your paths
# The dataset_path is already correct for the fb10dl03 node
# The checkpoint_path currently points to the rollout trained model with boundaries
# The output_dir should be changed to your desired output path
DATASET_PATH="/mnt/ssd/datasets/cerra_2022.zarr"
BOUNDARY_DATASET_PATH="/mnt/ssd/datasets/hres_forecasts_2022_projected.zarr/"
OUTPUT_DIR="/mnt/ssd/datasets/forecast_aurora_6h_rollout_long_w_boundaries_final.zarr/"

rm -rf "$OUTPUT_DIR"

# task.use_wb2_format is used to control whether to save in the normal CERRA format or in
# a format compatible with WeatherBench2 (WB2). When we plan to evaluate using our own
# compute_forecast_rmse.py script, we set this to False. When we plan to evaluate using
# the WB2 evaluation scripts, we set this to True.
# Comment the dataset.common.boundary_path line out if not using boundaries.
python main.py --config-name forecast \
    task.load_from_hf=True \
    task.hf_repo="HPI-MML/cerrora" \
    task.hf_checkpoint="cerrora-rollout.ckpt" \
    task.distributed=False \
    task.model_name=Aurora \
    task.output_dir=$OUTPUT_DIR \
    task.use_wb2_format=True \
    task.lead_times="[6,12,18,24,30]" \
    task.save_fp64=False \
    model=$MODEL \
    model.patch_size=8 \
    model.use_lora=False \
    dataset=$DATASET \
    dataset.common.lead_time_hours=6 \
    dataset.common.data_path=$DATASET_PATH \
    dataset.common.boundary_path=$BOUNDARY_DATASET_PATH \
    dataset.common.boundary_size="[168,171,168,171]" \
    task.max_rollout_steps=5 \
    dataset.val.start_time="2022-01-01T00:00:00" \
    dataset.val.end_time="2022-12-31T21:00:00" \
    dataloader.num_workers=4
