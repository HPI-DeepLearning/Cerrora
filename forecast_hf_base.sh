#!/bin/bash

# Initialize conda:
export OMP_NUM_THREADS=4

# Config options - should remain fixed for now
MODEL="standard_Aurora"
DATASET="cerra"

# Paths - change these to your paths
DATASET_PATH="hf://HPI-MML/cerrora/cerra_excerpt.zarr"
OUTPUT_DIR=$1

rm -rf "$OUTPUT_DIR"

# task.use_wb2_format is used to control whether to save in the normal CERRA format or in
# a format compatible with WeatherBench2 (WB2). When we plan to evaluate using our own
# compute_forecast_rmse.py script, we set this to False. When we plan to evaluate using
# the WB2 evaluation scripts, we set this to True.
python main.py --config-name forecast \
    task=forecast \
    task.load_from_hf=True \
    task.hf_repo="HPI-MML/cerrora" \
    task.hf_checkpoint="cerrora-base.ckpt" \
    task.distributed=False \
    task.model_name=Aurora \
    task.output_dir=$OUTPUT_DIR \
    task.use_wb2_format=True \
    task.lead_times="[6]" \
    task.save_fp64=False \
    model=$MODEL \
    model.patch_size=8 \
    model.use_lora=False \
    dataset=$DATASET \
    dataset.common.lead_time_hours=6 \
    dataset.common.data_path=$DATASET_PATH \
    task.max_rollout_steps=1 \
    dataset.val.start_time="2022-01-01T06:00:00" \
    dataset.val.end_time="2022-01-07T21:00:00" \
    dataloader.num_workers=0
