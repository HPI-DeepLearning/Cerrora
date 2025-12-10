#!/bin/bash

# Initialize conda:
# wandb login xxx
export OMP_NUM_THREADS=4

ulimit -n 8192
DATASET_PATH="/sc/projects/sci-aisc/ekapex/cerra_full_derived_1984_2023.zarr"
BOUNDARY_DATASET_PATH="/sc/projects/sci-aisc/ekapex/ifs_forecasts_projected.zarr"
OUTPUT_DIR="/sc/projects/sci-aisc/ekapex/trained_models_on_cerra_jona/Aurora_large_6h_rollout_short_w_boundaries_lora"
CHECKPOINT_PATH="put_your_checkpoint_filename_here.ckpt"
mkdir -p "$OUTPUT_DIR"

MODEL="standard_Aurora"
VARIABLE_WEIGHTS="finetuning"
DATASET="cerra_with_boundaries"

echo "Running with model.patch_size=8..."
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master_port=25794 main.py \
    task=rollout_short_boundary \
    task.distributed=True \
    task.model_name=Aurora \
    task.total_steps=25000 \
    task.max_rollout_steps=5 \
    task.checkpoint_encoder_decoder=True \
    model=$MODEL \
    model.patch_size=8 \
    model.autocast=True \
    model.use_lora=True \
    model.lora_mode=all \
    variable_weights=$VARIABLE_WEIGHTS \
    optimizer.weight_decay=0 \
    optimizer.constant_lr=1e-5 \
    lr_scheduler.warmup_steps=1000 \
    checkpoint.continue_training=True \
    checkpoint.ckpt_dir="$OUTPUT_DIR" \
    checkpoint.ckpt_file="$CHECKPOINT_PATH" \
    dataset=$DATASET \
    dataset.common.data_path="$DATASET_PATH" \
    dataset.common.boundary_path="$BOUNDARY_DATASET_PATH" \
    dataset.common.boundary_size="[168,171,168,171]" \
    dataset.common.lead_time_hours=6 \
    dataloader.num_workers=4 \
    dataloader.prefetch_factor=1 \
    logging.project_name=Aurora_large_6h_rollout_short_w_boundaries \
    checkpoint.reset_steps=True