#!/bin/bash

# Initialize conda:
# wandb login xxx
export OMP_NUM_THREADS=4

ulimit -n 8192
DATASET_PATH="/sc/projects/sci-aisc/ekapex/cerra_full_derived_1984_2023.zarr"
OUTPUT_DIR="/sc/projects/sci-aisc/ekapex/trained_models_on_cerra_jona/Aurora_large_6h_final"
mkdir -p "$OUTPUT_DIR"

MODEL="standard_Aurora"
VARIABLE_WEIGHTS="finetuning"
DATASET="cerra"

echo "Running with model.patch_size=8..."
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 --master_port=25793 main.py \
    task.distributed=True \
    task.model_name=Aurora \
    task.phase=finetuning \
    task.total_steps=200000 \
    task.max_rollout_steps=1 \
    model=$MODEL \
    model.patch_size=8 \
    model.autocast=True \
    variable_weights=$VARIABLE_WEIGHTS \
    optimizer.weight_decay=0 \
    optimizer.constant_lr=1e-4 \
    lr_scheduler.warmup_steps=1000 \
    checkpoint.ckpt_dir="$OUTPUT_DIR" \
    dataset=$DATASET \
    dataset.common.data_path="$DATASET_PATH" \
    dataset.common.lead_time_hours=6 \
    dataloader.num_workers=8 \
    dataloader.prefetch_factor=1 \
    logging.project_name=Aurora_large_6h