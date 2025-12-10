# -*- coding: utf-8 -*-
"""
Aurora Training Module with Selective Mixed Precision

This module implements a modified mixed precision training strategy where:
- Backbone (Swin3DTransformerBackbone) operates in BF16 for memory efficiency 
- All pre-backbone computations (encoder, data preprocessing) use float32
- All post-backbone computations (decoder, loss calculation) use float32

This approach provides better numerical stability for low-magnitude values while
maintaining the memory benefits of BF16 in the computationally intensive backbone.

Training Flow:
1. Input data loaded in float32
2. Encoder processes data in float32  
3. Backbone operates in BF16 (with autocast)
4. Decoder processes backbone output in float32
5. Loss computation in float32
6. Gradient scaling and optimization as usual

Key Changes from Full BF16:
- Removed global autocast from training loop
- Model.autocast=True enables backbone-only BF16
- Explicit dtype=torch.float32 for data loading and targets
- Loss tensors initialized in float32
"""
import torch
from torch.optim.lr_scheduler import LambdaLR, LinearLR, ConstantLR, SequentialLR
from torch.amp import GradScaler, autocast
import torch.distributed as dist
import time
import os

from aurora import rollout, Batch
from aurora.data.utils import get_train_dataloader, get_val_dataloader
from aurora.rollout import merge_boundaries, remove_boundaries
from aurora.training.loss import AuroraMeanAbsoluteError, compute_latitude_weights
from aurora.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)
from aurora.training.logging import (
    log_metrics,
    log_validation,
    log_validation_rmse,
    print_time,
    log_message,
    visualize_surf_vars,
)

from aurora.evaluation.metrics import mse

import warnings
# Disable the specific warning related to `epoch` in scheduler.step()
warnings.filterwarnings("ignore", message=".*epoch parameter in .scheduler.step().*")

# The following settings are to solve the problem:
# RuntimeError: CUDA error: invalid configuration argument (if image size >= 1024x1024)
# refer to https://stackoverflow.com/questions/77343471/pytorch-cuda-error-invalid-configuration-argument
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# set cudnn to be deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

RMSE_VARIABLES = [
    {"type": "surf", "name": "2t"},
    {"type": "surf", "name": "10u"},
    {"type": "surf", "name": "10v"},
    {"type": "surf", "name": "msl"},
    {"type": "surf", "name": "tp"},
    {"type": "atmos", "name": "z", "level": 500},
    {"type": "atmos", "name": "t", "level": 850},
    {"type": "atmos", "name": "q", "level": 700},
    {"type": "atmos", "name": "u", "level": 850},
    {"type": "atmos", "name": "v", "level": 850},
]

#  Added: Helper functions for parameter statistics
def count_parameters(model, only_trainable=False):
    """Count the number of model parameters"""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def print_parameter_statistics(model):
    """Print detailed parameter statistics"""
    total_params = count_parameters(model, only_trainable=False)
    trainable_params = count_parameters(model, only_trainable=True)
    frozen_params = total_params - trainable_params

    log_message(f"Parameter Statistics:")
    log_message(f"  Total parameters: {total_params:,}")
    log_message(f"  Trainable parameters: {trainable_params:,}")
    log_message(f"  Frozen parameters: {frozen_params:,}")
    log_message(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
    return total_params, trainable_params, frozen_params

def get_lr_scheduler(optimizer, num_batches, cfg):
    def lr_lambda(current_step):
        # Half cosine decay (after warm-up)
        progress = float(current_step - cfg.lr_scheduler.warmup_steps) / float(
            max(1, cfg.task.total_steps - cfg.lr_scheduler.warmup_steps)
        )
        progress_tensor = torch.tensor(progress)  # Use the model's device

        return (
            0.5
            * (1.0 + torch.cos(torch.pi * progress_tensor))
            * (cfg.lr_scheduler.start_lr - cfg.lr_scheduler.final_lr)
            + cfg.lr_scheduler.final_lr
        )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-10, end_factor=1.0, total_iters=cfg.lr_scheduler.warmup_steps
    )

    # Scheduler setup is based on phase
    if cfg.task.phase == "pretraining":
        second_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    else:
        second_scheduler = ConstantLR(
            optimizer, factor=1.0, total_iters=num_batches - cfg.lr_scheduler.warmup_steps
        )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, second_scheduler],
        milestones=[cfg.lr_scheduler.warmup_steps],
    )

def get_current_max_rollout_steps(step, cfg):
    """
    Get the maximum rollout step based on the current training step and the
    rollout curriculum defined in the configuration.
    """

    if "rollout" not in cfg.task.phase:
        # If the task phase is not related to rollout, return 1 as no rollouts are performed
        assert cfg.task.max_rollout_steps == 1, "Max rollout steps must be 1 for non-rollout tasks"
        return 1

    if cfg.task.phase == "rollout_long_buffer":
        # For now, we don't support rollout curriculum for rollout_long_buffer task
        # because that would require modifications to the replay buffer logic.
        # And since we don't use rollout_long_buffer anyway, there is no point in implementing it.
        assert len(cfg.task.rollout_curriculum) == 1, "Rollout curriculum not supported for rollout_long_buffer"

        curiculum_rollout_steps = cfg.task.rollout_curriculum[0][1]
        max_rollout_steps = cfg.task.max_rollout_steps

        # Make sure that the config is consistent
        assert curiculum_rollout_steps == max_rollout_steps, \
            "Rollout curriculum steps must match max rollout steps for rollout_long_buffer phase"

        # If using a replay buffer, there is no need to perform the rollout explicitly
        # The replay buffer will handle the rollout internally
        # So we return 1 to indicate that we are not performing any rollouts
        return 1

    max_rollout_steps = cfg.task.rollout_curriculum[0][1]  # Default to the first rollout step

    for start_step, rollout_step in cfg.task.rollout_curriculum:
        if step >= start_step:
            max_rollout_steps = rollout_step
        else:
            break

    return max_rollout_steps

def get_initial_input_batch(sample: dict[str, Batch], cfg) -> Batch:
    """
    Get the initial input batch from the sample dictionary.
    If boundary conditions are provided, they are merged into the input batch.

    Args:
        sample (dict[str, Batch]): The sample dictionary containing input data.

    Returns:
        Batch: The initial input batch.
    """
    if "input_boundary" not in sample:
        return sample["input"]

    cerra_initial_conditions = sample["input"]
    boundary_conditions = sample["input_boundary"]
    boundary_size = cfg.dataset.common.boundary_size

    return merge_boundaries(cerra_initial_conditions, boundary_conditions, boundary_size)

def train(model, cfg, device):
    """
    Train the given model using mixed precision where only the backbone operates in bf16 
    and all other computations (before and after backbone) are performed in float32.

    Args:
        model (torch.nn.Module): The model to be trained.
        cfg (DictConfig): Hydra configuration object containing hyperparameters.
        device (torch.device): The device to which the model and data is moved.
    Returns:
        None: The function trains the model in-place

        - Mixed precision is used where only backbone operates in bf16, other computations in float32.

    Example:
        >>> train(model, cfg, torch.device("cuda:0"))
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        if cfg.task.distributed:
            raise RuntimeError("Please use torchrun to start the training.")
        else:
            local_rank = 0

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    if local_rank >= num_gpus:
        raise RuntimeError(f"Local rank {local_rank} exceeds the number of available GPUs {num_gpus}.")

    # Added: LoRA parameter filtering logic
    if cfg.model.use_lora:
        if local_rank == 0: 
            # When using LoRA, only optimize LoRA-related parameters
            log_message("Using LoRA: Only optimizing LoRA parameters, freezing pretrained weights.")

        # Filter LoRA parameters (parameters containing "lora" keyword)
        lora_params = []
        frozen_params = []
        
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                lora_params.append(param)
                param.requires_grad = True  # Ensure LoRA parameters are trainable

            else:
                frozen_params.append(param)
                param.requires_grad = False  # Freeze non-LoRA parameters
        
        # Apply weight decay strategy to LoRA parameters
        lora_decay_params = []
        lora_no_decay_params = []
        
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                if 'weight' in name and 'norm' not in name:
                    lora_decay_params.append(param)
                else:
                    lora_no_decay_params.append(param)
        
        # Create optimizer with only LoRA parameters
        optimizer_params = []
        if lora_decay_params:
            optimizer_params.append({'params': lora_decay_params, 'weight_decay': cfg.optimizer.weight_decay})
        if lora_no_decay_params:
            optimizer_params.append({'params': lora_no_decay_params, 'weight_decay': 0.0})
            
        if not optimizer_params:
            raise RuntimeError("No LoRA parameters found! Please check your model configuration.")
            
        optimizer = torch.optim.AdamW(
            params=optimizer_params,
            lr=cfg.optimizer.constant_lr
        )
        
    else:
        if local_rank == 0:
            # Original logic: optimize all parameters
            log_message("Not using LoRA: Optimizing all model parameters.")
        
        decay_params = {k: True for k, v in model.named_parameters() if 'weight' in k and not 'norm' in k}
        decay_opt_params = [v for k, v in model.named_parameters() if k in decay_params]
        no_decay_opt_params = [v for k, v in model.named_parameters() if k not in decay_params]
        optimizer = torch.optim.AdamW(
            params=[{'params': decay_opt_params}, {'params': no_decay_opt_params, 'weight_decay': 0}],
            lr=cfg.optimizer.constant_lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    
    if local_rank == 0:
        # Added: Display detailed parameter statistics
        print_parameter_statistics(model)

    val_interval = cfg.validation.validation_interval

    evaluate_rmse = True
    if cfg.dataset.common._target_ == "aurora.data.dummy.DummyDataset":
        evaluate_rmse = False

    scheduler = get_lr_scheduler(optimizer, cfg.task.total_steps, cfg)
    scaler = GradScaler()  # proper scaling of gradients for mixed precision training.

    if cfg.checkpoint.continue_training:
        ckpt_path = os.path.join(cfg.checkpoint.ckpt_dir, cfg.checkpoint.ckpt_file)
        try:
            # Modified: When using LoRA, allow partial checkpoint loading when resetting steps
            # This is for the case where we transition from pretraining to fine-tuning
            strict_loading = not (cfg.model.use_lora and cfg.checkpoint.reset_steps)
            # Don't load optimizer state when resuming in LoRA mode
            # The old optimizer state would not be compatible with the new LoRA parameters
            load_optimizer_state = not (cfg.model.use_lora and cfg.checkpoint.reset_steps)
            start_epoch, global_step = load_checkpoint(local_rank, model, optimizer, scheduler, scaler, 
                                                       ckpt_path, strict=strict_loading, 
                                                       load_optimizer_state=load_optimizer_state)
            
            if cfg.model.use_lora and cfg.checkpoint.reset_steps and local_rank == 0:
                log_message("LoRA mode: Successfully loaded pretrained weights, LoRA parameters will use random initialization")
                log_message("LoRA mode: Optimizer and scheduler state not loaded, starting fresh optimization")
            elif cfg.model.use_lora and local_rank == 0:
                log_message("LoRA mode: Successfully loaded pretrained weights, continuing training with existing LoRA parameters")
                log_message("LoRA mode: Optimizer and scheduler state loaded, continuing optimization")
                
            wandb_step = global_step

            if cfg.task.distributed:
                dist.barrier()
        except RuntimeError as e:
            raise RuntimeError(f"Error loading checkpoint: {e}")

        if cfg.checkpoint.reset_steps:
            start_epoch = 0
            global_step = 0
            wandb_step = 0

            # Reset optimizer, scheduler, and scaler to make sure we start from scratch
            # Modified: Apply the same LoRA parameter filtering logic
            if cfg.model.use_lora:
                # Also only optimize LoRA parameters when resetting
                log_message("Resetting optimizer: Only optimizing LoRA parameters.")
                
                # Re-filter LoRA parameters
                lora_decay_params = []
                lora_no_decay_params = []
                
                for name, param in model.named_parameters():
                    if "lora" in name.lower() and param.requires_grad:
                        if 'weight' in name and 'norm' not in name:
                            lora_decay_params.append(param)
                        else:
                            lora_no_decay_params.append(param)
                
                # Create reset optimizer
                optimizer_params = []
                if lora_decay_params:
                    optimizer_params.append({'params': lora_decay_params, 'weight_decay': cfg.optimizer.weight_decay})
                if lora_no_decay_params:
                    optimizer_params.append({'params': lora_no_decay_params, 'weight_decay': 0.0})
                    
                optimizer = torch.optim.AdamW(
                    params=optimizer_params,
                    lr=cfg.optimizer.constant_lr
                )
            else:
                # Original logic: reset optimizer for all parameters
                decay_params = {k: True for k, v in model.named_parameters() if 'weight' in k and not 'norm' in k}
                decay_opt_params = [v for k, v in model.named_parameters() if k in decay_params]
                no_decay_opt_params = [v for k, v in model.named_parameters() if k not in decay_params]
                optimizer = torch.optim.AdamW(
                    params=[{'params': decay_opt_params}, {'params': no_decay_opt_params, 'weight_decay': 0}],
                    lr=cfg.optimizer.constant_lr,
                    weight_decay=cfg.optimizer.weight_decay
                )
            
            scheduler = get_lr_scheduler(optimizer, cfg.task.total_steps, cfg)
            scaler = GradScaler()
    else:
        start_epoch = 0
        global_step = 0
        wandb_step = 0

    # To make sure we don't load more data than necessary, we get the current maximum rollout steps
    dataset_max_rollout_steps_old = get_current_max_rollout_steps(global_step, cfg)
    log_message(f"Instantiating dataloaders with max_rollout_steps={dataset_max_rollout_steps_old}")
    train_dataloader = get_train_dataloader(cfg, max_rollout_steps=dataset_max_rollout_steps_old)
    val_dataloader = get_val_dataloader(cfg)
    num_batches = len(train_dataloader)
    num_epochs = (cfg.task.total_steps + num_batches - 1) // num_batches
    num_val_batches = len(val_dataloader)

    model.train()
    best_val_loss = float('inf')
    criterion = None

    start_time = time.time()
    log_message("Data is loaded")
    print_time("training_start", start_time)

    max_rollout_steps = cfg.task.max_rollout_steps

    for epoch in range(start_epoch, num_epochs):
        # Check if the maximum rollout steps will change during this epoch
        dataset_max_rollout_steps_new = get_current_max_rollout_steps(global_step + len(train_dataloader), cfg)
        if dataset_max_rollout_steps_new != dataset_max_rollout_steps_old:
            # If the maximum rollout steps have changed, we need to update the dataloader
            log_message(f"Updating dataloader: max_rollout_steps changed from {dataset_max_rollout_steps_old} to {dataset_max_rollout_steps_new}")

            # Just in case the curriculum decreased the max_rollout_steps, though I don't expect this to happen
            train_dl_max_rollout_steps = max(dataset_max_rollout_steps_new, dataset_max_rollout_steps_old)
            train_dataloader = get_train_dataloader(cfg, max_rollout_steps=train_dl_max_rollout_steps)
            dataset_max_rollout_steps_old = dataset_max_rollout_steps_new

        if cfg.task.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        epoch_start_time = time.time()

        for i, sample in enumerate(train_dataloader):
            # Ensure input batch is in float32 for pre-backbone computations
            batch = get_initial_input_batch(sample, cfg)
            batch = batch.to(device, non_blocking=True).type(torch.float32)
            current_rollout_steps = get_current_max_rollout_steps(global_step, cfg)

            rollout_batches = None
            boundary_size = cfg.dataset.common.boundary_size
            if "input_boundary_0" in sample:
                rollout_batches = []
                # Collect the input boundary batches
                for rollout_step in range(current_rollout_steps - 1):
                    step_batch = sample[f"input_boundary_{rollout_step}"]
                    step_batch = step_batch.to(device, non_blocking=True).type(torch.float32)
                    rollout_batches.append(step_batch)

            # remove non-existent RMSE variables from batch input
            if (epoch==start_epoch) and (i==0):
                # Get the variable information from the batch
                surf_vars = batch.surf_vars
                atmos_vars = batch.atmos_vars
                atmos_levels = batch.metadata.atmos_levels

                # Iterate backwards through the list to avoid index issues during deletion
                for k in range(len(RMSE_VARIABLES) - 1, -1, -1):
                    var = RMSE_VARIABLES[k]
                    if var["type"] == "surf":
                        if var["name"] not in surf_vars:
                            log_message(f"Surface variable {var['name']} does not exist, deleting this entry.")
                            del RMSE_VARIABLES[k]
                    elif var["type"] == "atmos":
                        # Check if the variable name and its corresponding level exist
                        if var["name"] not in atmos_vars or var["level"] not in atmos_levels:
                            log_message(f"Atmospheric variable {var['name']} or its level {var['level']} does not exist, deleting this entry.")
                            del RMSE_VARIABLES[k]

                cerra_batch = sample["input"].to(device, non_blocking=True).type(torch.float32)
                latitude_weights = compute_latitude_weights(cerra_batch.metadata.lat)
                criterion = AuroraMeanAbsoluteError(variable_weights=cfg.variable_weights,
                                                    latitude_weights=latitude_weights)


            optimizer.zero_grad(set_to_none=True)
            loss = torch.tensor(0.0, device=device, dtype=torch.float32)

            # Forward pass - backbone uses bf16, pre/post computations use float32
            if cfg.task.phase == "rollout_long":
                rollout_steps_without_grad_batch = torch.randint(0, current_rollout_steps, (1,), device=device)
                if cfg.task.distributed and dist.is_initialized():
                    torch.distributed.broadcast(rollout_steps_without_grad_batch, src=0)  # Sync the rollout steps across all GPUs
                rollout_steps_without_grad_batch = rollout_steps_without_grad_batch.cpu().item()
                with torch.no_grad():
                    # We also do standard one-step training to make sure the model doesn't forget
                    # how to do one-step predictions. In this case, no rollout is performed.
                    if rollout_steps_without_grad_batch == 0:
                        rollout_input = batch
                    else:
                        for _, rollout_input in rollout(
                                model,
                                batch,
                                rollout_steps_without_grad_batch,
                                boundary_size,
                                rollout_batches,
                                return_last_input=True,
                        ):
                            continue

                # For rollout_long phase, compute loss separately
                # Ensure targets are in float32 for loss computation
                target_key = f"target_{rollout_steps_without_grad_batch}"
                targets = sample[target_key].to(device, non_blocking=True).type(torch.float32)
                pred = model(rollout_input)
                pred = remove_boundaries(pred, boundary_size)
                loss = criterion(pred, targets)
            else:
                step = 0
                if cfg.task.phase == "rollout_short":
                    loss_val_per_step = [0] * cfg.task.max_rollout_steps
                for prediction, _ in rollout(
                        model,
                        batch,
                        current_rollout_steps,
                        boundary_size,
                        rollout_batches,
                        return_last_input=False,
                ):
                    # Ensure targets are in float32 for loss computation
                    targets = sample[f"target_{step}"].to(device, non_blocking=True).type(torch.float32)
                    current_loss = criterion(prediction, targets)

                    if cfg.task.phase == "rollout_short":
                        loss_val_per_step[step] = current_loss.item()
                    loss += (current_loss / current_rollout_steps)
                    step += 1

                if cfg.task.phase == "rollout_long_buffer":
                    train_dataloader.add_rollout_samples(sample, prediction)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            global_step += 1

            # For logging purposes, synchronize the loss across all GPUs.
            if cfg.task.distributed and dist.is_initialized():
                # 1. Create a new variable for logging and detach it from the computation graph.
                log_loss = loss.clone().detach()
                
                # 2. Synchronize this new variable to get the average value across all GPUs.
                dist.all_reduce(log_loss, op=dist.ReduceOp.AVG)
            else:
                # In non-distributed mode, the logged loss is simply the current loss.
                log_loss = loss


            if local_rank == 0:
                wandb_step += 1
                elapsed_time = time.time() - start_time
                steps_per_sec = global_step / elapsed_time
                remaining_steps = cfg.task.total_steps - global_step
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                step_stats = {
                    "epoch": epoch + 1,
                    "num_epochs": num_epochs,
                    "global_step": global_step,
                    "total_steps": cfg.task.total_steps,
                    "wandb_step": wandb_step,
                    "batch_index": i,
                    "num_batches": num_batches,
                    "steps_per_sec": steps_per_sec,
                }
                if cfg.task.phase == "rollout_short":
                    for i, value in enumerate(loss_val_per_step):
                        key = f"loss_step_{i}"
                        step_stats[key] = value
                timing_stats = {"eta_seconds": eta_seconds}
                lr = optimizer.param_groups[0]["lr"]
                log_metrics(step_stats, log_loss, lr, timing_stats)

            # 12904 steps  ==> 1 epoch
            if global_step % val_interval == 0 or global_step >= cfg.task.total_steps:
                model.eval()
                total_val_loss = 0.0

                if evaluate_rmse:
                    total_mse_results = {f"{var['name']}{'_' + str(var['level']) if 'level' in var else ''}_step{step}": 0.0
                                for var in RMSE_VARIABLES for step in range(max_rollout_steps)}
                    stats_dict = val_dataloader.dataset.stats

                num_nan_batches = 0  # Number of validation batches skipped due to NaNs
                with torch.no_grad():  # No gradient computation for evaluation
                    for j, sample in enumerate(val_dataloader):
                        # Ensure validation batch is in float32
                        batch = get_initial_input_batch(sample, cfg)
                        batch = batch.to(device, non_blocking=True).type(torch.float32)
                        val_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                        rollout_batches = None
                        boundary_size = cfg.dataset.common.boundary_size
                        if "input_boundary_0" in sample:
                            rollout_batches = []
                            contains_nan = torch.tensor(0, device=device, dtype=torch.int)

                            # Collect the input boundary batches
                            for rollout_step in range(max_rollout_steps - 1):
                                step_batch = sample[f"input_boundary_{rollout_step}"]
                                step_batch = step_batch.to(device, non_blocking=True).type(torch.float32)
                                rollout_batches.append(step_batch)

                                # Check for NaNs in boundary conditions
                                # For some reason there are some NaNs in the boundary conditions in the validation set
                                # This is not due to anything we can control, it's already in the data in the GCP bucket
                                # See e.g. geopotential for 2020-02-22T12:00:00
                                is_nan_atmos_vars = [value.isnan().any() for key, value in step_batch.atmos_vars.items()]
                                is_nan_surf_vars = [value.isnan().any() for key, value in step_batch.surf_vars.items()]
                                is_nan_static_vars = [value.isnan().any() for key, value in step_batch.static_vars.items()]
                                if any(is_nan_atmos_vars) or any(is_nan_surf_vars) or any(is_nan_static_vars):
                                    contains_nan = torch.tensor(1, device=device, dtype=torch.int)
                                    break

                            if cfg.task.distributed and dist.is_initialized():
                                # If any of the batches contains NaN, we skip this entire batch
                                dist.all_reduce(contains_nan, op=dist.ReduceOp.SUM)
                            if contains_nan.item() > 0:
                                num_nan_batches += 1
                                log_message(f"Warning: Found NaN in boundary conditions of validation batch {j}, skipping this batch.")
                                continue

                        # Forward pass - backbone uses bf16, pre/post computations use float32
                        previous_input = batch

                        step = 0
                        for prediction, rollout_input in rollout(
                                model,
                                batch,
                                max_rollout_steps,
                                boundary_size,
                                rollout_batches,
                                return_last_input=False,
                        ):
                            # Ensure target is in float32 for loss computation
                            target = sample[f"target_{step}"].to(device, non_blocking=True).type(torch.float32)
                            val_loss += criterion(prediction, target) / max_rollout_steps

                            if j == 0:
                                visualize_surf_vars(
                                    previous_input,
                                    prediction,
                                    target,
                                    train_dataloader.dataset.stats,
                                    epoch + 1,
                                    rollout_step=step,
                                    save_on_next_step=True,
                                )

                            previous_input = rollout_input

                            if evaluate_rmse:
                                prediction_unnormalized = prediction.unnormalise(stats=stats_dict)
                                target_unnormalized = target.unnormalise(stats=stats_dict)

                                for var in RMSE_VARIABLES:
                                    if var['type'] == 'surf':
                                        current_mse = mse(prediction_unnormalized, target_unnormalized,
                                                        variable=var['name'], latitude_weights=latitude_weights)
                                    else:
                                        current_mse = mse(prediction_unnormalized, target_unnormalized,
                                                        variable=var['name'], level=var['level'], latitude_weights=latitude_weights)

                                    if cfg.task.distributed and dist.is_initialized():
                                        dist.all_reduce(current_mse)
                                        current_mse = current_mse / dist.get_world_size()

                                    result_key = f"{var['name']}{'_' + str(var['level']) if 'level' in var else ''}_step{step}"
                                    total_mse_results[result_key] += current_mse.item()

                            step += 1

                        if cfg.task.distributed and dist.is_initialized():
                            dist.all_reduce(val_loss)
                            val_loss = val_loss / dist.get_world_size()
                        total_val_loss += val_loss.item()



                val_loss = total_val_loss / (num_val_batches - num_nan_batches)
                if evaluate_rmse:
                    rmse_results = {
                        key: (total_mse / (num_val_batches - num_nan_batches)) ** 0.5
                        for key, total_mse in total_mse_results.items()
                    }

                if cfg.task.distributed and dist.is_initialized():
                    dist.barrier()

                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                # Save best model if validation loss improved
                os.makedirs(cfg.checkpoint.ckpt_dir, exist_ok=True)
                best_model_file = f"aurora-{cfg.task.model_name}-{cfg.task.phase}-step{global_step}-best.ckpt"
                best_model_path = os.path.join(cfg.checkpoint.ckpt_dir, best_model_file)
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_model_path, global_step)

                if local_rank == 0:
                    # Log validation intermediate results
                    step_stats = {
                        "epoch": epoch + 1,
                        "num_epochs": num_epochs,
                        "batch_index": j,
                        "num_batches": (num_val_batches - num_nan_batches),
                        "global_step": global_step,
                    }
                    log_validation(step_stats, val_loss, best_val_loss)
                    if evaluate_rmse:
                        log_validation_rmse(step_stats, rmse_results)

                    # Print information only on the main process
                    epoch_time = time.time() - epoch_start_time
                    print_time(f"epoch_{epoch + 1}_duration", epoch_time)

            if global_step >= cfg.task.total_steps:
                # Reached the total number of steps
                break

    train_duration = time.time() - start_time
    print_time("train_duration", train_duration)
