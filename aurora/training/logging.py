import sys
from logging import info

import torch
import wandb
import datetime
import json

import matplotlib.pyplot as plt
from aurora.batch import Batch
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.transforms import Bbox

from aurora.normalisation import unnormalise_surf_var

UNITS_ATMOS = {
    "msl": "[Pa]",
    "tp": "[mm/m2]",
    "t": "[K]",
    "u": "[m/s]",
    "v": "[m/s]",
    "z": "[m2/s2]",
    "q": "[g/kg]",
}


VARIABLE_NAMES_SURFACE = {
    "msl": "Mean Sea Level Pressure",
    "2t": "2m Temperature",
    "10u": "10m U Wind Component",
    "10v": "10m V Wind Component",
    "tp": "Total Precipitation",
}

UNITS_SURFACE = {
    "msl": "hPa ",
    "2t": "°C",
    "10u": "m/s",
    "10v": "m/s",
    "tp": "mm",
}


def log_on_rank_zero(func):
    """
    Decorator to ensure that the decorated function is only executed
    when rank is 0 (i.e., the main process in a distributed setup).
    """

    def wrapper(*args, **kwargs):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            return func(*args, **kwargs)
        return None  # Skip logging if not rank 0

    return wrapper


def is_serializable(value):
    """
    Check if a value is JSON-serializable.

    Args:
        value: The value to check.

    Returns:
        bool: True if the value is JSON-serializable, False otherwise.
    """
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False


def sanitize_cfg(cfg):
    """
    Forcefully sanitize the configuration to ensure all nested dictionaries 
    are recognized as standard Python dictionaries.
    """
    sanitized = {}
    for key, value in cfg.items():
        if isinstance(value, dict) or hasattr(value, "items"):  # Check if it's dictionary-like
            sanitized[key] = sanitize_cfg(dict(value))  # Convert to a standard dictionary
        elif isinstance(value, (int, float, str, bool, list)):  # Supported basic types
            sanitized[key] = value
        else:
            print(f"Warning: Key '{key}' has non-serializable value '{value}', converting to string.")
            sanitized[key] = str(value)  # Convert other types to string
    return sanitized


@log_on_rank_zero
def initialize_wandb(cfg):
    """
    Initialize Weights & Biases logging.

    Args:
        cfg: The Hydra configuration object.
    """
    wandb_cfg = sanitize_cfg(cfg)
    wandb_mode = "online" if cfg.logging.use_wandb else "disabled"

    # Run name generation fails when forecasting, so we use a generic name in that case
    run_name = generate_run_name(cfg) if cfg.logging.use_wandb else "offline"

    wandb.init(
        project=cfg.logging.project_name,
        config=wandb_cfg,
        name=run_name,
        group=cfg.logging.group_name,
        job_type=cfg.task.task,
        entity="hpi-deep-learning",
        mode=wandb_mode,
    )


@log_on_rank_zero
def generate_run_name(cfg):
    """
    Generate a descriptive and unique run name based on configuration parameters.

    Args:
        cfg: The Hydra configuration object.
    Returns:
        str: A formatted run name.
    """
    task = cfg.task.task
    batch_size = cfg.dataloader.batch_size
    warmup_steps = cfg.lr_scheduler.warmup_steps
    start_lr = cfg.lr_scheduler.start_lr
    final_lr = cfg.lr_scheduler.final_lr
    total_steps = cfg.task.total_steps

    return f"{task}_bs{batch_size}_wu{warmup_steps}_lr{start_lr}-{final_lr}_steps{total_steps}"


@log_on_rank_zero
def log_metrics(step_stats, loss, lr, timing_stats):
    """
    Log metrics to Weights & Biases.

    Args:
        step_stats (dict): Dictionary containing step_stats such as global_step, epoch, etc.
        loss (Tensor): The current loss value.
        lr (float): The current learning rate.
        timing_stats (dict): Dictionary containing timing statistics.
    """
    eta_seconds = timing_stats.get("eta_seconds", 0)
    eta_hours = eta_seconds / 3600
    sys.stdout.write("\r"
        f"Epoch {step_stats['epoch']}/{step_stats['num_epochs']} | "
        f"Global Step: {step_stats['global_step']}/{step_stats['total_steps']} | "
        f"Batch {step_stats['batch_index'] + 1}/{step_stats['num_batches']} | "
        f"Loss: {loss.item():.4f} | "
        f"LR: {lr:.2e} | "
        f"Steps/sec: {step_stats['steps_per_sec']:.2f} | "
        f"ETA: {eta_hours:.2f} hours"
    )
    sys.stdout.flush()
    if step_stats["batch_index"] + 1 == step_stats["num_batches"] or step_stats["global_step"] >= step_stats["total_steps"]:
        sys.stdout.write("\n")
    log_data = {
        "global_step": step_stats["global_step"],
        "epoch": step_stats["epoch"],
        "step": step_stats["wandb_step"],
        "loss": loss.item(),
        "batch": step_stats["batch_index"] + 1,
        "learning_rate": lr,
        "steps/sec": step_stats["steps_per_sec"],
        "eta": eta_hours,
    }
    step_loss_count = sum(key.startswith('loss_step_') for key in step_stats.keys())
    if step_loss_count > 0:
        step_losses = {f"loss_step_{i}": step_stats[f"loss_step_{i}"] for i in range(step_loss_count)}
        log_data.update(step_losses)
    wandb.log(log_data)


@log_on_rank_zero
def log_validation(step_stats, val_loss, best_val_loss):
    """
    Display validation progress to stdout.
    Args:
        epoch (int): Current epoch number
        num_epochs (int): Total number of epochs
        val_loss (float): Current validation loss
        best_val_loss (float): Best validation loss so far
    """
    sys.stdout.write("\r"
        f"Epoch {step_stats['epoch']}/{step_stats['num_epochs']} | "
        f"Batch {step_stats['batch_index'] + 1}/{step_stats['num_batches']} | "
        f"Validation Loss: {val_loss:.4f} | "
        f"Best Loss: {best_val_loss:.4f}"
    )
    sys.stdout.flush()
    if step_stats["batch_index"] + 1 == step_stats["num_batches"]:
        sys.stdout.write("\n")
    wandb.log(
        {
            "epoch": step_stats["epoch"],
            "global_step": step_stats["global_step"],
            "val/val_loss": val_loss,
            "val/best_val_loss": best_val_loss,
        }
    )


def get_unit(key):
    for substring, unit in UNITS_ATMOS.items():
        if substring in key:
            return unit
    return ""


@log_on_rank_zero
def log_validation_rmse(step_stats, val_rmse, stdout=False, units=False):
    """
    Display RMSE results to stdout and log them using wandb.
    Args:
        epoch (int): Current epoch number
        num_epochs (int): Total number of epochs
        val_rmse (dict): Dictionary containing RMSE values for each variable
    """
    if stdout:
        sys.stdout.write(
            f"Epoch {step_stats['epoch']}/{step_stats['num_epochs']} | RMSE Results" + " | " +
            " | ".join([f"{key}: {value:.4f}" for key, value in val_rmse.items()]) + "\n"
        )
        sys.stdout.flush()
    wandb.log(
        {
            "epoch": step_stats["epoch"],
            "global_step": step_stats["global_step"],
            **{f"val/rmse_{key} {get_unit(key)}" if units and get_unit(key) else f"val/rmse_{key}": value
            for key, value in val_rmse.items()},
        }
    )


@log_on_rank_zero
def print_time(description: str, timestamp: float):
    """
    Print a human-readable timestamp to the console.

    Args:
        description (str): The description of the event being logged.
        timestamp (float): The timestamp as returned by time.time().
    """
    # Convert the timestamp to a datetime object
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    info(f"{description}: { dt_object.strftime('%Y-%m-%d %H:%M:%S')}")


@log_on_rank_zero
def log_message(description):
    info(description)


def get_vmin_vmax(data: list[np.ndarray]) -> tuple[float, float]:
    """
    Compute the global min and max across a list of arrays.

    Args:
        data (list[np.ndarray]): List of NumPy arrays.

    Returns:
        tuple[float, float]: The global minimum and maximum values.
    """
    all_data = np.concatenate([arr.flatten() for arr in data])
    return all_data.min(), all_data.max()


@log_on_rank_zero
def visualize_surf_vars(
    input_batch: Batch,
    pred_batch: Batch,
    gt_batch: Batch,
    stats: dict[str, tuple[float, float]],
    epoch: int,
    rollout_step: int,
    save_on_next_step: bool = False,
):
    """
    Visualizes surface variables and logs them with Weights & Biases.

    Args:
        input_batch (Batch): Input data batch.
        pred_batch (Batch): Prediction batch.
        gt_batch (Batch): Ground truth batch.
        stats (dict): Normalization statistics.
        epoch (int): Current epoch.
        rollout_step (int): Current rollout step.
        save_on_next_step (bool):
            Flag whether plots saved on next wandb step
    Returns:
        None
    """
    # Unnormalize batches
    input_batch = input_batch.unnormalise(stats)
    pred_batch = pred_batch.unnormalise(stats)
    gt_batch = gt_batch.unnormalise(stats)

    for variable_key in input_batch.surf_vars.keys():
        log_message(f"Visualizing variable: {variable_key}")

        batch_size = len(input_batch.metadata.time)
        fig, axes = plt.subplots(batch_size, 5, figsize=(20, 5 * batch_size))
        axes = np.atleast_2d(axes)

        # Prepare data for `vmin` and `vmax` calculation
        all_data = []
        diff_data = []

        for i in range(batch_size):
            input_var_1 = input_batch.surf_vars[variable_key][i, 0].cpu().numpy()
            input_var_2 = input_batch.surf_vars[variable_key][i, 1].cpu().numpy()
            pred_var = (
                pred_batch.surf_vars[variable_key][i, 0]
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            gt_var = gt_batch.surf_vars[variable_key][i, 0].cpu().numpy()
            difference = np.abs(pred_var - gt_var)

            # Aggregate data for global scaling
            # We exclude the prediction from the scaling to prevent the colormap from changing over time
            all_data.extend([input_var_1, input_var_2, gt_var])
            diff_data.append(difference)

        # Compute global vmin and vmax
        # diff_max is 3 standard deviations
        var_max = unnormalise_surf_var(torch.tensor(3), variable_key, stats=stats).item()
        diff_min = 0
        diff_max = var_max - unnormalise_surf_var(torch.tensor(0), variable_key, stats=stats).item()
        var_min, var_max = get_vmin_vmax(all_data)

        # Separate colormap normalizations
        norm_main = mcolors.Normalize(vmin=var_min, vmax=var_max)
        norm_diff = mcolors.Normalize(vmin=diff_min, vmax=diff_max)

        # Visualization
        for i in range(batch_size):
            input_var_1 = input_batch.surf_vars[variable_key][i, 0].cpu().numpy()
            input_var_2 = input_batch.surf_vars[variable_key][i, 1].cpu().numpy()
            pred_var = (
                pred_batch.surf_vars[variable_key][i, 0]
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            gt_var = gt_batch.surf_vars[variable_key][i, 0].cpu().numpy()

            difference = np.abs(pred_var - gt_var)

            times = [
                input_batch.metadata.time[i],
                input_batch.metadata.time[i],
                pred_batch.metadata.time[i],
                gt_batch.metadata.time[i],
            ]

            # Plot first four variables
            for j, (var, title) in enumerate(
                zip(
                    [input_var_1, input_var_2, pred_var, gt_var],
                    ["Input1", "Input2", "Prediction", "Ground Truth"],
                )
            ):
                im = axes[i, j].imshow(var, norm=norm_main, cmap="coolwarm")
                axes[i, j].set_title(f"{title}: {variable_key}")
                if j < 4:  # Add time labels for inputs, prediction, and ground truth
                    axes[i, j].set_ylabel(str(times[j]))
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

            im_diff = axes[i, 4].imshow(difference, norm=norm_diff, cmap="inferno")
            axes[i, 4].set_title(f"Difference: {variable_key}")
            axes[i, 4].set_xticks([])
            axes[i, 4].set_yticks([])

        # Dynamically calculate the position for the first color bar
        unit_label = UNITS_SURFACE.get(variable_key, "Unknown Unit")
        variable_name = VARIABLE_NAMES_SURFACE.get(variable_key, "Unknown Variable")

        bbox_main = Bbox.union([axes[-1, j].get_position() for j in range(4)])
        cax_main = fig.add_axes(
            [bbox_main.x0, bbox_main.y0 - 0.08, bbox_main.width, 0.02]
        )  # Below the first 4 subplots
        cbar_main = fig.colorbar(im, cax=cax_main, orientation="horizontal")
        cbar_main.set_label(f"{unit_label} ({variable_name})")

        # Dynamically calculate the position for the second color bar
        bbox_diff = Bbox.union([axes[-1, j].get_position() for j in range(4, 5)])
        cax_error = fig.add_axes(
            [bbox_diff.x0, bbox_diff.y0 - 0.08, bbox_diff.width, 0.02]
        )  # Below the last 2 subplots
        cbar_err = fig.colorbar(im_diff, cax=cax_error, orientation="horizontal")
        cbar_err.set_label(f"Difference {unit_label}")

        # Log the figure
        wandb.log(
            {f"{variable_key}_step{rollout_step}": wandb.Image(fig), "epoch": epoch},
            commit=not save_on_next_step,
        )

        plt.close(fig)
