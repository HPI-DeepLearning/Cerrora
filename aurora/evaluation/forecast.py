import os
import torch
import torch.distributed as dist
import xarray as xr
import numpy as np
from numcodecs import Blosc
from tqdm import tqdm

from aurora import rollout
from aurora.data.cerra import ATMOS_VAR_MAPPING, SURFACE_VAR_MAPPING
from aurora.data.utils import get_val_dataloader
from aurora.training.checkpoint import load_checkpoint
from aurora.training.train import get_initial_input_batch

def save_forecast(inputs, preds, cfg):
    """
    Saves the forecast batch as zarr file in the output directory.
    """

    # Remove timezone from the string because numpy doesn't support it
    time = [np.datetime64(str(t)[:-6]) for t in inputs.metadata.time]
    time = np.array(time).astype("datetime64[ns]")
    timedeltas = [np.timedelta64(lt, 'h').astype("timedelta64[ns]") for lt in cfg.task.lead_times]
    timedeltas = np.array(timedeltas)

    lat_name = "latitude" if cfg.task.use_wb2_format else "y"
    lon_name = "longitude" if cfg.task.use_wb2_format else "x"
    level_name = "level" if cfg.task.use_wb2_format else "pressure_level"

    all_levels = inputs.metadata.atmos_levels
    kept_levels = [lvl for lvl in all_levels if lvl in cfg.task.save_levels]
    kept_levels_idx = [all_levels.index(lvl) for lvl in kept_levels]

    data_vars = {}

    for var in preds[0].atmos_vars:
        if var not in cfg.task.save_variables:
            continue
        var_mapped = ATMOS_VAR_MAPPING[var]
        var_values = np.concatenate([pred.atmos_vars[var] for pred in preds], axis=1)
        var_values = var_values[:, :, kept_levels_idx, :, :]
        data_vars[var_mapped] = (["time", "prediction_timedelta", level_name, lat_name, lon_name], var_values)

    for var in preds[0].surf_vars:
        if var not in cfg.task.save_variables:
            continue
        var_mapped = SURFACE_VAR_MAPPING[var]
        var_values = np.concatenate([pred.surf_vars[var] for pred in preds], axis=1)
        data_vars[var_mapped] = (["time", "prediction_timedelta", lat_name, lon_name], var_values)

    if cfg.task.use_wb2_format:
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                latitude=inputs.metadata.lat.cpu().numpy()[:, 0],
                longitude=inputs.metadata.lon.cpu().numpy()[0, :],
                cerra_latitude=([lat_name, lon_name], inputs.metadata.lat.cpu().numpy()),
                cerra_longitude=([lat_name, lon_name], inputs.metadata.lon.cpu().numpy()),
                level=np.array(kept_levels),
                time=time,
                prediction_timedelta=timedeltas,
            ),
            attrs=dict(description="Weather forecasts for Europe."),
        )

    else:
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                latitude=(["y", "x"], inputs.metadata.lat.cpu().numpy()),
                longitude=(["y", "x"], inputs.metadata.lon.cpu().numpy()),
                pressure_level=np.array(kept_levels),
                time=time,
                prediction_timedelta=timedeltas,
            ),
            attrs=dict(description="Weather forecasts for Europe."),
        )

    num_timedeltas = len(timedeltas)
    num_pressure_levels = len(kept_levels)
    num_y = ds[lat_name].values.shape[0]
    num_x = ds[lon_name].values.shape[0]
    ds = ds.chunk({
        "time": 1,
        "prediction_timedelta": num_timedeltas,
        level_name: num_pressure_levels,
        lat_name: num_y,
        lon_name: num_x
    })

    outdir = cfg.task.output_dir

    if os.path.exists(outdir):
        ds.to_zarr(outdir, append_dim="time", mode="a")
    else:
        # We use lz4hc because it combines good compression with fast decompression
        # It's relatively slow to compress, but we only need to do that once
        compressor = Blosc(cname='lz4hc', clevel=9, shuffle=Blosc.SHUFFLE)
        encoding = {var: {"compressor": compressor} for var in ds.data_vars}

        # It is important to set the encoding for the time variable explicitly, otherwise xarray might select
        # an encoding that is not compatible with the later values, resulting in wrong time values
        # See https://github.com/pydata/xarray/issues/5969 and https://github.com/pydata/xarray/issues/3942
        encoding.update({"time": {"units": "nanoseconds since 1970-01-01"}})

        ds.to_zarr(outdir, mode="w-", encoding=encoding)


def forecast(model, cfg, device):
    """
    Forecasts values using the trained model.

    Args:
        model (torch.nn.Module): The model to be trained.
        cfg (DictConfig): Hydra configuration object containing hyperparameters.
        device (torch.device): The device to which the model and data is moved.
    Returns:
        None: The forecasted values are saved to the output directory.

    Example:
        >>> forecast(model, cfg, torch.device("cuda:0"))
    """

    if cfg.task.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank != 0:
            raise RuntimeError("Cannot use multiple GPUs for forecasting.")
    else:
        local_rank = 0

    if cfg.task.use_quantized_model:
        print("\nPreparing quantized model...")
        from aurora.quantization.packing import replace_linear_layers
        from aurora.quantization.packing import CalibrationArgs
        args = CalibrationArgs()

        # Quantization configuration
        args.weight_bits = cfg.task.quantization_config.weight_bits
        args.group_size = cfg.task.quantization_config.group_size
        args.scale_groups = cfg.task.quantization_config.group_size
        args.output_bits = cfg.task.quantization_config.output_bits
        args.symmetric = cfg.task.quantization_config.symmetric
        args.use_shift = cfg.task.quantization_config.use_shift
        model.backbone = replace_linear_layers(model.backbone, args)
        model.backbone = model.backbone.to(device)

    # load checkpoint either from HF or local path
    if cfg.task.load_from_hf:
        model.load_checkpoint(cfg.task.hf_repo, cfg.task.hf_checkpoint, strict=True)
        print(f"Loaded model checkpoint from HF: {cfg.task.hf_repo} - {cfg.task.hf_checkpoint}")
    else:
        ckpt_path = cfg.task.checkpoint_path
        model.load_checkpoint_local(ckpt_path, strict=True)
        print(f"Loaded model checkpoint from local path: {ckpt_path}")

    lead_times = cfg.task.lead_times
    max_lead_time = max(lead_times)
    max_lead_time_steps = max_lead_time // cfg.model.lead_time_hours
    assert all([lead_time % cfg.model.lead_time_hours == 0 for lead_time in lead_times]), \
        "Lead times must be multiples of model lead time."

    # Disable gradients and set model to eval mode
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    dataloader = get_val_dataloader(cfg)

    stats_dict = dataloader.dataset.stats

    with torch.no_grad(): 
        for sample in tqdm(dataloader):
            batch = get_initial_input_batch(sample, cfg)
            batch = batch.to(device, non_blocking=True).type(torch.float32)

            rollout_batches = None
            boundary_size = cfg.dataset.common.boundary_size
            if "input_boundary_0" in sample:
                rollout_batches = []
                # Collect the input boundary batches
                for rollout_step in range(max_lead_time_steps - 1):
                    step_batch = sample[f"input_boundary_{rollout_step}"]
                    step_batch = step_batch.to(device, non_blocking=True).type(torch.float32)
                    rollout_batches.append(step_batch)

            preds = [pred.to("cpu") for pred, _ in rollout(
                model,
                batch,
                max_lead_time_steps,
                boundary_size,
                rollout_batches,
                return_last_input=False,
            )]

            if cfg.task.save_fp64:
                # Convert predictions to float64 if required
                preds = [pred.to(torch.float64) for pred in preds]

            preds = [pred.unnormalise(stats=stats_dict) for pred in preds]

            # Save forecasted values
            save_forecast(sample["input"], preds, cfg)
