import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from aurora.data.collate import collate_fn
from aurora.evaluation.metrics import mse
from aurora.training.loss import compute_latitude_weights
from aurora.training.train import RMSE_VARIABLES
from main import check_and_start_debugger


@hydra.main(config_name="evaluate", config_path="../configs", version_base="1.3.2")
def compute_forecast_rsme(cfg: DictConfig) -> None:
    """
    Compute the RMSE between the forecast and ground truth data.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    assert not (cfg.task.save_csv and cfg.task.csv_save_path is None), "task.csv_save_path needs to be set to save CSV"

    dataset_cfg = OmegaConf.merge(
        OmegaConf.to_container(cfg.dataset.common, resolve=True),
        OmegaConf.to_container(cfg.dataset.val, resolve=True)
    )

    forecast_dataset = instantiate(
        dataset_cfg,
        data_path=cfg.task.forecast_data_path,
        is_forecast_dataset=True
    )

    # The gt dataset needs to be offset by the lead time, so that we can compare
    # the forecast with the ground truth at the same time
    start_datetime = np.datetime64(dataset_cfg.start_time)
    end_datetime = np.datetime64(dataset_cfg.end_time)
    lead_time_timedelta = np.timedelta64(dataset_cfg.lead_time_hours, "h")
    dataset_cfg.start_time = str(start_datetime + lead_time_timedelta)
    dataset_cfg.end_time = str(end_datetime + lead_time_timedelta)
    gt_dataset = instantiate(
        dataset_cfg,
        is_forecast_dataset=False
    )

    gt_dataloader = DataLoader(
        gt_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        multiprocessing_context="forkserver",
        num_workers=4,
    )

    forecast_dataloader = DataLoader(
        forecast_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        multiprocessing_context="forkserver",
        num_workers=4,
    )

    filtered_rmse_variables = []

    example_batch = forecast_dataset[0]["input"]
    for var in RMSE_VARIABLES:
        if var["type"] == "surf" and var["name"] in example_batch.surf_vars:
            filtered_rmse_variables.append(var)
        elif var["type"] == "atmos" and var["name"] in example_batch.atmos_vars:
            if "level" in var and var["level"] in example_batch.metadata.atmos_levels:
                filtered_rmse_variables.append(var)
            elif "level" not in var:
                filtered_rmse_variables.append(var)

    latitude_weights = None

    total_mse_results = {f"{var['name']}{'_' + str(var['level']) if 'level' in var else ''}": 0.0
                         for var in filtered_rmse_variables}

    for gt_batch, forecast_batch in tqdm(zip(gt_dataloader, forecast_dataloader)):
        gt_batch = gt_batch["input"]
        forecast_batch = forecast_batch["input"]

        if cfg.task.use_latitude_weighting and latitude_weights is None:
            latitude_weights = compute_latitude_weights(gt_batch.metadata.lat)

        for var in filtered_rmse_variables:
            if var['type'] == 'surf':
                current_mse = mse(forecast_batch, gt_batch,
                                  variable=var['name'], latitude_weights=latitude_weights)
            else:
                current_mse = mse(forecast_batch, gt_batch,
                                  variable=var['name'], level=var['level'], latitude_weights=latitude_weights)


            result_key = f"{var['name']}{'_' + str(var['level']) if 'level' in var else ''}"
            total_mse_results[result_key] += current_mse.item()

    num_samples = len(gt_dataloader)
    total_rmse_results = {key: 0.0 for key in total_mse_results}
    for key in total_mse_results:
        total_mse_results[key] = total_mse_results[key] / num_samples
        total_rmse_results[key] = total_mse_results[key] ** 0.5

    print("Results:")
    for key in total_rmse_results.keys():
        print(f"{key}: MSE: {total_mse_results[key]} RMSE: {total_rmse_results[key]}")

    # Format the lead time for the CSV file
    lead_time_days = dataset_cfg.lead_time_hours // 24
    lead_time_hours = dataset_cfg.lead_time_hours % 24
    lead_time_str = f"{lead_time_days} days {lead_time_hours:2}h"

    if cfg.task.save_csv:
        # The file might already exist, so we append to it
        with open(cfg.task.csv_save_path, "a") as f:
            for key in total_rmse_results.keys():
                var = key.split('_')[0]
                level = key.split('_')[1] if '_' in key else None

                f.write(f"{var},{lead_time_str},")

                if level is not None:
                    f.write(f"{level},")
                else:
                    f.write(",")

                f.write(f"{total_mse_results[key]},")
                f.write(f"{total_rmse_results[key]},")
                f.write("\n")


if __name__ == "__main__":
    check_and_start_debugger()
    compute_forecast_rsme()
