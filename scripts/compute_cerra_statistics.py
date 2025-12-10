"""Compute CERRA mean/std statistics."""
import pickle

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from aurora.data.collate import collate_fn
from main import check_and_start_debugger


@hydra.main(config_name="cerra", config_path="../configs/dataset", version_base="1.3.2")
def main(cfg: DictConfig) -> None:
    """Compute mean, std, min, and max statistics for ERA5 data.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """

    if hasattr(cfg, "train"):
        train_cfg = OmegaConf.merge(
            OmegaConf.to_container(cfg.common, resolve=True),
            OmegaConf.to_container(cfg.train, resolve=True)
        )
        dataset = instantiate(train_cfg, rollout_steps=0, normalize=False, stats_path=None)
    else:
        dataset = instantiate(cfg.common, rollout_steps=0, normalize=False, stats_path=None)  # For DummyDataset

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        num_workers=8,
        collate_fn=collate_fn,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )

    num_samples = len(dataloader)

    # Compute per variable mean
    mean_per_var: dict[str, Tensor] = {}
    for batch in tqdm(dataloader):
        batch = batch["input"][0]  # Input returns two consecutive steps, and we don't want to count samples twice

        for atmos_var in batch.atmos_vars.keys():
            for i, atmos_level in enumerate(batch.metadata.atmos_levels):
                var_key = f"{atmos_var}_{atmos_level}"
                var_value = batch.atmos_vars[atmos_var][0, 0, i]

                if var_key not in mean_per_var:
                    mean_per_var[var_key] = torch.tensor(0, dtype=torch.float32)

                mean_per_var[var_key] += var_value.mean() / num_samples

        for surf_var in batch.surf_vars.keys():
            var_value = batch.surf_vars[surf_var][0, 0]

            if surf_var not in mean_per_var:
                mean_per_var[surf_var] = torch.tensor(0, dtype=torch.float32)

            mean_per_var[surf_var] += var_value.mean() / num_samples

        for static_var in batch.static_vars.keys():
            var_value = batch.static_vars[static_var]

            if static_var not in mean_per_var:
                mean_per_var[static_var] = torch.tensor(0, dtype=torch.float32)

            mean_per_var[static_var] += var_value.mean() / num_samples

    # Compute per variable variance
    var_per_var: dict[str, Tensor] = {}

    for batch in tqdm(dataloader):
        batch = batch["input"][0]

        for atmos_var in batch.atmos_vars.keys():
            for i, atmos_level in enumerate(batch.metadata.atmos_levels):
                var_key = f"{atmos_var}_{atmos_level}"
                var_value = batch.atmos_vars[atmos_var][0, 0, i]

                if var_key not in var_per_var:
                    var_per_var[var_key] = torch.tensor(0, dtype=torch.float32)

                var_per_var[var_key] += ((var_value - mean_per_var[var_key]) ** 2).mean() / num_samples

        for surf_var in batch.surf_vars.keys():
            var_value = batch.surf_vars[surf_var][0, 0]

            if surf_var not in var_per_var:
                var_per_var[surf_var] = torch.tensor(0, dtype=torch.float32)

            var_per_var[surf_var] += ((var_value - mean_per_var[surf_var]) ** 2).mean() / num_samples

        for static_var in batch.static_vars.keys():
            var_value = batch.static_vars[static_var]

            if static_var not in var_per_var:
                var_per_var[static_var] = torch.tensor(0, dtype=torch.float32)

            var_per_var[static_var] += ((var_value - mean_per_var[static_var]) ** 2).mean() / num_samples

    std_per_var = {k: torch.sqrt(v) for k, v in var_per_var.items()}

    stat_dict: dict[str, tuple[float, float]] = {}
    for k, v in mean_per_var.items():
        stat_dict[k] = (v.item(), std_per_var[k].item())

    # Save to disk
    with open("cerra_stats.pkl", "wb") as f:
        pickle.dump(stat_dict, f)


if __name__ == '__main__':
    check_and_start_debugger()
    main()
