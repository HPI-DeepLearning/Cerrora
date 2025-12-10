import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler

from aurora.data.collate import collate_fn
from aurora.data.replay_buffer import ReplayBuffer


def get_train_dataloader(cfg: DictConfig, max_rollout_steps: int) -> DataLoader:
    cfg_rollout_steps = cfg.dataset.common.rollout_steps
    cfg.dataset.common.rollout_steps = max_rollout_steps

    if hasattr(cfg.dataset, "train"):
        train_cfg = OmegaConf.merge(
            OmegaConf.to_container(cfg.dataset.common, resolve=True),
            OmegaConf.to_container(cfg.dataset.train, resolve=True)
        )
        train_dataset = instantiate(train_cfg)
    else:
        train_dataset = instantiate(cfg.dataset.common)  # For DummyDataset

    train_sampler = DistributedSampler(train_dataset, drop_last=True) if cfg.task.distributed else None

    if cfg.task.task == "train" and cfg.task.phase == "rollout_long_buffer":
        # If using a replay buffer, we need to set the batch size to 1 for the dataloader
        # The batching is handled by the replay buffer
        dl_batch_size = 1
    else:
        dl_batch_size = cfg.dataloader.batch_size

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=dl_batch_size,
        sampler=train_sampler,
        shuffle=not cfg.task.distributed,
        drop_last=not cfg.task.distributed,
        pin_memory=False,
        collate_fn=collate_fn,
        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
        multiprocessing_context="forkserver" if cfg.dataloader.num_workers > 0 else None,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )

    if cfg.task.task == "train" and cfg.task.phase == "rollout_long_buffer":
        train_dataloader = ReplayBuffer(
            train_dataloader,
            cfg.dataloader.batch_size,
            buffer_size=cfg.task.buffer_size,
            refresh_freq=cfg.task.refresh_freq,
            max_rollout_steps=cfg.task.max_rollout_steps
        )

    cfg.dataset.common.rollout_steps = cfg_rollout_steps  # Restore original rollout steps

    return train_dataloader

def get_val_dataloader(cfg: DictConfig) -> DataLoader:
    if hasattr(cfg.dataset, "val"):
        val_cfg = OmegaConf.merge(
            OmegaConf.to_container(cfg.dataset.common, resolve=True),
            OmegaConf.to_container(cfg.dataset.val, resolve=True)
        )

        if cfg.task.task == "forecast":
            # When forecasting, we want the start time of the first forecast to be the same as val_cfg.start_time
            # Since the model needs two timesteps to make a forecast, we need to offset the start time by the lead time
            start_datetime = np.datetime64(val_cfg.start_time)
            lead_time_timedelta = np.timedelta64(val_cfg.lead_time_hours, "h")
            val_cfg.start_time = str(start_datetime - lead_time_timedelta)

        val_dataset = instantiate(val_cfg)
    else:
        val_dataset = instantiate(cfg.dataset.common)  # For DummyDataset

    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if cfg.task.distributed else None

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        collate_fn=collate_fn,
        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
        multiprocessing_context="forkserver" if cfg.dataloader.num_workers > 0 else None,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )

    return val_dataloader
