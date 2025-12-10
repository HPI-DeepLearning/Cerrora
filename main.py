import hydra
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from aurora import Aurora, AuroraSmall
from aurora.data.collate import collate_fn
from aurora.data.replay_buffer import ReplayBuffer
from aurora.evaluation.forecast import forecast
from aurora.training.train import train
from aurora.training.logging import initialize_wandb

# Disable DDP optimization for now
# See https://github.com/pytorch/pytorch/issues/134182
torch._dynamo.config.optimize_ddp = False

MODEL_REGISTRY = {"Aurora": Aurora, "AuroraSmall": AuroraSmall}
MODEL_CHECKPOINT_REGISTRY = {"Aurora": "aurora-0.25-pretrained.ckpt", "AuroraSmall": "aurora-0.25-small-pretrained.ckpt"}


def check_and_start_debugger():
    """Check if PyCharm remote debugger should be started."""
    import os
    debug_port = int(os.environ.get("REMOTE_PYCHARM_DEBUG_PORT", 12034))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if os.environ.get("REMOTE_PYCHARM_DEBUG_SESSION", False) and local_rank == 0:
        import pydevd_pycharm
        pydevd_pycharm.settrace(
            "localhost",
            port=debug_port,
            stdout_to_server=True,
            stderr_to_server=True,
            suspend=False,
        )

def cleanup():
    dist.destroy_process_group()


def setup_distributed(local_rank):
    """Initializes the distributed environment."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    dist.barrier(device_ids=[local_rank])


@hydra.main(config_name="train", config_path="configs", version_base="1.3.2")
def main(cfg):
    check_and_start_debugger()
    if cfg.task.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        setup_distributed(local_rank)
    else:
        local_rank = 0

    model_class = MODEL_REGISTRY.get(cfg.task.model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {cfg.task.model_name}")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = model_class(**cfg.model)

    if cfg.task.task == "train" and cfg.task.phase == "finetuning":
        if not cfg.checkpoint.continue_training:
            model_checkpoint = MODEL_CHECKPOINT_REGISTRY[cfg.task.model_name]
            model.load_checkpoint("microsoft/aurora", model_checkpoint, strict=False)

    model.to(device)

    if cfg.task.use_activation_checkpointing:
        model.configure_activation_checkpointing(cfg.task.checkpoint_encoder_decoder)

    if cfg.task.distributed:
        dist.barrier()

        model = model.to(local_rank)
        # Add find_unused_parameters=True to handle LoRA parameter freezing
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True)

    model = torch.compile(model, dynamic=False) if cfg.task.use_torch_compile else model

    if cfg.task.task == "train":
        initialize_wandb(cfg)
        train(model, cfg, device)
    elif cfg.task.task == "forecast":
        forecast(model, cfg, device)

    if cfg.task.distributed:
        cleanup()


if __name__ == "__main__":
    main()
