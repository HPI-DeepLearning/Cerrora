import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, ckpt_path, global_step):
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    
    # Create success tensor on the same device as model
    success = torch.zeros(1, device=model.device)
    
    if rank == 0:
        try:
            checkpoint = {
                "model": model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch,
                "global_step": global_step,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all()
            }

            # Save to a temporary file first
            tmp_path = f"{ckpt_path}.tmp"
            torch.save(checkpoint, tmp_path)
            # Atomic rename
            os.replace(tmp_path, ckpt_path)
            
            print(f"Rank 0: Saved checkpoint to {ckpt_path}")
            success.fill_(1)
        except Exception as e:
            print(f"Rank 0: Failed to save checkpoint: {str(e)}")
            success.fill_(0)

    if not dist.is_initialized():
        return
    
    # First broadcast success status
    dist.broadcast(success, src=0)
    
    # Then barrier to ensure all processes are synced
    dist.barrier()
    
    if success.item() == 0:
        raise RuntimeError("Checkpoint saving failed on rank 0")
    

def save_final_checkpoint(model, ckpt_path):
    """Save the final model state dict to the specified path.
    
    This saves the model weights in a format compatible with Aurora.load_checkpoint_local().
    Only saves the model parameters without optimizer state or training metadata.
    
    Args:
        model (nn.Module): The Aurora model to save
        ckpt_path (str): Path where to save the checkpoint
    """
    # Only save on rank 0 if using distributed training
    if dist.is_initialized() and dist.get_rank() != 0:
        return
        
    try:
        # Extract the state dict, handling DistributedDataParallel case
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()

        # Wrap in a checkpoint dict for compatibility
        checkpoint = {
            "model": state_dict
        }

        # Save to temporary file first for atomic save
        tmp_path = f"{ckpt_path}.tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, ckpt_path)
        
        print(f"Saved final model checkpoint to {ckpt_path}")
        
    except Exception as e:
        print(f"Failed to save final checkpoint: {str(e)}")
        raise
    

def load_checkpoint(
        rank,
        model,
        optimizer,
        scheduler,
        scaler,
        ckpt_path,
        strict=True,
        load_optimizer_state=True,
        load_model_only=False,
):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Create a list to store the checkpoint
    object_list = [None]
    
    # Only rank 0 loads from file
    if rank == 0:
        object_list[0] = torch.load(ckpt_path, map_location="cpu")

    if dist.is_initialized():
        # Synchronize before broadcasting
        dist.barrier()

        # Broadcast the checkpoint to all ranks
        dist.broadcast_object_list(object_list, src=0)

    checkpoint = object_list[0]

    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(checkpoint["model"], strict=strict)
    else:
        model.load_state_dict(checkpoint["model"], strict=strict)

    if load_model_only:
        return

    if optimizer is not None and load_optimizer_state:
        # Load optimizer state only if requested
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state if available and requested
    if scheduler is not None and checkpoint["scheduler"] is not None and load_optimizer_state:
        scheduler.load_state_dict(checkpoint["scheduler"])

    # Load scaler state if using AMP and requested
    if scaler is not None and checkpoint["scaler"] is not None and load_optimizer_state:
        scaler.load_state_dict(checkpoint["scaler"])

    # Load random states
    torch.set_rng_state(checkpoint["rng_state"])
    if torch.cuda.device_count() == len(checkpoint["cuda_rng_state"]):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    # Return training state info
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    print(f"Rank {rank}: Loaded checkpoint from {ckpt_path} (epoch {epoch + 1})")

    return epoch, global_step
