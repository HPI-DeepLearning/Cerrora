import os
import logging
from datetime import datetime
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from aurora.model.aurora import Aurora 
from aurora.training.loss import AuroraMeanAbsoluteError
from aurora.batch import Batch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AuroraTrainer:
    def __init__(self, config):
        self.config = config
        self.setup_distributed()
        self.setup_model()
        self.setup_training()
        self.setup_logging()

    def setup_distributed(self):
        """Setup distributed training"""
        dist.init_process_group(backend='nccl')
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.device = torch.device(f'cuda:{self.rank}')
        torch.cuda.set_device(self.device)

    def setup_model(self):
        """Initialize model"""
        self.model = Aurora(
            surf_vars=("2t", "10u", "10v", "msl", "tp"),  # Added tp
            static_vars=("lsm", "z", "slt"),
            atmos_vars=("z", "u", "v", "t", "q")
        ).to(self.device)
        
        self.model = DistributedDataParallel(
            self.model, 
            device_ids=[self.rank],
            find_unused_parameters=True
        )

    def setup_training(self):
        """Setup training components"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=5e-6
        )
        
        self.criterion = AuroraMeanAbsoluteError(phase="pretraining")
        
        warmup_steps = 1000
        total_steps = 150000
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config['learning_rate'] / 10
        )
        
        self.scaler = GradScaler()

    def setup_logging(self):
        """Setup wandb logging"""
        if self.rank == 0:
            wandb.init(
                project="aurora-training",
                config=self.config,
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for i, batch in enumerate(dataloader):
            batch = batch.to(self.device)
            
            with autocast(dtype=torch.bfloat16):
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            
            if i % self.config['log_interval'] == 0 and self.rank == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"Step: {i}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
                wandb.log({
                    'train_loss_step': loss.item(),
                    'learning_rate': current_lr,
                    'step': i
                })
        
        return total_loss / len(dataloader)

    def validate(self, dataloader):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                with autocast(dtype=torch.bfloat16):
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)

    def train(self, train_loader, val_loader):
        """Complete training loop"""
        logger.info("Starting training...")
        logger.info(f"Config: {self.config}")
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            if self.rank == 0:
                logger.info(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                wandb.log({
                    'epoch': epoch,
                    'train_loss_epoch': train_loss,
                    'val_loss': val_loss
                })
                
                if epoch % self.config['save_interval'] == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.module.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    }
                    torch.save(
                        checkpoint,
                        os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pt')
                    )

if __name__ == "__main__":
    config = {
        'learning_rate': 5e-4,
        'batch_size': 1,  # per GPU
        'epochs': 1000,  # Adjusted to equivalent number of epochs
        'log_interval': 100,
        'save_interval': 10,
        'checkpoint_dir': './checkpoints',
    }
    
    trainer = AuroraTrainer(config)
    
    # Load datasets
    train_dataset = YourDataset(...)  # Need to implement
    val_dataset = YourDataset(...)    # Need to implement
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    trainer.train(train_loader, val_loader)