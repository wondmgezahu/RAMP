import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from deps.torch_robotics.torch_robotics.torch_utils.torch_timer import TimerCUDA
from deps.torch_robotics.torch_robotics.torch_utils.torch_utils import dict_to_device, to_numpy
import numpy as np
from math import ceil
import copy

import os
import logging
import time
import torch
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast



def get_num_epochs(num_train_steps, batch_size, dataset_len):
    return ceil(num_train_steps * batch_size / dataset_len)


def save_models_to_disk(models_prefix_l, epoch, total_steps, checkpoints_dir=None):
    for model, prefix in models_prefix_l:
        if model is not None:
            save_model_to_disk(model, epoch, total_steps, checkpoints_dir, prefix=f'{prefix}_')
            for submodule_key, submodule_value in model.submodules.items():
                save_model_to_disk(submodule_value, epoch, total_steps, checkpoints_dir,
                                   prefix=f'{prefix}_{submodule_key}_')


def save_checkpoint(model, ema_model, optimizer, epoch, step, checkpoints_dir):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if ema_model is not None:
        checkpoint['ema_model_state_dict'] = ema_model.module.state_dict()
    
    torch.save(checkpoint, os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch}_step_{step}.pt'))
    print(f"Checkpoint saved at epoch {epoch}, step {step}")

def save_checkpoint_latest(model, ema_model, optimizer, epoch, step, checkpoints_dir):
    torch.save(model.module.state_dict(), os.path.join(checkpoints_dir, "model_current_state_dict.pth"))
    torch.save(model, os.path.join(checkpoints_dir, "model_current.pth"))
    if ema_model is not None:
        torch.save(ema_model.module.state_dict(), os.path.join(checkpoints_dir, "ema_model_current_state_dict.pth"))       
        # Save the latest full EMA model
        torch.save(ema_model, os.path.join(checkpoints_dir, "ema_model_current.pth"))  
    # print('Latest checkpoint saved')
def save_model_to_disk(model, epoch, total_steps, checkpoints_dir=None, prefix='model_'):
    # If the model is frozen we do not save it again, since the parameters did not change
    if hasattr(model, 'is_frozen') and model.is_frozen:
        return

    torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'{prefix}current_state_dict.pth'))
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'{prefix}epoch_{epoch:04d}_iter_{total_steps:06d}_state_dict.pth'))
    torch.save(model, os.path.join(checkpoints_dir, f'{prefix}current.pth'))
    torch.save(model, os.path.join(checkpoints_dir, f'{prefix}epoch_{epoch:04d}_iter_{total_steps:06d}.pth'))


def save_losses_to_disk(train_losses, val_losses, checkpoints_dir=None):
    np.save(os.path.join(checkpoints_dir, f'train_losses.npy'), train_losses)
    np.save(os.path.join(checkpoints_dir, f'val_losses.npy'), val_losses)


class EarlyStopper:
    # https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience  # use -1 to deactivate it
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf') #torch.inf

    def early_stop(self, validation_loss):
        if self.patience == -1:
            return
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def create_ema_model(model, rank):
    print(f"Rank {rank}: Entering create_ema_model")
    try:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            print(f"Rank {rank}: Model is DDP, accessing .module")
            orig_model = model.module
        else:
            print(f"Rank {rank}: Model is not DDP")
            orig_model = model
        
        print(f"Rank {rank}: Creating deep copy of model")
        ema_model = copy.deepcopy(orig_model)
        
        print(f"Rank {rank}: Getting device of original model")
        device = next(orig_model.parameters()).device
        print(f"Rank {rank}: Device is {device}")
        
        print(f"Rank {rank}: Moving EMA model to device")
        ema_model.to(device)
        
        print(f"Rank {rank}: Wrapping EMA model in DDP")
        ema_model = torch.nn.parallel.DistributedDataParallel(ema_model, device_ids=[rank])
        
        print(f"Rank {rank}: EMA model creation successful")
        return ema_model
    except Exception as e:
        print(f"Rank {rank}: Error in create_ema_model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

class EMA:
    def __init__(self, beta=0.995):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ema_model, current_model):
        for ema_params, current_params in zip(ema_model.parameters(), current_model.parameters()):
            old_weight, up_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def do_summary(
        summary_fn,
        train_steps_current,
        model,
        batch_dict,
        loss_info,
        datasubset,
        **kwargs
):
    if summary_fn is None:
        return

    with torch.no_grad():
        # set model to evaluation mode
        model.eval()

        summary_fn(train_steps_current,
                   model,
                   batch_dict=batch_dict,
                   loss_info=loss_info,
                   datasubset=datasubset,
                   **kwargs
                   )

    # set model to training mode
    model.train()



def setup_logging(model_dir, rank, seed):
    log_dir = os.path.join(model_dir, f'seed_{seed}', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(f'trainer_rank_{rank}')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(os.path.join(log_dir, f'train_rank_{rank}.log'))
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    if rank == 0:
        tensorboard_dir = os.path.join(log_dir, 'tensorboard')
        writer = SummaryWriter(tensorboard_dir)
    else:
        writer = None
    
    return logger, writer
import time 



def train(model, train_dataloader, epochs, lr, steps_til_summary, model_dir, loss_fn,
          train_subset, summary_fn, steps_til_checkpoint, val_dataloader, val_subset,
          clip_grad, clip_grad_max_norm=1.0, val_loss_fn=None, optimizers=None,
          use_ema=True, ema_decay=0.995, step_start_ema=1000, update_ema_every=10,
          use_amp=False, debug=False, tensor_args=None, rank=0, world_size=1, seed=None):

    logger, writer = setup_logging(model_dir, rank, seed)
    logger.info(f"Rank {rank}: Starting training")

    ema_model = None
    if use_ema:
        ema = EMA(beta=ema_decay)
        ema_model = create_ema_model(model, rank)

    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    scaler = GradScaler(enabled=use_amp)

    checkpoints_dir = os.path.join(model_dir, f'seed_{seed}', 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    train_steps_current = 0

    for epoch in range(epochs):
        # logger.info(f"Rank {rank}: Starting epoch {epoch}")
        
        train_dataloader.sampler.set_epoch(epoch)
        if val_dataloader:
            val_dataloader.sampler.set_epoch(epoch)
        
        model.train()
        for step, train_batch_dict in enumerate(train_dataloader):
            with autocast(enabled=use_amp):
                train_losses, train_losses_info = loss_fn(model, train_batch_dict, train_subset.dataset)
                train_loss_batch = sum(loss.mean() for loss in train_losses.values())

            # Synchronize loss across all GPUs
            dist.all_reduce(train_loss_batch, op=dist.ReduceOp.SUM)
            train_loss_batch /= world_size

            for optim in optimizers:
                optim.zero_grad()

            scaler.scale(train_loss_batch).backward()

            if clip_grad:
                scaler.unscale_(optimizers[0])
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_max_norm)

            for optim in optimizers:
                scaler.step(optim)

            scaler.update()

            if ema_model is not None and train_steps_current % update_ema_every == 0:
                if train_steps_current < step_start_ema:
                    ema_model.module.load_state_dict(model.module.state_dict())
                else:
                    ema.update_model_average(ema_model.module, model.module)

            if train_steps_current % steps_til_summary == 0 and rank==0: # same loss for all ranks 
                logger.info(f"Rank {rank}, Step: {train_steps_current}, Loss: {train_loss_batch.item():.4f}")
                if rank == 0 and writer:
                    writer.add_scalar('Loss/train', train_loss_batch.item(), train_steps_current)

            if rank == 0 and (steps_til_checkpoint is not None) and (train_steps_current % steps_til_checkpoint == 0):
                save_checkpoint(model, ema_model, optimizers[0], epoch, train_steps_current, checkpoints_dir)
                logger.info(f"Rank {rank}: Saved checkpoint at step {train_steps_current}")
            if rank==0 and (train_steps_current%1000==0):
                save_checkpoint_latest(model, ema_model, optimizers[0], epoch, train_steps_current, checkpoints_dir)
                logger.info("Saved current checkpoint")
            train_steps_current += 1
        # logger.info(f"Rank {rank}: Completed epoch {epoch}")

    logger.info(f'Rank {rank}: Training finished')
    if rank == 0 and writer:
        writer.close()

    dist.barrier()
    logger.info(f"Rank {rank}: Final barrier passed, exiting train function")
