import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from mpd import models, losses, datasets, summaries
from mpd.utils import model_loader, pretrain_helper
from deps.torch_robotics.torch_robotics.torch_utils.torch_utils import freeze_torch_model_params

@model_loader
def get_model(model_class=None, checkpoint_path=None,
              freeze_loaded_model=False,
              tensor_args=None,
              **kwargs):
    if checkpoint_path is not None:
        model = torch.load(checkpoint_path, map_location=tensor_args['device'])
        if freeze_loaded_model:
            freeze_torch_model_params(model)
    else:
        ModelClass = getattr(models, model_class)
        model = ModelClass(**kwargs).to(tensor_args['device'])
    return model

@pretrain_helper
def get_pretrain_model(model_class=None, device=None, checkpoint_path=None, **kwargs):
    Model = getattr(models, model_class)
    model = Model(**kwargs).to(device)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model

def build_module(model_class=None, submodules=None, **kwargs):
    if submodules is not None:
        for key, value in submodules.items():
            kwargs[key] = build_module(**value)
    Model = getattr(models, model_class)
    model = Model(**kwargs)
    return model

def get_loss(loss_class=None, **kwargs):
    LossClass = getattr(losses, loss_class)
    loss = LossClass(**kwargs)
    loss_fn = loss.loss_fn
    return loss_fn

def get_dataset(dataset_class=None,
                dataset_subdir=None,
                batch_size=2,
                val_set_size=0.05,
                results_dir=None,
                save_indices=False,
                rank=0,
                num_workers=8,
                world_size=1,
                # training=False,
                **kwargs):
    DatasetClass = getattr(datasets, dataset_class)
    print('\n---------------Loading data')
    full_dataset = DatasetClass(dataset_subdir=dataset_subdir, **kwargs)
    print(full_dataset)
    
    dataset_size = len(full_dataset)
    val_size = int(val_set_size * dataset_size)
    train_size = dataset_size - val_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank,shuffle=True)
    val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank,shuffle=False)
    
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, sampler=val_sampler)
    
    if save_indices and rank == 0:
        torch.save(train_subset.indices, os.path.join(results_dir, f'train_subset_indices.pt'))
        torch.save(val_subset.indices, os.path.join(results_dir, f'val_subset_indices.pt'))

    return train_subset, train_dataloader, val_subset, val_dataloader

def get_summary(summary_class=None, **kwargs):
    if summary_class is None:
        return None
    SummaryClass = getattr(summaries, summary_class)
    summary_fn = SummaryClass(**kwargs).summary_fn
    return summary_fn