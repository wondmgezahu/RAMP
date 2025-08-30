import os
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from mpd import trainerddp
from mpd.models import UNET_DIM_MULTS, TemporalUnetTrain
from mpd.trainerddp import get_dataset, get_model, get_loss, get_summary
from mpd.trainerddp.trainer import get_num_epochs
from deps.torch_robotics.torch_robotics.torch_utils.seed import fix_random_seed
import torch.distributed as dist
import time
import random
import logging
# torch.cuda.empty_cache()
def setup_logging(rank):
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
def get_device_id(rank, world_size):
    return rank % torch.cuda.device_count()
def setup(rank, world_size, logger):
    os.environ['MASTER_ADDR'] = 'localhost'
    base_port = 12355 #12355
    max_retries = 20
    port_range=100
    
    for attempt in range(max_retries):
        port = base_port + attempt
        # port = base_port + random.randint(0,port_range)
        os.environ['MASTER_PORT'] = str(port)
        logger.info(f"Rank {rank}: Attempting to initialize process group with port {port}")
        try:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            # dist.init_process_group("gloo", rank=rank, world_size=world_size)
            logger.info(f"Rank {rank}: Process group initialized successfully")
            
            # Set cuda device explicitly
            if torch.cuda.is_available():
                # torch.cuda.set_device(rank)
                device=torch.device(f"cuda:{rank}")
                torch.cuda.set_device(device)
                logger.info(f"Rank {rank}: CUDA device set to cuda:{device}")
            
            # Barrier to ensure all processes have initialized
            # dist.barrier(device_ids=[rank]) # cust
            dist.barrier() # cust
            logger.info(f"Rank {rank}: Passed initialization barrier")
            return
        except RuntimeError as e:
            logger.warning(f"Rank {rank}: Failed to initialize process group: {str(e)}")
            if "Address already in use" in str(e):
                if attempt < max_retries - 1:
                    wait_time = random.uniform(1, 5)
                    logger.info(f"Rank {rank}: Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rank {rank}: Max retries reached. Unable to initialize process group.")
                    raise
            else:
                logger.error(f"Rank {rank}: Unexpected error during initialization.")
                raise

def cleanup():
    dist.destroy_process_group()

def load_model_weights(self, ckpt_path):
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt 
        
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Load the state dict
        incompatible = self.load_state_dict(state_dict, strict=False)
        
        if incompatible.missing_keys:
            print('Warning: Missing keys:', incompatible.missing_keys)
        if incompatible.unexpected_keys:
            print('Warning: Unexpected keys:', incompatible.unexpected_keys)
        
        print(f'Successfully loaded the model weights from {ckpt_path}')
    else:
        print('No checkpoint provided. Initializing model from scratch!')    

def run_training(rank, world_size, config):
    # setup(rank, world_size)
    logger = setup_logging(rank)
    logger.info(f"Rank {rank}: Starting run_training")
    try:
        setup(rank, world_size, logger)
        
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        logger.info(f"Rank {rank}: Using device {device}")
        dist.barrier()
        # fix_random_seed(config['seed'] + rank)
        logger.info(f"Rank {rank}: Random seed fixed")
        
        tensor_args = {'device': device, 'dtype': torch.float32}

        dataset_class = 'TrajectoryDataset3d' if config['3d'] else 'TrajectoryDataset'
        diffusion_model_class = 'GaussianDiffusionModel3d' if config['3d'] else config['diffusion_model_class']
        obstacle_3d = config['3d']

        train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
            dataset_class=dataset_class,
            include_velocity=config['include_velocity'],
            dataset_subdir=config['dataset_subdir'],
            batch_size=config['batch_size'],
            results_dir=config['results_dir'],
            save_indices=False,
            tensor_args=tensor_args,
            rank=rank,
            world_size=world_size,
            training=config['train'],
            static=True,
            # dataset_base_dir=config['dataset_path']
        )

        dataset = train_subset.dataset

        diffusion_configs = dict(
            variance_schedule=config['variance_schedule'],
            n_diffusion_steps=config['n_diffusion_steps'],
            predict_epsilon=config['predict_epsilon'],
            training=config['train'],
        )

        unet_configs = dict(
            state_dim=dataset.state_dim,
            n_support_points=dataset.n_support_points,
            unet_input_dim=config['unet_input_dim'],
            dim_mults=UNET_DIM_MULTS[config['unet_dim_mults_option']],
            obstacle_3d=obstacle_3d
        )
        # print(f'state dimension is {dataset.state_dim}')
        model = get_model(
            model_class=diffusion_model_class,
            model=TemporalUnetTrain(**unet_configs),
            tensor_args=tensor_args,
            **diffusion_configs,
            **unet_configs
        )
        
        
        model = model.to(device)
        logger.info(f"Rank {rank}: Model moved to device, creating DDP wrapper")
            # dist.barrier()  # Ensure all processes reach this point
        try:
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
            logger.info(f"Rank {rank}: DDP wrapper created successfully")
        except Exception as e:
            logger.error(f"Rank {rank}: Failed to create DDP wrapper: {str(e)}")
            raise

        # model = model.to(device)
        # model = DDP(model, device_ids=[rank]) 
        # dist.barrier()
        # logger.info(f"Rank {rank}: Model initialized and wrapped with DDP")
        
        loss_fn = val_loss_fn = get_loss(loss_class=config['loss_class'])
        summary_fn = None

        dist.barrier()
        logger.info(f"Rank {rank}: Starting training")

        trainerddp.train(
            model=model,
            train_dataloader=train_dataloader,
            train_subset=train_subset,
            val_dataloader=val_dataloader,
            val_subset=train_subset,
            epochs=get_num_epochs(config['num_train_steps'], config['batch_size'], len(dataset)),
            model_dir=config['results_dir'],
            summary_fn=summary_fn,
            lr=config['lr'],
            loss_fn=loss_fn,
            val_loss_fn=val_loss_fn,
            steps_til_summary=config['steps_til_summary'],
            steps_til_checkpoint=config['steps_til_ckpt'],
            clip_grad=True,
            use_ema=config['use_ema'],
            use_amp=config['use_amp'],
            debug=config['debug'],
            tensor_args=tensor_args,
            rank=rank,
            world_size=world_size,
            seed=config['seed']
        )
        logger.info(f"Rank {rank}: Training completed")
    
    except Exception as e:
        logger.error(f"Rank {rank}: An error occurred in run_training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        cleanup()
        logger.info(f"Rank {rank}: Cleanup completed")

if __name__ == '__main__':
    config = {
        'dataset_subdir': 'EnvSmall3D', #EnvSmall3D', #'EnvParameterized500small'  
        'include_velocity': True,
        'train': True,
        '3d': True, # True for maze3d obstacle set
        'diffusion_model_class': 'StaticGaussianDiffusionModel',
        'variance_schedule': 'exponential',
        'n_diffusion_steps': 25, 
        'predict_epsilon': True, 
        'unet_input_dim': 32,
        'unet_dim_mults_option': 1,
        'loss_class': 'GaussianDiffusionLossDDP',
        'batch_size': 16, 
        'lr': 1e-4,
        'num_train_steps': 2800000, 
        'use_ema': True,
        'use_amp': True,
        'steps_til_summary': 100,
        'steps_til_ckpt': 100000,
        'debug': True,
        'seed': 101, 
        # 'dataset_path':'../../dataset/',
        'results_dir': '../../checkpoints/',
    }

    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")
    
    try:
        print("processes")
        mp.spawn(run_training, args=(world_size, config), nprocs=world_size, join=True)
        print("All processes completed successfully")
    except Exception as e:
        print(f"An error occurred during process spawning: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("Main process exiting")