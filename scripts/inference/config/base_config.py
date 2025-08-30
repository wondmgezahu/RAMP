from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
import torch
import os
from typing import List

@dataclass
class BaseInferenceConfig:
    """Base configuration for inference experiments"""
    # Model configuration
    model_id: str = 'maze2d'
    planner_alg: str = 'mpd'
    
    # Training/Model parameters 
    batch_size: int = 48
    dataset_subdir: str = 'EnvHard2dquant'  # EnvSimple2dquant, EnvPredator, EnvCompBoxCompare
    debug: bool = True
    diffusion_model_class: str = 'StaticGaussianDiffusionModel'
    include_velocity: bool = True
    loss_class: str = 'GaussianDiffusionLoss'
    lr: float = 0.0001
    n_diffusion_steps: int = 100
    num_train_steps: int = 1600000
    predict_epsilon: bool = True
    steps_til_ckpt: int = 40000
    steps_til_summary: int = 100
    summary_class: str = 'SummaryTrajectoryGeneration'
    unet_dim_mults_option: int = 1
    unet_input_dim: int = 32
    use_amp: bool = True
    use_ema: bool = False
    variance_schedule: str = 'exponential'
    wandb_entity: str = 'scoreplan'
    wandb_mode: str = 'disabled'
    wandb_project: str = 'test_train'
    
    # Sampling configuration
    n_samples: int = 20
    start_guide_steps_fraction: float = 0.25
    n_guide_steps: int = 1
    n_diffusion_steps_without_noise: int = 0
    
    # Trajectory configuration
    trajectory_duration: float = 5.0
    
    # Hardware configuration
    device: str = 'cuda'
    
    # Experiment configuration
    seed: int = 100  # Updated to match model default
    n_environments: int = 100  # number of obstacle configurations
    n_contexts_per_env: int = 20  # number of start/goal pairs per configuration
    
    # Output configuration
    render: bool = True
    results_dir: str = 'logs_new/seed_100'  
    
    # Paths
    trained_models_dir: str = '../../checkpoints/'
    dataset_path: str = '../../dataset/'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items()}
    
    def get_model_dir(self) -> str:
        """Get the model directory path"""
        return os.path.join(self.trained_models_dir, self.model_id)
    
    def get_checkpoint_path(self, checkpoint_type: str = 'ema') -> str:
        """Get checkpoint path for model loading"""
        checkpoint_name = f'{checkpoint_type}_model_current_state_dict.pth' if checkpoint_type == 'ema' else 'model_current_state_dict.pth'
        return os.path.join(self.get_model_dir(), 'checkpoints', checkpoint_name)

@dataclass 
class StaticConfig(BaseInferenceConfig):
    """Configuration for static obstacle experiments"""
    compose: bool = False  # For maze2D-10 obstacle sets
    # Override defaults for static experiments
    dataset_subdir: str = 'EnvHard2dquant'  # or 'EnvSimple2dquant'
    diffusion_model_class: str = 'StaticGaussianDiffusionModel'
    use_apf: bool =False 
    
    def __post_init__(self):
        """Post-initialization setup for static config"""
    
        if 'Simple2d' in self.dataset_subdir:
            self.compose = False
            self.n_diffusion_steps_without_noise = 5
        elif 'Hard2d' in self.dataset_subdir:
            self.compose = True
            self.n_diffusion_steps_without_noise = 0
@dataclass 
class Config3d(BaseInferenceConfig):
    """Configuration for static obstacle experiments"""
    compose: bool = False 
    model_id: str = 'maze3d'
    dataset_subdir: str = 'EnvSmall3D'  
    diffusion_model_class: str = 'GaussianDiffusionModel3d'
    use_apf: bool =False 
    n_samples: int =1
    n_diffusion_steps: int =25
    include_velocity: bool = True


@dataclass
class DynamicConfig(BaseInferenceConfig):
    """Configuration for dynamic obstacle experiments"""
    # Dynamic-specific parameters
    use_guide_on_extra_objects_only: bool = False
    weight_grad_cost_collision: float = 3e-2
    weight_grad_cost_smoothness: float = 1e-7
    factor_num_interpolated_points_for_collision: float = 1.5
    
    # Pursuer configuration
    pursuer_radius: float = 0.05
    pursuer_threshold: float = 0.2
    goal_safe_threshold: float = 0.25
    pursuer_pos: List[float] = None

    # Velocity configuration
    velocity_max_pursuer: float = 0.5
    
    # Pursuit dynamics
    pursuit_strength: float = 0.8
    random_strength: float = 0.2
    
    # Override defaults for dynamic experiments
    dataset_subdir: str = 'EnvPredator'
    diffusion_model_class: str = 'DynamicGaussianDiffusionModel'
    n_diffusion_steps_without_noise: int = 4
    n_samples: int = 35  # Higher for dynamic scenarios
    
    def __post_init__(self):
        """Post-initialization setup for dynamic config"""
        # Set pursuer position (can be overridden)
        # self.pursuer_pos = torch.zeros(2)
        if self.pursuer_pos is None:
            self.pursuer_pos = [0.0, 0.0]
        self.distance_threshold_pred = self.pursuer_radius + self.pursuer_threshold