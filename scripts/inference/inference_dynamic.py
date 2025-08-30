import os
import json
from math import ceil
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from mpd.models import TemporalUnetInference, UNET_DIM_MULTS
from mpd.models.diffusion_models.sample_functionsdynamic import  ddpm_sample_fn
from mpd.trainer import get_dataset, get_model,get_dataset_explicit_dir
from deps.torch_robotics.torch_robotics.torch_utils.seed import fix_random_seed
from deps.torch_robotics.torch_robotics.torch_utils.torch_timer import TimerCUDA
from deps.torch_robotics.torch_robotics.torch_utils.torch_utils import freeze_torch_model_params
import numpy as np


from config.base_config import DynamicConfig
from core.utils import StateGenerator, ContextManager, DynamicsGenerator
from core.metrics import DynamicMetrics
from core.visualization import DynamicVisualizer

allow_ops_in_compiled_graph()


class DynamicInference:
    def __init__(self, config: DynamicConfig):
        self.config = config
        self.device = config.device
        self.tensor_args = {'device': config.device, 'dtype': torch.float32}
        
        # Initialize utilities
        self.metrics_calculator = DynamicMetrics()
        self.visualizer = DynamicVisualizer()
        self.context_manager = ContextManager()
        
        # Setup dynamics
        self.dynamics_fn, self.velocity = DynamicsGenerator.create_pursuit_dynamics(
            velocity_max=config.velocity_max_pursuer,
            pursuit_strength=config.pursuit_strength,
            random_strength=config.random_strength
        )
        
        self.model = None
        self.dataset = None
        self.robot = None
        self.env = None
        
    def setup_model_and_dataset(self):
        """Setup model and dataset"""
        torch.cuda.set_device(0)
        
        # Load dataset
        train_subset, _, _, _ = get_dataset(
            dataset_class='TrajectoryDataset',
            use_extra_objects=True,
            obstacle_cutoff_margin=0.05,
            dynamics_fn=self.dynamics_fn,
            velocity=self.velocity,
            dataset_subdir=self.config.dataset_subdir,
            include_velocity=self.config.include_velocity,
            training=False,
            static=False,
            tensor_args=self.tensor_args
        )
        
        self.dataset = train_subset.dataset
        self.robot = self.dataset.robot
        self.env = self.dataset.env
        self.train_subset = train_subset
        
        # Setup robot timing
        n_support_points = self.dataset.n_support_points
        dt = self.config.trajectory_duration / n_support_points
        self.robot.dt = dt
        
        # Load diffusion model
        diffusion_configs = dict(
            variance_schedule=self.config.variance_schedule,
            n_diffusion_steps=self.config.n_diffusion_steps,
            predict_epsilon=self.config.predict_epsilon,
        )
        unet_configs = dict(
            state_dim=self.dataset.state_dim,
            n_support_points=n_support_points,
            unet_input_dim=self.config.unet_input_dim,
            dim_mults=UNET_DIM_MULTS[self.config.unet_dim_mults_option]
        )
        
        self.model = get_model(
            model_class=self.config.diffusion_model_class,
            model=TemporalUnetInference(**unet_configs),
            tensor_args=self.tensor_args,
            **diffusion_configs,
            **unet_configs
        )
        
        # Load model weights
        self.model.load_state_dict(
            torch.load(os.path.join(self.config.trained_models_dir, 'checkpoints', 
                    'ema_model_current_state_dict.pth' if self.config.use_ema else 'model_current_state_dict.pth'),
                    map_location=self.device)
        )
        
        self.model.eval()
        freeze_torch_model_params(self.model)
        
    def run_single_experiment(self, context_idx: int):
        """Run single experiment"""
        
        torch.cuda.set_device(0)
        
        # Load dataset
        train_subset, _, _, _ = get_dataset(
            dataset_class='TrajectoryDataset',
            use_extra_objects=True,
            obstacle_cutoff_margin=0.05,
            dynamics_fn=self.dynamics_fn,
            velocity=self.velocity,
            dataset_subdir=self.config.dataset_subdir,
            include_velocity=self.config.include_velocity,
            training=False,
            static=False,
            dataset_base_dir=self.config.dataset_path,
            tensor_args=self.tensor_args,
            pursuer_pos=self.config.pursuer_pos,
        )
        
        self.dataset = train_subset.dataset
        self.robot = self.dataset.robot
        self.env = self.dataset.env
        # self.train_subset = train_subset
        # Get random trajectory data
        traj_id = np.random.choice(len(train_subset), 1).item()
        print('traj_id', traj_id)
        
        data_normalized = train_subset[traj_id]
        traj_normalized = data_normalized['traj_normalized']
        obstacle_pts = data_normalized['obstacle_points']
        box_centers = data_normalized['box_centers']
        box_size = data_normalized['box_sizes']
        box_centers_visual=box_centers[:4] # only use the first 4 obstacles for dynamic
        box_size_visual=box_size[:4]    
        # Use only 4 obstacles + 2 random
        obstacle_pts = torch.cat([obstacle_pts[:4], obstacle_pts[torch.randint(0, 4, (2,))]], dim=0)
        # breakpoint() 
        # Setup robot timing
        n_support_points = self.dataset.n_support_points
        dt = self.config.trajectory_duration / n_support_points
        self.robot.dt = dt
        
        # Load or generate start/goal states
        env_dir = os.path.join(self.config.dataset_path, self.config.dataset_subdir, 'contexts')
        # Load existing context
        start_state_pos, goal_state_pos = self.context_manager.load_context(
            os.path.join(env_dir, 'contexts'), context_idx, self.device
        )
        
        print(f'start_state_pos: {start_state_pos}')
        print(f'goal_state_pos: {goal_state_pos}')


        # Load diffusion model
        diffusion_configs = dict(
            variance_schedule=self.config.variance_schedule,
            n_diffusion_steps=self.config.n_diffusion_steps,
            predict_epsilon=self.config.predict_epsilon,
        )
        unet_configs = dict(
            state_dim=self.dataset.state_dim,
            n_support_points=n_support_points,
            unet_input_dim=self.config.unet_input_dim,
            dim_mults=UNET_DIM_MULTS[self.config.unet_dim_mults_option]
        )
        
        self.model = get_model(
            model_class=self.config.diffusion_model_class,
            model=TemporalUnetInference(**unet_configs),
            tensor_args=self.tensor_args,
            **diffusion_configs,
            **unet_configs
        )
        
        # Load model weights

        self.model.load_state_dict(
            torch.load(os.path.join(self.config.trained_models_dir, self.config.model_id, 'checkpoints',
                    'ema_model_current_state_dict.pth' if self.config.use_ema else 'model_current_state_dict.pth'),
                    map_location=self.device)
        )

        self.model.eval()
        freeze_torch_model_params(self.model)

        hard_conds = StateGenerator.get_hard_cond_custom(
            torch.vstack((start_state_pos, goal_state_pos)), 
            horizon=n_support_points,
            include_velocity=self.config.include_velocity
        )
        context = {'dataset': self.dataset}
        
        # Update start/goal from hard conditions
        start_state_pos = hard_conds[0][:2]
        goal_state_pos = hard_conds[n_support_points-1][:2]
        
        # Run inference
        t_start_guide = ceil(self.config.start_guide_steps_fraction * self.model.n_diffusion_steps)
        sample_fn_kwargs = dict(
            guide=None,  
            n_guide_steps=self.config.n_guide_steps,
            t_start_guide=t_start_guide,
            noise_std_extra_schedule_fn=lambda x: 0.5,
        )
        # breakpoint()
        with TimerCUDA() as timer:
            trajs_normalized_iters, chain_obs, chain_start = self.model.run_inference(
                context, hard_conds,
                n_samples=self.config.n_samples, 
                horizon=n_support_points,
                return_chain=True,
                traj_normalized=traj_normalized,
                obstacle_pts=obstacle_pts,
                sample_fn=ddpm_sample_fn,
                **sample_fn_kwargs,
                n_diffusion_steps_without_noise=self.config.n_diffusion_steps_without_noise,
            )
        # breakpoint()
        # Process chains
        chain_obs.pop()
        chain_obs = [tensor.cpu().detach().numpy() for tensor in chain_obs]
        chain_start = [np.around(tensor[:,:2].cpu().detach().numpy(), decimals=4) for tensor in chain_start]
        
        # Get collision info
        trajs_final = trajs_normalized_iters[-1]
        # Check collision intensity
        if isinstance(chain_start, list):
            trajs = torch.tensor([state.squeeze() for state in chain_start]).unsqueeze(0)
        elif isinstance(chain_start, np.ndarray):
            trajs = torch.tensor(chain_start).unsqueeze(0)   
        else:
            trajs = chain_start.unsqueeze(0)
            
        collision_intensities = self.metrics_calculator.compute_collision_intensity(
            trajs.to(box_centers.device), 
            box_centers[:4], 
            box_size[:4].unsqueeze(1).repeat(1,2)
        )
        
        metrics = {
            'chain_start': chain_start,
            'chain_obs': chain_obs,
            'start_state_pos': start_state_pos,
            'goal_state_pos': goal_state_pos,
            'computation_time': timer.elapsed,
            'collision_intensity': bool(collision_intensities.any().item())
        }
        # breakpoint()
        # Render if needed
        if self.config.render:
            pos_trajs_iters = self.robot.get_position(trajs_normalized_iters)
            self.visualizer.create_animation(
                box_centers_visual,
                box_size_visual,
                start_state_pos,
                goal_state_pos,
                pos_trajs_iters,
                obstacle_pts,
                trajs,
                chain_start,
                chain_obs,
                self.config.pursuer_radius,
                self.config.distance_threshold_pred,
                context_idx
                # f'/tmp/context_{context_idx}.gif'  
            )
        # breakpoint()
        return metrics
    
    def run_multiple_experiments(self, n_contexts=100, n_experiments=10):
        """Run multiple experiments for each context"""
        all_experiment_results = []
        
        for exp_idx in range(n_experiments):
            print(f'Running experiment {exp_idx + 1}/{n_experiments}')
            # self.setup_model_and_dataset()
            
            experiment_metrics = []
            for context_idx in range(n_contexts):
            # for context_idx in range(20, 20 + n_contexts):  
                print(f'  Processing context {context_idx + 1}/{n_contexts}')
                
                # try:
                metrics = self.run_single_experiment(context_idx)
                if metrics is None:
                    continue
                    
                episode_metrics = self.metrics_calculator.calculate_single_episode_metrics(
                    chain_start=metrics['chain_start'],
                    chain_obs=metrics['chain_obs'],
                    start_state_pos=metrics['start_state_pos'],
                    goal_state_pos=metrics['goal_state_pos'],
                    goal_safe_threshold=self.config.goal_safe_threshold,
                    static_collision=metrics['collision_intensity'],
                    pursuer_radius=self.config.pursuer_radius
                )
                episode_metrics['computation_time'] = metrics['computation_time']
                experiment_metrics.append(episode_metrics)
                    
                # except Exception as e:
                #     print(f"Error in experiment {exp_idx}, context {context_idx}: {e}")
                #     continue
            
            # Process results for this experiment
            experiment_results = self.process_experiment_results(experiment_metrics)
            all_experiment_results.append(experiment_results)
        
        # Calculate final averaged results
        final_averaged_results = self.average_experiment_results(all_experiment_results)
        return final_averaged_results
    
    def process_experiment_results(self,all_metrics):
        """Process metrics from all experiments"""
        valid_paths = [m['path_length'] for m in all_metrics if m['path_length'] is not None] # modified
        final_metrics = {
            'detection_rate': {
                'mean': np.mean([m['captured'] for m in all_metrics]),
                'std': np.std([m['captured'] for m in all_metrics])
            },
            'goal_success': {
                'mean': np.mean([m['goal_reached'] for m in all_metrics]),
                'std': np.std([m['goal_reached'] for m in all_metrics])
            },
            'path_length': {
                'mean': np.mean(valid_paths) if valid_paths else None,
                'std': np.std(valid_paths) if valid_paths else None
            },
            'score': {
                'mean': np.mean([m['score'] for m in all_metrics]),
                'std': np.std([m['score'] for m in all_metrics])
            }
        }
        return final_metrics
    
    def average_experiment_results(self,all_results):
        """
        Average metrics across multiple experiments.
        
        Args:
            all_results: List of processed results from each experiment
        """
        # Initialize structure for averaged results
        avg_results = {
            'detection_rate': {'mean': [], 'std': []},
            'goal_success': {'mean': [], 'std': []},
            'path_length': {'mean': [], 'std': []},
            'score': {'mean': [], 'std': []}
        }
        # Collect values across experiments
        for result in all_results:
            for metric, values in result.items():
                if isinstance(values, dict):
                    if 'mean' in values:
                        avg_results[metric]['mean'].append(values['mean'])
                        avg_results[metric]['std'].append(values['std'])
                    else:
                        for submetric, value in values.items():
                            avg_results[metric][submetric].append(value)

        final_results = {}
        for metric, values in avg_results.items():
            final_results[metric] = {}
            for submetric, subvalues in values.items():
                if submetric in ['mean', 'std']:
                    if metric=='path_length':
                        valid_values=[v for v in subvalues if v is not None]
                        if valid_values:
                            final_results[metric][submetric]={
                                'value':np.mean(valid_values),
                                'uncertainty':np.std(valid_values)
                            }
                        else:
                            final_results[metric][submetric]={
                                'value':None,
                                'uncertainty':None
                            }    
                    else:    
                        final_results[metric][submetric] = {
                        'value': np.mean(subvalues), # 
                        'uncertainty': np.std(subvalues) # 
                    }
                else:
                    final_results[metric][submetric] = {
                        'value': np.mean(subvalues),
                        'uncertainty': np.std(subvalues)
                    }
        return final_results

    
    def save_results(self, results, save_dir=None):
        """Save results to file"""
        if save_dir is None:
            save_dir = '/tmp'
            
        json_filename = "eval_results_dynamic.json"
        json_path = os.path.join(save_dir, json_filename)
        os.makedirs(save_dir, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {json_path}")

def main():
    """Main execution function"""
    # Create configuration
    config = DynamicConfig(
        model_id='model2d',
        dataset_subdir='EnvPredator',
        n_samples=35,
        render=True,
        device='cuda',
        velocity_max_pursuer=0.5,
        pursuit_strength=0.8,
        random_strength=0.2,
        n_diffusion_steps_without_noise=4,
        pursuer_pos=[0.0, 0.0]
    )
    
    # Run inference
    inference = DynamicInference(config)
    
    n_contexts = 5  # 100 for paper 
    n_experiments = 2  # 10 for paper
    save_path = 'dynamic_results'
    os.makedirs(save_path, exist_ok=True) 
    results = inference.run_multiple_experiments(
        n_contexts=n_contexts,
        n_experiments=n_experiments
    )
    
    inference.save_results(results, save_path)
    print("Dynamic inference completed!")

if __name__ == '__main__':
    main()

