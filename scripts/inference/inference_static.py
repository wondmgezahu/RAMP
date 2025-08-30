import os
import json
from math import ceil
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from mpd.models import TemporalUnetInference, UNET_DIM_MULTS
from mpd.models.diffusion_models.sample_functions import  ddpm_sample_fn
from mpd.trainer import get_dataset, get_model,get_dataset_explicit_dir
from deps.torch_robotics.torch_robotics.torch_utils.seed import fix_random_seed
from deps.torch_robotics.torch_robotics.torch_utils.torch_timer import TimerCUDA
from deps.torch_robotics.torch_robotics.torch_utils.torch_utils import freeze_torch_model_params
import numpy as np

from config.base_config import StaticConfig
from core.utils import StateGenerator, ContextManager
from core.metrics import Metrics
from core.visualization import BaseVisualizer

allow_ops_in_compiled_graph()

class StaticInference:
    def __init__(self, config: StaticConfig):
        self.config = config
        self.device = config.device
        self.tensor_args = {'device': config.device, 'dtype': torch.float32}
        
        # Initialize utilities
        self.metrics_calculator = Metrics()
        self.visualizer = BaseVisualizer()
        self.context_manager = ContextManager()
        
        # Will be set during setup
        self.model = None
        self.dataset = None
        self.robot = None
        self.env = None
        
    def run_single_experiment(self, current_dir: int, context_idx: int):
        """Run single experiment """
        
        """Setup model and dataset """
        torch.cuda.set_device(0)
        
        # Load dataset
        train_subset, _, _, _ = get_dataset_explicit_dir(
            dataset_class='ExpDirectoryDataset',
            current_dir=current_dir,
            use_extra_objects=False,
            obstacle_cutoff_margin=0.05,
            dataset_subdir=self.config.dataset_subdir,
            include_velocity=self.config.include_velocity,
            training=False,
            static=True,
            dataset_base_dir=self.config.dataset_path,
            tensor_args=self.tensor_args,
        )
        
        self.dataset = train_subset.dataset
        self.robot = self.dataset.robot
        self.env = self.dataset.env
               # Get data
        data_normalized = train_subset[0]
        traj_normalized = data_normalized['traj_normalized']
        obstacle_pts = data_normalized['obstacle_points']
        obstacle_pts_visual = obstacle_pts.clone()

        # Handle obstacle composition
        if self.config.compose:
            first_batch = obstacle_pts[:6]
            remaining = obstacle_pts[6:] 
            indices = torch.randperm(4)[:2]
            second_batch = torch.cat([remaining, remaining[indices]], dim=0)
            obstacle_pts = torch.stack([first_batch, second_batch], dim=0)
        
        box_centers = data_normalized['box_centers']
        box_size = data_normalized['box_sizes']
        self.env.update_box_centers(box_centers)
        
        # Warmup model
        n_support_points = self.dataset.n_support_points
        dt = self.config.trajectory_duration / n_support_points
        self.robot.dt = dt
        
        # Load diffusion model
        diffusion_configs = dict(
            variance_schedule=self.config.variance_schedule,
            n_diffusion_steps=self.config.n_diffusion_steps,
            predict_epsilon=self.config.predict_epsilon,
            compose=self.config.compose,
            use_apf=self.config.use_apf,
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
        # model_id
        self.model.load_state_dict(
            torch.load(os.path.join(self.config.trained_models_dir, self.config.model_id, 'checkpoints',
                    'ema_model_current_state_dict.pth' if self.config.use_ema else 'model_current_state_dict.pth'),
                    map_location=self.device)
        )
        self.model.eval()
        freeze_torch_model_params(self.model)
        self.model = torch.compile(self.model, mode='reduce-overhead') 
        self.model.warmup(
            horizon=n_support_points,
            traj_normalized=traj_normalized, 
            obstacle_pts=obstacle_pts, 
            batch_size=self.config.n_samples,
            device=self.device
        )
        
        # Load context
        env_dir = os.path.join(self.config.dataset_path, self.config.dataset_subdir, str(current_dir))
        start_state_pos, goal_state_pos = self.context_manager.load_context(
            os.path.join(env_dir, 'contexts'), context_idx, self.device
        )
        
        # Setup hard conditions
        hard_conds = StateGenerator.get_hard_cond_custom(
            torch.vstack((start_state_pos, goal_state_pos)), 
            horizon=n_support_points,
            include_velocity=self.config.include_velocity
        )
        context = {'dataset': self.dataset}
        
        # Run inference
        t_start_guide = ceil(self.config.start_guide_steps_fraction * self.model.n_diffusion_steps)
        sample_fn_kwargs = dict(
            guide=None,
            n_guide_steps=self.config.n_guide_steps,
            t_start_guide=t_start_guide,
            noise_std_extra_schedule_fn=lambda x: 0.5,
        )
        
        with TimerCUDA() as timer:
            trajs_normalized_iters = self.model.run_inference(
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
        
        trajs_final = trajs_normalized_iters[-1]
        
        # Compute metrics
        collision_intensities = self.metrics_calculator.compute_collision_intensity(
            trajs_final, box_centers, box_size
        )
        
        metrics = self.metrics_calculator.trajectory_success_and_metrics(
            trajs_final, collision_intensities
        )
        metrics['total_time'] = timer.elapsed
        
        # Render if needed
        if self.config.render:
            pos_trajs_iters = self.robot.get_position(trajs_normalized_iters)
            self.visualizer.save_static_plot(
                box_centers.cpu().numpy(),
                box_size.cpu().numpy(),
                start_state_pos.cpu().numpy(),
                goal_state_pos.cpu().numpy(),
                pos_trajs_iters[-1].cpu().numpy(),
                obstacle_pts_visual.cpu().numpy(),
                os.path.join(env_dir, f'robot-traj-dir{current_dir}.png')
            )
        
        return metrics
    
    def run_full_evaluation(self):
        """Run full evaluation across all environments and contexts"""
        env_metrics = []
        
        for env_idx in range(self.config.n_environments):
            print(f'Processing Environment {env_idx}')
            
            context_metrics = []
            for context_idx in range(self.config.n_contexts_per_env):
                print(f'  Processing Context {context_idx}')
                try:
                    metrics = self.run_single_experiment(env_idx, context_idx)
                    context_metrics.append(metrics)
                except Exception as e:
                    print(f"Error in env {env_idx}, context {context_idx}: {e}")
                    continue
            
            
            env_result = self.process_environment_metrics(context_metrics)
            env_metrics.append(env_result)
        
       
        final_results = self.calculate_final_results(env_metrics)
        return final_results
    
    def process_environment_metrics(self,context_metrics):
        """
        Process metrics from multiple contexts in an environment.
        Handles None values and provides detailed statistics.
        """
        metrics_summary = {
            'success_rates': [],
            'collision_intensities': [],
            'path_lengths': [],
            'path_length_stds': [],
            'variances': [],
            'times': [],
            'n_valid_variance_contexts': 0,
            'n_single_traj_contexts': 0,
            'n_multi_traj_contexts': 0
        }
        
        for metrics in context_metrics:
            if metrics is None:
                continue
            # breakpoint()    
            # Always include basic metrics
            metrics_summary['success_rates'].append(metrics['success'])
            metrics_summary['collision_intensities'].append(metrics['collision_intensity'])
            
            # Handle path length metrics
            if metrics['path_length'] is not None:
                metrics_summary['path_lengths'].append(metrics['path_length'])
                metrics_summary['path_length_stds'].append(metrics['path_length_std'])
            
            # Handle variance metrics
            if metrics['waypoint_variance'] is not None:
                metrics_summary['variances'].append(metrics['waypoint_variance'])
                metrics_summary['n_valid_variance_contexts'] += 1
                
                # Count single vs multi trajectory cases
                if len(metrics['free_trajectories']) == 1:
                    metrics_summary['n_single_traj_contexts'] += 1
                else:
                    metrics_summary['n_multi_traj_contexts'] += 1
            
            if 'total_time' in metrics:
                metrics_summary['times'].append(metrics.get('total_time', 0))
        
        # Calculate means only if we have valid data
        results = {}
        for key in metrics_summary:
            if isinstance(metrics_summary[key], list):
                if metrics_summary[key]:
                    results[f'{key}_mean'] = np.mean(metrics_summary[key])
                else:
                    results[f'{key}_mean'] = None
                    
        # Add counts
        results.update({
            'n_valid_variance_contexts': metrics_summary['n_valid_variance_contexts'],
            'n_single_traj_contexts': metrics_summary['n_single_traj_contexts'],
            'n_multi_traj_contexts': metrics_summary['n_multi_traj_contexts']
        })
        
        return results
    
    def calculate_final_results(self, env_metrics):
        final_results = {
            'success_rates': [],
            'collision_intensities': [],
            'path_lengths': [],
            'path_length_stds': [],
            'variances': [],
            'times': [],
            'total_valid_variance_contexts': 0,
            'total_single_traj_contexts': 0,
            'total_multi_traj_contexts': 0
        }
        
        # Collect all valid metrics across environments
        for env_result in env_metrics:
            if env_result['success_rates_mean'] is not None:
                final_results['success_rates'].append(env_result['success_rates_mean'])
            if env_result['collision_intensities_mean'] is not None:
                final_results['collision_intensities'].append(env_result['collision_intensities_mean'])
            if env_result['path_lengths_mean'] is not None:
                final_results['path_lengths'].append(env_result['path_lengths_mean'])
            if env_result['path_length_stds_mean'] is not None:
                final_results['path_length_stds'].append(env_result['path_length_stds_mean'])
            if env_result['variances_mean'] is not None:
                final_results['variances'].append(env_result['variances_mean'])
            if env_result['times_mean'] is not None:
                final_results['times'].append(env_result['times_mean'])
                
            final_results['total_valid_variance_contexts'] += env_result['n_valid_variance_contexts']
            final_results['total_single_traj_contexts'] += env_result['n_single_traj_contexts']
            final_results['total_multi_traj_contexts'] += env_result['n_multi_traj_contexts']

            # Calculate final statistics
        def compute_mean_std(values):
            if not values:
                return 0.0, 0.0
            values = np.array(values)
            return np.mean(values), np.std(values)

        # Compute final metrics
        final_success_rate, final_success_std = compute_mean_std(final_results['success_rates'])
        final_collision_intensity, final_collision_intensity_std = compute_mean_std(final_results['collision_intensities'])
        final_path_length, final_path_length_std = compute_mean_std(final_results['path_lengths'])
        final_var_mean, final_var_std = compute_mean_std(final_results['variances'])
        final_time_mean, final_time_std = compute_mean_std(final_results['times'])

        # Print final results
        print("\nFinal Results:")
        print(f'Success rate: {final_success_rate * 100:.2f}% ± {final_success_std * 100:.2f}%')
        print(f'Collision intensity: {final_collision_intensity:.2f}% ± {final_collision_intensity_std:.2f}%')
        print(f'Path length: {final_path_length:.3f} ± {final_path_length_std:.3f}')
        print(f'Waypoint variance: {final_var_mean:.4f} ± {final_var_std:.4f}')
        print(f'Computation time: {final_time_mean:.3f} ± {final_time_std:.3f} seconds')
        
        # Save results with additional statistics
        results = {
            'success_rate': (final_success_rate, final_success_std),
            'collision_intensity': (final_collision_intensity, final_collision_intensity_std),
            'path_length': (final_path_length, final_path_length_std),
            'waypoint_variance': (final_var_mean, final_var_std),
            'computation_time': (final_time_mean, final_time_std),
            'context_statistics': {
                'valid_variance_contexts': final_results['total_valid_variance_contexts'],
                'single_traj_contexts': final_results['total_single_traj_contexts'],
                'multi_traj_contexts': final_results['total_multi_traj_contexts']
            }
        }
        
        with open('eval_results_static.json', 'w') as f:
            json.dump(results, f, indent=2)    
        
        return results
    
def main():
    """Main execution function"""
    # Create configuration
    config = StaticConfig(
        model_id='model2d',
        dataset_subdir='EnvSimple2dquant', # EnvHard2dquant for 10 obstacles
        n_samples=20,  # 20 N_TRAJECTORIES_PER_CONTEXT
        n_environments=2,  # 100 N_ENVIRONMENTS 
        n_contexts_per_env=2, # 20 N_CONTEXTS_PER_ENV
        compose=False, #True for maze2d-10 obstacle, and set dataset_subdir to EnvHard2dquant
        use_apf=False, # True for use of apf 
        render=True, # visualize results
        device='cuda'
    )
    # breakpoint()
    # Run inference
    inference = StaticInference(config)
    results = inference.run_full_evaluation()
  
if __name__ == '__main__':
    main()

