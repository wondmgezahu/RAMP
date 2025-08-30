import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class Metrics:
    """Unified metrics calculation for both static and dynamic experiments"""
    
    @staticmethod
    def compute_variance_waypoints(trajs, eps=1e-8):
        """Compute variance of waypoints across trajectories"""
        assert trajs.ndim == 3  # batch, horizon, state_dim
        trajs_pos=trajs[...,:2]
        sum_var_waypoints = 0.
        for via_points in trajs_pos.permute(1, 0, 2):  # horizon, batch, position
            parwise_distance_between_points_via_point = torch.cdist(via_points, via_points, p=2)
            distances = torch.triu(parwise_distance_between_points_via_point, diagonal=1).view(-1)
            sum_var_waypoints += torch.var(distances+eps)
        return sum_var_waypoints

    @staticmethod
    def compute_smoothness(trajs: torch.Tensor, trajs_vel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute trajectory smoothness"""
        if trajs_vel is None:
            assert trajs.ndim == 3
            trajs_vel = trajs[..., 2:]
        else:
            assert trajs_vel.ndim == 3
        smoothness = torch.linalg.norm(torch.diff(trajs_vel, dim=-2), dim=-1)
        smoothness = smoothness.sum(-1)  # sum over trajectory horizon
        return smoothness

    @staticmethod
    def compute_path_length(trajectories: torch.Tensor) -> torch.Tensor:
        """Compute path length using only position dimensions"""
        assert trajectories.ndim==3
        if len(trajectories) == 0:
            return torch.tensor(0.0, device=trajectories.device)
        
        # Only use position dimensions [x,y]
        positions = trajectories[..., :2]  
        
        # Calculate euclidean distance between consecutive waypoints
        diffs = positions[:, 1:] - positions[:, :-1]
        lengths = torch.sqrt((diffs ** 2).sum(dim=-1)).sum(dim=-1)
        return lengths


    @staticmethod
    def compute_collision_intensity(trajs: torch.Tensor, box_centers: torch.Tensor, 
                                  box_sizes: torch.Tensor) -> torch.Tensor:
        """Compute collision intensity between trajectories and obstacles"""
        n_samples, horizon, state_dim = trajs.shape
        n_boxes = box_centers.shape[0]
        
        # Ensure inputs are PyTorch tensors with correct device and dtype
        if not isinstance(box_centers, torch.Tensor):
            box_centers = torch.tensor(box_centers, dtype=torch.float32, device=trajs.device)
        if not isinstance(box_sizes, torch.Tensor):
            box_sizes = torch.tensor(box_sizes, dtype=torch.float32, device=trajs.device)
        
        # Handle case where box_sizes is [n_boxes] instead of [n_boxes, 2]
        if len(box_sizes.shape) == 1:
            box_sizes = box_sizes.unsqueeze(-1).repeat(1, 2)

        trajs_reshaped = trajs[:, :, None, :2]  # Add box dimension
        box_centers_reshaped = box_centers.view(1, 1, n_boxes, 2)
        box_sizes_reshaped = box_sizes.view(1, 1, n_boxes, 2)
        
        # Compute bounds
        lower_bounds = box_centers_reshaped - box_sizes_reshaped/2
        upper_bounds = box_centers_reshaped + box_sizes_reshaped/2
        
        # Check collisions
        # Results in tensor of shape [n_samples, horizon, n_boxes]
        collisions = ((trajs_reshaped >= lower_bounds) & 
                    (trajs_reshaped <= upper_bounds)).all(dim=-1)
        
        # If any timestep collides with any box, that timestep has a collision
        # Then take mean over timesteps to get collision intensity
        collision_intensities = collisions.any(dim=-1).float().mean(dim=1)
        
        return collision_intensities

    def trajectory_success_and_metrics(self, trajs_final: torch.Tensor, 
                                     collision_intensities: torch.Tensor, 
                                     threshold: float = 0.01) -> Dict[str, Any]:
        """
        Determine success based on collision intensities and return metrics
        """
        successful_trajectories = collision_intensities <= threshold 
        success = 1 if torch.any(successful_trajectories) else 0
        
        # Get collision-free trajectories
        free_trajectory_indices = torch.where(successful_trajectories)[0]
        trajs_final_free = trajs_final[free_trajectory_indices]
        n_free_trajectories = len(trajs_final_free)
        collision_intensity = collision_intensities.mean().item() * 100

        # Initialize metrics
        metrics = {
            'success': success,
            'collision_intensity': collision_intensity,
            'path_length': None,
            'path_length_std': None,
            'waypoint_variance': None,
            'free_trajectories': trajs_final_free,
            'n_free_trajectories': n_free_trajectories
        }


        if n_free_trajectories > 0:
            # Compute path length metrics
            path_lengths = self.compute_path_length(trajs_final_free)
            metrics['path_length'] = path_lengths.mean().item()
            metrics['path_length_std'] = path_lengths.std().item()

            # Variance handling
            if n_free_trajectories == 1:
                metrics['waypoint_variance'] = 0.0
            else:
                variance = self.compute_variance_waypoints(trajs_final_free)
                if torch.is_tensor(variance):
                    variance = variance.item()
                metrics['waypoint_variance'] = variance if not np.isnan(variance) else None
        
        return metrics

class DynamicMetrics(Metrics):
    """Extended metrics for dynamic obstacle experiments"""
    
    def calculate_single_episode_metrics(self, chain_start: List, chain_obs: List, 
                                       start_state_pos: torch.Tensor, goal_state_pos: torch.Tensor,
                                       goal_safe_threshold: float, static_collision: bool,
                                       pursuer_radius: float) -> Dict[str, Any]:
        """Calculate metrics for a single dynamic episode"""
        goal_pos = goal_state_pos.cpu().numpy() if torch.is_tensor(goal_state_pos) else goal_state_pos
        
        # Calculate pursuer capture
        safety_margin = 0.02
        capture_threshold = pursuer_radius + safety_margin
        pursuer_capture = False
        
        for i in range(len(chain_obs)):
            evader_idx = i + 2
            if evader_idx >= len(chain_start):
                break
                
            distance = np.linalg.norm(chain_start[evader_idx] - chain_obs[i])
            if distance <= capture_threshold:
                pursuer_capture = True
                break

        # Calculate metrics
        captured = static_collision or pursuer_capture
        final_evader_pos = chain_start[-1]
        distance_to_goal = np.linalg.norm(final_evader_pos - goal_pos)
        goal_reached = (distance_to_goal <= goal_safe_threshold) and (not captured)
        
        # Calculate path length
        path_length = 0
        for i in range(len(chain_start)-1):
            path_length += np.linalg.norm(chain_start[i+1] - chain_start[i])
        
        return {
            'static_collision': static_collision,
            'pursuer_capture': pursuer_capture,
            'captured': captured,
            'goal_reached': goal_reached,
            'path_length': path_length if not captured else None,
            'score': 0.5 * float(goal_reached) + 0.5 * float(not captured)
        }