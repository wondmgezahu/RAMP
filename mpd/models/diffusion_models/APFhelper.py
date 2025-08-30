import torch
from scipy.spatial import cKDTree
import numpy as np

class ObstacleField:
    def __init__(self, obstacle_pts, distance_threshold=0.1, max_leaf_size=16):
        
        self.distance_threshold = distance_threshold
        self.static_obstacle_points = obstacle_pts
        self.static_kdtree = cKDTree(self.static_obstacle_points, leafsize=max_leaf_size)
        
        if isinstance(obstacle_pts, torch.Tensor):
            self.obstacle_points_tensor = obstacle_pts
        else:
            self.obstacle_points_tensor = torch.tensor(obstacle_pts)
            
    def query_static(self, points, chunk_size=1024):
    
        n_points = len(points)
        if n_points <= chunk_size:
            return self.static_kdtree.query(points, distance_upper_bound=self.distance_threshold)
        

        distances_list = []
        indices_list = []
        for i in range(0, n_points, chunk_size):
            chunk_points = points[i:i + chunk_size]
            chunk_distances, chunk_indices = self.static_kdtree.query(
                chunk_points, 
                distance_upper_bound=self.distance_threshold
            )
            distances_list.append(chunk_distances)
            indices_list.append(chunk_indices)
            
        return np.concatenate(distances_list), np.concatenate(indices_list)

def avoidance(trajectories, obstacle_field, avoidance_window=7, avoidance_strength=0.2):
    """Optimized vectorized obstacle avoidance"""
    batch_size, seq_len = trajectories.shape[:2]
    device = trajectories.device
    
    window_weights = torch.exp(-0.5 * torch.square(
        torch.arange(-avoidance_window, avoidance_window + 1, device=device)
    ) / (avoidance_window / 2) ** 2)
    
    points = trajectories[..., :2].reshape(-1, 2).cpu().numpy()
    
    distances, indices = obstacle_field.query_static(points)
    
    if np.min(distances) > obstacle_field.distance_threshold:
        return trajectories
    
    distances = torch.from_numpy(distances).to(device).reshape(batch_size, seq_len)
    indices = torch.from_numpy(indices).to(device).reshape(batch_size, seq_len)
 
    force_field = torch.zeros_like(trajectories[..., :2])
    collision_mask = distances < obstacle_field.distance_threshold
    
    collision_batches, collision_times = torch.where(collision_mask)
    if len(collision_batches) == 0:
        return trajectories
    
    valid_indices = indices[collision_batches, collision_times] < len(obstacle_field.obstacle_points_tensor)
    if not valid_indices.any():
        return trajectories
        
    collision_batches = collision_batches[valid_indices]
    collision_times = collision_times[valid_indices]

    indices=indices.cpu()

    nearest_obstacles = obstacle_field.obstacle_points_tensor[
        indices[collision_batches.cpu(), collision_times.cpu()]
    ].to(device)
    traj_points = trajectories[collision_batches, collision_times, :2]
    
    # Vectorized direction and magnitude computation
    avoid_directions = traj_points - nearest_obstacles
    avoid_distances = torch.norm(avoid_directions, dim=1, keepdim=True)
    avoid_directions = avoid_directions / (avoid_distances + 1e-8)
    
    force_magnitudes = avoidance_strength * torch.exp(
        -distances[collision_batches, collision_times, None] / 
        obstacle_field.distance_threshold
    )
   
    time_indices = collision_times[:, None] + torch.arange(
        -avoidance_window, 
        avoidance_window + 1, 
        device=device
    )
    valid_time_mask = (time_indices >= 0) & (time_indices < seq_len)

    for i, (batch_idx, time_idx) in enumerate(zip(collision_batches, collision_times)):
        valid_times = valid_time_mask[i]
        curr_indices = time_indices[i][valid_times]
        curr_weights = window_weights[valid_times]
        
        force_field[batch_idx, curr_indices] += (
            force_magnitudes[i] * avoid_directions[i] * curr_weights[:, None]
        )
    trajectories_modified = trajectories.clone()
    trajectories_modified[..., :2] += force_field
    return trajectories_modified