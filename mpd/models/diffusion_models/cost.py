import torch

def compute_path_length(trajs):
    assert trajs.ndim == 3  # batch, horizon, state_dim
    trajs_pos = trajs[:,:,:2]
    path_length = torch.linalg.norm(torch.diff(trajs_pos, dim=-2), dim=-1).sum(-1)
    return path_length

def compute_variance_waypoints(trajs, robot):
    assert trajs.ndim == 3  # batch, horizon, state_dim
    trajs_pos = robot.get_position(trajs)
    sum_var_waypoints = 0.
    for via_points in trajs_pos.permute(1, 0, 2):  # horizon, batch, position
        parwise_distance_between_points_via_point = torch.cdist(via_points, via_points, p=2)
        distances = torch.triu(parwise_distance_between_points_via_point, diagonal=1).view(-1)
        sum_var_waypoints += torch.var(distances)
    return sum_var_waypoints

def compute_smoothness(trajs):
    assert trajs.ndim == 3
    trajs_vel = trajs[:,:,2:]
    smoothness = torch.linalg.norm(torch.diff(trajs_vel, dim=-2), dim=-1)
    smoothness = smoothness.sum(-1)  # sum over trajectory horizon
    return smoothness
def compute_collision_with_pointcloud(trajs, obstacle_points, collision_threshold=0.0,safety_margin=0.05):
    """
    Compute collisions between trajectories and obstacle pointclouds.
    
    Args:
    trajs (torch.Tensor): Batch of trajectories with shape (batch_size, horizon, state_dim)
    obstacle_points (torch.Tensor): Obstacle pointcloud with shape (n_obstacles, n_points, 2)
    collision_threshold (float): Distance threshold for collision detection
    
    Returns:
    torch.Tensor: Boolean tensor indicating collisions for each trajectory
    """
    batch_size, horizon, _ = trajs.shape
    n_obstacles, n_points, _ = obstacle_points.shape
    # breakpoint()
    # Reshape trajectories and obstacle points for broadcasting
    trajs_xy = trajs[:, :, :2].unsqueeze(2).unsqueeze(3)  # (batch_size, horizon, 1, 1, 2)
    obstacle_points = obstacle_points.unsqueeze(0).unsqueeze(1)  # (1, 1, n_obstacles, n_points, 2)
    
    # Compute distances between trajectory points and obstacle points
    distances = torch.norm(trajs_xy - obstacle_points, dim=-1)  # (batch_size, horizon, n_obstacles, n_points)
    
    # Check if any point is closer than the collision threshold
    # collisions = (distances < collision_threshold).any(dim=(-1, -2))  # (batch_size, horizon)
    collisions = (distances < collision_threshold).any(dim=-1).any(dim=-1)  # (batch_size, horizon)
    
    # A trajectory is in collision if any of its points are in collision
    trajectory_collisions = collisions.any(dim=-1)  # (batch_size,)
    
    return trajectory_collisions

def compute_trajectory_costs(trajs, obstacle_points,smoothness_weight=.1, path_length_weight=.9,collision_threshold=0.0,normalize=True): #0.1,0.9
    collision_free_mask = ~compute_collision_with_pointcloud(trajs, obstacle_points, collision_threshold)
    # breakpoint()
    if not collision_free_mask.any():
        print(f'No collision-free trajectories found!')
        return None, None, None, collision_free_mask,None
    collision_free_trajs = trajs[collision_free_mask]

    path_lengths = compute_path_length(collision_free_trajs)
    smoothness = compute_smoothness(collision_free_trajs)
    if normalize:
        # Normalize both metrics to [0,1] range
        path_lengths = (path_lengths - path_lengths.min()) / (path_lengths.max() - path_lengths.min())
        smoothness = (smoothness - smoothness.min()) / (smoothness.max() - smoothness.min())
    
    # breakpoint()
    total_costs = (
        smoothness_weight * smoothness +
        path_length_weight * path_lengths
    )
    
    best_index = torch.argmin(total_costs)
    best_trajectory = collision_free_trajs[best_index]
    best_cost = total_costs[best_index]

    # Get top 10 trajectories (lowest costs)
    # top_k = min(7, len(total_costs))  # Handle case where fewer than 10 trajectories
    top_k=1
    top_costs, top_indices = torch.topk(total_costs, top_k, largest=False)  # largest=False for lowest costs
    top_trajectories = collision_free_trajs[top_indices]

    # breakpoint()
    return best_trajectory, best_cost, total_costs,collision_free_mask,best_index #,top_trajectories, top_indices


