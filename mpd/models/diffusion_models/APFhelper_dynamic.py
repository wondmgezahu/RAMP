import torch
import numpy as np
from scipy.spatial import cKDTree

def apply_hard_conditioning(x, conditions):
    for t, val in conditions.items():
        x[:, t, :] = val.clone()
    return x

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def quadratic_bezier(p0, p1, p2, t):
    return (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2

def generate_sphere_points(center, radius, num_points, surface_ratio=0.9):

    surface_points = int(num_points * surface_ratio)
    inner_points = num_points - surface_points
    
    temp_angle = np.pi * (3 - np.sqrt(5))  
    theta_surface = temp_angle * np.arange(surface_points)
    x_surface = radius * np.cos(theta_surface) + center[0]
    y_surface = radius * np.sin(theta_surface) + center[1]
    
    if inner_points > 0:
        r_inner = radius * np.sqrt(np.random.uniform(0, 1, inner_points))
        theta_inner = np.random.uniform(0, 2*np.pi, inner_points)
        x_inner = r_inner * np.cos(theta_inner) + center[0]
        y_inner = r_inner * np.sin(theta_inner) + center[1]
        
        x = np.concatenate((x_surface, x_inner))
        y = np.concatenate((y_surface, y_inner))
    else:
        x, y = x_surface, y_surface
    
    return np.column_stack((x, y))

def generate_box_points(center, size, num_points):
    center_x, center_y = center
    width, height = size

    left = center_x - width / 2
    right = center_x + width / 2
    top = center_y + height / 2
    bottom = center_y - height / 2
    boundary_points = np.random.randint(2*num_points // 3, 1*num_points + 1)
    inside_points = num_points - boundary_points

    # Generate boundary points
    edges = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
    edge_lengths = np.array([width, height, width, height]).repeat(2)
    edge_points = np.random.rand(boundary_points) * edge_lengths.sum()
    
    cumulative_lengths = np.cumsum(edge_lengths)
    edge_indices = np.searchsorted(cumulative_lengths, edge_points)
    
    t = (edge_points - np.concatenate(([0], cumulative_lengths[:-1]))[edge_indices]) / edge_lengths[edge_indices]
    start_points = edges[edge_indices % 4]
    end_points = edges[(edge_indices + 1) % 4]
    boundary = start_points + t[:, np.newaxis] * (end_points - start_points)
    inside = np.random.rand(inside_points, 2)
    inside[:, 0] = inside[:, 0] * width + left
    inside[:, 1] = inside[:, 1] * height + bottom
    # Combine boundary and inside points
    return np.concatenate([boundary, inside], axis=0)

class ObstacleField:
    def __init__(self, static_obstacle_centers, static_obstacle_sizes, dynamic_obstacle_fn, points_per_obstacle=32, distance_threshold=0.1,distance_threshold_pred=0.2):
        self.static_obstacle_centers = static_obstacle_centers
        self.static_obstacle_sizes = static_obstacle_sizes
        self.dynamic_obstacle_fn = dynamic_obstacle_fn
        self.points_per_obstacle = points_per_obstacle
        self.distance_threshold = distance_threshold
        self.distance_threshold_pred = distance_threshold_pred
        self.static_kdtree = None
        self.dynamic_kdtree = None
        self.last_update_time = None

        # Generate static obstacle points
        self.static_obstacle_points = np.vstack([
            generate_box_points(center, size, self.points_per_obstacle)
            for center, size in zip(static_obstacle_centers, static_obstacle_sizes)
        ])
        self.static_kdtree = cKDTree(self.static_obstacle_points)

    def update_dynamic(self, t, start_pos,replan_guide=False,best_idx=None):
        if self.last_update_time != t:
            # Get dynamic obstacle center and radius
            dynamic_center, dynamic_radius = self.dynamic_obstacle_fn(t, start_pos,replan_guide,best_idx)
            self.dynamic_center = dynamic_center
            # Generate points for dynamic obstacle
            dynamic_obstacle_points = generate_sphere_points(dynamic_center, dynamic_radius, self.points_per_obstacle)
            # Update KD-tree for dynamic obstacle
            self.dynamic_kdtree = cKDTree(dynamic_obstacle_points)
            self.last_update_time = t
    def query_static(self, points):
        return self.static_kdtree.query(points, distance_upper_bound=self.distance_threshold)

    def query_dynamic(self, points):
        if self.dynamic_kdtree is None:
            return np.inf * np.ones(len(points)), np.arange(len(points))
        return self.dynamic_kdtree.query(points, distance_upper_bound=self.distance_threshold_pred)

def avoidance(trajectory, obstacle_field, is_dynamic=False, avoidance_window=5, avoidance_strength=0.1, avoidance_strength_pred=0.3,affected_states=5, goal_state=None,stepp=None):
    if affected_states is None:
        affected_states=len(trajectory)
    if is_dynamic:
        distances, indices = obstacle_field.query_dynamic(trajectory[:affected_states, :2].cpu().numpy())
        avoidance_strength = avoidance_strength_pred  # Strength for predator
    else:
        distances, indices = obstacle_field.query_static(trajectory[:, :2].cpu().numpy())

    collision_index = np.argmin(distances)
    if is_dynamic:
        start_idx = 0 
        end_idx = min(affected_states, len(trajectory))
    else:
        start_idx = max(0, collision_index - avoidance_window)
        end_idx = min(len(trajectory) - 1, collision_index + avoidance_window)
    for idx in range(start_idx, end_idx):
        if indices[idx] < len(obstacle_field.static_obstacle_points if not is_dynamic else obstacle_field.dynamic_kdtree.data):
            nearest_obstacle = (obstacle_field.static_obstacle_points if not is_dynamic else obstacle_field.dynamic_kdtree.data)[indices[idx]]
            avoid_direction = trajectory[idx, :2] - torch.from_numpy(nearest_obstacle).to(trajectory.device)
            avoid_distance = torch.norm(avoid_direction)
            avoid_direction = avoid_direction / (avoid_distance + 1e-8)
            # include goal direction
            if goal_state is not None:
                goal_direction = goal_state[:2] - trajectory[idx, :2]
                goal_distance = torch.norm(goal_direction)
                goal_direction = goal_direction / (goal_distance + 1e-8)
                # Combine avoidance and goal directions
                combined_direction = 0.9*avoid_direction +0.1* goal_direction #0.65,0.35
                combined_direction = combined_direction / (torch.norm(combined_direction) + 1e-8)
            else:
                combined_direction = avoid_direction
            distances = torch.tensor(distances).to(trajectory.device)
            force = avoidance_strength * torch.exp(-distances[idx] / obstacle_field.distance_threshold)  
            trajectory[idx, :2] += force * combined_direction    
    return trajectory
