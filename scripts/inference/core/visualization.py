import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Rectangle, Circle
from typing import List, Optional, Tuple, Any
import torch
import os

class BaseVisualizer:
    """Base visualization class for trajectory plotting"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 10)):
        self.figsize = figsize
    
    def visualize_obstacles_and_trajectory(self, ax, box_centers: np.ndarray, box_size: np.ndarray,
                                         start_state: np.ndarray, goal_state: np.ndarray,
                                         trajectories: List[np.ndarray], obstacle_pts: np.ndarray,
                                         box_centers2: Optional[np.ndarray] = None,
                                         box_size2: Optional[np.ndarray] = None,
                                         obstacle_pts2: Optional[np.ndarray] = None):
        """Base trajectory visualization method"""
        def get_box_dimension(box_size_array, index):
            """Helper function to get box dimension, handling both 1D and 2D cases"""
            if len(box_size_array.shape) == 1:
                # 1D case: box_size[i]
                return box_size_array[index]
            else:
                # 2D case: box_size[i][0] 
                return box_size_array[index][0]
        
        
        # Configuration 1
        color1 = 'dimgray'
        # Plot the obstacles as boxes with point cloud for configuration 1
        for i, center in enumerate(box_centers):
            x, y = center
            box_dim = get_box_dimension(box_size, i)
            rect = Rectangle((x - box_dim/2, y - box_dim/2), box_dim, box_dim,
                        fill=True, facecolor='lightgray', edgecolor=color1, linewidth=1.5, alpha=0.7)
            # breakpoint()
            ax.add_patch(rect)
            # ax.scatter(obstacle_pts[i, :, 0], obstacle_pts[i, :, 1], color=color1, s=8, alpha=0.8)
  
        # Configuration 2 (if provided)
        if box_centers2 is not None and box_size2 is not None and obstacle_pts2 is not None:
            color2 = 'red'
            for i, center in enumerate(box_centers2):
                x, y = center
                rect = Rectangle((x - box_size2[i]/2, y - box_size2[i]/2), box_size2[i], box_size2[i], 
                                fill=True, facecolor='lightblue', edgecolor=color2, linewidth=1.5, alpha=0.5)
                ax.add_patch(rect)
                # ax.scatter(obstacle_pts2[i, :, 0], obstacle_pts2[i, :, 1], color=color2, s=8, alpha=0.6)
        
        # Plot start and goal states
        ax.scatter(start_state[0], start_state[1], color='green', s=150, zorder=5, label='Start')
        ax.scatter(goal_state[0], goal_state[1], color='purple', s=150, zorder=5, label='Goal')
        
        # Plot the trajectory
        # ax.plot(trajectory[:, 0], trajectory[:, 1], color='orange', linewidth=1.5, zorder=3, label='Trajectory')
        if trajectories is not None:
            for i, trajectory in enumerate(trajectories):
                ax.plot(trajectory[:, 0], trajectory[:, 1], color='orange', linewidth=1.5, zorder=3, 
                        label='Trajectory')
                ax.scatter(trajectory[:, 0], trajectory[:, 1], color='darkorange', s=15, zorder=4)
        # Set labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # ax.set_title('Robot Trajectory with Obstacles (Two Configurations)')
        # ax.legend()
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    
    def save_static_plot(self, box_centers: np.ndarray, box_size: np.ndarray,
                        start_state: np.ndarray, goal_state: np.ndarray,
                        trajectories: List[np.ndarray], obstacle_pts: np.ndarray,
                        save_path: str, **kwargs):
        """Save a static trajectory plot"""
        fig, ax = plt.subplots(figsize=self.figsize)
        self.visualize_obstacles_and_trajectory(
            ax, box_centers, box_size, start_state, goal_state,
            trajectories, obstacle_pts, **kwargs
        )
        fig.savefig(save_path)
        plt.close(fig)

class DynamicVisualizer(BaseVisualizer):
    """Extended visualizer for dynamic obstacle experiments"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 10)):
        super().__init__(figsize)
    
    def create_animation(self, 
                        box_centers: torch.Tensor, 
                        box_size: torch.Tensor,
                        start_state_pos: torch.Tensor, 
                        goal_state_pos: torch.Tensor,
                        pos_trajs_iters: torch.Tensor, 
                        obstacle_pts: torch.Tensor,
                        trajs: torch.Tensor,
                        chain_start: List, 
                        chain_obs: List,
                        pursuer_radius: float, 
                        distance_threshold: float,
                        context_idx: int,
                        result_dir: str = 'dynamic_results',
                        format: str = 'gif', 
                        interval: int = 350):
        """Create animation for dynamic experiments"""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert tensors to numpy arrays
        box_centers_np = box_centers.cpu().numpy()
        box_size_np = box_size.cpu().numpy()
        obstacle_pts_np = obstacle_pts.cpu().numpy()
        start_state_np = start_state_pos.cpu().numpy()
        goal_state_np = goal_state_pos.cpu().numpy()
        
        # Prepare evader state trajectory
        evader_state = trajs.squeeze(0)
        
        def animate(frame_index):
            ax.clear()
            
            # Basic visualization using the unified function
            self.visualize_obstacles_and_trajectory(
                ax, 
                box_centers_np, 
                box_size_np,
                start_state_np,
                goal_state_np,
                None, #[pos_trajs_iters[frame_index].cpu().numpy()[0]],  # Convert to list for trajectories
                obstacle_pts_np
            )
            
            # Current evader position
            current_pos = evader_state[frame_index].cpu().numpy()
            ax.plot(current_pos[0], current_pos[1], 
                   marker='*', 
                   markersize=11.5,
                   color='lime',
                   linestyle='none',
                   zorder=5)
            
            # Show evader trajectory history
            if frame_index >= pos_trajs_iters.shape[0] - len(chain_start):
                start_idx = frame_index - (pos_trajs_iters.shape[0] - len(chain_start))
                # current_start = chain_start[start_idx]
                # ax.plot(current_start[0, 0], current_start[0, 1], 'o',
                #        markersize=3.25, color='green')
                trajectory_x = []
                trajectory_y = []
                for prev_idx in range(start_idx):
                    prev_start = chain_start[prev_idx]
                    x_hist, y_hist = prev_start[0, 0], prev_start[0, 1]
                    trajectory_x.append(x_hist)
                    trajectory_y.append(y_hist)
                    ax.plot(prev_start[0, 0], prev_start[0, 1], 'o',
                           markersize=3.5, color='navy')
                current_start = chain_start[start_idx]
                # ax.plot(current_start[0, 0], current_start[0, 1], 'o',
                #        markersize=3.25, color='green')
                x, y = current_start[0, 0], current_start[0, 1]
                trajectory_x.append(x)
                trajectory_y.append(y)
                ax.plot(x, y, 'o', markersize=3.25, color='green')  
                        # Draw connecting line
                if len(trajectory_x) > 1:
                    ax.plot(trajectory_x, trajectory_y, '-', 
                        color='lightgreen', linewidth=2.0, alpha=0.7, zorder=3)  
            # Show pursuer dynamics
            if frame_index >= pos_trajs_iters.shape[0] - len(chain_obs):
                obs_idx = frame_index - (pos_trajs_iters.shape[0] - len(chain_obs))
                current_obstacle_pos = chain_obs[obs_idx]
                
                # Add pursuer circle
                ax.add_patch(Circle((current_obstacle_pos[0, 0], current_obstacle_pos[0, 1]), 
                                  pursuer_radius, color='red'))
                
                # Add potential field visualization
                self._add_potential_field(ax, current_obstacle_pos, distance_threshold,pursuer_radius)
                
                # Show pursuer history
                for prev_idx in range(obs_idx):
                    prev_obstacle_pos = chain_obs[prev_idx]
                    ax.plot(prev_obstacle_pos[0, 0], prev_obstacle_pos[0, 1], 'o', 
                           markersize=3, color='peachpuff')
            
            # Add initial trajectory samples
            # plt.scatter(pos_trajs_iters[0, :, :, 0].cpu(), pos_trajs_iters[0, :, :, 1].cpu(), 
                    #    s=4, color='salmon')
            ax.set_title(f"t: {frame_index}")
        
        # Create animation
        ani = animation.FuncAnimation(fig, animate, frames=len(pos_trajs_iters), 
                                    interval=interval, blit=False)
        
        # Save animation
        if format == 'gif':
            save_path = os.path.join(result_dir, f'context_{context_idx}.gif')
            ani.save(save_path, writer='pillow')
        elif format == 'mp4':
            save_path = os.path.join(result_dir, f'context_{context_idx}.mp4')
            ani.save(save_path, writer='ffmpeg')
        
        plt.close(fig)
        return ani, save_path
    
    def _add_potential_field(self, ax, obstacle_pos: np.ndarray, distance_threshold: float,pursuer_radius: float):
        """Add potential field visualization around pursuer"""
        x = np.linspace(obstacle_pos[0, 0] - distance_threshold, 
                       obstacle_pos[0, 0] + distance_threshold, 100)
        y = np.linspace(obstacle_pos[0, 1] - distance_threshold, 
                       obstacle_pos[0, 1] + distance_threshold, 100)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distances from obstacle
        distances = np.sqrt((X - obstacle_pos[0, 0])**2 + (Y - obstacle_pos[0, 1])**2)
        
        # Create potential field
        Z = np.exp(-distances / distance_threshold)
        mask = distances <= distance_threshold
        Z = np.ma.masked_where(~mask, Z)
        
        if hasattr(ax, '_potential_field_mesh'):
            ax._potential_field_mesh.remove()
        ax._potential_field_mesh = ax.pcolormesh(X, Y, Z, cmap='YlOrRd', alpha=0.1, shading='auto')
        
        # Add boundary circle
        circle = Circle((obstacle_pos[0, 0], obstacle_pos[0, 1]), 
                       distance_threshold, fill=True, color='peachpuff', linestyle='--')
        ax.add_artist(circle)
        ax.add_patch(plt.Circle((obstacle_pos[0, 0], obstacle_pos[0, 1]), pursuer_radius, color='red'))