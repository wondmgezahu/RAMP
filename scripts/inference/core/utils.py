import os
import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any

class StateGenerator:

    @staticmethod
    def get_hard_cond_custom(traj: torch.Tensor, horizon: int, include_velocity: bool = True): # -> Dict[int, torch.Tensor]:
        """Generate hard conditions (start/goal states) for trajectory optimization"""
        start_state_pos = traj[0]
        goal_state_pos = traj[-1]
        
        if include_velocity:
            # Set velocities to zero at beginning and end
            start_state = torch.cat((start_state_pos, torch.zeros_like(start_state_pos)), dim=-1)
            goal_state = torch.cat((goal_state_pos, torch.zeros_like(goal_state_pos)), dim=-1)
        else:
            start_state = start_state_pos
            goal_state = goal_state_pos

        hard_conds = {
            0: start_state,
            horizon - 1: goal_state
        }
        return hard_conds

class ContextManager:
    """Utility class for saving/loading experiment contexts"""
    
    @staticmethod
    def save_context(start_state_pos: torch.Tensor, goal_state_pos: torch.Tensor,
                    env_dir: str, dataset_id: str, context_idx: int) -> str:
        """Save start and goal states for a specific context"""
        try:
            # Validate inputs
            if not torch.is_tensor(start_state_pos) or not torch.is_tensor(goal_state_pos):
                raise ValueError("Start and goal positions must be torch tensors")
                
            if not isinstance(context_idx, int) or context_idx < 0:
                raise ValueError(f"Invalid context_idx: {context_idx}")
                
            # Create directory structure
            contexts_dir = os.path.join(env_dir, 'contexts')
            os.makedirs(contexts_dir, exist_ok=True)
            
            # Prepare context data
            context_data = {
                'start_pos': start_state_pos.cpu(),
                'goal_pos': goal_state_pos.cpu(),
                'metadata': {
                    'context_idx': context_idx,
                    'dataset_id': dataset_id
                }
            }
            
            # Save context
            context_filename = f'context_{context_idx:03d}.pt'
            context_path = os.path.join(contexts_dir, context_filename)
            
            torch.save(context_data, context_path)
            print(f"Successfully saved context {context_idx} to {context_path}")
            
            return context_path
            
        except Exception as e:
            print(f"Error saving context: {str(e)}")
            raise

    @staticmethod
    def load_context(contexts_dir: str, context_idx: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """Load start and goal states for a specific context"""
        context_path = os.path.join(contexts_dir, f'context_{context_idx:03d}.pt')
        context_config = torch.load(context_path, map_location=device)
        
        start_state_pos = context_config['start_pos'].to(device)
        goal_state_pos = context_config['goal_pos'].to(device)
        
        return start_state_pos, goal_state_pos

class DynamicsGenerator:
    """Generate dynamics functions for different scenarios"""
    
    @staticmethod
    def create_pursuit_dynamics(velocity_max: float = 0.5, 
                              pursuit_strength: float = 0.8,
                              random_strength: float = 0.2):
        """Create pursuit dynamics function for dynamic obstacles"""
        velocity = np.array([[velocity_max/np.sqrt(2), velocity_max/np.sqrt(2)]])
        
        def dynamics_fn(t, prev_center, robot_position, velocity_input):
            """
            Simulate the motion of a predator (obstacle) pursuing a prey (robot).
            """
            dt = 0.1
            
            def pursuit_dynamics(x, y, robot_x, robot_y):
                dx = robot_x - x
                dy = robot_y - y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 0:
                    dx /= distance
                    dy /= distance
                
                return dx, dy
            
            def random_motion(t):
                dx = np.sin(2*np.pi*t)
                dy = np.cos(2*np.pi*t)
                return dx, dy
            
            x, y = prev_center[0]
            vx, vy = velocity_input[0]
            robot_x, robot_y = robot_position[0]
            
            # Calculate pursuit direction
            dx_pursuit, dy_pursuit = pursuit_dynamics(x, y, robot_x, robot_y)
            
            # Calculate random motion
            dx_random, dy_random = random_motion(t)
            
            # Combine pursuit and random motion
            dx = pursuit_strength * dx_pursuit + random_strength * dx_random
            dy = pursuit_strength * dy_pursuit + random_strength * dy_random
            
            # Update positions
            new_x = x + dx * vx * dt
            new_y = y + dy * vy * dt
            
            # Limit to range [-1, 1]
            new_x = np.clip(new_x, -1, 1)
            new_y = np.clip(new_y, -1, 1)
            
            new_center = np.array([[new_x, new_y]])
            return new_center
        
        return dynamics_fn, velocity