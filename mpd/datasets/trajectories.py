import abc
import os.path
import git
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from mpd.datasets.normalization import DatasetNormalizer
from mpd.utils.loading import load_params_from_yaml
# from torch_robotics import environments, robots
from deps.torch_robotics.torch_robotics import environments, robots
# from torch_robotics.tasks.tasks import PlanningTask

# repo = git.Repo('.', search_parent_directories=True)
# datase=t_base_dir = os.path.join(repo.working_dir, 'dataset')
# dataset_base_dir
def safe_load_yaml(file_path):
    class CustomSafeLoader(yaml.SafeLoader):
        def ignore_unknown(self, node):
            return None

    CustomSafeLoader.add_constructor(None, CustomSafeLoader.ignore_unknown)

    with open(file_path, 'r') as file:
        try:
            return yaml.load(file, Loader=CustomSafeLoader)
        except yaml.YAMLError as e:
            print(f"Error loading YAML file {file_path}: {e}")
            return None

class TrajectoryDatasetBase(Dataset, abc.ABC):

    def __init__(self,
                 dataset_subdir=None,
                 include_velocity=False,
                 normalizer='LimitsNormalizer',
                 use_extra_objects=False,
                 obstacle_cutoff_margin=None,
                 tensor_args=None,
                 dynamics_fn=None, 
                 velocity=None,
                 training=False,
                 static=True,
                 dataset_base_dir='/home/wondm/RAMP/dataset',
                 **kwargs):

        self.tensor_args = tensor_args
        self.dataset_subdir = dataset_subdir
        self.base_dir = os.path.join(dataset_base_dir, self.dataset_subdir)
        # self.args = load_params_from_yaml(os.path.join(self.base_dir, '0', 'args.yaml'))
        # self.metadata = load_params_from_yaml(os.path.join(self.base_dir, '0', 'metadata.yaml'))
        
        self.training=training
        self.static = static
        
        if self.static:
            self.metadata = load_params_from_yaml(os.path.join(self.base_dir, '0', 'metadata.yaml'))
        else:
            self.args = safe_load_yaml(os.path.join(self.base_dir, '0', 'args.yaml'))
            self.metadata = safe_load_yaml(os.path.join(self.base_dir, '0', 'metadata.yaml'))
        
        # breakpoint()
        # -------------------------------- Load env, robot, task ---------------------------------
        # Environment

        self.include_velocity = include_velocity
        self.map_task_id_to_trajectories_id = {}
        self.map_trajectory_id_to_task_id = {}

        # -------------------------------- Load trajectories ---------------------------------
        
        self.field_key_traj = 'traj'
        self.field_key_task = 'task'
        self.fields = {}
        self.use_extra_objects=use_extra_objects
        self.dynamics_fn=dynamics_fn
        self.velocity=velocity
        self.load_basic_data()

        env_class = getattr(environments, self.metadata['env_id'] + 'Obstacles' if self.use_extra_objects else self.metadata['env_id'])
        # self.env = env_class(box_centers=self.box_centers[0],box_sizes=self.box_sizes[0],tensor_args=tensor_args, dynamics_fn=self.dynamics_fn, velocity=self.velocity)
        env_kwargs = {
            'box_centers': self.box_centers[0],
            'box_sizes': self.box_sizes[0],
            'tensor_args': tensor_args, 
            'dynamics_fn': self.dynamics_fn, 
            'velocity': self.velocity
        }
        # Add pursuer_pos only for EnvPredatorObstacles
        if 'Predator' in self.metadata['env_id'] and 'pursuer_pos' in kwargs:
            env_kwargs['pursuer_pos'] = kwargs['pursuer_pos']

        self.env = env_class(**env_kwargs)

        # Robot
        robot_class = getattr(robots, self.metadata['robot_id'])
        self.robot = robot_class(tensor_args=tensor_args)
        self.process_loaded_data()
        # Task
        # self.task = PlanningTask(env=self.env, robot=self.robot, tensor_args=tensor_args, **self.args)
     
        self.min_x=self.fields['traj'].min()
        self.max_x=self.fields['traj'].max()
        b, h, d = self.dataset_shape = self.fields[self.field_key_traj].shape
        self.n_trajs = b
        self.n_support_points = h
        self.state_dim = d  # state dimension used for the diffusion model
        self.trajectory_dim = (self.n_support_points, d)
        self.normalizer = DatasetNormalizer(self.fields, normalizer=normalizer)
        self.normalizer_keys = [self.field_key_traj, self.field_key_task]
        # breakpoint()
        self.normalize_all_data(*self.normalizer_keys) # commented 

    def load_basic_data(self):
        trajs_free_l = []
        obstacle_points_l = []
        box_centers_l = []
        box_sizes_l = []
        task_id = 0
        n_trajs = 0

        for current_dir, subdirs, files in os.walk(self.base_dir, topdown=True):
            if self.static:
                required_files = ['trajs-free.pt', 'obstacle_points.pt', 'box_centers.npy']
            else:
                required_files = ['trajs-free.pt', 'obstacle_pointsORG.pt', 'obstacle_config.npy']           
            # if 'trajs-free.pt' in files and 'obstacle_points.pt' in files and 'box_centers.npy' in files:
            if all(file in files for file in required_files):
                trajs_free_tmp = torch.load(os.path.join(current_dir, 'trajs-free.pt'), map_location=self.tensor_args['device'])
                
                if self.static:
                    # Static mode - load from separate files
                    obstacle_points_tmp = torch.load(os.path.join(current_dir, 'obstacle_points.pt'), 
                                                   map_location=self.tensor_args['device'])
                    box_centers_tmp = torch.from_numpy(np.load(os.path.join(current_dir, 'box_centers.npy'))).to(**self.tensor_args)
                    
                    # Load box sizes from metadata
                    metadata = load_params_from_yaml(os.path.join(current_dir, 'metadata.yaml'))
                    box_sizes_tmp = torch.tensor(metadata['box_sizes'], **self.tensor_args)
                else:
                    # Dynamic mode - load from obstacle config
                    obstacle_points_tmp = torch.load(os.path.join(current_dir, 'obstacle_pointsORG.pt'), 
                                                   map_location=self.tensor_args['device'])
                    
                    # Load obstacle configuration
                    obstacle_config = np.load(os.path.join(current_dir, 'obstacle_config.npy'), allow_pickle=True)
                    
                    # Extract box and sphere data
                    box_centers = []
                    box_sizes = []
                    for obstacle in obstacle_config:
                        if obstacle['type'] == 'box':
                            box_centers.append(obstacle['center'])
                            box_sizes.append(obstacle['size'])
        
                    box_centers_tmp = torch.tensor(np.array(box_centers), **self.tensor_args)
                    box_sizes_tmp = torch.tensor(np.array(box_sizes), **self.tensor_args)    

                num_trajs_in_dir = trajs_free_tmp.shape[0]                
                trajectories_idx = n_trajs + np.arange(num_trajs_in_dir)
                self.map_task_id_to_trajectories_id[task_id] = trajectories_idx
                for j in trajectories_idx:
                    self.map_trajectory_id_to_task_id[j] = task_id
                task_id += 1
                n_trajs += num_trajs_in_dir
                trajs_free_l.append(trajs_free_tmp)
                obstacle_points_l.extend([obstacle_points_tmp] * num_trajs_in_dir)
                box_centers_l.extend([box_centers_tmp] * num_trajs_in_dir)
                box_sizes_l.extend([box_sizes_tmp] * num_trajs_in_dir)

        self.trajs_free = torch.cat(trajs_free_l)
        self.obstacle_points = torch.stack(obstacle_points_l)
        self.box_centers = torch.stack(box_centers_l)
        self.box_sizes = torch.stack(box_sizes_l)
        self.n_trajs = n_trajs
        
    def process_loaded_data(self):
        trajs_free_pos = self.robot.get_position(self.trajs_free)
        
        if self.include_velocity:
            trajs = self.trajs_free
        else:
            trajs = trajs_free_pos
        
        self.fields[self.field_key_traj] = trajs
        self.fields[self.field_key_task] = torch.cat((trajs_free_pos[..., 0, :], trajs_free_pos[..., -1, :]), dim=-1)
        self.fields['box_centers'] = self.box_centers

        print(f"Loaded {self.n_trajs} trajectories with corresponding obstacle points and box centers")
        assert len(self.fields[self.field_key_traj]) == len(self.obstacle_points) == len(self.box_centers), \
            "Mismatch in the number of trajectories, obstacle points, and box centers"

    def normalize_all_data(self, *keys):
        for key in keys:
            self.fields[f'{key}_normalized'] = self.normalizer(self.fields[f'{key}'], key)

    
    def __repr__(self):
        msg = f'TrajectoryDataset\n' \
              f'n_trajs: {self.n_trajs}\n' \
              f'trajectory_dim: {self.trajectory_dim}\n'
        return msg

    def __len__(self):
        return self.n_trajs

    def __getitem__(self, index):
        # Generates one sample of data - one trajectory and tasks
        field_traj_normalized = f'{self.field_key_traj}_normalized'
        field_task_normalized = f'{self.field_key_task}_normalized'
        traj_normalized = self.fields[field_traj_normalized][index]
        task_normalized = self.fields[field_task_normalized][index]
        
        obs_points = self.obstacle_points[index]
        box_centers = self.box_centers[index]
        box_sizes=self.box_sizes[index]

        if self.training:
            data = {
                field_traj_normalized: traj_normalized,
                field_task_normalized: task_normalized,
                'obstacle_points': obs_points,
                'box_centers': box_centers,
                # 'task': task,
                # 'box_sizes' :box_sizes
            }
        else:    
            data={
                field_traj_normalized: traj_normalized,
                field_task_normalized: task_normalized,
                'obstacle_points': obs_points,
                'box_centers': box_centers,
                'box_sizes' :box_sizes
            }
        if not self.static:
            hard_conds = self.get_hard_conditions(traj_normalized, horizon=len(traj_normalized))
            data.update({'hard_conds': hard_conds})
        return data

    def get_hard_conditions(self, traj, horizon=None, normalize=False):
        raise NotImplementedError

    def get_unnormalized(self, index):
        raise NotImplementedError
        traj = self.fields[self.field_key_traj][index][..., :self.state_dim]
        task = self.fields[self.field_key_task][index]
        if not self.include_velocity:
            task = task[self.task_idxs]
        data = {self.field_key_traj: traj,
                self.field_key_task: task,
                }
        if self.variable_environment:
            data.update({self.field_key_env: self.fields[self.field_key_env][index]})

        # hard conditions
        # hard_conds = self.get_hard_conds(tasks)
        hard_conds = self.get_hard_conditions(traj)
        data.update({'hard_conds': hard_conds})

        return data

    def unnormalize(self, x, key):
        return self.normalizer.unnormalize(x, key)

    def normalize(self, x, key):
        return self.normalizer.normalize(x, key)

    def unnormalize_trajectories(self, x):
        return self.unnormalize(x, self.field_key_traj)

    def normalize_trajectories(self, x):
        return self.normalize(x, self.field_key_traj)

    def unnormalize_tasks(self, x):
        return self.unnormalize(x, self.field_key_task)

    def normalize_tasks(self, x):
        return self.normalize(x, self.field_key_task)


class TrajectoryDataset(TrajectoryDatasetBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_hard_conditions(self, traj, horizon=None, normalize=False):
        # start and goal positions
        start_state_pos = self.robot.get_position(traj[0])
        goal_state_pos = self.robot.get_position(traj[-1])

        if self.include_velocity:
            # If velocities are part of the state, then set them to zero at the beggining and end of a trajectory
            start_state = torch.cat((start_state_pos, torch.zeros_like(start_state_pos)), dim=-1)
            goal_state = torch.cat((goal_state_pos, torch.zeros_like(goal_state_pos)), dim=-1)
        else:
            start_state = start_state_pos
            goal_state = goal_state_pos

        if normalize:
            start_state = self.normalizer.normalize(start_state, key=self.field_key_traj)
            goal_state = self.normalizer.normalize(goal_state, key=self.field_key_traj)

        if horizon is None:
            horizon = self.n_support_points
        hard_conds = {
            0: start_state,
            horizon - 1: goal_state
        }
        return hard_conds

    
class ExpDirectoryDataset(TrajectoryDatasetBase):
    def __init__(self, current_dir_id=0, **kwargs):
        self.specified_dir_id = int(current_dir_id)
        super().__init__(**kwargs)
    
    def load_basic_data(self):
        """Load environment data from the specific directory"""
        # Construct path to the specified directory
        current_dir = os.path.join(self.base_dir, str(self.specified_dir_id))
        print(f"\nLoading environment data from directory: {current_dir}")
        
        # Load obstacle points
        obstacle_points_tmp = torch.load(
            os.path.join(current_dir, 'obstacle_points.pt'),
            map_location=self.tensor_args['device']
        )
        
        # Load box centers
        box_centers_tmp = torch.from_numpy(
            np.load(os.path.join(current_dir, 'box_centers.npy'))
        ).to(**self.tensor_args)
        
        # Load box sizes from metadata
        metadata = load_params_from_yaml(os.path.join(current_dir, 'metadata.yaml'))
        box_sizes_tmp = torch.tensor(metadata['box_sizes'], **self.tensor_args)
        
        # We still need trajs_free for dimension purposes, but it won't be used
        trajs_free_tmp = torch.load(
            os.path.join(current_dir, 'trajs-free.pt'),
            map_location=self.tensor_args['device']
        )
        
        # Store the environment data
        num_trajs_in_dir = trajs_free_tmp.shape[0]
        self.trajs_free = trajs_free_tmp
        self.obstacle_points = obstacle_points_tmp.unsqueeze(0).repeat(num_trajs_in_dir, 1, 1, 1)
        self.box_centers = box_centers_tmp.unsqueeze(0).repeat(num_trajs_in_dir, 1, 1)
        self.box_sizes = box_sizes_tmp.unsqueeze(0).repeat(num_trajs_in_dir, 1, 1)
        self.n_trajs = num_trajs_in_dir        
        print(f"Loaded environment data from directory {self.specified_dir_id}")
        
    def get_random_traj_from_current_dir(self):
        """Simple function to return current directory info"""
        return 0, self.specified_dir_id  # Always return first trajectory index