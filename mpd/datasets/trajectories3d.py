import abc
import os.path
import numpy as np
import torch
from torch.utils.data import Dataset
from mpd.datasets.normalization import DatasetNormalizer
from mpd.utils.loading import load_params_from_yaml
from deps.torch_robotics.torch_robotics import environments, robots

class TrajectoryDatasetBase(Dataset, abc.ABC):

    def __init__(self,
                 dataset_subdir=None,
                 include_velocity=False,
                 normalizer='LimitsNormalizer',
                 use_extra_objects=False,
                 obstacle_cutoff_margin=None,
                 tensor_args=None,
                 static=True,
                 training=True,
                 base_dataset_dir='/home/wondm/RAMP/dataset',
                 **kwargs):

        self.tensor_args = tensor_args
        self.dataset_subdir = dataset_subdir
        self.base_dir = os.path.join(base_dataset_dir, self.dataset_subdir)
        self.args = load_params_from_yaml(os.path.join(self.base_dir, '0', 'args.yaml'))
        self.metadata = load_params_from_yaml(os.path.join(self.base_dir, '0', 'metadata.yaml'))
   
        # -------------------------------- Load env, robot, task ---------------------------------
                # load data
        self.include_velocity = include_velocity
        self.map_task_id_to_trajectories_id = {}
        self.map_trajectory_id_to_task_id = {}

        # -------------------------------- Load trajectories ---------------------------------
        self.threshold_start_goal_pos = self.args['threshold_start_goal_pos']

        self.field_key_traj = 'traj'
        self.field_key_task = 'task'
        self.fields = {}
        self.use_extra_objects=use_extra_objects
        self.load_basic_data()
        env_class = getattr(environments, self.metadata['env_id'] + 'Obstacles' if self.use_extra_objects else self.metadata['env_id'])
        # breakpoint()
        self.env = env_class(box_centers=self.box_centers[0],box_sizes=self.box_sizes[0],sphere_centers=self.sphere_centers[0],
                 sphere_radii=self.sphere_radi[0],tensor_args=tensor_args)

        # Robot
        robot_class = getattr(robots, self.metadata['robot_id'])
        self.robot = robot_class(tensor_args=tensor_args)
        self.process_loaded_data()
        # Task
        self.min_x=self.fields['traj'].min()
        self.max_x=self.fields['traj'].max()
        b, h, d = self.dataset_shape = self.fields[self.field_key_traj].shape
        self.n_trajs = b
        self.n_support_points = h
        self.state_dim = d  # state dimension used for the diffusion model
        self.trajectory_dim = (self.n_support_points, d)
        self.normalizer = DatasetNormalizer(self.fields, normalizer=normalizer)
        self.normalizer_keys = [self.field_key_traj, self.field_key_task]
        self.normalize_all_data(*self.normalizer_keys) # commented 

    def load_basic_data(self):
        trajs_free_l = []
        obstacle_points_l = []
        box_centers_l = []
        box_sizes_l = []
        sphere_centers_l = []
        sphere_radi_l = []
        task_id = 0
        n_trajs = 0

        for current_dir, subdirs, files in os.walk(self.base_dir, topdown=True):
            if 'trajs-free.pt' in files and 'obstacle_points.pt' in files and 'metadata.yaml' in files:
                trajs_free_tmp = torch.load(os.path.join(current_dir, 'trajs-free.pt'), map_location=self.tensor_args['device'])
                obstacle_points_tmp = torch.load(os.path.join(current_dir, 'obstacle_points.pt'), map_location=self.tensor_args['device'])
                # Load box sizes from metadata
                metadata = load_params_from_yaml(os.path.join(current_dir, 'metadata.yaml'))
                box_sizes_tmp = torch.tensor(metadata['box_sizes'], **self.tensor_args)
                sphere_radi_tmp = torch.tensor(metadata['sphere_radii'], **self.tensor_args)
                box_centers_tmp = torch.tensor(metadata['box_centers'], **self.tensor_args)
                sphere_centers_tmp = torch.tensor(metadata['sphere_centers'], **self.tensor_args)
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
                sphere_centers_l.extend([sphere_centers_tmp] * num_trajs_in_dir)
                sphere_radi_l.extend([sphere_radi_tmp] * num_trajs_in_dir)
                
        self.trajs_free = torch.cat(trajs_free_l)
        self.obstacle_points = torch.stack(obstacle_points_l)
        self.box_centers = torch.stack(box_centers_l)
        self.box_sizes = torch.stack(box_sizes_l)
        self.sphere_centers = torch.stack(sphere_centers_l)
        self.sphere_radi = torch.stack(sphere_radi_l)

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
        sphere_centers=self.sphere_centers[index]
        sphere_radi=self.sphere_radi[index]

        data = {
            field_traj_normalized: traj_normalized,
            field_task_normalized: task_normalized,
            'obstacle_points': obs_points,
            'box_centers': box_centers,
            'box_sizes' :box_sizes,
            'sphere_centers':sphere_centers,
            'sphere_radii':sphere_radi
        }
        hard_conds = self.get_hard_conditions(traj_normalized, horizon=len(traj_normalized))
        data.update({'hard_conds': hard_conds})
        return data

    def get_hard_conditions(self, traj, horizon=None, normalize=False):
        raise NotImplementedError
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


class TrajectoryDataset3d(TrajectoryDatasetBase):

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
