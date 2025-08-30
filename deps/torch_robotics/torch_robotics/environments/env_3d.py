import numpy as np
import torch
from deps.torch_robotics.torch_robotics.environments.env_base import EnvBase
from deps.torch_robotics.torch_robotics.environments.primitives import ObjectField, MultiBoxField, MultiSphereField
from deps.torch_robotics.torch_robotics.robots import RobotPointMass3D

class EnvParameterized3D(EnvBase):
    def __init__(self,
                 box_centers,
                 box_sizes,
                 sphere_centers,
                 sphere_radii,
                 points_per_obstacle=128,
                 name='EnvParameterized3D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=False,
                 sdf_cell_size=0.005,
                 **kwargs):
        
        self.box_centers = box_centers
        self.box_sizes=box_sizes
        self.sphere_centers=sphere_centers
        self.sphere_radii=sphere_radii
        self.tensor_args=tensor_args

        super().__init__(
            name=name,
            limits=torch.tensor([[-1,-1,-1], [1,1, 1]], **tensor_args),
            # obj_fixed_list=[ObjectField(obj_list, 'parameterizedmix')],
            obj_fixed_list=[self.create_object_field()],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )
    def create_object_field(self):
        if isinstance(self.box_centers, torch.Tensor):
            box_centers_np = self.box_centers.cpu().numpy()
        else:
            box_centers_np = np.array(self.box_centers)
        
        # Convert box_sizes to numpy if it's a tensor
        if isinstance(self.box_sizes, torch.Tensor):
            box_sizes_np = self.box_sizes.cpu().numpy()
        else:
            box_sizes_np = np.array(self.box_sizes)
        
        # Convert sphere_centers to numpy if it's a tensor
        if isinstance(self.sphere_centers, torch.Tensor):
            sphere_centers_np = self.sphere_centers.cpu().numpy()
        else:
            sphere_centers_np = np.array(self.sphere_centers)
        if isinstance(self.sphere_radii, torch.Tensor):
            sphere_radii_np = self.sphere_radii.cpu().numpy()
        else:
            sphere_radii_np = np.array(self.sphere_radii)
        
        obj_list = [
            MultiBoxField(
            box_centers_np,
            box_sizes_np,
            tensor_args=self.tensor_args
            ),
            MultiSphereField(
            sphere_centers_np,
            sphere_radii_np,
            tensor_args=self.tensor_args
        )
    ]
        return ObjectField(obj_list, 'parameterized3d')
    
    def update_box_centers(self, new_box_centers,new_sphere_centers):
        self.box_centers = new_box_centers
        self.sphere_centers=new_sphere_centers
        self.obj_fixed_list = [self.create_object_field()]

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=0.05,
            n_radius=0.1,
            n_pre_samples=50000,
            max_time=60
        )

        if isinstance(robot, RobotPointMass3D):
            return params
        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            n_interpolated_points=None,
            dt=0.1,
            opt_iters=500, # 100
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )

        if isinstance(robot, RobotPointMass3D):
            return params
        else:
            raise NotImplementedError

    def get_chomp_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.1,
            opt_iters=1,  # Keep this 1 for visualization
            weight_prior_cost=1e-4,
            step_size=0.05,
            grad_clip=0.05,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=0.3,
            pos_only=False,
        )

        if isinstance(robot, RobotPointMass3D):
            return params
        else:
            raise NotImplementedError