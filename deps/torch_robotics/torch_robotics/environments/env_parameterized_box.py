import numpy as np
import torch
from matplotlib import pyplot as plt

from deps.torch_robotics.torch_robotics.environments.env_base import EnvBase
from deps.torch_robotics.torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from deps.torch_robotics.torch_robotics.environments.utils import create_grid_spheres
from deps.torch_robotics.torch_robotics.robots import RobotPointMass
from deps.torch_robotics.torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from deps.torch_robotics.torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvParameterizedBox(EnvBase):

    def __init__(self,
                 box_centers,
                 box_sizes,
                 points_per_box=64,
                 name='EnvParameterizedBox',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.005,
                 **kwargs
                 ):
        
        self.box_centers = box_centers
        self.box_sizes = box_sizes
        self.points_per_box = points_per_box
        self.tensor_args = tensor_args

        
        # box_centers= np.random.uniform(-0.8, 0.8, (num_boxes, 2))
        # print(f'center_max{box_centers.max()},center_min{box_centers.min()}')
        # box_sizes= np.full((num_boxes, 2), box_size)
        # obj_list = [
        #     MultiBoxField(
        #         np.array(box_centers
        #         ),
        #         np.array(box_sizes
        #         )
        #         ,
        #         tensor_args=tensor_args
        #         )
        # ]

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environments limits
            obj_fixed_list=[self.create_object_field()], #[ObjectField(obj_list, 'parameterizedbox')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )
        # self.obstacle_points = self.generate_obstacle_points(box_centers, box_sizes, points_per_box)
    def create_object_field(self):
        obj_list = [
            MultiBoxField(
                np.array(self.box_centers.cpu().numpy()),
                np.array(self.box_sizes.cpu().numpy()),
                tensor_args=self.tensor_args
            )
        ]
        return ObjectField(obj_list, 'parameterizedbox')
    def update_box_centers(self, new_box_centers):
        self.box_centers = new_box_centers
        self.obj_fixed_list = [self.create_object_field()]
        # self.update_obstacle_points()

    def generate_obstacle_points(self, centers, sizes, points_per_box):
        all_points = []
        # breakpoint()
        for center, size in zip(centers, sizes):
            box_points = self.generate_box_points(center, size, points_per_box)
            # breakpoint()
            all_points.append(box_points.unsqueeze(0))
        return torch.cat(all_points, dim=0) 

    def generate_box_points(self,center,size,num_points):
        
        center_x,center_y=center
        width,height=size

        left = center_x - width / 2
        right = center_x + width / 2
        top = center_y + height / 2
        bottom = center_y - height / 2

        # Randomly decide the number of boundary points (between 1/3 and 2/3 of total points)
        boundary_points = torch.randint(num_points // 3, 2 * num_points // 3 + 1, (1,)).item()
        inside_points = num_points - boundary_points

        # Generate boundary points
        edges = torch.tensor([[left, top], [right, top], [right, bottom], [left, bottom]])
        edge_lengths = torch.tensor([width, height, width, height]).repeat(2)
        edge_points = torch.rand(boundary_points) * edge_lengths.sum()
        
        cumulative_lengths = torch.cumsum(edge_lengths, 0)
        edge_indices = torch.searchsorted(cumulative_lengths, edge_points)
        
        t = (edge_points - torch.cat([torch.tensor([0.]), cumulative_lengths[:-1]])[edge_indices]) / edge_lengths[edge_indices]
        start_points = edges[edge_indices % 4]
        end_points = edges[(edge_indices + 1) % 4]
        boundary = start_points + t.unsqueeze(1) * (end_points - start_points)

        # Generate inside points
        inside = torch.rand(inside_points, 2)
        inside[:, 0] = inside[:, 0] * width + left
        inside[:, 1] = inside[:, 1] * height + bottom

        # Combine boundary and inside points
        obs_points = torch.cat([boundary, inside], dim=0)
        return obs_points
    
    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=0.01,
            n_radius=0.3,
            n_pre_samples=50000,
            max_time=50
        )

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            n_interpolated_points=None,
            dt=0.04,
            opt_iters=300,
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

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError

    def get_chomp_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=1,  # Keep this 1 for visualization
            weight_prior_cost=1e-4,
            step_size=0.05,
            grad_clip=0.05,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=0.3,
            pos_only=False,
        )

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError


if __name__ == '__main__':
    env = EnvCompBox1(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.grid()
    plt.show()
    plt.savefig('comp.png')

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    # plt.show()
    # plt.savefig('comp.png')

