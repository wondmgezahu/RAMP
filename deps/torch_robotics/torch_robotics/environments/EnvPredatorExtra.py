import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd.functional import jacobian

from deps.torch_robotics.torch_robotics.environments.EnvPredator import EnvPredator
from deps.torch_robotics.torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField,MultiSphereFieldDynamics
from deps.torch_robotics.torch_robotics.environments.utils import create_grid_spheres
from deps.torch_robotics.torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from deps.torch_robotics.torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvPredatorObstacles(EnvPredator):

    def __init__(self,  tensor_args=None, dynamics_fn=None, velocity=None,pursuer_pos=None, **kwargs):
        if pursuer_pos is None:
            pursuer_pos = [0.0, 0.0]
        obj_extra_list = [
            MultiSphereFieldDynamics(
                # np.array(
                #     [
                #     [0.0,0.0], #  [0.6,0.0], #  [0.4,0.5] for path weight comparison rebuttal,#[0.3, -0.5], # [0.,-0.25],[-0.6, 0.8],[0.2,0.5] for dynamic quant
                #     ]),
                np.array([pursuer_pos]),
                np.array(
                    [
                        0.05,#0.05
                    ]
                )
                ,
                dynamics_fn=dynamics_fn,
                velocity=velocity,
                tensor_args=tensor_args
            )
        ]

        super().__init__(
            name=self.__class__.__name__,
            obj_extra_list=[ObjectField(obj_extra_list, 'PredatorObstacles')],
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    env = EnvPredatorObstacles(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()
