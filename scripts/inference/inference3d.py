import os
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from mpd.models import TemporalUnetInference, UNET_DIM_MULTS
from mpd.models.diffusion_models.sample_functions import  ddpm_sample_fn
from mpd.trainer import get_dataset, get_model
from deps.torch_robotics.torch_robotics.torch_utils.seed import fix_random_seed
from deps.torch_robotics.torch_robotics.torch_utils.torch_timer import TimerCUDA
from deps.torch_robotics.torch_robotics.torch_utils.torch_utils import  freeze_torch_model_params
from matplotlib.animation import FuncAnimation  
from config.base_config import Config3d

allow_ops_in_compiled_graph()


def main(args): 
    fix_random_seed(args.seed)
    tensor_args = {'device': args.device, 'dtype': torch.float32}
    model_dir = os.path.join(args.trained_models_dir, args.model_id)
    res_dir=os.path.join(args.trained_models_dir,args.model_id)
    results_dir = os.path.join(res_dir, 'results_inference', str(args.seed))
    os.makedirs(results_dir, exist_ok=True)
    
    # Load dataset with env, robot, task
    train_subset, _,_, _ = get_dataset(
        dataset_class='TrajectoryDataset3d',
        use_extra_objects=False,
        include_velocity=args.include_velocity,
        dataset_subdir=args.dataset_subdir,
        obstacle_cutoff_margin=0.05,
        tensor_args=tensor_args,
        # base_dataset_dir=args.dataset_path
    )
    traj_id=np.random.choice(len(train_subset),1).item()
    print('traj_id',traj_id)
    data_normalized = train_subset[traj_id]
    traj_normalized=data_normalized['traj_normalized']
    obstacle_pts=data_normalized['obstacle_points']
    box_centers = data_normalized['box_centers']
    sphere_centers=data_normalized['sphere_centers']
    # box_size_single = data_normalized['box_sizes']
    box_size = data_normalized['box_sizes']
    # sphere_radii_single=data_normalized['sphere_radii']
    sphere_radii=data_normalized['sphere_radii']
    # breakpoint()
    if args.compose:
        # For sphere radii (1D tensor)
        box_size = box_size.repeat(2, 1)  # for 3 obstacles 
        # For sphere radii - repeat to exactly 10 items
        sphere_radii = sphere_radii.repeat(2)  # for 3 obstacles 
        obs1=14 # 4 final
        obs2=15 # 4 final 
        # for 3 obstacle sets
        # obs1=5745 
        # obs2=4485 
        # obs3=145
        base_dir='/home/wondm/RAMP/dataset/EnvSmall3D'
        obstacle_1=torch.load(f'{base_dir}/{obs1}/obstacle_points.pt')
        obstacle_2=torch.load(f'{base_dir}/{obs2}/obstacle_points.pt') 
        # obstacle_3=torch.load(f'/data/wondm/maze3d/EnvSmall3D/{obs3}/obstacle_points.pt') 
        # mapp={0:0,1:1,2:2,3:3,4:4,5:0,6:1,7:2,8:3,9:4} # to replace obstacle 3 
        # for replace_idx, source_idx in mapp.items():    
            # obstacle_3[replace_idx]=obstacle_3[source_idx]
        obstacle_pts=torch.stack([obstacle_1, obstacle_2], dim=0).to(args.device)
        # obstacle_pts=torch.stack([obstacle_1, obstacle_2,obstacle_3], dim=0)
        box_center1=np.load(f'{base_dir}/{obs1}/box_centers.npy')
        box_center2=np.load(f'{base_dir}/{obs2}/box_centers.npy')
        # box_center3=np.load(f'/data/wondm/maze3d/EnvSmall3D/{obs3}/box_centers.npy')
        # breakpoint()
        box_centers=np.vstack((box_center1,box_center2))
        # box_centers=np.vstack((box_center1,box_center2,box_center3))
        sphere_center1=np.load(f'{base_dir}/{obs1}/sphere_centers.npy')
        sphere_center2=np.load(f'{base_dir}/{obs2}/sphere_centers.npy')
        # sphere_center3=np.load(f'/data/wondm/maze3d/EnvSmall3D/{obs3}/sphere_centers.npy')
        sphere_centers=np.vstack((sphere_center1,sphere_center2))
        # sphere_centers=np.vstack((sphere_center1,sphere_center2,sphere_center3)) # no need for spehre since we took the  first 5 obstacles from osbtacle 3
    # else:
    #     pass  
    # breakpoint()  
    dataset = train_subset.dataset
    env = dataset.env
    robot = dataset.robot
    env.update_box_centers(box_centers,sphere_centers)
    n_support_points = dataset.n_support_points
    dt = args.trajectory_duration / n_support_points  
    robot.dt = dt

    diffusion_configs = dict(
        variance_schedule=args.variance_schedule,
        n_diffusion_steps=args.n_diffusion_steps,
        predict_epsilon=args.predict_epsilon,
        training=False,
        compose=args.compose

    )
    unet_configs = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=args.unet_input_dim,
        dim_mults=UNET_DIM_MULTS[args.unet_dim_mults_option],
        obstacle_3d=True
    )
    diffusion_model = get_model(
        model_class=args.diffusion_model_class,
        model=TemporalUnetInference(**unet_configs),
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs
    )
    # breakpoint()
    diffusion_model.load_state_dict(
        torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args.use_ema else 'model_current_state_dict.pth'),
        map_location=tensor_args['device'])
    )
    # breakpoint()
    diffusion_model.eval()
    model = diffusion_model
    freeze_torch_model_params(model)
    model = torch.compile(model) 
     
    start_state_pos=torch.tensor([-0.8, -0.25, -0.8]).to(args.device) # single compose
    goal_state_pos=torch.tensor([0.8, -0.4, 0.9]).to(args.device) # single compose
    
    print(f'start_state_pos: {start_state_pos}')
    print(f'goal_state_pos: {goal_state_pos}')
 
    ########
    # normalize start and goal positions
    hard_conds = dataset.get_hard_conditions(torch.vstack((start_state_pos, goal_state_pos)), normalize=True)
    context = {'dataset': dataset}
    t_start_guide = ceil(args.start_guide_steps_fraction * model.n_diffusion_steps)
    sample_fn_kwargs = dict(
        guide=None ,
        n_guide_steps=args.n_guide_steps,
        t_start_guide=t_start_guide,
        noise_std_extra_schedule_fn=lambda x: 0.5,
    )

    with TimerCUDA() as timer_model_sampling:
        trajs_normalized_iters = model.run_inference(
            context, hard_conds,
            n_samples=args.n_samples, horizon=n_support_points,
            return_chain=True,
            traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,
            sample_fn=ddpm_sample_fn,
            **sample_fn_kwargs,
            n_diffusion_steps_without_noise=args.n_diffusion_steps_without_noise,
        )

    ########
    trajs_iters = dataset.unnormalize_trajectories(trajs_normalized_iters)
    pos_trajs_iters = robot.get_position(trajs_iters)
    fig, ax = plt.subplots()

    def visualize_environment_direct_3d(box_centers, box_size, sphere_centers, sphere_radii, pos_trajs_iters_final, start_state_pos, goal_state_pos, env_limits=None):
        plt.rcParams['font.family'] = 'DeJavu Serif'  
        fig = plt.figure(figsize=(10, 8), dpi=300)  
        ax = fig.add_subplot(111, projection='3d')
        box_color = '#1565C0'  
        sphere_color = '#1565C0'  
      
        for center, size in zip(box_centers, box_size):
            if isinstance(center, torch.Tensor):
                center = center.detach().cpu().numpy()
            if isinstance(size, torch.Tensor):
                size = size.detach().cpu().numpy()
                
            ax.bar3d(center[0] - size[0]/2,
                    center[1] - size[1]/2,
                    center[2] - size[2]/2,
                    size[0], size[1], size[2],
                    color=box_color, alpha=.75, #0.7
                    edgecolor='black', linewidth=0.75)
        
        for center, radius in zip(sphere_centers, sphere_radii):
            if isinstance(center, torch.Tensor):
                center = center.detach().cpu().numpy()
            if isinstance(radius, torch.Tensor):
                radius = radius.item()
            else:
                radius = float(radius)

            u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]  
            x = radius * np.cos(u) * np.sin(v) + center[0]
            y = radius * np.sin(u) * np.sin(v) + center[1]
            z = radius * np.cos(v) + center[2]
            ax.plot_surface(x, y, z, color=sphere_color, alpha=0.75, #0.7
                        edgecolor='black', linewidth=0.1)
        
        if isinstance(pos_trajs_iters_final, torch.Tensor):
            traj = pos_trajs_iters_final.detach().cpu().numpy()
        else:
            traj = np.array(pos_trajs_iters_final)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
        color='orange', linewidth=1.5, alpha=0.8,
        marker='o', markersize=3, markerfacecolor='orange',
        label='Trajectory')

        if isinstance(start_state_pos, torch.Tensor):
            start = start_state_pos.detach().cpu().numpy()
        else:
            start = np.array(start_state_pos)
            
        if isinstance(goal_state_pos, torch.Tensor):
            goal = goal_state_pos.detach().cpu().numpy()
        else:
            goal = np.array(goal_state_pos)
            
        ax.scatter(start[0], start[1], start[2], color='green', s=80, label='Start')
        ax.scatter(goal[0], goal[1], goal[2], color='purple', s=80, label='Goal')
        
        ax.grid(True, linestyle='--', alpha=0.2)
        ax.xaxis._axinfo["grid"].update({"color": "#CCCCCC", "linewidth": 0.5})
        ax.yaxis._axinfo["grid"].update({"color": "#CCCCCC", "linewidth": 0.5})
        ax.zaxis._axinfo["grid"].update({"color": "#CCCCCC", "linewidth": 0.5})
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
       
        ax.set_xlabel('X', fontsize=12, labelpad=10)
        ax.set_ylabel('Y', fontsize=12, labelpad=10)
        ax.set_zlabel('Z', fontsize=12, labelpad=10)
        
        ax.set_title('3D Environment with Boxes, Spheres, and Trajectory', 
                    fontsize=14, pad=15)
        
        if env_limits is not None:
            if isinstance(env_limits, torch.Tensor):
                limits = env_limits.detach().cpu().numpy()
            else:
                limits = np.array(env_limits)
        else:
            limits = np.array([[-1, -1, -1], [1, 1, 1]])
        
        ax.set_xlim(limits[0, 0], limits[1, 0])
        ax.set_ylim(limits[0, 1], limits[1, 1])
        ax.set_zlim(limits[0, 2], limits[1, 2])
        ax.view_init(elev=25, azim=235)
        ax.set_box_aspect([1, 1, 1])
        
        plt.tight_layout()
        return fig, ax

    def render_environment_direct_anim(ax, box_centers, box_size, sphere_centers, sphere_radii, 
                                pos_trajs_iters_final, start_state_pos, goal_state_pos, env_limits=None):
        box_color = '#1565C0'  
        sphere_color = '#1565C0'
        for center, size in zip(box_centers, box_size):
            if isinstance(center, torch.Tensor):
                center = center.detach().cpu().numpy()
            if isinstance(size, torch.Tensor):
                size = size.detach().cpu().numpy()
            ax.bar3d(center[0] - size[0]/2,
                    center[1] - size[1]/2,
                    center[2] - size[2]/2,
                    size[0], size[1], size[2],
                    color=box_color, alpha=.75, #0.7
                    edgecolor='black', linewidth=0.75)
  
        for center, radius in zip(sphere_centers, sphere_radii):
            if isinstance(center, torch.Tensor):
                center = center.detach().cpu().numpy()
            if isinstance(radius, torch.Tensor):
                radius = radius.item()
            else:
                radius = float(radius)

            u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]  
            x = radius * np.cos(u) * np.sin(v) + center[0]
            y = radius * np.sin(u) * np.sin(v) + center[1]
            z = radius * np.cos(v) + center[2]
            ax.plot_surface(x, y, z, color=sphere_color, alpha=.75, #0.7
                        edgecolor='black', linewidth=0.1)
        
        if isinstance(pos_trajs_iters_final, torch.Tensor):
            traj = pos_trajs_iters_final.detach().cpu().numpy()
        else:
            traj = np.array(pos_trajs_iters_final)
     
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
           color='orange', linewidth=1.5, alpha=0.8,
           marker='o', markersize=3, markerfacecolor='orange',
           label='Trajectory')
        
        # Plot start and goal
        if isinstance(start_state_pos, torch.Tensor):
            start = start_state_pos.detach().cpu().numpy()
        else:
            start = np.array(start_state_pos)
            
        if isinstance(goal_state_pos, torch.Tensor):
            goal = goal_state_pos.detach().cpu().numpy()
        else:
            goal = np.array(goal_state_pos)
            
        ax.scatter(start[0], start[1], start[2], color='green', s=80, label='Start')
        ax.scatter(goal[0], goal[1], goal[2], color='purple', s=80, label='Goal')
        ax.grid(True, linestyle='--', alpha=0.2)
        ax.xaxis._axinfo["grid"].update({"color": "#CCCCCC", "linewidth": 0.5})
        ax.yaxis._axinfo["grid"].update({"color": "#CCCCCC", "linewidth": 0.5})
        ax.zaxis._axinfo["grid"].update({"color": "#CCCCCC", "linewidth": 0.5})

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        if env_limits is not None:
            if isinstance(env_limits, torch.Tensor):
                limits = env_limits.detach().cpu().numpy()
            else:
                limits = np.array(env_limits)
        else:
            limits = np.array([[-1, -1, -1], [1, 1, 1]])
        
        ax.set_xlim(limits[0, 0], limits[1, 0])
        ax.set_ylim(limits[0, 1], limits[1, 1])
        ax.set_zlim(limits[0, 2], limits[1, 2])
        ax.set_box_aspect([1, 1, 1])
        ax.set_facecolor('#F0F0F0')
    
    def visualize_and_save_direct_3d(box_centers, box_size, sphere_centers, sphere_radii, 
                            pos_trajs_iters_final, start_state_pos, goal_state_pos, 
                            results_dir, env_limits=None, show_plot=False):
    
        fig, ax = visualize_environment_direct_3d(
            box_centers, box_size, sphere_centers, sphere_radii,
            pos_trajs_iters_final, start_state_pos, goal_state_pos, env_limits
        )

        filename = 'final_trajectory.png'
        plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Figure saved as {os.path.join(results_dir, filename)}")
        plt.close(fig)

    def visualize_environment_direct_rotating_anim(box_centers, box_size, sphere_centers, sphere_radii,
                                        final_trajectory, start_state_pos, goal_state_pos, 
                                        save_gif=False, results_dir=None, env_limits=None):
        plt.rcParams['font.family'] = 'DeJavu Serif'
        fig = plt.figure(figsize=(10, 8), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        
        render_environment_direct_anim(
            ax, box_centers, box_size, sphere_centers, sphere_radii,
            final_trajectory, start_state_pos, goal_state_pos, env_limits
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Environment with Boxes, Spheres, and Trajectory')
        
        def rotate(angle):
            ax.view_init(elev=25, azim=angle)
            return fig,
        
        ax.view_init(elev=25, azim=235)
        ani = FuncAnimation(fig, rotate, frames=np.linspace(0, 360, 201), interval=50, blit=True)
        
        if save_gif:
            filename = 'final_trajectory_animation.gif'
            full_path = os.path.join(results_dir, filename) if results_dir else filename
            ani.save(full_path, writer='pillow', fps=20)
            print(f"Animation saved as {full_path}")
        else:
            plt.show()

    visualize_and_save_direct_3d(
        box_centers, box_size, sphere_centers, sphere_radii,
        pos_trajs_iters[-1].squeeze(0), start_state_pos, goal_state_pos,
        results_dir=results_dir, env_limits=torch.tensor([[-1,-1,-1], [1,1,1]]), show_plot=True
    )
 
    visualize_environment_direct_rotating_anim(
        box_centers, box_size, sphere_centers, sphere_radii,
        pos_trajs_iters[-1].squeeze(0), start_state_pos, goal_state_pos,
        save_gif=True, results_dir=results_dir, env_limits=torch.tensor([[-1,-1,-1], [1,1,1]])
    )


if __name__ == '__main__':
        config = Config3d(
        model_id='model3d',
        dataset_subdir='EnvSmall3D',
        n_diffusion_steps=25,  
        compose=False, )
        main(config)

