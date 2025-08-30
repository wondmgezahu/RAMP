import torch
class GaussianDiffusionLoss:

    def __init__(self):
        pass

    @staticmethod
    def loss_fn(diffusion_model, input_dict, dataset, step=None):
        """
        Loss function for training diffusion-based generative models.
        """
        traj_normalized = input_dict[f'{dataset.field_key_traj}_normalized']

        # context = build_context(diffusion_model, dataset, input_dict)
        context=None
        # breakpoint()
        obstacle_points=input_dict['obstacle_points']
        hard_conds = input_dict.get('hard_conds', {})
        box_centers=input_dict['box_centers']
        # obstacle_centers
        # breakpoint()
        # loss, info = diffusion_model.loss(traj_normalized, context, hard_conds)
        loss, info = diffusion_model.loss(traj_normalized, context, hard_conds,obstacle_points,box_centers)

        loss_dict = {'diffusion_loss': loss}

        return loss_dict, info


class GaussianDiffusionLossDDP:
    def __init__(self):
        pass

    @staticmethod
    def loss_fn(diffusion_model, input_dict, dataset, step=None):
        """
        Loss function for training diffusion-based generative models.
        Compatible with DistributedDataParallel.
        """
        traj_normalized = input_dict[f'{dataset.field_key_traj}_normalized']
        context = None  
        obstacle_points = input_dict['obstacle_points']
        hard_conds = input_dict.get('hard_conds', {})
        box_centers=input_dict['box_centers']
        # obstacle_center=input_dict['obstacle_centers']

        # breakpoint()
        # Access the underlying model if it's wrapped in DistributedDataParallel
        if isinstance(diffusion_model, torch.nn.parallel.DistributedDataParallel):
            model = diffusion_model.module
        else:
            model = diffusion_model

        # loss, info = model.loss(traj_normalized, context, hard_conds, obstacle_points,box_centers=box_centers)
        loss, info = model.loss(traj_normalized, context, hard_conds, obstacle_points)
        loss_dict = {'diffusion_loss': loss}

        return loss_dict, info