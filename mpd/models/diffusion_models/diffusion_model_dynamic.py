"""
Adapted from https://github.com/jannerm/diffuser
"""

import time
from copy import copy
import einops
import numpy as np
import torch
import torch.nn as nn
from abc import ABC

from mpd.models.diffusion_models.helpers import cosine_beta_schedule, Losses, exponential_beta_schedule
from mpd.models.diffusion_models.sample_functionsdynamic import extract, apply_hard_conditioning, \
    ddpm_sample_fn
from .cost import compute_trajectory_costs
from .APFhelper_dynamic import avoidance,ObstacleField,generate_sphere_points

def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class DynamicGaussianDiffusionModel(nn.Module, ABC):

    def __init__(self,
                 model=None,
                 variance_schedule='exponential',
                 n_diffusion_steps=100,
                 clip_denoised=True,
                 predict_epsilon=False,
                 loss_type='l2', # 'l2smooth', # l2
                 context_model=None,
                 mask_type=None,
                 traj_len=None,
                 **kwargs):
        super().__init__()
        # breakpoint()
        self.model = model
        self.mask_type=mask_type

        self.context_model = context_model
        self.traj_len=traj_len
        self.n_diffusion_steps = n_diffusion_steps
        self.training=False 
        self.ddim=True # for paper results
        self.ddim_num_inference_steps_high=10#10
        self.ddim_num_inference_steps_low=5 # 5
        self.state_dim = self.model.state_dim

        if variance_schedule == 'cosine':
            betas = cosine_beta_schedule(n_diffusion_steps, s=0.008, a_min=0, a_max=0.999)
        elif variance_schedule == 'exponential':
            betas = exponential_beta_schedule(n_diffusion_steps, beta_start=1e-4, beta_end=1.0)
        else:
            raise NotImplementedError

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ddim_set_alpha_to_one = True
        self.final_alpha_cumprod = torch.tensor([1.0,], ) \
            if ddim_set_alpha_to_one else torch.clone(self.alphas_cumprod[0:1]) # tensor of size (1,)
        self.loss_fn = Losses[loss_type]()
 
    # ------------------------------------------ sampling ------------------------------------------#
    def predict_noise_from_start(self, x_t, t, x0):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return x0
        else:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
            ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def deep_repeat_tensor(self,x, t, traj_normalized, obstacle_pts, n_rp):
        '''
        Deep copy tensors along batch dimension for evaluation pipeline
        '''
        # Repeat x
        x_repeated = x.repeat((n_rp,) + (1,) * (x.dim() - 1))
        
        # Repeat t
        t_repeated = t.repeat((n_rp,))
        
        # Repeat context (assuming it's a dictionary of tensors)

        # Repeat traj_normalized
        traj_normalized_repeated = traj_normalized.repeat((n_rp,) + (1,) * (traj_normalized.dim() - 1))
        
        # Repeat obstacle_pts
        obstacle_pts_repeated = obstacle_pts.repeat((n_rp,) + (1,) * (obstacle_pts.dim() - 1))
        
        return x_repeated, t_repeated, traj_normalized_repeated, obstacle_pts_repeated

# Example usage:
# x_rep, t_rep, context_rep, traj_norm_rep, obs_pts_rep = deep_repeat_tensor(x, t, context, traj_normalized, obstacle_pts, n_rp=3)

    def p_mean_variance(self, x, hard_conds, context, t,traj_normalized=None,obstacle_pts=None,forward_t=None):
        x2,t2,traj_normalized2,obstacle_pts2=self.deep_repeat_tensor(x, t, traj_normalized, obstacle_pts, 2)
        # uncond=True
        out=self.model(x2, t2, context,x_start=traj_normalized2,obstacle_pts=obstacle_pts2)
        half_batch=x2.shape[0] //2
        w=2.5
        # if uncond:
        e_comb=(1+w)*out[:half_batch]-w*out[half_batch:]
            # e_comb=e_comb.unsqueeze(0)
        # else:
            # e_comb=out
        x_recon = self.predict_start_from_noise(x, t=t, noise=e_comb) 
        # self.clip_denoised=False
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        # breakpoint()
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        # breakpoint()
        if self.ddim:
            return model_mean, posterior_variance, posterior_log_variance,x_recon,e_comb
        else:                # x_recon=apply_hard_conditioning(x_recon,hard_conds)
            return model_mean, posterior_variance, posterior_log_variance
    
    
    def replan_scratch(self,shape,obstacle_pts,context,hard_conds,traj_normalized,sample_fn,**sample_kwargs):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.n_diffusion_steps)):
            t = make_timesteps(batch_size, i, device)
            x, _, obstacle_pos = sample_fn(self, x, hard_conds, context, t, 
                                           traj_normalized=traj_normalized,
                                           obstacle_pts=obstacle_pts,
                                            replan_guide=False,**sample_kwargs)
            x = apply_hard_conditioning(x, hard_conds)
       # consider adding the APF for the predator only
        return x
    
    def sm(self, s1, s2, dt=0.1, num_steps=3, max_vel=.8): #0.56 non-compose
        batch_size = s1.shape[0]
        
        delta_pos = s2[:, :2] - s1[:, :2]
        dist = torch.norm(delta_pos, dim=1, keepdim=True)
        dir = torch.where(dist > 1e-6, delta_pos / dist, torch.zeros_like(delta_pos))
        
        # Adjust for total_time = num_steps * dt
        total_time = num_steps * dt
        desired_v = delta_pos / total_time  # Correct scaling
        desired_v_mag = torch.norm(desired_v, dim=1, keepdim=True)
        base_v = torch.where(desired_v_mag > max_vel, dir * max_vel, desired_v)
        
        # t = torch.arange(1, num_steps + 1, device=s1.device).float()  # Start from 1 to skip s1 original
        t = torch.arange(1, num_steps+1, device=s1.device).float()  # Start from 1 to skip s1
        t = t.view(1, num_steps, 1) * dt  # Cumulative time
        # breakpoint()
        # Positions start from s1 and apply smoothed velocities
        smooth_pos = s1[:, None, :2] + t * base_v[:, None, :]
        smooth_vel = base_v.unsqueeze(1).expand(-1, num_steps, -1)
        
        smooth_states = torch.cat([smooth_pos, smooth_vel], dim=-1)
        return smooth_states

    @torch.no_grad()
    def p_sample_loop(self, shape, hard_conds, context=None, return_chain=False,traj_normalized=None,obstacle_pts=None,
                      sample_fn=ddpm_sample_fn,
                      n_diffusion_steps_without_noise=0,
                      **sample_kwargs):
        device = self.betas.device
        batch_size = shape[0]
        # breakpoint()
        obstacle_pts=obstacle_pts.unsqueeze(0)
        obstacle_pts=obstacle_pts.repeat(batch_size,1,1,1)
        x = torch.randn(shape, device=device) # for energy use 1*torch.randd else 0.5 incomplete
        x = apply_hard_conditioning(x, hard_conds)
        chain = [] if return_chain else None
        dataset = context['dataset']
        context['static_obstacle_centers']=dataset.env.obj_fixed_list[0].fields[0].centers.cpu().numpy()[:4]
        context['static_obstacle_sizes']=dataset.env.obj_fixed_list[0].fields[0].sizes.cpu().numpy()[:4]
        multi_sphere_field = dataset.env.obj_extra_list[0].fields[0]
        # chain_obs = [multi_sphere_field.centers]
        chain_obs=[]
        chain_start = [hard_conds[0][0].unsqueeze(0)]
        updated_start_state = hard_conds[0].clone()
        high_level_plan = None
        predator_start_time=25
        replan_guide=False
        total_steps = n_diffusion_steps_without_noise + self.n_diffusion_steps
        # high level plan generation
        high_time=time.time() 
        for i in reversed(range(-n_diffusion_steps_without_noise, self.n_diffusion_steps)):
            t = make_timesteps(batch_size, i, device)
            forward_t = total_steps - (i + n_diffusion_steps_without_noise) - 1
            x, _, obstacle_pos = sample_fn(self, x, hard_conds, context, t, traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,forward_t=forward_t,replan_guide=replan_guide,**sample_kwargs)
            x = apply_hard_conditioning(x, hard_conds)
            if forward_t == predator_start_time :
                high_level_plan = x.clone()
                break 
        print(time.time()-high_time)
        # breakpoint(s)
        # refine the high level plan further: select the best
        replan_high_level=5
        stepp=0
        x=self.q_sample(x_start=x, t=make_timesteps(x.shape[0],replan_high_level,device)) 
        x = apply_hard_conditioning(x, hard_conds)
        for j in reversed(range(replan_high_level)): 
            t = make_timesteps(batch_size, j, device)
            x = apply_hard_conditioning(x, hard_conds)
            x, _, obstacle_pos = sample_fn(self, x, hard_conds, context, t, traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,forward_t=forward_t,predator_start_time=predator_start_time,replan_guide=replan_guide,**sample_kwargs)
            x = apply_hard_conditioning(x, hard_conds) 
            high_level_plan=x.clone()

        collision_threshold=0.05
        best_traj,best_cost, total_costs,collision_free_mask,_=compute_trajectory_costs(high_level_plan, obstacle_pts[0],collision_threshold=collision_threshold)
        while best_traj is None:
            x=self.replan_scratch(shape,obstacle_pts,context,hard_conds,traj_normalized,sample_fn,**sample_kwargs)
            best_traj,best_cost, total_costs,collision_free_mask,_=compute_trajectory_costs(high_level_plan, obstacle_pts[0],collision_threshold=collision_threshold)
        
        high_level_plan=best_traj.clone()
        x=best_traj.clone()
        # introduce the dynamic predator and moving prey
        replan_step=20
        safe_threshold=0.25
        best_idx=None
        max_iteration=50
        executed_history = [x[0].clone().unsqueeze(0)]
        for i in range(max_iteration):
            forward_t = i #(-i -1)
            t = make_timesteps(1, i, device) #  make_timesteps(batch_size, i, device)
            st_time=time.time() 
            x_clean=x.clone() 
            x=x.unsqueeze(0).repeat(batch_size, 1, 1) 
            x=self.q_sample(x_start=x, t=make_timesteps(x.shape[0],replan_step,device)) 
            x[:,0,2:]=0
            history_length = len(executed_history)
            for hist_idx in range(history_length):
                x[:, hist_idx] = executed_history[hist_idx]
            x[:, -1] = x_clean[-1]   
            for j in reversed(range(replan_step)): 
                t = make_timesteps(batch_size, j, device) # make_timesteps(batch_size, j, device)
                if j==0:
                    window=5
                    smooth_segment = self.sm(x[:, stepp], x[:, stepp + window], num_steps=window)
                    x[:, stepp+1:stepp+1+window] = smooth_segment  # Keep x[:, step] unchanged
                x, _, obstacle_pos = sample_fn(self, x, hard_conds, context, t, 
                                               traj_normalized=traj_normalized,
                                               obstacle_pts=obstacle_pts,
                                               forward_t=forward_t,
                                               predator_start_time=predator_start_time,
                                               replan_guide=(j==0),
                                               best_idx=best_idx,**sample_kwargs)
                # Maintain conditions on history and goal
                for hist_idx in range(history_length):
                    x[:, hist_idx] = executed_history[hist_idx]
                # x[:,:stepp+1]=x_clean[stepp]
                x[:, -1] = x_clean[-1]  # Changed from x[:, -stepp-1:]
                x[:,0,2:]=0.0
            window=2
            smooth_segment = self.sm(x[:, stepp], x[:, stepp + window], num_steps=window)
            x[:, stepp+1:stepp+1+window] = smooth_segment 
            collision_threshold=0.06 #0.06
            x,_, _,_,bes_idx=compute_trajectory_costs(x, obstacle_pts[0],collision_threshold=collision_threshold)
            x[0,2:]=0.0
            # updated_start_state=x[0].clone()
            next_state = x[stepp + 1].clone().unsqueeze(0)  # Get next state based on step counter
            executed_history.append(next_state)  
            updated_start_state=x[stepp].clone()
            ed_time=time.time()
            print(f'resample time {ed_time-st_time} sec')
            stepp+=1
            if return_chain:
                if stepp==1:
                    chain.append(high_level_plan.unsqueeze(0).clone())
                # else:
                chain.append(x.unsqueeze(0).clone())   
            chain_obs.append(multi_sphere_field.centers)  
            chain_start.append(updated_start_state.unsqueeze(0).clone())
            # if torch.norm(x[0,:2]-x[-1,:2]) < safe_threshold:
            if torch.norm(x[stepp-1,:2]-x[-1,:2]) < safe_threshold:
                break
        if return_chain:
            chain = torch.stack(chain, dim=1)
        return x, chain, chain_obs, chain_start

    # from pdm 
    def ddim_p_sample(self, x, hard_conds,context, t, obstacle_pts,traj_normalized=None, forward_t=None,eta=0.0,use_apf=False, use_clipped_model_output=False):
        ''' NOTE follow diffusers ddim, any post-processing *NOT CHECKED yet*
        t (cuda tensor [B,]) must be same
        eta: weight for noise'''
#  x = self.ddim_p_sample(x, hard_conds, timesteps, obstacle_pts, eta=0.0, use_clipped_model_output=True)
        # # 1. get previous step value (=t-1), (B,)
        prev_timestep = t - self.n_diffusion_steps // self.ddim_num_inference_steps_high
        # # 2. compute alphas, betas
        alpha_prod_t = extract(self.alphas_cumprod, t, x.shape) # 
        if prev_timestep[0] >= 0:
            alpha_prod_t_prev = extract(self.alphas_cumprod, prev_timestep, x.shape) # tensor 
        else:
            # extract from a tensor of size 1, cuda tensor [80, 1, 1]
            alpha_prod_t_prev = extract(self.final_alpha_cumprod.to(t.device), torch.zeros_like(t), x.shape)
            
            # print(f'alpha_prod_t_prev {alpha_prod_t_prev[0:3]}')
        assert alpha_prod_t.shape == alpha_prod_t_prev.shape

        beta_prod_t = 1 - alpha_prod_t

        b, *_, device = *x.shape, x.device
        
        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # 4. Clip "predicted x_0"
        ## model_mean is clipped x_0, 
        ## model_output: model prediction, should be the epsilon (noise)
        # breakpoint()
        model_mean, _, model_log_variance, x_recon, model_output = self.p_mean_variance(x=x, hard_conds=hard_conds,
                                                                                        context=context, t=t,
                                                                                        traj_normalized=traj_normalized,
                                                                                        obstacle_pts=obstacle_pts)
        ## 5. compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) \
            * ( 1 - alpha_prod_t / alpha_prod_t_prev )
        std_dev_t = eta * variance ** (0.5)
        assert use_clipped_model_output
        if use_apf: 
            x_start=x[:,forward_t].clone()
            x_goal=x[:,-1].clone()
            dataset = context['dataset']
            multi_sphere_field = dataset.env.obj_extra_list[0].fields[0]
            obs_radius=0.1 
            points_per_obstacle=64 
            distance_threshold_static=0.2 # 0.1 # trigger the potential for static obstacles within this threshold
            distance_threshold_pred=0.4+obs_radius #  trigger the potential for predator within this threshold
            avoidance_strength_static=0.15 # 0.1 # strength of potential for static obstacles
            avoidance_strength_pred= 0.15 # strength of potential for predator obstacle
            avoidance_window_static=8 # 
            avoidance_window_pred=5 # 
            
            def dynamic_obstacle_fn(t, start_pos,replan_guide=True,best_idx=None):
                if replan_guide:
                    if best_idx is not None:
                        start_pos=start_pos[best_idx].unsqueeze(0)
                    else:
                        start_pos=start_pos

                multi_sphere_field.update_centers(t, start_pos)
                center = multi_sphere_field.centers[0].cpu().numpy()
                radius = obs_radius  
                return center, radius
            
            if 'obstacle_field' not in context:
                context['obstacle_field'] = ObstacleField(
                context['static_obstacle_centers'],
                context['static_obstacle_sizes'],
                dynamic_obstacle_fn,
                points_per_obstacle,
                distance_threshold=distance_threshold_static,
                distance_threshold_pred=distance_threshold_pred
            )
            
            obstacle_field = context['obstacle_field']
            obstacle_field.update_dynamic(forward_t, x_start[:,:2], replan_guide=use_apf)
            predator_pos = torch.tensor(obstacle_field.dynamic_center).to(x.device)
            distances = torch.norm(x_start[:,:2] - predator_pos.unsqueeze(0), dim=1)
            obstacle_pos = multi_sphere_field.centers   
            # stt=time.time() if diffusion_only --> comment this below avoidance
            for i in range(x.shape[0]):  # For each batch
                x_recon[i] = avoidance(x_recon[i].detach(),
                obstacle_field, 
                is_dynamic=False,
                avoidance_window=avoidance_window_static, 
                avoidance_strength=avoidance_strength_static, 
                avoidance_strength_pred=avoidance_strength_pred)
                dist = distances[i]
                if dist < distance_threshold_pred:
                    x_recon[i] = avoidance(
                        x_recon[i].detach(), obstacle_field, is_dynamic=True, 
                        avoidance_window=avoidance_window_pred,
                        avoidance_strength=avoidance_strength_static,
                        avoidance_strength_pred=avoidance_strength_pred,
                        affected_states=x.shape[1], 
                        goal_state=x_goal[0],
                    )

            x_recon[:,-1]=x_goal
        if use_clipped_model_output:
            sample = x
            pred_original_sample = x_recon
            model_output = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
            
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        sample = prev_sample
        return sample
        


    def ddim_set_timesteps(self, num_inference_steps) -> np.ndarray: 

        self.num_inference_steps = num_inference_steps
        step_ratio = self.n_diffusion_steps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        # e.g., 10: [90, 80, 70, 60, 50, 40, 30, 20, 10,  0]
        
        return timesteps
    def ddim_replan_scratch(self,shape,
                            hard_conds,
                            context=None, 
                            traj_normalized=None,
                            forward_t=None,
                            obstacle_pts=None,
                            use_apf=False,
                            executed_history=None
                        ):
        device=self.betas.device
        batch_size=shape[0]
        history_length = len(executed_history)
        x = torch.randn(shape, device=device)
        x = apply_hard_conditioning(x, hard_conds)
        for hist_idx in range(history_length):
            x[:, hist_idx] = executed_history[hist_idx]
        time_idx_high = self.ddim_set_timesteps(self.ddim_num_inference_steps_high)        
        for i in time_idx_high: 
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if i==0:
                use_apf=True
            assert not obstacle_pts.requires_grad
            x = self.ddim_p_sample(x, hard_conds, context,timesteps, 
                                   obstacle_pts,traj_normalized=traj_normalized,
                                   forward_t=forward_t,
                                   eta=0.0,use_apf=use_apf, 
                                   use_clipped_model_output=True)
            x = apply_hard_conditioning(x, hard_conds)
            for hist_idx in range(history_length):
                x[:, hist_idx] = executed_history[hist_idx]
            x = x.detach()
        return x


    def ddim_p_sample_loop(self, shape, hard_conds,
        context=None, 
        return_chain=False,
        traj_normalized=None,
        obstacle_pts=None,
        t_start_guide=float('inf'), 
        guide=None,
        n_guide_steps=1,
        **sample_kwargs,
    ):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        # x = apply_conditioning(x, cond, 0) # start from dim 0, different from diffuser
        x = apply_hard_conditioning(x, hard_conds)
        if return_chain: diffusion = [x]
        # 100 // 20 = 5
        time_idx_high = self.ddim_set_timesteps(self.ddim_num_inference_steps_high)
        time_idx_low = self.ddim_set_timesteps(self.ddim_num_inference_steps_low)
        # breakpoint() 
        dataset = context['dataset']
        context['static_obstacle_centers']=dataset.env.obj_fixed_list[0].fields[0].centers.cpu().numpy()[:4]
        context['static_obstacle_sizes']=dataset.env.obj_fixed_list[0].fields[0].sizes.cpu().numpy()[:4]
        multi_sphere_field = dataset.env.obj_extra_list[0].fields[0]
        # breakpoint()
        obstacle_pts=obstacle_pts.unsqueeze(0)
        obstacle_pts=obstacle_pts.repeat(batch_size,1,1,1)
        chain_obs=[]
        chain_start = [hard_conds[0][0].unsqueeze(0)]
        updated_start_state = hard_conds[0].clone()
        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        max_iteration=60 # 50
        # ddim_num_inference_steps_low=5 # dn_steps
        safe_threshold=0.2
        distance_threshold_pred=0.4
        chain = [] if return_chain else None
        stepp=0 
        collision_threshold_high=0.02 #
        collision_threshold_low=0.05# 
        replan_scratch_shape=(30,48,4) # BxHxD
        use_apf=False
        # STAGE I: generate high-level plan
        high_time=time.time()
        for i in time_idx_high: # if np array, i is <class 'numpy.int64'>
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            assert not obstacle_pts.requires_grad
            x = self.ddim_p_sample(x, hard_conds, context,timesteps, 
                                   obstacle_pts,traj_normalized=traj_normalized,
                                   eta=0.0,use_apf=use_apf, #use_apf
                                   use_clipped_model_output=True)
            x = apply_hard_conditioning(x, hard_conds)
            x = x.detach()
        best_traj,best_cost, total_costs,collision_free_mask,_=compute_trajectory_costs(x, obstacle_pts[0],collision_threshold=collision_threshold_high)
        high_plan=best_traj.clone()
        x=best_traj.clone()
        executed_history = [x[0].clone().unsqueeze(0)]
        # STAGE II: low-level plan
        for k in range(max_iteration):
            use_apf=False # introduce apf in the later stages of the sampling
            x_clean=x.clone()
            x=x.unsqueeze(0).repeat(batch_size, 1, 1) 
            assert self.ddim_num_inference_steps_low <= self.ddim_num_inference_steps_high * 0.51
            ts = self.ddim_set_timesteps(self.ddim_num_inference_steps_high)  
            noise_t = ts[-self.ddim_num_inference_steps_low].item()
            # breakpoint()
            noise_t = torch.tensor([noise_t,], device=device)
            x = self.q_sample(x, noise_t)
            x[:,0,2:]=0 
            history_length = len(executed_history)
            for hist_idx in range(history_length):
                x[:, hist_idx] = executed_history[hist_idx]
            x[:, -1] = x_clean[-1]  
            for forward_idx, i in enumerate(ts[-self.ddim_num_inference_steps_low:]):
            # for forward_t,i in enumerate(time_idx_low): # if np array, i is <class 'numpy.int64'>
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
                assert not obstacle_pts.requires_grad
                if i==0: # controls when to introduce apf
                    use_apf=True # 
                    window=3
                    # if stepp>44:
                        # window=1    
                    smooth_segment = self.sm(x[:, stepp], x[:, stepp + window], num_steps=window) 
                    x[:, stepp+1:stepp+1+window] = smooth_segment  
                x = self.ddim_p_sample(x, hard_conds, context,timesteps, 
                                       obstacle_pts,traj_normalized=traj_normalized,
                                       forward_t=k,eta=0.0, use_apf=use_apf,
                                       use_clipped_model_output=True)
                x = apply_hard_conditioning(x, hard_conds)
                for hist_idx in range(history_length):
                    x[:, hist_idx] = executed_history[hist_idx]
                x[:, -1] = x_clean[-1] 
                x[:,0,2:]=0.0    
                x = x.detach()
            
            window=2
            smooth_segment = self.sm(x[:, stepp], x[:, stepp + window], num_steps=window)
            x[:, stepp+1:stepp+1+window] = smooth_segment  
            # for fully observable pursuer, we can use the pursuer obstacle points besides the static obstacle points all the time in compute_trajectory_costs.
            # we can use the following when the evader is within distance_threshold_pred of pursuer, which will be partially observable
            if np.linalg.norm(x[0,stepp,:2].cpu().numpy()-multi_sphere_field.centers[0].cpu().numpy()) <distance_threshold_pred: 
                pursuer_obstacle_pts=generate_sphere_points(multi_sphere_field.centers[0].cpu().numpy(),multi_sphere_field.radii[0].cpu().numpy(),64)
                x,_, _,_,_=compute_trajectory_costs(x, torch.vstack([obstacle_pts[0],torch.from_numpy(pursuer_obstacle_pts).unsqueeze(0).to(obstacle_pts.device)]),collision_threshold=collision_threshold_low) 
            else:
                x,_, _,_,_=compute_trajectory_costs(x, obstacle_pts[0],collision_threshold=collision_threshold_low)
            while x is None: 
                print(f'replanning from scratch')
                new_hard_conds = {kk: v[:replan_scratch_shape[0]].clone() for kk, v in hard_conds.items()}
                new_obstacle_pts=obstacle_pts[:replan_scratch_shape[0]]
                x=self.ddim_replan_scratch(replan_scratch_shape,new_hard_conds,context,traj_normalized,forward_t=k,obstacle_pts=new_obstacle_pts,use_apf=False,executed_history=executed_history)
                window=2
                smooth_segment = self.sm(x[:, stepp], x[:, stepp + window], num_steps=window)
                x[:, stepp+1:stepp+1+window] = smooth_segment 
                x,_, _,_,_=compute_trajectory_costs(x, obstacle_pts[0],collision_threshold=collision_threshold_low)
                # x,_, _,_,_=compute_trajectory_costs(x, torch.vstack([obstacle_pts[0],torch.from_numpy(pursuer_obstacle_pts).unsqueeze(0).to(obstacle_pts.device)]),collision_threshold=collision_threshold_low) 
            x[0,2:]=0.0
            next_state = x[stepp + 1].clone().unsqueeze(0)  # Get next state 
            executed_history.append(next_state)  
            updated_start_state=x[stepp].clone()
            stepp+=1
            if return_chain:
                if stepp==1:
                    chain.append(high_plan.unsqueeze(0).clone())
                chain.append(x.unsqueeze(0).clone())   
            chain_obs.append(multi_sphere_field.centers)  
            chain_start.append(updated_start_state.unsqueeze(0).clone())
            if torch.norm(x[stepp-1,:2]-x[-1,:2]) < safe_threshold:
                break
        if return_chain:
            chain = torch.stack(chain, dim=1)
        return x, chain, chain_obs, chain_start            

    @torch.no_grad()
    def conditional_sample(self, hard_conds, horizon=None, batch_size=1, ddim=False, traj_normalized=None,obstacle_pts=None,**sample_kwargs):
        '''
            hard conditions : hard_conds : { (time, state), ... }
        '''
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.state_dim)
        with torch.set_grad_enabled(False):
            if self.ddim:
                return self.ddim_p_sample_loop(shape, hard_conds, traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,**sample_kwargs) 
            return self.p_sample_loop(shape, hard_conds, traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,**sample_kwargs)

    def forward(self, cond, *args, **kwargs):
        raise NotImplementedError
        return self.conditional_sample(cond, *args, **kwargs)

    @torch.no_grad()
    def warmup(self, horizon=64, device='cuda'):
        shape = (2, horizon, self.state_dim)
        x = torch.randn(shape, device=device)
        t = make_timesteps(2, 1, device)
        self.model(x, t, context=None)

    @torch.no_grad()
    def run_inference(self, context=None, hard_conds=None, n_samples=1, return_chain=False,traj_normalized=None,obstacle_pts=None, **diffusion_kwargs):
        hard_conds = copy(hard_conds)
        context = copy(context)
        for k, v in hard_conds.items():
            new_state = einops.repeat(v, 'd -> b d', b=n_samples)
            hard_conds[k] = new_state
        samples, chain,chain_obs,chain_start = self.conditional_sample(
            hard_conds, context=context, batch_size=n_samples, return_chain=True, traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,**diffusion_kwargs
        )
        trajs_chain_normalized = chain

        # trajs: [ (n_diffusion_steps + 1) x n_samples x horizon x state_dim ]
        trajs_chain_normalized = einops.rearrange(trajs_chain_normalized, 'b diffsteps h d -> diffsteps b h d')
        if return_chain:
            return trajs_chain_normalized,chain_obs,chain_start

        # return the last denoising step
        return trajs_chain_normalized[-1]

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample
