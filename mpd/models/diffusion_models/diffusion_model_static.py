"""
Adapted from https://github.com/jannerm/diffuser
"""
from copy import copy
import einops
import numpy as np
import torch
import torch.nn as nn
from abc import ABC

from mpd.models.diffusion_models.helpers import cosine_beta_schedule, Losses, exponential_beta_schedule
from mpd.models.diffusion_models.sample_functions import extract, apply_hard_conditioning, \
    ddpm_sample_fn
from mpd.models.diffusion_models.APFhelper import ObstacleField,avoidance

def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class StaticGaussianDiffusionModel(nn.Module, ABC):

    def __init__(self,
                 model=None,
                 variance_schedule='exponential',
                 n_diffusion_steps=100,
                 clip_denoised=True,
                 predict_epsilon=False,
                 loss_type='l2', 
                 context_model=None,
                 compose=False,
                 use_apf=False,
                 training=False,
                 **kwargs):
        super().__init__()
        self.model = model
        self.context_model = context_model
        self.n_diffusion_steps = n_diffusion_steps
        # self.ddim_num_inference_steps=20 # 5 for non-APF and simple2d apf
        self.ddim_num_inference_steps = 8 if (compose and use_apf) else 5
        self.ddim=True # for paper results
        self.compose=compose
        self.APF=use_apf
        self.energy_mode=True # for paper results
        self.training=training 
        self.state_dim = self.model.state_dim
        # breakpoint()
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

        # added for ddim from pdm:
        ddim_set_alpha_to_one = True
        self.final_alpha_cumprod = torch.tensor([1.0,], ) \
            if ddim_set_alpha_to_one else torch.clone(self.alphas_cumprod[0:1]) # tensor of size (1,)

        ## get loss coefficients and initialize objective
        # breakpoint()
        # self.loss_type='l2smooth'
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

    def deep_repeat_tensor(self, x, t, traj_normalized, obstacle_pts, n_rp):
        '''
        Deep copy tensors along batch dimension for evaluation pipeline
        Args:
            x: Input tensor of shape [batch_size, 48, 4]
            t: Time tensor of shape [batch_size]
            traj_normalized: Normalized trajectory tensor
            obstacle_pts: Obstacle points tensor of shape [batch_size, 6, 64, 2]
            n_rp: Number of repetitions (2 for CFG)
        '''
        # Repeat each sample in the batch for conditional/unconditional evaluation
        x_repeated = x.repeat_interleave(n_rp, dim=0)
        t_repeated = t.repeat_interleave(n_rp, dim=0)
        traj_normalized_repeated = traj_normalized.repeat_interleave(n_rp, dim=0)
        obstacle_pts_repeated = obstacle_pts.repeat_interleave(x.shape[0]*n_rp, dim=0)
        
        return x_repeated, t_repeated, traj_normalized_repeated, obstacle_pts_repeated

    def p_mean_variance(self, x, hard_conds, context, t,traj_normalized=None,obstacle_pts=None,forward_t=None,compose=False):
        batch_size=x.shape[0]
        # breakpoint()
        # x=x.detach()
        x2,t2,traj_normalized2,obstacle_pts2=self.deep_repeat_tensor(x, t, traj_normalized, obstacle_pts, 2)
        uncond=True
        # st_time=time.time()
        # breakpoint()   
        # x2=x2.detach() 
        out=self.model(x2, t2, context,x_start=traj_normalized2,obstacle_pts=obstacle_pts2,compose=compose)
        # print(f'time {time.time()-st_time} seconds')
        # breakpoint()
        out = out.view(batch_size, 2, *out.shape[1:])  # [batch_size, 2, 48, 4]
        cond_out = out[:, 0]    # [batch_size, 48, 4]
        uncond_out = out[:, 1]  # [batch_size, 48, 4]
        w=2.
        e_comb = (1 + w) * cond_out - w * uncond_out  # [batch_size, 48, 4]
        # t=t.detach()
        x_recon = self.predict_start_from_noise(x, t=t, noise=e_comb) 
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if self.ddim:
            return model_mean, posterior_variance, posterior_log_variance,x_recon,e_comb
        else:
            if self.APF and forward_t>20: 
            # apply apf 
                distance_threshold=0.07
                avoidance_strength=0.1
                avoidance_window=5
                obstacle_pts_stack = obstacle_pts.squeeze(0).reshape(-1, 2)
                obstacle_field = ObstacleField(obstacle_pts_stack.cpu().numpy(),distance_threshold=distance_threshold)
                # breakpoint()
                model_mean = avoidance(model_mean, obstacle_field,avoidance_window=avoidance_window,avoidance_strength=avoidance_strength)
                # x_recon=apply_hard_conditioning(x_recon,hard_conds)
            return model_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance_compose(self, x, hard_conds, context, t, traj_normalized=None, obstacle_pts=None, forward_t=None,compose=True):
        n_samples = x.shape[0]  # Should be 5 in this case
        
        # Repeat each sample 3 times (for two environments and unconditional)
        x2 = x.repeat_interleave(3, dim=0)
        t2 = t.repeat_interleave(3, dim=0)
        traj_normalized2 = traj_normalized.repeat_interleave(3, dim=0) if traj_normalized is not None else None
        
        # Handle obstacle_pts
        assert obstacle_pts.shape == (2, 6, 64, 2), "Obstacle points should have shape (2, 6, 64, 2)"
        
        # Use the first configuration for unconditional (change this to the second if preferred)
        obstacle_pts_uncond = obstacle_pts[0].unsqueeze(0)  # Shape: (1, 6, 64, 2)
        
        # Concatenate the two configurations and the unconditional
        obstacle_pts_all = torch.cat([obstacle_pts, obstacle_pts_uncond], dim=0)  # Shape: (3, 6, 64, 2)
        
        # Repeat for all samples
        obstacle_pts2 = obstacle_pts_all.repeat(n_samples, 1, 1, 1)  # Shape: (n_samples * 3, 6, 64, 2)
        
        w1, w2 = 2., 2.
        # breakpoint()
        out = self.model(x2, t2, context, x_start=traj_normalized2, obstacle_pts=obstacle_pts2, forward_t=forward_t,compose=compose)
        
        # Reshape out to separate samples: (n_samples, 3, 48, 2)
        out = out.view(n_samples, 3, *out.shape[1:])
        
        # Compute e_comb for each sample
        e_comb = out[:, 2] + w1 * (out[:, 0] - out[:, 2]) + w2 * (out[:, 1] - out[:, 2])
        
        x_recon = self.predict_start_from_noise(x, t=t, noise=e_comb)
        
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if self.ddim:
            return model_mean, posterior_variance, posterior_log_variance,x_recon,e_comb
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, hard_conds, context=None, return_chain=False,traj_normalized=None,obstacle_pts=None,
                      sample_fn=ddpm_sample_fn,
                      n_diffusion_steps_without_noise=0,
                      **sample_kwargs):
        device = self.betas.device
        batch_size = shape[0]
        if not self.compose:
            obstacle_pts=obstacle_pts.unsqueeze(0)
        x = torch.randn(shape, device=device) # for energy use 1*torch.randd else 0.5 incomplete
        x = apply_hard_conditioning(x, hard_conds)
        chain = [x] if return_chain else None
        forward_t=0
        # breakpoint()
        for i in reversed(range(-n_diffusion_steps_without_noise, self.n_diffusion_steps)):
        # for i in reversed(range(20)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, hard_conds, context, t, traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,forward_t=forward_t,compose=self.compose,**sample_kwargs)
            x = apply_hard_conditioning(x, hard_conds)          
            if return_chain:
                chain.append(x)
            forward_t+=1
        if return_chain:
            chain = torch.stack(chain, dim=1)
            return x, chain
        return x
    
    # from pdm 
    def ddim_p_sample(self, x, hard_conds,context, t, obstacle_pts,traj_normalized=None, forward_t=None,eta=0.0, use_clipped_model_output=False):
        ''' NOTE follow diffusers ddim, any post-processing *NOT CHECKED yet*
        t (cuda tensor [B,]) must be same
        eta: weight for noise'''
#  x = self.ddim_p_sample(x, hard_conds, timesteps, obstacle_pts, eta=0.0, use_clipped_model_output=True)
        # # 1. get previous step value (=t-1), (B,)
        prev_timestep = t - self.n_diffusion_steps // self.ddim_num_inference_steps
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
        if self.compose:
            model_mean, _, model_log_variance, x_recon, model_output = self.p_mean_variance_compose(x=x, hard_conds=hard_conds,context=context, t=t, traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,compose=self.compose)
        else:
            model_mean, _, model_log_variance, x_recon, model_output = self.p_mean_variance(x=x, hard_conds=hard_conds,context=context, t=t, traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,compose=self.compose)
        ## 5. compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) \
            * ( 1 - alpha_prod_t / alpha_prod_t_prev )

        std_dev_t = eta * variance ** (0.5)

        assert use_clipped_model_output
        # breakpoint()
        temp_forward=2 # 0.2*self.ddim_num_inference_steps
        if self.APF and forward_t>=temp_forward: # >=2 for the table results  
            # breakpoint()
            distance_threshold=0.07 # 0.07 for simple2d
            base_strength = 0.1 # 0.1 for simple2d
            avoidance_window=7 # 5 for simple2d
            # avoidance_window = min(5 + forward_t // 2, 7)
            # timestep_factor = 1 + 2*(forward_t / self.ddim_num_inference_steps)
            avoidance_strength = base_strength #* timestep_factor
            if self.compose:
                first_six = obstacle_pts[0]
                next_four = obstacle_pts[1][:4]
                obstacle_pts=torch.cat([first_six, next_four], dim=0)
                obstacle_pts_stack=obstacle_pts.reshape(-1,2)
            else:
                obstacle_pts_stack = obstacle_pts.squeeze(0).reshape(-1, 2)
            obstacle_field = ObstacleField(obstacle_pts_stack.cpu().numpy(),distance_threshold=distance_threshold)
            for _ in range(3): # 3 for simple2d
                x_recon_apf = avoidance(x_recon.detach(), obstacle_field,avoidance_window=avoidance_window,avoidance_strength=avoidance_strength)
            # blend_factor = torch.sigmoid(torch.tensor((forward_t - 0) / 2)).item()
            # x_recon = blend_factor * x_recon_apf + (1 - blend_factor) * x_recon
                x_recon=apply_hard_conditioning(x_recon_apf,hard_conds)
            # print(time.time()-st)
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
        time_idx = self.ddim_set_timesteps(self.ddim_num_inference_steps)
        # breakpoint()
        if not self.compose:
            obstacle_pts = obstacle_pts.unsqueeze(0)
        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for forward_t,i in enumerate(time_idx): # if np array, i is <class 'numpy.int64'>
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            assert not obstacle_pts.requires_grad
            # breakpoint()
            x = self.ddim_p_sample(x, hard_conds, context,timesteps, obstacle_pts,traj_normalized=traj_normalized, forward_t=forward_t,eta=0.0, use_clipped_model_output=True)
            x = apply_hard_conditioning(x, hard_conds)
            # progress.update({'t': i})
            # breakpoint()
            x = x.detach()

            if return_chain: diffusion.append(x)
        # progress.close()
        if return_chain:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, hard_conds, horizon=None, batch_size=1, ddim=False, traj_normalized=None,obstacle_pts=None,**sample_kwargs):
        '''
            hard conditions : hard_conds : { (time, state), ... }
        '''
        
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.state_dim)
        with torch.set_grad_enabled(self.energy_mode):
            if self.ddim:
                # return self.ddim_sample(shape, hard_conds, traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,**sample_kwargs)
                return self.ddim_p_sample_loop(shape, hard_conds, traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,**sample_kwargs) 
            # breakpoint()
            return self.p_sample_loop(shape, hard_conds, traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,**sample_kwargs)

    def forward(self, cond, *args, **kwargs):
        raise NotImplementedError
        return self.conditional_sample(cond, *args, **kwargs)

    @torch.no_grad()
    def warmup(self, horizon=64, traj_normalized=None,obstacle_pts=None,batch_size=None,device='cuda'):
        num_warmup=1
        shape = (batch_size, horizon, self.state_dim)
        x = torch.randn(shape, device=device)
        t = make_timesteps(batch_size, 1, device)
        # breakpoint()
        if not self.compose:
            obstacle_pts = obstacle_pts.unsqueeze(0)
        if self.compose:
            n_samples = x.shape[0]  # Should be 5 in this case 
            # Repeat each sample 3 times (for two environments and unconditional)
            x2 = x.repeat_interleave(3, dim=0)
            t2 = t.repeat_interleave(3, dim=0)
            traj_normalized2 = traj_normalized.repeat_interleave(3, dim=0) if traj_normalized is not None else None  
        # Handle obstacle_pts
            assert obstacle_pts.shape == (2, 6, 64, 2), "Obstacle points should have shape (2, 6, 64, 2)"
        # Use the first configuration for unconditional (change this to the second if preferred)
            obstacle_pts_uncond = obstacle_pts[0].unsqueeze(0)  # Shape: (1, 6, 64, 2)
        # Concatenate the two configurations and the unconditional
            obstacle_pts_all = torch.cat([obstacle_pts, obstacle_pts_uncond], dim=0)  # Shape: (3, 6, 64, 2)
        # Repeat for all samples
            obstacle_pts2 = obstacle_pts_all.repeat(n_samples, 1, 1, 1)  # Shape: (n_samples * 3, 6, 64, 2)
        else:
            x2,t2,traj_normalized2,obstacle_pts2=self.deep_repeat_tensor(x, t, traj_normalized, obstacle_pts, 2)
        uncond=True 
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(x2, t2, context=None,x_start=traj_normalized2,obstacle_pts=obstacle_pts2,compose=self.compose)
        # torch.cuda.synchronize()  #
        # self.model(x, t, context=None)
        
    @torch.no_grad()
    def run_inference(self, context=None, hard_conds=None, n_samples=1, return_chain=False,traj_normalized=None,obstacle_pts=None, **diffusion_kwargs):
        # context and hard_conds must be normalized
        hard_conds = copy(hard_conds)
        context = copy(context)
        # breakpoint()
        # repeat hard conditions and contexts for n_samples
        for k, v in hard_conds.items():
            new_state = einops.repeat(v, 'd -> b d', b=n_samples)
            hard_conds[k] = new_state
        # breakpoint()
        samples, chain = self.conditional_sample(
            hard_conds, context=context, batch_size=n_samples, ddim=False, return_chain=True, traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,**diffusion_kwargs
        )

        # chain: [ n_samples x (n_diffusion_steps + 1) x horizon x (state_dim)]
        # extract normalized trajectories
        trajs_chain_normalized = chain

        # trajs: [ (n_diffusion_steps + 1) x n_samples x horizon x state_dim ]
        trajs_chain_normalized = einops.rearrange(trajs_chain_normalized, 'b diffsteps h d -> diffsteps b h d')

        if return_chain:
            return trajs_chain_normalized

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

    def p_losses(self, x_start, context, t, hard_conds,obstacle_pts): # c 

        noise = torch.randn_like(x_start)
        # breakpoint() # if normalize change here both the traj and the hard conds 
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # if not self.predict_epsilon:
        # x_noisy = apply_hard_conditioning(x_noisy, hard_conds) # removed
        x_noisy[:,0,:]=x_start[:,0,:]  # added
        x_noisy[:,-1,:]=x_start[:,-1,:] # added 
        
        if context is not None:
            context = self.context_model(context)
        # diffusion model
        # breakpoint()
        if self.energy_mode and self.training:
            # x_recon,engy_batch=self.model(x_noisy, t, context,x_start=apply_hard_conditioning(x_start,hard_conds),obstacle_pts=obstacle_pts)
            x_recon,engy_batch=self.model(x_noisy, t, context,obstacle_pts=obstacle_pts)
        else:
            # x_recon = self.model(x_noisy, t, context,x_start=apply_hard_conditioning(x_start,hard_conds),obstacle_pts=obstacle_pts)
            x_recon=self.model(x_noisy, t, context,obstacle_pts=obstacle_pts)
        x_recon[:,0,:]=x_start[:,0,:]
        x_recon[:,-1,:]=x_start[:,-1,:]

        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)
        return loss, info

    def loss(self, x, context, *args): # called from losses.GaussianDiffusionLoss 
        batch_size = x.shape[0]
        t = torch.randint(0, self.n_diffusion_steps, (batch_size,), device=x.device).long()
        return self.p_losses(x, context, t, *args) # args is hard conds 

