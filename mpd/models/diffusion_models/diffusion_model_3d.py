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

def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t

class GaussianDiffusionModel3d(nn.Module, ABC):

    def __init__(self,
                 model=None,
                 variance_schedule='exponential',
                 n_diffusion_steps=100,
                 clip_denoised=True,
                 predict_epsilon=False,
                 loss_type='l2', # 'l2smooth', # l2
                 context_model=None,
                 compose=False,
                 training=False,
                 **kwargs):
        super().__init__()
        # breakpoint()
        self.model = model
    
        self.context_model = context_model
        self.n_diffusion_steps = n_diffusion_steps
        self.energy_mode=True 
        self.training=training
        self.compose=compose 
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

    # def 
    # import torch

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

    def p_mean_variance(self, x, hard_conds, context, t,traj_normalized=None,obstacle_pts=None,forward_t=None,compose=False):
        x2,t2,traj_normalized2,obstacle_pts2=self.deep_repeat_tensor(x, t, traj_normalized, obstacle_pts, 2)
        out=self.model(x2, t2, context,x_start=traj_normalized2,obstacle_pts=obstacle_pts2,compose=compose)
        w=5.75 
        e_comb=(1+w)*out[0]-w*out[1] # cfg
        e_comb=e_comb.unsqueeze(0)
        x_recon = self.predict_start_from_noise(x, t=t, noise=e_comb) 
        # self.clip_denoised=False
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_mean_variance_compose(self, x, hard_conds, context, t,traj_normalized=None,obstacle_pts=None,forward_t=None,compose=True):
        x2,t2,traj_normalized2,obstacle_pts2=self.deep_repeat_tensor(x, t, traj_normalized, obstacle_pts, 3) # for a single compose
        # x2,t2,traj_normalized2,obstacle_pts2=self.deep_repeat_tensor(x, t, traj_normalized, obstacle_pts, 4) # for 2 compose
        obstacle_pts_uncond=obstacle_pts[1].unsqueeze(0) # for single compose
        # obstacle_pts_uncond=obstacle_pts[2].unsqueeze(0) # for 2 compose (more than two obstacle sets)
        obstacle_pts=torch.cat([obstacle_pts,obstacle_pts_uncond],dim=0)
        w1=5.
        w2=5.
        # w3=5.
        out=self.model(x2, t2, context,x_start=traj_normalized2,obstacle_pts=obstacle_pts,forward_t=forward_t,compose=compose)
        e_comb=out[2]+w1*(out[0]-out[2])+w2*(out[1]-out[2]) # conjunction composable diff  for a single compose
        # e_comb=out[3]+w1*(out[0]-out[3])+w2*(out[1]-out[3])+w3*(out[2]-out[3]) # conjunction composable diff  for 2 compose
        x_recon = self.predict_start_from_noise(x, t=t, noise=e_comb) 
        # self.clip_denoised=False
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, hard_conds, context=None, return_chain=False,traj_normalized=None,obstacle_pts=None,
                      sample_fn=ddpm_sample_fn,
                      n_diffusion_steps_without_noise=0,
                      **sample_kwargs):
        device = self.betas.device
        # for inference keep the n_samples to 1
        batch_size = shape[0]
        if not self.compose:
            obstacle_pts=obstacle_pts.unsqueeze(0)
        x = torch.randn(shape, device=device) # for energy use 1*torch.randd else 0.5 incomplete
        x = apply_hard_conditioning(x, hard_conds)
        chain = [x] if return_chain else None
        resample_steps=1
        forward_t=0
        for i in reversed(range(-n_diffusion_steps_without_noise, self.n_diffusion_steps)):
            t = make_timesteps(batch_size, i, device)
            # if i>15: # and i>0:
                # resample_steps=1 #24
            for r in range(resample_steps):
                x, values = sample_fn(self, x, hard_conds, context, t, traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,forward_t=forward_t,compose=self.compose,**sample_kwargs)
                x = apply_hard_conditioning(x, hard_conds)
                if r<resample_steps-1:
                    if t[0]<0:
                        t = torch.zeros_like(t)
                    x=self.q_sample(x_start=x, t=t)
                    x = apply_hard_conditioning(x, hard_conds)
            if return_chain:
                chain.append(x)
            forward_t+=1
        if return_chain:
            chain = torch.stack(chain, dim=1)
            return x, chain

        return x

    @torch.no_grad()
    def ddim_sample(
        self, shape, hard_conds,
        context=None, return_chain=False,
        t_start_guide=float('inf'), # cust torch.inf,
        guide=None,
        n_guide_steps=1,
        **sample_kwargs,
    ):
        # Adapted from https://github.com/ezhang7423/language-control-diffusion/blob/63cdafb63d166221549968c662562753f6ac5394/src/lcd/models/diffusion.py#L226
        device = self.betas.device
        batch_size = shape[0]
        total_timesteps = self.n_diffusion_steps
        sampling_timesteps = self.n_diffusion_steps // 5
        eta = 0.

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(0, total_timesteps - 1, steps=sampling_timesteps + 1, device=device)
        times = torch.cat((torch.tensor([-1], device=device), times))
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)
        x = apply_hard_conditioning(x, hard_conds)

        chain = [x] if return_chain else None

        for time, time_next in time_pairs:
            t = make_timesteps(batch_size, time, device)
            t_next = make_timesteps(batch_size, time_next, device)

            model_out = self.model(x, t, context)

            x_start = self.predict_start_from_noise(x, t=t, noise=model_out)
            pred_noise = self.predict_noise_from_start(x, t=t, x0=model_out)

            if time_next < 0:
                x = x_start
                x = apply_hard_conditioning(x, hard_conds)
                if return_chain:
                    chain.append(x)
                break

            alpha = extract(self.alphas_cumprod, t, x.shape)
            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            x = x_start * alpha_next.sqrt() + c * pred_noise

            # guide gradient steps before adding noise
     
            # add noise
            noise = torch.randn_like(x)
            x = x + sigma * noise
            x = apply_hard_conditioning(x, hard_conds)

            if return_chain:
                chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
            return x, chain

        return x

    @torch.no_grad()
    def conditional_sample(self, hard_conds, horizon=None, batch_size=1, ddim=False, traj_normalized=None,obstacle_pts=None,**sample_kwargs):
        '''
            hard conditions : hard_conds : { (time, state), ... }
        '''
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.state_dim)
        with torch.set_grad_enabled(self.energy_mode):
            if ddim:
                return self.ddim_sample(shape, hard_conds, **sample_kwargs)
            # breakpoint()
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
        # context and hard_conds must be normalized
        hard_conds = copy(hard_conds)
        context = copy(context)
        # breakpoint()
        # repeat hard conditions and contexts for n_samples
        for k, v in hard_conds.items():
            new_state = einops.repeat(v, 'd -> b d', b=n_samples)
            hard_conds[k] = new_state

        # if context is not None:
        #     for k, v in context.items():
        #         context[k] = einops.repeat(v, 'd -> b d', b=n_samples)
        # breakpoint()
        # Sample from diffusion model
        samples, chain = self.conditional_sample(
            hard_conds, context=context, batch_size=n_samples, return_chain=True, traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,**diffusion_kwargs
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
        x_noisy[:,0,:]=x_start[:,0,:]  
        x_noisy[:,-1,:]=x_start[:,-1,:] 
        # context model
        if context is not None:
            context = self.context_model(context)
        if self.energy_mode and self.training:
            x_recon,engy_batch=self.model(x_noisy, t, context,x_start=apply_hard_conditioning(x_start,hard_conds),obstacle_pts=obstacle_pts)
        else:
            x_recon = self.model(x_noisy, t, context,x_start=apply_hard_conditioning(x_start,hard_conds),obstacle_pts=obstacle_pts)
        # if not self.predict_epsilon: #  from potential diffusion planning
            # x_recon = apply_hard_conditioning(x_recon, hard_conds)
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

