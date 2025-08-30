import torch
from matplotlib import pyplot as plt


def apply_hard_conditioning(x, conditions):
    for t, val in conditions.items():
        # breakpoint()
        x[:, t, :] = val.clone()
        # x[:,[2,-3],2:]=0
    return x


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


@torch.no_grad()
def ddpm_sample_fn(
        model, x, hard_conds, context, t,traj_normalized=None,obstacle_pts=None,forward_t=None,compose=None,
        noise_std_extra_schedule_fn=None,  # 'linear'
        **kwargs
):
    t_single = t[0]
    if t_single < 0:
        t = torch.zeros_like(t)
    if compose:
        model_mean, _, model_log_variance = model.p_mean_variance_compose(x=x, hard_conds=hard_conds, context=context, t=t,traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,forward_t=forward_t,compose=compose)
    else:
        model_mean, _, model_log_variance = model.p_mean_variance(x=x, hard_conds=hard_conds, context=context, t=t,traj_normalized=traj_normalized,obstacle_pts=obstacle_pts,forward_t=forward_t,compose=compose)
    x = model_mean

    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    if noise_std_extra_schedule_fn is None:
        noise_std = 1.0
    else:
        noise_std = noise_std_extra_schedule_fn(t_single)

    values = None
    return x + model_std * noise * noise_std, values

