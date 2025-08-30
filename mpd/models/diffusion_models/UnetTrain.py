import einops
import numpy as np
import torch
import torch.nn as nn
from .obstacle_encoder import ObstacleEncoderSet
from .obstacle_encoder3d import ObstacleEncoder
from mpd.models.layers.layers import GaussianFourierProjection, Downsample1d, Conv1dBlock, Upsample1d, \
    ResidualTemporalBlock, TimeEncoder, MLP, group_norm_n_groups, LinearAttention, PreNorm, Residual, TemporalBlockMLP
from mpd.models.layers.layers_attention_mini import SpatialTransformer

UNET_DIM_MULTS = {
    0: (1, 2, 4),
    1: (1, 2, 4, 8)
}


class TemporalUnetTrain(nn.Module):

    def __init__(
            self,
            n_support_points=None,
            state_dim=None,
            unet_input_dim=32,
            dim_mults=(1, 2, 4, 8),
            time_emb_dim=32,
            self_attention=False,
            conditioning_embed_dim=4,
            conditioning_type='attention',
            attention_num_heads=4,
            attention_dim_head=64,
            obstacle_3d=False, 
            **kwargs
    ):
        super().__init__()

        self.state_dim = state_dim
        input_dim = state_dim
        self.energy_mode=True
        self.training=True
        self.drop_concept=True
        self.concept_drop_prob=0.2 # condition drop probability
        self.scene_encoder=ObstacleEncoder() if obstacle_3d else ObstacleEncoderSet()
        self.context_dim=256 if obstacle_3d else 320 # condition encoder dim
        if conditioning_type is None or conditioning_type == 'None':
            conditioning_type = None
        elif conditioning_type == 'concatenate':
            if self.state_dim < conditioning_embed_dim // 4:
                # Embed the state in a latent space HxF if the conditioning embedding is much larger than the state
                state_emb_dim = conditioning_embed_dim // 4
                self.state_encoder = MLP(state_dim, state_emb_dim, hidden_dim=state_emb_dim//2, n_layers=1, act='silu') # mish
            else:
                state_emb_dim = state_dim
                self.state_encoder = nn.Identity()
            input_dim = state_emb_dim + conditioning_embed_dim
        elif conditioning_type == 'attention':
            pass
        elif conditioning_type == 'default':
            pass
        else:
            raise NotImplementedError
        self.conditioning_type = conditioning_type

        dims = [input_dim, *map(lambda m: unet_input_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        # Networks
        self.time_mlp = TimeEncoder(32, time_emb_dim)
        cond_dim=time_emb_dim 
        self.depth_attn=2 # 3 #2 # 4 for mini and insideBig, 2 for inside
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            # breakpoint()
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, cond_dim, n_support_points=n_support_points),
                ResidualTemporalBlock(dim_out, dim_out, cond_dim, n_support_points=n_support_points),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if self_attention else nn.Identity(),
                SpatialTransformer(dim_out, attention_num_heads, attention_dim_head, depth=self.depth_attn,
                                   context_dim=self.context_dim), #  if conditioning_type == 'attention' else None,
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                n_support_points = n_support_points // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim, n_support_points=n_support_points)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if self_attention else nn.Identity()
        self.mid_attention = SpatialTransformer(mid_dim, attention_num_heads, attention_dim_head, depth=self.depth_attn,
                                                context_dim=self.context_dim) #  if conditioning_type == 'attention' else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim, n_support_points=n_support_points)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, cond_dim, n_support_points=n_support_points),
                ResidualTemporalBlock(dim_in, dim_in, cond_dim, n_support_points=n_support_points),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if self_attention else nn.Identity(),
                SpatialTransformer(dim_in, attention_num_heads, attention_dim_head, depth=self.depth_attn,
                                   context_dim=self.context_dim), #  if conditioning_type == 'attention' else None,
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                n_support_points = n_support_points * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(unet_input_dim, unet_input_dim, kernel_size=5, n_groups=group_norm_n_groups(unet_input_dim)),
            nn.Conv1d(unet_input_dim, state_dim, 1),
        )

    def forward(self, x, time, context,x_start=None,obstacle_pts=None,forward_t=None,uncond=False):
        """
        x : [ batch x horizon x state_dim ] [Bx64x4]
        context: [batch x context_dim]
        obstacle_pts: [batch x num_obstacle x n_point x dim] [Bx6x64x4]
        """
        obstacle_pts=obstacle_pts.to(x.dtype)
        if self.energy_mode:
            x_inp = x.requires_grad_(True)
        b, h, d = x.shape
        scene_latents=self.scene_encoder(obstacle_pts)
        if self.drop_concept and self.training: # drop concepts 
            b = scene_latents.shape[0]
            scene_latents[np.random.rand(b,) < self.concept_drop_prob] = 0. # uncond 
        t_emb = self.time_mlp(time)
        # scene_latents=scene_latents.unsqueeze(-1)
        x = einops.rearrange(x, 'b h c -> b c h')  # batch, horizon, channels (state_dim)
        h = []
        for resnet, resnet2, attn_self, attn_conditioning, downsample in self.downs:
            x = resnet(x, t_emb)
            # if self.conditioning_type == 'attention':
            #     x = attention1(x, context=conditioning_emb)
            x = resnet2(x, t_emb)
            x = attn_self(x)
            if self.conditioning_type == 'attention':
                # breakpoint()
                x = attn_conditioning(x, context=scene_latents)
                # breakpoint()
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        if self.conditioning_type == 'attention':
            x = self.mid_attention(x, context=scene_latents)
        x = self.mid_block2(x, t_emb)

        for resnet, resnet2, attn_self, attn_conditioning, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t_emb)
            x = resnet2(x, t_emb)
            x = attn_self(x)
            if self.conditioning_type == 'attention':
                x = attn_conditioning(x, context=scene_latents)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b c h -> b h c')
        if self.energy_mode:
            unet_out=x
            energy_norm=0.5*(unet_out**2)
            energy_norm=energy_norm.sum(dim=(1,2))
            if self.training:
                eps = torch.autograd.grad([energy_norm.sum()], [x_inp], create_graph=True)[0]
            else:
                # with torch.enable_grad():    
                eps = torch.autograd.grad([energy_norm.sum()], [x_inp], create_graph=False)[0]
            if self.training:
                engy_batch = energy_norm.sum()
                return eps, engy_batch.detach()
            else:
                return eps
        return x
