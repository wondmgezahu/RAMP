import torch
from torch import nn
import torch.nn.functional as F
import math

class MultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SetTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ObstaclePositionalEncoding(nn.Module):

    def __init__(self, d_model, num_obstacles=6):
        super().__init__()
        self.d_model = d_model
        self.num_obstacles = num_obstacles
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        """
        x: tensor of shape (batch, num_obstacles, num_points, 2)
        """
        batch_size, num_obstacles, num_points, _ = x.shape
        
        # Compute center of each obstacle
        obstacle_centers = x.mean(dim=2)  # (batch, num_obstacles, 2)
        
        # Compute PE for obstacle centers
        pe_obstacles = torch.zeros(batch_size, num_obstacles, self.d_model, device=x.device)
        pe_obstacles[..., 0::2] = torch.sin(obstacle_centers[..., 0, None] * self.div_term)
        pe_obstacles[..., 1::2] = torch.cos(obstacle_centers[..., 0, None] * self.div_term)
        pe_obstacles[..., 0::2] += torch.sin(obstacle_centers[..., 1, None] * self.div_term)
        pe_obstacles[..., 1::2] += torch.cos(obstacle_centers[..., 1, None] * self.div_term)
        
        # Compute relative positions of points within each obstacle
        relative_positions = x - obstacle_centers.unsqueeze(2)
        
        # Normalize relative positions to be in [-1, 1] range
        max_distance, _ = torch.max(torch.abs(relative_positions).view(batch_size, num_obstacles, -1), dim=-1, keepdim=True)
        relative_positions_normalized = relative_positions / (max_distance.unsqueeze(-1) + 1e-8)
        
        # Compute PE for normalized relative positions
        pe_relative = torch.zeros(batch_size, num_obstacles, num_points, self.d_model, device=x.device)
        pe_relative[..., 0::2] = torch.sin(relative_positions_normalized[..., 0, None] * self.div_term)
        pe_relative[..., 1::2] = torch.cos(relative_positions_normalized[..., 0, None] * self.div_term)
        pe_relative[..., 0::2] += torch.sin(relative_positions_normalized[..., 1, None] * self.div_term)
        pe_relative[..., 1::2] += torch.cos(relative_positions_normalized[..., 1, None] * self.div_term)
        
        return pe_obstacles, pe_relative

class ObstacleEncoderSet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64,output_dims=[64, 96, 160], num_obstacles=6, num_points=64, num_blocks=3):
        super().__init__()
        self.num_obstacles = num_obstacles
        self.num_points = num_points
        
        self.pos_encoder = ObstaclePositionalEncoding(hidden_dim, num_obstacles)
        
        self.point_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.combined_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.set_transformers = nn.ModuleList([
            nn.Sequential(*[SetTransformerBlock(hidden_dim) for _ in range(num_blocks)])
            for _ in output_dims
        ])
        self.poolings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            ) for dim in output_dims
        ])
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Generate positional encodings
        pe_obstacles, pe_relative = self.pos_encoder(x)
        # breakpoint()
        # Embed points
        x = x.view(batch_size * self.num_obstacles * self.num_points, -1)
        point_embedding = self.point_embedding(x)
        point_embedding = point_embedding.view(batch_size, self.num_obstacles, self.num_points, -1)
        
        # Combine point embeddings with positional encodings
        combined = torch.cat([
            point_embedding, 
            pe_obstacles.unsqueeze(2).expand(-1, -1, self.num_points, -1), 
            pe_relative
        ], dim=-1)
        combined = self.combined_encoder(combined)
        
        # Reshape for set transformer
        combined = combined.view(batch_size, self.num_obstacles * self.num_points, -1)
        outputs = []
        for transformer, pooling in zip(self.set_transformers, self.poolings):
            transformed = transformer(combined)
            pooled = pooling(transformed.mean(dim=1))
            outputs.append(pooled)
        
        return torch.cat(outputs,dim=-1)