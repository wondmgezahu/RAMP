import torch
import torch.nn as nn
import torch.nn.functional as F

class PointProcessor(nn.Module):
    def __init__(self, input_dim=3, output_dim=128):
        super(PointProcessor, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.activation = nn.SELU()
        
    def forward(self, x):
        # x: (batch_size, num_points, input_dim)
        x = x.transpose(2, 1)  # (batch_size, input_dim, num_points)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = torch.max(x, 2)[0]  # Global max pooling for permutation invariance
        return x

class SetTransformerBlock(nn.Module):
    
    def __init__(self, dim=128, num_heads=4, dropout=0.1):
        super(SetTransformerBlock, self).__init__()

        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, num_obstacles, dim) -> need to transpose for MHA
        x_norm = self.norm1(x)
        x_norm = x_norm.transpose(0, 1)  # (num_obstacles, batch_size, dim)
        
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        attn_out = attn_out.transpose(0, 1)  # (batch_size, num_obstacles, dim)
        
        x = x + self.dropout(attn_out)
        
        # FFN with residual
        x_norm = self.norm2(x)
        x = x + self.dropout(self.ffn(x_norm))
        
        return x

class ObstacleEncoder(nn.Module):
   
    def __init__(self, embedding_dim=256, point_dim=3, num_layers=2):
        super(ObstacleEncoder, self).__init__()
        
        self.point_processor = PointProcessor(input_dim=point_dim, output_dim=embedding_dim)
        
        self.set_transformer_blocks = nn.ModuleList([
            SetTransformerBlock(dim=embedding_dim) for _ in range(num_layers)
        ])
        
        # Final projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.global_pooling = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, obstacle_points):
        # obstacle_points: (batch_size, num_obstacles, num_points, 3)
        batch_size, num_obstacles, num_points, _ = obstacle_points.size()
        
        x = obstacle_points.reshape(-1, num_points, obstacle_points.size(-1))
        x = self.point_processor(x)  # (batch_size * num_obstacles, embedding_dim)
        
        x = x.view(batch_size, num_obstacles, self.embedding_dim)
        
        for block in self.set_transformer_blocks:
            x = block(x)

        obstacle_features = self.output_proj(x)  # (batch_size, num_obstacles, embedding_dim)
    
        scene_embedding = torch.max(obstacle_features, dim=1)[0]  # (batch_size, embedding_dim)
        scene_embedding = self.global_pooling(scene_embedding)
        # breakpoint()
        return scene_embedding