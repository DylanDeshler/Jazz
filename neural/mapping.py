import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Callable

from fm import FM, FMEulerSampler

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLUMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    # @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class ResBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, bias=False, dropout=0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mlp = SwiGLUMlp(dim, int(2 / 3 * mlp_ratio * dim), bias=bias, drop=dropout)
    
    def forward(self, x):
        return x + self.mlp(self.norm(x))

class ResNetMLP(nn.Module):
    def __init__(self, dim=768, n_blocks=4, mlp_ratio=4, bias=False, dropout=0):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(dim, mlp_ratio=mlp_ratio, bias=bias, dropout=dropout) for _ in range(n_blocks)
        ])
        self.norm = RMSNorm(dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)
        for block in self.blocks:
            torch.nn.init.zeros_(block.mlp.w3.weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, y):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        loss = F.mse_loss(x, y)
        return loss, x

class ResNetDiffusion(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = ResNetMLP(**kwargs)
        
        self.diffusion = FM(timescale=1000)
        self.sampler = FMEulerSampler()
    
    def forward(self, x, y):
        return self.diffusion.target_loss(self.net, x, y), None

def DiffusionMLP_B(**kwargs):
    return ResNetDiffusion(n_blocks=12, **kwargs)

def MLP_B(**kwargs):
    return ResNetMLP(n_blocks=12, **kwargs)

def MLP_S(**kwargs):
    return ResNetMLP(n_blocks=8, **kwargs)

def MLP_T(**kwargs):
    return ResNetMLP(n_blocks=4, **kwargs)