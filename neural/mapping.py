import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, Callable

from fm import FM, FMEulerSampler

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000, bias=True, swiglu=False):
        super().__init__()
        if swiglu:
            self.mlp = SwiGLUMlp(frequency_embedding_size, int(2 / 3 * 4 * hidden_size), hidden_size, bias=bias)
        else:
            self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of (N) indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, max_period=self.max_period)
        t_emb = self.mlp(t_freq)
        return t_emb

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
        self.drop = nn.Dropout(drop)

    # @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.drop(hidden))

class ResBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, bias=False, dropout=0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mlp = SwiGLUMlp(dim, int(2 / 3 * mlp_ratio * dim), bias=bias, drop=dropout)
        self.gate = nn.Parameter(torch.ones(dim) * 1e-2)
    
    def forward(self, x):
        return x + self.gate * self.mlp(self.norm(x))

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift

class DiffusionResBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, bias=False, dropout=0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mlp = SwiGLUMlp(dim, int(2 / 3 * mlp_ratio * dim), bias=bias, drop=dropout)
        self.scale_shift_table = nn.Parameter(
            torch.randn(3, dim) / dim ** 0.5,
        )
        
    def forward(self, x, t):
        biases = self.scale_shift_table[None] + t.reshape(x.size(0), 3, -1)
        (
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = [chunk for chunk in biases.chunk(3, dim=1)]
        
        x = x + gate_mlp * self.mlp(modulate(self.norm(x), shift_mlp, scale_mlp))
        return x

class ResNetMLP(nn.Module):
    def __init__(self, dim=768, n_blocks=4, mlp_ratio=4, bias=True, dropout=0):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(dim, mlp_ratio=mlp_ratio, bias=bias, dropout=dropout) for _ in range(n_blocks)
        ])
        self.norm = RMSNorm(dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)
        # for block in self.blocks:
        #     torch.nn.init.zeros_(block.mlp.w3.weight)

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
    
    def forward(self, x, y=None):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        if y is not None:
            loss = F.mse_loss(x, y)
            return loss, x
        return x

class DiffusionResNetMLP(nn.Module):
    def __init__(self, dim=768, n_blocks=4, mlp_ratio=4, bias=False, dropout=0):
        super().__init__()
        self.t_embedder = TimestepEmbedder(dim, bias=False, swiglu=True)
        self.fuse_conditioning = SwiGLUMlp(dim * 2, int(2 / 3 * mlp_ratio * dim), dim, bias=False)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 3, bias=True),
        )
        
        self.blocks = nn.ModuleList([
            DiffusionResBlock(dim, mlp_ratio=mlp_ratio, bias=bias, dropout=dropout) for _ in range(n_blocks)
        ])
        self.norm = RMSNorm(dim)
        self.final_layer_scale_shift_table = nn.Parameter(
            torch.randn(2, dim) / dim ** 0.5,
        )
        self.fc = nn.Linear(dim, dim, bias=False)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.t_block[-1].weight)
        nn.init.zeros_(self.t_block[-1].bias)
        
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
    
    def forward(self, x, t, actions):
        t = self.t_embedder(t)
        t = torch.cat([t, actions.squeeze(1)], dim=-1)
        t = self.fuse_conditioning(t)
        t0 = self.t_block(t)
        
        for block in self.blocks:
            x = block(x, t0)
        
        shift, scale = (self.final_layer_scale_shift_table[None] + F.silu(t[:, None])).chunk(
            2, dim=1
        )
        x = modulate(self.norm(x), shift, scale)
        x = self.fc(x)
        
        return x

class ResNetDiffusion(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = DiffusionResNetMLP(**kwargs)
        
        self.diffusion = FM(timescale=1000)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, actions, x):
        return self.diffusion.loss(self.net, x, net_kwargs={'actions': actions}), None

    def sample(self, actions, x, n_steps=50, noise=None):
        return self.sampler.sample(self.net, x.shape, net_kwargs={'actions': actions}, n_steps=n_steps, noise=noise)

def DiffusionMLP_B(**kwargs):
    return ResNetDiffusion(n_blocks=12, **kwargs)

def MLP_L(**kwargs):
    return ResNetMLP(n_blocks=24, **kwargs)

def MLP_B(**kwargs):
    return ResNetMLP(n_blocks=12, **kwargs)

def MLP_S(**kwargs):
    return ResNetMLP(n_blocks=8, **kwargs)

def MLP_T(**kwargs):
    return ResNetMLP(n_blocks=4, **kwargs)