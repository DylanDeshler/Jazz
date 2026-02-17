import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from typing import Optional
from einops import rearrange
import math

from pytorch_metric_learning import losses

class ToMel(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels):
        super().__init__()
        self.transform = torch.nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=0.0,
                f_max=sample_rate // 2,
                power=2.0,
                normalized=True,      # Normalizes the STFT to be magnitude invariant
                center=True,          # Padding to keep time/length consistent
                pad_mode='reflect'    # Better for audio boundary artifacts
            ),
            T.AmplitudeToDB(top_db=80.0)
        )
    
    def forward(self, x):
        return self.transform(x)

class SpecAugment(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.time_mask = T.TimeMasking(30)
        self.freq_mask = T.FrequencyMasking(30)
    
    @torch.compiler.disable
    def forward(self, x):
        x = self.time_mask(x)
        x = self.freq_mask(x)
        return x

def apply_scaling(freqs: torch.Tensor):
    # RoPE scaling (values obtained from grid search)
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    freqs_cis_real = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return freqs_cis_real

import torch

def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0):
    """
    Precompute 2D rotary position embeddings.
    
    Args:
        dim: Dimension of the head (must be even and divisible by 2 for split).
        height: Maximum height (H).
        width: Maximum width (W).
        theta: Base period for the angles.
    
    Returns:
        freqs_cis: [H*W, dim/2, 2] complex64 tensor representing the flattened 2D position grid.
    """
    # Split the dimension into two halves: one for H (Freq), one for W (Time)
    dim_h = dim // 2
    dim_w = dim // 2
    
    # 1. Compute frequencies for Height (Y-axis / Frequency)
    freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
    t_h = torch.arange(height, device=freqs_h.device)
    freqs_h = torch.outer(t_h, freqs_h)  # [H, dim_h/2]
    
    # 2. Compute frequencies for Width (X-axis / Time)
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
    t_w = torch.arange(width, device=freqs_w.device)
    freqs_w = torch.outer(t_w, freqs_w)  # [W, dim_w/2]

    # 3. Broadcast to create the 2D grid
    # We want every row (H) to have the same Width embeddings, 
    # and every column (W) to have the same Height embeddings.
    
    # freqs_h: [H, 1, dim_h/2] -> Broadcast over W
    freqs_h_grid = freqs_h.unsqueeze(1).repeat(1, width, 1)
    
    # freqs_w: [1, W, dim_w/2] -> Broadcast over H
    freqs_w_grid = freqs_w.unsqueeze(0).repeat(height, 1, 1)
    
    # 4. Concatenate the frequencies [H, W, dim/2]
    # This aligns with the channel split: first half Y, second half X
    freqs_2d = torch.cat([freqs_h_grid, freqs_w_grid], dim=-1)
    
    # 5. Convert to complex form (polar)
    freqs_cis = torch.polar(torch.ones_like(freqs_2d), freqs_2d)
    
    # 6. Return as real/imag stack for efficient multiplication
    # Shape: [H, W, dim/2, 2]
    freqs_cis_real = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    
    # Flatten spatial dims to match the transformer sequence [N, dim/2, 2]
    # N = H * W
    return freqs_cis_real.reshape(-1, freqs_cis_real.shape[-2], 2)

def apply_rotary_emb_2d(x, freqs_cis):
    """
    Apply 2D RoPE.
    Args:
        x: [B, N, num_heads, head_dim] input tensor (where N = H*W).
        freqs_cis: [N, head_dim/2, 2] precomputed frequencies.
    """
    # x: (bs, seqlen, n_heads, head_dim)
    # freqs_cis: (seq_len, head_dim/2, 2)
    
    # Reshape x into pairs for complex multiplication
    # xshaped: (bs, seqlen, n_heads, head_dim/2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    
    # Broadcast freqs_cis to match batch and heads
    # freqs_cis: (1, seqlen, 1, head_dim/2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    
    # Complex multiplication (rotation)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    
    # Flatten back to original shape
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

def apply_rotary_emb(x, freqs_cis):
    # shape gymnastics let's go
    # x is (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)
    # freqs_cis is (seq_len, head_dim/2, 2), e.g. (8, 64, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    # xshaped is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    # freqs_cis becomes (1, seqlen, 1, head_dim/2, 2), e.g. (1, 8, 1, 64, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    # x_out2 at this point is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    x_out2 = x_out2.flatten(3)
    # x_out2 is now (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)
    return x_out2.type_as(x)

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

class SelfAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            freqs_cis: Optional[torch.Tensor] = None,
            attn_mask = None,
            is_causal: bool = False
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # RoPE
        if freqs_cis is not None:
            q = apply_rotary_emb_2d(q.transpose(1, 2), freqs_cis).transpose(1, 2)
            k = apply_rotary_emb_2d(k.transpose(1, 2), freqs_cis).transpose(1, 2)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            raise NotImplementedError()
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor,
            freqs_cis: Optional[torch.Tensor] = None,
            attn_mask = None,
            is_causal: bool = False
    ) -> torch.Tensor:
        B, N, C = x.shape
        B, M, C = context.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            raise NotImplementedError()
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwiGLUMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=False, **block_kwargs)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=False)
    
    def forward(self, x, freqs_cis=None, is_causal=False):
        x = x + self.attn(self.norm1(x), is_causal=is_causal, freqs_cis=freqs_cis[:x.shape[1]] if freqs_cis is not None else None)
        x = x + self.mlp(self.norm2(x))
        return x

import numpy as np
means, stds = [], []
class Transformer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 patch_size,
                 sample_rate,
                 n_fft,
                 hop_length,
                 n_mels,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.head_dim = hidden_size // num_heads
        
        self.to_mel = ToMel(sample_rate, n_fft, hop_length, n_mels)
        self.augment = SpecAugment()
        self.x_embedder = nn.Linear(in_channels * patch_size * patch_size, hidden_size, bias=False)
        
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.pool = CrossAttention(hidden_size, num_heads, qkv_bias=False, proj_bias=False)

        self.norm = RMSNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.criterion = losses.SelfSupervisedLoss(losses.NTXentLoss(temperature=0.5), symmetric=True)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        nn.init.zeros_(self.fc.weight)
        # zero out c_proj weights in all blocks
        for block in self.blocks:
            nn.init.zeros_(block.mlp.w3.weight)
            nn.init.zeros_(block.attn.proj.weight)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    @torch.compiler.disable
    def _compute_mel(self, x):
        return self.to_mel(x)
    
    def _compute_loss(self, x):
        embs = x[::2]
        ref_embs = x[1::2]
        loss = self.criterion(embs, ref_embs)
        return loss
    
    def forward(self, x):
        x = self._compute_mel(x)
        
        means.append(x.mean())
        stds.append(x.std())
        print(np.mean(means), np.mean(stds))
        
        if self.training:
            x = self.augment(x)
        
        x = (x + 40) / 40
        # mu = x.mean((-1, -2), keepdims=True)
        # std = x.std((-1, -2), keepdims=True)
        # x = (x - mu) / (std + 1e-6)
        
        B, C, H, W = x.shape
        
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
        x = self.x_embedder(x)
        
        freqs_cis = precompute_freqs_cis_2d(
            dim=self.head_dim, 
            height=H // self.patch_size, 
            width=W // self.patch_size
        ).to(x.device)
        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis)
        
        x = self.norm(x)
        
        x = self.pool(x.mean(1, keepdims=True), x).squeeze(1)
        x = self.fc(x)
        x = F.normalize(x, dim=-1)
        
        loss = self._compute_loss(x)
        
        out = {'loss': loss, 'z': x}
        
        return out

if __name__ == '__main__':
    depth = 12
    hidden_size = 768
    num_heads = 12
    patch_size = 16
    sample_rate = 16000
    n_fft = 1024
    hop_length = 512
    n_mels = 192
    
    model = Transformer(
        in_channels=1,
        patch_size=patch_size,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        depth=depth,
        hidden_size=hidden_size,
        num_heads=num_heads
    ).to('cuda')
    
    x = torch.randn(16, 1, 10 * sample_rate).to('cuda')
    out = model(x)
    
    print(out.shape)