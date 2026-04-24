import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

import torch
import torch.nn as nn
from typing import Optional
from einops import rearrange

class DropPath(nn.Module):
    """Stochastic Depth: Randomly drops paths (blocks) per sample during training."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ToMel(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels):
        super().__init__()
        self.transform = torch.nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=40.0,
                f_max=sample_rate // 2,
                power=2.0,
                normalized=True,      # Normalizes the STFT to be magnitude invariant
                center=True,          # Padding to keep time/length consistent
                pad_mode='reflect'    # Better for audio boundary artifacts
            ),
            T.AmplitudeToDB(top_db=80.0)
        )
        # self.mu = -34.36543
        # self.std = 15.82586
    
    @torch.compiler.disable
    def forward(self, x):
        x = self.transform(x)
        
        mu = x.mean((-1, -2), keepdims=True)
        std = x.std((-1, -2), keepdims=True)
        x = (x - mu) / (std + 1e-6)
        # x = (x - self.mu) / (self.std + 1e-6)
        return x

class SpecAugment(nn.Module):
    def __init__(self, time_length=32, frequency_length=64):
        super().__init__()
        self.time_mask = T.TimeMasking(time_length)
        self.freq_mask = T.FrequencyMasking(frequency_length)
    
    @torch.compiler.disable
    def forward(self, x):
        x = self.time_mask(x)
        x = self.time_mask(x)
        x = self.freq_mask(x)
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
    
    def forward(self, x, freqs_cis=None):
        x = x + self.attn(self.norm1(x), freqs_cis=freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        
        return input + self.drop_path(x)

class MIL(nn.Module):
    def __init__(self, num_instruments, n_fft=1024, hop_length=512, n_mels=192, in_chans=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., num_heads=8, transformer_layers=2, mlp_ratio=4, time_length=32, frequency_length=64):
        super().__init__()
        
        self.to_mel = ToMel(16000, n_fft, hop_length, n_mels)
        self.augment = SpecAugment(time_length, frequency_length)
        
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample = nn.Sequential(
                nn.LayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        
        # self.transformer_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        # self.transformer_proj = nn.Linear(dims[-1], dims[-1])
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(dims[-1], num_heads, mlp_ratio=mlp_ratio) for _ in range(transformer_layers)
        ])
        
        self.queries = nn.Parameter(torch.randn(num_instruments, dims[-1]) * 0.02)
        self.query_norm = RMSNorm(dims[-1])
        self.proj = nn.Linear(dims[-1], num_instruments)
        
        self.head_dim = dims[-1] // num_heads
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        nn.init.zeros_(self.proj.weight)
        # zero out c_proj weights in all blocks
        for block in self.blocks:
            nn.init.zeros_(block.mlp.w3.weight)
            nn.init.zeros_(block.attn.proj.weight)

    def forward(self, x, targets=None):
        x = self.to_mel(x)
        
        if self.training:
            x = self.augment(x)
        
        for i in range(4):
            if i == 0:
                x = self.downsample_layers[i][0](x)
                x = x.permute(0, 2, 3, 1)
                x = self.downsample_layers[i][1](x)
                x = x.permute(0, 3, 1, 2)
            else:
                x = x.permute(0, 2, 3, 1)
                x = self.downsample_layers[i][0](x)
                x = x.permute(0, 3, 1, 2)
                x = self.downsample_layers[i][1](x)
                
            x = self.stages[i](x)
        
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        freqs_cis = precompute_freqs_cis_2d(
            dim=self.head_dim, 
            height=H, 
            width=W
        ).to(x.device)
        # x = self.transformer_norm(x)
        # x = self.transformer_proj(x)
        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis)
        
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        frame_logits = torch.einsum('bftc,nc->bftn', x, self.query_norm(self.queries)).max(dim=1)[0]
        frame_probs = torch.sigmoid(frame_logits) 
        
        temporal_features = x.mean(dim=1)
        attn_logits = self.proj(temporal_features)
        
        attn_weights = torch.softmax(attn_logits, dim=1)
        clip_logits = torch.sum(frame_logits * attn_weights, dim=1)
        
        outputs = {}
        outputs['clip_logits'] = clip_logits
        outputs['frame_probs'] = frame_probs
        
        if targets is not None:
            alpha = 0.2
            smooth_targets = targets * (1.0 - alpha) + (alpha / 2.0)
            outputs['loss'] = F.binary_cross_entropy_with_logits(clip_logits, smooth_targets)
        
        return outputs

class ImageEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels=320) -> None:
        super().__init__()
        self.f = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        return self.f(x)


class ImageUnembedding(nn.Module):
    def __init__(self, in_channels=320, out_channels=3) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(32, in_channels)
        self.f = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        return self.f(F.silu(self.gn(x)))

class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        kernel_size: int = 3
    ):
        super().__init__()
        assert out_channels % factor == 0, f'{out_channels}, {factor}'
        self.norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels // factor, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pixel_unshuffle = nn.PixelUnshuffle(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.conv(x)
        x = self.pixel_unshuffle(x)
        return x

class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert in_channels * factor % out_channels == 0, f'{in_channels} {factor} {out_channels}'
        self.group_size = in_channels * factor // out_channels
        self.pixel_unshuffle = nn.PixelUnshuffle(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixel_unshuffle(x)
        B, C, L = x.shape
        x = x.view(B, self.out_channels, self.group_size, L)
        x = x.mean(dim=2)
        return x

class DownsampleV3(nn.Module):
    def __init__(self, in_channels, out_channels, ratio):
        super().__init__()
        self.conv = ConvPixelUnshuffleDownSampleLayer(in_channels, out_channels, ratio, ratio * 2 + 1)
        self.shortcut = PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, out_channels, ratio)
    
    def forward(self, x, t=None):
        x = self.conv(x) + self.shortcut(x)
        return x

class ConvPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        kernel_size: int = 3
    ):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels * factor, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pixel_shuffle = nn.PixelShuffle(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels * factor % in_channels == 0, f'{out_channels} {factor} {in_channels}'
        self.repeats = out_channels * factor // in_channels
        self.pixel_shuffle = nn.PixelShuffle(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = self.pixel_shuffle(x)
        return x

class UpsampleV3(nn.Module):
    def __init__(self, in_channels, out_channels, ratio):
        super().__init__()
        self.conv = ConvPixelShuffleUpSampleLayer(in_channels, out_channels, ratio, ratio * 2 + 1)
        self.shortcut = ChannelDuplicatingPixelUnshuffleUpSampleLayer(in_channels, out_channels, ratio)
    
    def forward(self, x, t=None):
        x = self.conv(x) + self.shortcut(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_instruments, n_fft=1024, hop_length=512, n_mels=192, in_chans=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., time_length=32, frequency_length=64):
        super().__init__()
        
        self.to_mel = ToMel(16000, n_fft, hop_length, n_mels)
        self.augment = SpecAugment(time_length, frequency_length)
        
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            if i == len(depths) - 1:
                downsample = DownsampleV3(dims[i], dims[i], 2)
            else:
                downsample = DownsampleV3(dims[i], dims[i+1], 2)
            self.downsample_layers.append(downsample)

        self.down_stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.down_stages.append(stage)
            cur += depths[i]
        
        self.skip_projs = nn.ModuleList([])
        self.upsample_layers = nn.ModuleList([])
        # self.upsample_layers.append(nn.ModuleList([AdaLNConvBlock(channels[-1], channels[-1], type=type)]))
        for i in reversed(range(3)):
            if i == len(depths) - 1:
                upsample = UpsampleV3(dims[i], dims[i], 2)
            else:
                upsample = UpsampleV3(dims[i+1], dims[i], 2)
            self.upsample_layers.insert(0, upsample)
            self.skip_projs.insert(0, nn.Conv1d(dims[i] * 2, dims[i], kernel_size=1))
        
        self.up_stages = nn.ModuleList()
        cur = 0
        for i in reversed(range(4)):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.up_stages.insert(0, stage)
            cur += depths[i]
        
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.proj = nn.Linear(dims[-1], num_instruments)
        
        self.apply(self._init_weights)
    
    
        # depths = [3] * len(channels)
        # self.skip_projs = nn.ModuleList([])
        # self.up = nn.ModuleList([])
        # self.up.append(nn.ModuleList([AdaLNConvBlock(channels[-1], channels[-1], type=type)]))
        # for i, (channel, depth, ratio) in reversed(list(enumerate(zip(channels, depths, ratios)))):
        #     blocks = nn.ModuleList([])
        #     if ratio > 1:
        #         if i == len(channels) - 1:
        #             blocks.append(UpsampleV3(channel, channel, ratio))
        #         else:
        #             blocks.append(UpsampleV3(channels[i+1], channel, ratio))
        #     self.skip_projs.insert(0, nn.Conv1d(channel * 2, channel, kernel_size=1))
        #     for _ in range(depth):
        #         blocks.append(AdaLNConvBlock(channel, channel, dilation=2 ** _, type=type))
        #     self.up.insert(0, blocks)

        # self.output = ImageUnembedding(in_channels=channels[0], out_channels=1)
    
        # self.initialize_weights()
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        nn.init.zeros_(self.proj.weight)
        # zero out c_proj weights in all blocks
        # for block in self.blocks:
        #     nn.init.zeros_(block.mlp.w3.weight)
        #     nn.init.zeros_(block.attn.proj.weight)

    def forward(self, x, targets=None):
        x = self.to_mel(x)
        
        if self.training:
            x = self.augment(x)
        
        # down
        skips = [x]
        for i in range(4):
            print(i, x.shape)
            if i == 0:
                x = self.downsample_layers[i][0](x)
                x = x.permute(0, 2, 3, 1)
                x = self.downsample_layers[i][1](x)
                x = x.permute(0, 3, 1, 2)
            else:
                x = self.downsample_layers[i](x)
                
            x = self.down_stages[i](x)
            skips.append(x)
            print(i, x.shape)
        
        # middle
        

        # up
        skips.pop()
        for i in reversed(range(4)):
            x = torch.cat([x, skips.pop()], dim=1)
            x = self.skip_projs[-i](x)
            x = self.up_stages[i](x)
            print(x.shape)
            
            x = x.permute(0, 2, 3, 1)
            x = self.downsample_layers[i][0](x)
            x = x.permute(0, 3, 1, 2)
            x = self.downsample_layers[i][1](x)
        
        x = x.mean(2, 3)
        x = self.norm(x)
        x = self.proj(x)
        
        if targets is not None:
            alpha = 0.2
            smooth_targets = targets * (1.0 - alpha) + (alpha / 2.0)
            loss = F.binary_cross_entropy_with_logits(x, smooth_targets, reduction='none')
            loss_mask = targets > -1
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            return loss
        
        return x

    # def forward(self, x, t=None, z_dec=None) -> torch.Tensor:
        

    #     skips = [x]
    #     for i, down in enumerate(self.down):
    #         c = self.c_projs[i](t) + self.c_projs[i](self.interpolate(x, z_dec))
    #         for block in down:
    #             x = block(x, c)
    #         skips.append(x)

    #     c = self.c_projs[-1](t) + self.c_projs[-1](self.interpolate(x, z_dec))
    #     for mid in self.mid:
    #         x = mid(x, c)

    #     skips.pop()
    #     for i, up in enumerate(reversed(self.up)):
    #         for block in up:
    #             if isinstance(block, UpsampleV3):
    #                 x = block(x, c)
    #                 x = torch.cat([x, skips.pop()], dim=1)
    #                 x = self.skip_projs[-i](x)
    #                 c = self.c_projs[-i](t) + self.c_projs[-i](self.interpolate(x, z_dec))
    #             else:
    #                 x = block(x, c)

    #     return self.output(x)

if __name__ == '__main__':
    net = UNet(10)
    x = torch.randn(32, 1, 16383 * 5)
    y = net(x)