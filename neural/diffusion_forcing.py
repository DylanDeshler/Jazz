import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat

import math
from typing import Optional, List, Callable

class FM:
    
    def __init__(self, sigma_min=1e-5, timescale=1.0):
        self.sigma_min = sigma_min
        self.prediction_type = None
        self.timescale = timescale
    
    def alpha(self, t):
        return 1.0 - t
    
    def sigma(self, t):
        return self.sigma_min + t * (1.0 - self.sigma_min)
    
    def A(self, t):
        return 1.0
    
    def B(self, t):
        return -(1.0 - self.sigma_min)
    
    def get_betas(self, n_timesteps):
        return torch.zeros(n_timesteps) # Not VP and not supported
    
    def add_noise(self, x, t, noise=None):
        noise = torch.randn_like(x) if noise is None else noise
        s = [x.shape[0], x.shape[1], x.shape[2], 1]
        x_t = self.alpha(t).view(*s) * x + self.sigma(t).view(*s) * noise
        return x_t, noise
    
    def loss(self, net, x, t=None, net_kwargs=None, return_loss_unreduced=False, return_all=False):
        B, T, N, C = x.shape
        
        if net_kwargs is None:
            net_kwargs = {}
        
        if t is None:
            t = torch.rand(B, T, device=x.device)
            repeat_t = t.unsqueeze(2).repeat(1, 1, N)
        x_t, noise = self.add_noise(x, repeat_t)
        
        pred = net(x_t, t=t * self.timescale, **net_kwargs)
        
        target = self.A(repeat_t) * x + self.B(repeat_t) * noise # -dxt/dt
        if return_loss_unreduced:
            loss = ((pred.float() - target.float()) ** 2).mean(dim=[1, 2])
            if return_all:
                return loss, t, x_t, pred
            else:
                return loss, t
        else:
            loss = ((pred.float() - target.float()) ** 2).mean()
            if return_all:
                return loss, x_t, pred
            else:
                return loss
    
    def get_prediction(
        self,
        net,
        x_t,
        t,
        net_kwargs=None,
        uncond_net_kwargs=None,
        guidance=1.0,
    ):
        if net_kwargs is None:
            net_kwargs = {}
        pred = net(x_t, t=t * self.timescale, **net_kwargs)
        if guidance != 1.0:
            assert uncond_net_kwargs is not None
            uncond_pred = net(x_t, t=t * self.timescale, **uncond_net_kwargs)
            pred = uncond_pred + guidance * (pred - uncond_pred)
        return pred
    
    def convert_sample_prediction(self, x_t, t, pred):
        M = torch.tensor([
            [self.alpha(t), self.sigma(t)],
            [self.A(t), self.B(t)],
        ], dtype=torch.float64)
        M_inv = torch.linalg.inv(M)
        sample_pred = M_inv[0, 0].item() * x_t + M_inv[0, 1].item() * pred
        return sample_pred

class FMEulerSampler:

    def __init__(self, diffusion):
        self.diffusion = diffusion

    def sample(
        self,
        net,
        shape,
        n_steps,
        net_kwargs=None,
        uncond_net_kwargs=None,
        guidance=1.0,
        noise=None,
        mask=None,
    ):
        device = next(net.parameters()).device
        x_t = torch.randn(shape, device=device) if noise is None else noise
        t_steps = torch.linspace(1, 0, n_steps + 1, device=device)
        mask = torch.ones_like(x_t) if mask is None else mask

        with torch.no_grad():
            for i in range(n_steps):
                t = t_steps[i].repeat(x_t.shape[0])
                neg_v = self.diffusion.get_prediction(
                    net,
                    x_t,
                    t,
                    net_kwargs=net_kwargs,
                    uncond_net_kwargs=uncond_net_kwargs,
                    guidance=guidance,
                )
                x_t = x_t + mask.bool() * neg_v * (t_steps[i] - t_steps[i + 1])
        return x_t

# @torch.compile
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift

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

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
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
            q = apply_rotary_emb(q.transpose(1, 2), freqs_cis).transpose(1, 2)
            k = apply_rotary_emb(k.transpose(1, 2), freqs_cis).transpose(1, 2)

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
    """
    Multi-head cross-attention module.
    Query: x
    Key/Value: context
    """
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        # Separate linear layers for query (from x) and key/value (from context)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor,
            freqs_cis: Optional[torch.Tensor] = None,
            attn_mask = None,
    ) -> torch.Tensor:
        """
        x: [B, N, C] query sequence
        context: [B, M, C] key/value sequence
        attn_mask: optional [B, N, M] mask
        """
        B, N, C = x.shape
        _, M, _ = context.shape

        # Linear projections
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # [B, H, M, D]
        
        # RoPE
        if freqs_cis is not None:
            q = apply_rotary_emb(q.transpose(1, 2), freqs_cis[:q.shape[2]]).transpose(1, 2)
            k = apply_rotary_emb(k.transpose(1, 2), freqs_cis[:k.shape[2]]).transpose(1, 2)

        if self.fused_attn:
            # PyTorch 2.1+ scaled_dot_product_attention supports cross-attention
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            raise NotImplementedError()
            # fallback: manual attention
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # [B, H, N, M]
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask.bool(), float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v  # [B, H, N, D]

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
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=False, proj_bias=False, **block_kwargs)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=False)
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size ** 0.5,
        )
    
    def forward(self, x, t, freqs_cis=None, attn_mask=None):
        """
        Incredibly ugly but trades huge memory savings for time
        """
        B, TN, C = x.shape
        B, T, NC = t.shape
        N = TN // T
        
        biases = self.scale_shift_table[None, None] + t.reshape(x.size(0), T, 6, -1)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = [chunk.expand(-1, -1, N, -1) for chunk in biases.chunk(6, dim=-2)]
        
        # ugly but memory saving...
        x = rearrange(x, 'b (t n) c -> b t n c', t=T, n=N)
        x = x + gate_msa * rearrange(self.attn(rearrange(modulate(self.norm1(x), shift_msa, scale_msa), 'b t n c -> b (t n) c'), freqs_cis=freqs_cis, attn_mask=attn_mask), 'b (t n) c -> b t n c', t=T, n=N)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = rearrange(x, 'b t n c -> b (t n) c')
        return x

def create_block_causal_mask(block_size: int, num_blocks: int, dtype=torch.float32):
    """
    Creates a block causal mask where tokens can attend to their own block 
    and all previous blocks, but not future blocks.
    
    Args:
        block_size (int): The length of each block.
        num_blocks (int): The number of blocks.
        dtype: The data type for the mask (default: torch.float32).
        
    Returns:
        torch.Tensor: A mask of shape (seq_len, seq_len) where 
                      0.0 indicates 'attend' and -inf indicates 'mask'.
                      (seq_len = block_size * num_blocks)
    """
    # 1. Create a vector of block IDs: [0, 0, ..., 1, 1, ..., 2, 2, ...]
    block_ids = torch.arange(num_blocks).repeat_interleave(block_size)
    
    # 2. Broadcast to create a grid of block comparisons
    # Shape becomes (seq_len, 1) and (1, seq_len) for broadcasting
    row_ids = block_ids.unsqueeze(1)
    col_ids = block_ids.unsqueeze(0)
    
    # 3. Create boolean mask: True if row_block >= col_block (Past or Current Block)
    # This allows full bidirectional attention within the block
    mask_bool = row_ids >= col_ids
    return mask_bool

def token_drop(labels, null_token, training, p_uncond=0.1, p_full=0.3, p_ind_width=0.6):
    """
    Partitions the batch into three mutually exclusive training modes:
    1. Unconditional (Drop All)
    2. Full Conditional (Keep All)
    3. Partial Conditional (Drop Individual Tokens)
    
    Args:
        labels: (B, ...) Input tensor
        null_token: Learnable null vector
        p_uncond: Probability of the Unconditional mode.
        p_full: Probability of the Full Conditional mode.
        p_ind_drop: Probability of dropping a token *given* we are in Partial mode.
    """
    if not training:
        return labels
    
    B = labels.shape[0]
    device = labels.device
    
    batch_rand_shape = (B,) + (1,) * (labels.ndim - 2)
    batch_rand = torch.rand(batch_rand_shape, device=device)
    
    mask_drop_all = batch_rand < p_uncond
    mask_partial_mode = batch_rand >= (p_uncond + p_full)
    
    sample_specific_drop_rates = torch.rand(batch_rand_shape, device=device) * p_ind_width + (1 - p_ind_width) / 2
    token_noise = torch.rand(labels.shape[:-1], device=device)
    mask_token_drop = token_noise < sample_specific_drop_rates
    
    final_mask = mask_drop_all.unsqueeze(-1) | (mask_partial_mode.unsqueeze(-1) & mask_token_drop.unsqueeze(-1))
    null_token = null_token.to(labels.dtype)
    
    return torch.where(final_mask, null_token, labels)

class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        self.groupnorm = nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels
        )
        self.activation = nn.SiLU()
        self.project = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=kernel_size//2
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.groupnorm(x)
        x = self.activation(x)
        return self.project(x)

class ResnetBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            num_groups=num_groups,
        )

        self.block2 = ConvBlock1d(
            in_channels=out_channels,
            out_channels=out_channels,
            num_groups=num_groups,
        )

        self.to_out = (
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.to_out(x)

class Patcher(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.block = ResnetBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            num_groups=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x

class ModernDiT(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 spatial_window,
                 n_chunks,
                 style_dim,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 ):
        super().__init__()
        self.spatial_window = spatial_window
        max_input_size = spatial_window * n_chunks
        
        self.t_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True)
        self.bpm_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True, max_period=1000)
        self.action_embedder = nn.Linear(style_dim, hidden_size)
        self.x_embedder = Patcher(in_channels, hidden_size)
        
        self.fuse_conditioning = SwiGLUMlp(hidden_size * 2, int(2 / 3 * mlp_ratio * hidden_size), hidden_size, bias=False)
        
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6, bias=True),
        )
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.norm = RMSNorm(hidden_size)
        self.final_layer_scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size ** 0.5,
        )
        self.fc = nn.Linear(hidden_size, in_channels, bias=False)
        
        self.initialize_weights()
        self.register_buffer('block_causal_mask', create_block_causal_mask(spatial_window, n_chunks))
        self.register_buffer('freqs_cis',  precompute_freqs_cis(hidden_size // num_heads, max_input_size))
    
    def initialize_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.t_block[-1].weight)
        nn.init.zeros_(self.t_block[-1].bias)
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
    
    def forward(self, x, t, bpm, actions, attn_mask=None):
        B, T, N, C = x.shape
        
        if self.training and attn_mask is None:
            if torch.rand(1).item() < 0.2:
                attn_mask = self.block_causal_mask
        elif attn_mask:
            attn_mask = self.block_causal_mask
        
        x = rearrange(x, 'b t n c -> (b t) c n')
        x = self.x_embedder(x)
        x = rearrange(x, '(b t) c n -> b t n c', b=B, t=T)
        bpm = self.bpm_embedder(bpm.flatten()).view(B, T, 1, -1)
        
        x = x + bpm
        x = rearrange(x, 'b t n c -> b (t n) c')
        
        t = self.t_embedder(t.flatten()).view(B, T, -1)
        actions = self.action_embedder(actions)
        print(t.shape, actions.shape)
        t = torch.cat([t, actions], dim=-1)
        print(t.shape)
        t = self.fuse_conditioning(t)
        print(t.shape)
        t0 = self.t_block(t)
        print(t0.shape)
        
        freqs_cis = self.freqs_cis[:x.shape[1]]
        for block in self.blocks:
            x = block(x, t0, freqs_cis=freqs_cis, attn_mask=attn_mask)
        
        # SAM Audio does not use a non-linearity on t here
        shift, scale = (self.final_layer_scale_shift_table[None] + F.silu(t[:, None])).chunk(
            2, dim=2
        )
        x = modulate(self.norm(x), shift.squeeze(2), scale.squeeze(2))
        x = self.fc(x)
        return x

class ModernDiTWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = ModernDiT(**kwargs)
        
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x, bpm, actions, t=None, attn_mask=None):
        return self.diffusion.loss(self.net, x, t=t, net_kwargs={'actions': actions, 'bpm': bpm, 'attn_mask': attn_mask})
    
    def sample(self, x, bpm, actions, attn_mask=None, n_steps=50):
        return self.sampler.sample(self.net, x.shape, n_steps=n_steps, net_kwargs={'actions': actions, 'bpm': bpm, 'attn_mask': attn_mask})

def ModernDiT_large(**kwargs):
    return ModernDiTWrapper(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def ModernDiT_medium(**kwargs):
    return ModernDiTWrapper(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def ModernDiT_small(**kwargs):
    return ModernDiTWrapper(depth=16, hidden_size=1024, num_heads=16, **kwargs)

def ModernDiT_tiny(**kwargs):
    return ModernDiTWrapper(depth=16, hidden_size=768, num_heads=12, **kwargs)