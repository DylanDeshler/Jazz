import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List, Callable

from einops import rearrange, repeat

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
        s = [x.shape[0]] + [1] * (x.dim() - 1)
        x_t = self.alpha(t).view(*s) * x + self.sigma(t).view(*s) * noise
        return x_t, noise
    
    def loss(self, net, x, t=None, net_kwargs=None, return_loss_unreduced=False, return_all=False):
        if net_kwargs is None:
            net_kwargs = {}
        
        if t is None:
            t = torch.rand(x.shape[0], device=x.device)
        x_t, noise = self.add_noise(x, t)
        
        pred = net(x_t, t=t * self.timescale, **net_kwargs)
        
        target = self.A(t) * x + self.B(t) * noise # -dxt/dt
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
    ):
        device = next(net.parameters()).device
        x_t = torch.randn(shape, device=device) if noise is None else noise
        t_steps = torch.linspace(1, 0, n_steps + 1, device=device)

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
                x_t = x_t + neg_v * (t_steps[i] - t_steps[i + 1])
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

def top_k_softmax(scores, k=5):
    """
    scores: (Batch, Heads, Queries, Styles)
    Keeps top k values, masks the rest to -inf, then softmax.
    """
    # 1. Find the top K scores
    top_k_values, _ = torch.topk(scores, k=k, dim=-1)
    
    # 2. Determine the cutoff (smallest of the top K)
    # maintain shape for broadcasting
    cutoff = top_k_values[..., -1].unsqueeze(-1) 
    
    # 3. Mask everything below cutoff
    # 1e-9 or -inf ensures they become absolute zero after softmax
    scores_masked = scores.clone()
    scores_masked[scores < cutoff] = float('-inf')
    
    # 4. Standard Softmax on the remaining 5 items
    return F.softmax(scores_masked, dim=-1)

## Just for attention pooling
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_hidden, num_heads, dropout_w=0, dropout_e=0, bias=False, flash=True, top_k=None, **kwargs):
        super().__init__()
        self.Q = nn.Linear(dim_hidden, dim_hidden, bias)
        self.K = nn.Linear(dim_hidden, dim_hidden, bias)
        self.V = nn.Identity()#nn.Linear(dim_hidden, dim_hidden, bias)
        self.out = nn.Identity()#nn.Linear(dim_hidden, dim_hidden, bias)
        self.dropout_w = nn.Dropout(dropout_w)
        self.dropout_e = nn.Dropout(dropout_e)
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden
        self.dim_attn = dim_hidden // num_heads
        self.scale = self.dim_attn ** -0.5
        self.top_k = top_k
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and (flash and not top_k)

    def forward(self, query, key, value, mask=None, return_weights=False):

        ## PROJECT INPUTS
        q = self.Q(query)
        k = self.K(key)
        v = self.V(value)

        ## SPLIT ATTENTION HEADS
        b = query.size(0) # Assume [batch, seq_len, hidden]
        q = q.view(b, -1, self.num_heads, self.dim_attn).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.dim_attn).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.dim_attn).transpose(1, 2)

        ## COMPUTE ATTENTION
        if self.flash and not return_weights:
            e = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask, 
                dropout_p=self.dropout_w.p if self.training else 0, 
                is_causal=False,
                scale=self.scale,
            )
            e = e.transpose(1, 2).contiguous().view_as(query) 
            
        else:
            dot_product = torch.einsum("bhqa,bhka->bhqk", (q, k)) * self.scale
            if mask is not None:
                dot_product = dot_product.masked_fill_(mask.logical_not(), float("-inf"))
            if self.top_k:
                w = top_k_softmax(dot_product, self.top_k)
            else:
                w = torch.softmax(dot_product, dim=-1)
            w = self.dropout_w(w)
            e = torch.einsum("bhqv,bhva->bhqa", (w, v)).transpose(1, 2).contiguous().view_as(query) 
            
            if return_weights:
                return self.dropout_e(self.out(e)), w
        
        return self.dropout_e(self.out(e))

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

class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=False, **block_kwargs)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=False)
    
    def forward(self, x, freqs_cis=None, is_causal=False):
        x = x + self.attn(self.norm1(x), is_causal=is_causal, freqs_cis=freqs_cis[:x.shape[1]] if freqs_cis is not None else None)
        x = x + self.mlp(self.norm2(x))
        return x

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

class Transformer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 max_seq_len,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 min_log_std=1e-6,
                 max_log_std=1
                 ):
        super().__init__()
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.bpm_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True, max_period=1000)
        
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.norm = RMSNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, 2 * in_channels)
        
        self.initialize_weights()
        self.register_buffer('freqs_cis',  precompute_freqs_cis(hidden_size // num_heads, max_seq_len, theta=1000))
    
    def initialize_weights(self):
        self.apply(self._init_weights)
        # zero out c_proj weights in all blocks
        for block in self.blocks:
            torch.nn.init.zeros_(block.mlp.w3.weight)
            torch.nn.init.zeros_(block.attn.proj.weight)

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
    
    def forward(self, x, bpm, y=None):
        """
        x: (B, T, D) latents
        """
        B, T, D = x.shape
        
        x = self.x_embedder(x)
        bpm = self.bpm_embedder(bpm.flatten()).view(B, T, -1)
        
        x = x + bpm
        for block in self.blocks:
            x = block(x, freqs_cis=self.freqs_cis, is_causal=True)
        
        x = self.norm(x)
        mu, log_std = self.proj(x).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()
        
        dist = torch.distributions.Normal(mu, std)
        
        if y is not None:
            loss = -dist.log_prob(y).mean()
            return dist, loss
        
        return dist

def Transformer_large(**kwargs):
    return Transformer(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def Transformer_medium(**kwargs):
    return Transformer(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def Transformer_small(**kwargs):
    return Transformer(depth=16, hidden_size=1024, num_heads=16, **kwargs)

def Transformer_tiny(**kwargs):
    return Transformer(depth=16, hidden_size=768, num_heads=12, **kwargs)