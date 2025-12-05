import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import math
from typing import Optional, List, Callable

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

class WindowAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            window_size: int,
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
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            freqs_cis: Optional[torch.Tensor] = None,
            attn_mask = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # RoPE
        if freqs_cis is not None:
            q = apply_rotary_emb(q.transpose(1, 2), freqs_cis).transpose(1, 2)
            k = apply_rotary_emb(k.transpose(1, 2), freqs_cis).transpose(1, 2)

        if self.fused_attn:
            print(q.shape, k.shape)
            block_mask = create_block_mask(
                generate_window_mask_mod(self.window_size), 
                B=None, H=None, Q_LEN=q.shape[0], KV_LEN=k.shape[0], 
                device=q.device
            )
            print(block_mask.shape, block_mask.dtype)
            block_mask = block_mask & attn_mask
            
            x = flex_attention(q, k, v, block_mask=block_mask)
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

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size))
    
    def forward(self, x, context, freqs_cis=None):
        x = x + self.attn(self.norm1(x), context)
        x = x + self.mlp(self.norm2(x))
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size))
    
    def forward(self, x, context=None, freqs_cis=None):
        x = x + self.attn(self.norm1(x), freqs_cis=freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x

def generate_window_mask_mod(window_size):
    """
    Returns a score_mod function for flex_attention that enforces
    local window constraints.
    """
    def window_mask(b, h, q_idx, kv_idx):
        # Only attend if Query and Key are in the same window block
        return (q_idx // window_size) == (kv_idx // window_size)
    return window_mask

class ContinuousPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_freq=10):
        super().__init__()
        self.dim = dim
        
        half_dim = dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, dtype=torch.float32) * -(math.log(max_freq) / half_dim)
        )
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        """
        x: Input tensor of positions (Batch, Seq_Len) in range [0, 1]
        Returns: (Batch, Seq_Len, Dim)
        """
        # (B, L, 1) * (D/2) -> (B, L, D/2)
        args = x.unsqueeze(-1) * self.freqs * 2 * math.pi
        
        # Concatenate sin and cos -> (B, L, Dim)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class Perciever(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_heads, depth, n_interleave, n_latents):
        super().__init__()
        self.n_latents = n_latents
        
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.latents = nn.Embedding(n_latents, hidden_dim)
        self.pos_emb = ContinuousPositionalEmbeddings(hidden_dim)
        
        layers = []
        for d in range(depth):
            layers.append(CrossAttentionBlock(hidden_dim, n_heads))
            for _ in range(n_interleave):
                layers.append(SelfAttentionBlock(hidden_dim, n_heads))
        
        self.layers = nn.ModuleList(layers)
        
        self.norm = RMSNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        B, C, L = x.shape
        
        x = x.transpose(1, 2)
        data = self.in_proj(x)
        data = data + self.pos_emb(torch.linspace(0, 1, steps=L, device=x.device).unsqueeze(0))
        
        x = self.latents(torch.arange(self.n_latents, device=x.device, dtype=torch.long).unsqueeze(0))
        x = x + self.pos_emb(torch.linspace(0, 1, steps=x.shape[1], device=x.device).unsqueeze(0))
        x = x.repeat((B, 1, 1))
        
        for layer in self.layers:
            x = layer(x, data)
        
        x = self.norm(x)
        x = self.out_proj(x)
        return x

class Reciever(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_heads, depth, n_interleave, n_latents, window_size):
        super().__init__()
        self.n_latents = n_latents
        
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_emb = ContinuousPositionalEmbeddings(hidden_dim)
        
        layers = []
        for d in range(depth):
            layers.append(CrossAttentionBlock(hidden_dim, n_heads))
            for _ in range(n_interleave):
                layers.append(WindowAttention(hidden_dim, window_size, n_heads))
        
        self.layers = nn.ModuleList(layers)
        
        self.norm = RMSNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x, z):
        B, L, C = x.shape
        
        x = self.in_proj(x)
        x = x + self.pos_emb(torch.linspace(0, 1, steps=L, device=x.device).unsqueeze(0))
        
        z = z + self.pos_emb(torch.linspace(0, 1, steps=z.shape[1], device=z.device).unsqueeze(0))
        
        for layer in self.layers:
            x = layer(x, z)
        
        x = self.norm(x)
        x = self.out_proj(x)
        return x

if __name__ == '__main__':
    from torchinfo import summary
    
    with torch.no_grad():
        encoder = Perciever(1, 512, 16, 8, 4, 4, 32).to('cuda:1')
        summary(encoder)
        
        decoder = Reciever(16, 512, 1, 8, 4, 4, 32, 256).to('cuda:1')
        summary(decoder)
        
        x = torch.randn((64, 1, 16000)).to('cuda:1')
        y = encoder(x)
        print(x.shape, y.shape)
        z = decoder(x, y)
        print(z.shape)
        
        x = torch.randn((32, 1, 48000)).to('cuda:1')
        y = encoder(x)
        print(x.shape, y.shape)
        z = decoder(x, y)
        print(z.shape)