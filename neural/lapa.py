import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List, Callable

from einops import rearrange
from vector_quantize_pytorch import FSQ
from fm import FM, FMEulerSampler

@torch.compile
def modulate(x, shift, scale):
    if scale.ndim == 3:
        return x * (1 + scale) + shift
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

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
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
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
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ClassEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + (dropout_prob > 0), hidden_size)
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
    
    def forward(self, x, force_drop=False):
        if self.training:
            drop_ids = torch.rand(x.shape[0], device=x.device) < self.dropout_prob
            x = torch.where(drop_ids.unsqueeze(-1), self.num_classes, x)
        elif force_drop:
            drop_ids = torch.ones(x.shape[0], device=x.device).bool()
            x = torch.where(drop_ids.unsqueeze(-1), self.num_classes, x)
        
        x = self.embedding(x)
        return x

class NoiseEmbedder(nn.Module):
    def __init__(self, num_buckets, hidden_size, max_t=0.7):
        super().__init__()
        self.embedding = nn.Embedding(num_buckets, hidden_size)
        self.num_buckets = num_buckets
        self.max_t = max_t
    
    def forward(self, x):
        x = (torch.clamp(x / self.max_t, 0, 1) * self.num_buckets).long()
        x = (x - 0) / (self.max_t - 0)
        x = (x * self.num_buckets).floor().long()
        x = torch.clamp(x, 0, self.num_buckets - 1)
        x = self.embedding(x)
        return x

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

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = RMSNorm(hidden_size)
        self.cross = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm3 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size))
    
    def forward(self, x, context, freqs_cis=None):
        x = x + self.attn(self.norm1(x), freqs_cis=freqs_cis[:x.shape[1]] if freqs_cis is not None else None)
        x = x + self.cross(self.norm2(x), context, freqs_cis=freqs_cis[:max(x.shape[1], context.shape[1])] if freqs_cis is not None else None)
        x = x + self.mlp(self.norm3(x))
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size))
    
    def forward(self, x, freqs_cis=None, is_causal=False):
        x = x + self.attn(self.norm1(x), is_causal=is_causal, freqs_cis=freqs_cis[:x.shape[1]] if freqs_cis is not None else None)
        x = x + self.mlp(self.norm2(x))
        return x

class ActionTransformer(nn.Module):
    def __init__(self,in_channels,
                 hidden_size,
                 levels,
                 spatial_window,
                 temporal_window,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 ):
        super().__init__()
        
        self.x_embedder = nn.Sequential(nn.Linear(in_channels, hidden_size, bias=True), RMSNorm(hidden_size))
        
        self.spatial_blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.temporal_blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.to_vq = nn.Sequential(
            RMSNorm(spatial_window * hidden_size),
            nn.Linear(spatial_window * hidden_size, len(levels))
        )
        self.vq = FSQ(levels=levels)
        
        self.spatial_pos = nn.Embedding(spatial_window, hidden_size)
        self.temporal_pos = nn.Embedding(temporal_window, hidden_size)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.to_vq[-1].weight)
        # zero out c_proj weights in all blocks
        for block in self.spatial_blocks:
            torch.nn.init.zeros_(block.mlp.w3.weight)
            torch.nn.init.zeros_(block.attn.proj.weight)
        
        for block in self.temporal_blocks:
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
    
    def forward(self, x):
        """
        x: (B, 2, N, C) latents
        """
        B, T, N, C = x.shape
        
        x = self.x_embedder(x)
        
        x = rearrange(x, 'b t n c -> (b t) n c')
        x = x + self.spatial_pos(torch.arange(N, device=x.device, dtype=torch.long).unsqueeze(0))
        for block in self.spatial_blocks:
            x = block(x)
        
        x = rearrange(x, '(b t) n c -> (b n) t c', b=B, t=T)
        x = x + self.temporal_pos(torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0))
        for block in self.temporal_blocks:
            x = block(x, is_causal=True)
        
        x = rearrange(x, '(b n) t c -> b t n c', b=B, n=N)
        first_frame, last_frame = x[:, 0], x[:, 1]
        first_frame, last_frame = rearrange(first_frame, 'b n c -> b (n c)'), rearrange(last_frame, 'b n c -> b (n c)')
        x = last_frame - first_frame # LAPA subtracts at codebook dim but thats 2 for [8, 8] levels... not enough information?
        x = x.unsqueeze(1)
        
        x = self.to_vq(x)
        x, indices = self.vq(x)
        return indices

class DiT(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 num_actions,
                 max_input_size,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 ):
        super().__init__()
        
        self.x_embedder = nn.Sequential(nn.Linear(in_channels, hidden_size, bias=True), RMSNorm(hidden_size))
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.action_embedder = nn.Embedding(num_actions, hidden_size)
        
        self.x_pos = nn.Embedding(max_input_size, hidden_size)
        self.context_pos = nn.Embedding(2, hidden_size)

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.final_layer = nn.Sequential(RMSNorm(hidden_size), nn.Linear(hidden_size, in_channels, bias=True))

        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.final_layer[-1].weight)
        # zero out c_proj weights in all blocks
        for block in self.blocks:
            torch.nn.init.zeros_(block.mlp.w3.weight)
            torch.nn.init.zeros_(block.attn.proj.weight)
            torch.nn.init.zeros_(block.cross.proj.weight)

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
        """
        x: (B, N, C) latents to be denoised
        t: (B) noise level for x
        actions: (B, T) frame actions
        """
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        actions = self.action_embedder(actions)
        context = torch.cat([t.unsqueeze(1), actions], dim=1)
        
        print(x.shape, torch.arange(x.shape[1], device=x.device, dtype=torch.long).shape)
        x = x + self.x_pos(torch.arange(x.shape[1], device=x.device, dtype=torch.long).unsqueeze(0))
        context = context + self.context_pos(torch.arange(2, device=x.device, dtype=torch.long).unsqueeze(0))
        for block in self.blocks:
            x = block(x, context)
        
        x = self.final_layer(x)
        return x

class DiTWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = DiT(**kwargs)
        
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        self.net.initialize_weights()
    
    def forward(self, x, actions, t=None):
        return self.diffusion.loss(self.net, x, t=t, net_kwargs={'actions': actions})
    
    def sample(self, x, actions, n_steps=50):
        return self.sampler.sample(self.net, x[:, [1]].shape, n_steps, net_kwargs={'x': x, 'actions': actions})

class LAM(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 levels,
                 spatial_window,
                 temporal_window,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 ):
        super().__init__()
        self.action_model = ActionTransformer(in_channels=in_channels, 
                                              hidden_size=hidden_size, 
                                              levels=levels, 
                                              spatial_window=spatial_window, 
                                              temporal_window=temporal_window, 
                                              num_heads=num_heads, 
                                              depth=depth, 
                                              mlp_ratio=mlp_ratio)
        self.decoder = DiTWrapper(in_channels=in_channels, 
                                  hidden_size=hidden_size, 
                                  num_actions=math.prod(levels), 
                                  max_input_size=spatial_window,
                                  num_heads=num_heads, 
                                  depth=depth, 
                                  mlp_ratio=mlp_ratio)
        
        self.levels = levels
        
        # tie weights
        self.decoder.net.x_embedder[0].weight = self.action_model.x_embedder[0].weight
        self.decoder.net.x_embedder[0].bias = self.action_model.x_embedder[0].bias
        self.decoder.net.x_embedder[1].weight = self.action_model.x_embedder[1].weight
    
    def forward(self, x):
        """
        x: (B, T, N, C) latents
        alpha: (B) noise level for history latents
        """
        assert x.ndim == 4
        
        actions = self.action_model(x)
        
        x = self.decoder(x, actions)
        return x, actions
    
    def generate(self, x, actions, n_steps=50):
        return self.decoder.sample(x.shape, x, actions, n_steps=n_steps)
    
    def generate_random_different_actions(self, actions_indices, codebook_size, device):
        shape = actions_indices.shape
        random_actions = torch.randint(0, codebook_size, shape, device=device)

        while torch.any(random_actions == actions_indices):
            random_actions = torch.where(
                random_actions == actions_indices,
                torch.randint(0, codebook_size, shape, device=device),
                random_actions,
            )

        return random_actions
    
    def lam_vs_random_actions(self, x, n_steps=50):
        actions = self.action_model(x)
        
        random_actions = self.generate_random_different_actions(actions, math.prod(self.levels), x.device)
        recon = self.generate(x, actions, n_steps=n_steps)
        random = self.generate(x, random_actions, n_steps=n_steps)
        
        return recon, random

def LAM_L(**kwargs):
    return LAM(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def LAM_M(**kwargs):
    return LAM(depth=20, hidden_size=768, num_heads=12, **kwargs)

def LAM_B(**kwargs):
    return LAM(depth=12, hidden_size=768, num_heads=12, **kwargs)