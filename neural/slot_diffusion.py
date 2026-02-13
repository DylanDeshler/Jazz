import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from typing import Optional, Callable
from einops import rearrange
import numpy as np
import math

from fm import FM, FMEulerSampler

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
                f_max=sample_rate / 2,
                power=2.0,
                normalized=True,      # Normalizes the STFT to be magnitude invariant
                center=True,          # Padding to keep time/length consistent
                pad_mode='reflect'    # Better for audio boundary artifacts
            ),
            T.AmplitudeToDB(top_db=80.0)
        )
    
    def forward(self, x):
        return self.transform(x)

class SlotAttention(nn.Module):
    """Slot attention module that iteratively performs cross-attention."""

    def __init__(
        self,
        in_features,
        num_iterations,
        num_slots,
        slot_size,
        mlp_hidden_size,
        eps=1e-6,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.eps = eps
        self.attn_scale = self.slot_size**-0.5

        self.norm_inputs = nn.LayerNorm(self.in_features)

        # Linear maps for the attention module.
        self.project_q = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.slot_size, bias=False),
        )
        self.project_k = nn.Linear(in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(in_features, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

    def forward(self, inputs, slots):
        """Forward function.

        Args:
            inputs (torch.Tensor): [B, N, C], flattened per-pixel features.
            slots (torch.Tensor): [B, num_slots, C] slot inits.

        Returns:
            updated slots, same shape as `slots`.
        """
        # `inputs` has shape [B, num_inputs, inputs_size].
        # `num_inputs` is actually the spatial dim of feature map (H*W)
        bs, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [B, num_inputs, slot_size].
        k = self.project_k(inputs)
        # Shape: [B, num_inputs, slot_size].
        v = self.project_v(inputs)

        # Initialize the slots. Shape: [B, num_slots, slot_size].
        assert len(slots.shape) == 3

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots

            # Attention. Shape: [B, num_slots, slot_size].
            q = self.project_q(slots)

            attn_logits = self.attn_scale * torch.einsum('bnc,bmc->bnm', k, q)
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [B, num_inputs, num_slots].

            # Normalize along spatial dim and do weighted mean.
            attn = attn + self.eps
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.einsum('bnm,bnc->bmc', attn, v)
            # `updates` has shape: [B, num_slots, slot_size].

            # Slot update.
            # GRU is expecting inputs of size (N, L)
            # so flatten batch and slots dimension
            slots = self.gru(
                updates.view(bs * self.num_slots, self.slot_size),
                slots_prev.view(bs * self.num_slots, self.slot_size),
            )
            slots = slots.view(bs, self.num_slots, self.slot_size)
            slots = slots + self.mlp(slots)

        return slots

    @property
    def dtype(self):
        return self.project_k.weight.dtype

    @property
    def device(self):
        return self.project_k.weight.device

class SlotAttentionWMask(SlotAttention):
    """Slot attention module that iteratively performs cross-attention.

    We return the last attention map from SA as the segmentation mask.
    """

    def forward(self, inputs, slots):
        """Forward function.

        Args:
            inputs (torch.Tensor): [B, N, C], flattened per-pixel features.
            slots (torch.Tensor): [B, num_slots, C] slot inits.

        Returns:
            slots (torch.Tensor): [B, num_slots, C] slot inits.
            masks (torch.Tensor): [B, num_slots, N] segmentation mask.
        """
        # `inputs` has shape [B, num_inputs, inputs_size].
        # `num_inputs` is actually the spatial dim of feature map (H*W)
        bs, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [B, num_inputs, slot_size].
        k = self.project_k(inputs)
        # Shape: [B, num_inputs, slot_size].
        v = self.project_v(inputs)

        # Initialize the slots. Shape: [B, num_slots, slot_size].
        assert len(slots.shape) == 3

        # Multiple rounds of attention.
        for attn_iter in range(self.num_iterations):
            slots_prev = slots

            # Attention. Shape: [B, num_slots, slot_size].
            q = self.project_q(slots)

            attn_logits = self.attn_scale * torch.einsum('bnc,bmc->bnm', k, q)
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [B, num_inputs, num_slots].

            # attn_map normalized along slot-dim is treated as seg_mask
            if attn_iter == self.num_iterations - 1:
                seg_mask = attn.detach().clone().permute(0, 2, 1)

            # Normalize along spatial dim and do weighted mean.
            attn = attn + self.eps
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.einsum('bnm,bnc->bmc', attn, v)
            # `updates` has shape: [B, num_slots, slot_size].

            # Slot update.
            # GRU is expecting inputs of size (N, L)
            # so flatten batch and slots dimension
            slots = self.gru(
                updates.view(bs * self.num_slots, self.slot_size),
                slots_prev.view(bs * self.num_slots, self.slot_size),
            )
            slots = slots.view(bs, self.num_slots, self.slot_size)
            slots = slots + self.mlp(slots)

        # [B, num_slots, slot_size], [B, num_slots, num_inputs (H*W)]
        return slots, seg_mask

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

class SelfAttention(nn.Module):
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

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn1 = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=False, proj_bias=False, **block_kwargs)
        self.norm2 = RMSNorm(hidden_size)
        self.attn2 = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=False, proj_bias=False, **block_kwargs)
        self.norm3 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=False)
        
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size ** 0.5,
        )
    
    def forward(self, x, context, t, freqs_cis=None):
        biases = self.scale_shift_table[None] + t.reshape(x.size(0), 6, -1)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = biases.chunk(6, dim=-2)
        
        x = x + gate_msa * self.attn1(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis=freqs_cis)
        x = x + self.attn2(self.norm2(x), context)
        x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x

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

class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 max_seq_len,
                 patch_size,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 ):
        super().__init__()
        self.patch_size = patch_size
        
        self.x_embedder = nn.Linear(in_channels * patch_size * patch_size, hidden_size, bias=False)
        
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.norm = RMSNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.initialize_weights()
        self.register_buffer('freqs_cis',  precompute_freqs_cis(hidden_size // num_heads, max_seq_len))
    
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
    
    def forward(self, x):
        print(x.shape)
        
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
        print(x.shape)
        x = self.x_embedder(x)
        print(x.shape)
        
        freqs_cis = self.freqs_cis[:x.shape[1]]
        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis)
        
        x = self.norm(x)
        x = self.fc(x)
        
        return x

class ModernDiT(nn.Module):
    def __init__(self,
                 in_channels,
                 slot_size,
                 hidden_size,
                 max_seq_len,
                 patch_size,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 ):
        super().__init__()
        self.patch_size = patch_size

        self.slot_embedder = SwiGLUMlp(slot_size, hidden_size, hidden_size, bias=False)
        self.t_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True)
        self.x_embedder = nn.Linear(in_channels * patch_size * patch_size, hidden_size, bias=False)
        
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
        self.fc = nn.Linear(hidden_size, in_channels * patch_size * patch_size, bias=False)
        
        self.initialize_weights()
        self.register_buffer('freqs_cis',  precompute_freqs_cis(hidden_size // num_heads, max_seq_len))
    
    def initialize_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.t_block[-1].weight)
        nn.init.zeros_(self.t_block[-1].bias)
        # zero out c_proj weights in all blocks
        for block in self.blocks:
            nn.init.zeros_(block.mlp.w3.weight)
            nn.init.zeros_(block.attn1.proj.weight)
            nn.init.zeros_(block.attn2.proj.weight)
    
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
    
    def forward(self, x, t, slots):
        B, C, H, W = x.shape
        
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
        x = self.x_embedder(x)
        
        slots = self.slot_embedder(slots)
        t = self.t_embedder(t)
        t0 = self.t_block(t)
        
        freqs_cis = self.freqs_cis[:x.shape[1]]
        for block in self.blocks:
            x = block(x, slots, t0, freqs_cis=freqs_cis)
        
        # SAM Audio does not use a non-linearity on t here
        shift, scale = (self.final_layer_scale_shift_table[None] + F.silu(t[:, None])).chunk(
            2, dim=1
        )
        x = modulate(self.norm(x), shift, scale)
        x = self.fc(x)
        
        x = rearrange(x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)', c=C, h=H // self.patch_size, w=W // self.patch_size, p1=self.patch_size, p2=self.patch_size)
        
        return x

class ModernDiTWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = ModernDiT(**kwargs)
        
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x, slots, t=None):
        return self.diffusion.loss(self.net, x, t=t, net_kwargs={'slots': slots})
    
    def generate(self, shape, slots, n_steps=50):
        return self.sampler.sample(self.net, shape, n_steps=n_steps, net_kwargs={'slots': slots})

class SADiffusion(nn.Module):
    """SlotDiffusion model on images."""

    def __init__(
            self,
            resolution,
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            num_slots=7,
            slot_size=128,
            slot_mlp_size=256,
            num_iterations=3,
            encoder_dict={},
            decoder_dict={},
            eps=1e-6,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.num_slots = num_slots
        self.resolution = resolution
        self.width = (sample_rate // hop_length) + 1
        
        self.to_mel = ToMel(sample_rate, n_fft, hop_length, n_mels)
        self.encoder = Encoder(**encoder_dict)

        self.init_latents = nn.Parameter(
            nn.init.normal_(torch.empty(1, num_slots, slot_size)))

        self.slot_attention = SlotAttentionWMask(
            in_features=encoder_dict['hidden_size'],
            num_iterations=num_iterations,
            num_slots=num_slots,
            slot_size=slot_size,
            mlp_hidden_size=slot_mlp_size,
            eps=eps,
        )
        
        self.decoder = ModernDiTWrapper(**decoder_dict)

    def encode(self, img, init_slots=None):
        """Encode from img to slots."""
        B = img.shape[0]

        img = self.to_mel(img)
        encoder_out = self.encoder(img).type(self.dtype)
        # `encoder_out` has shape: [B, H*W, out_features]

        # init slots
        if init_slots is None:
            init_slots = self.init_latents.repeat(B, 1, 1)

        # perform slot attention operation
        slots, masks = self.slot_attention(encoder_out, init_slots)
        print(slots.shape, masks.shape)
        masks = masks.unflatten(-1, (self.n_mels // self.resolution, self.width // self.resolution))
        # [B, N, C], [B, N, h, w]
        print(slots.shape, masks.shape)

        # resize masks to the original resolution
        if not self.training:
            with torch.no_grad():
                masks = masks.flatten(0, 1).unsqueeze(1)  # [BN, 1, h, w]
                masks = F.interpolate(
                    masks,
                    self.resolution,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1).unflatten(0, (B, self.num_slots))  # [B, N, H, W]

        # [B, N, C], [B, N, H, W]
        return img, slots, masks

    def forward(self, img):
        """Forward function."""

        img, slots, masks = self.encode(img)
        # `slots` has shape: [B, self.num_slots, self.slot_size]
        # `masks` has shape: [B, self.num_slots, H, W]
        
        if not self.training:
            samples = self.decoder.generate(img.shape, slots)
            return {'masks': masks, 'slots': slots, 'samples': samples}
        
        loss = self.decoder(img, slots)

        return {'masks': masks, 'slots': slots, 'loss': loss}

    # def decode(self, slots):
    #     """Decode from slots to reconstructed images and masks."""
    #     # `slots` has shape: [B, self.num_slots, self.slot_size].
    #     bs, num_slots, slot_size = slots.shape
    #     height = width = self.resolution
    #     num_channels = 1

    #     print(slots.shape)
    #     # spatial broadcast
    #     decoder_in = slots.view(bs * num_slots, slot_size, 1, 1)
    #     print(decoder_in.shape)
    #     decoder_in = decoder_in.repeat(1, 1, self.resolution, self.resolution)
    #     print(decoder_in.shape)

    #     out = self.decoder.sample(decoder_in)
    #     # `out` has shape: [B*num_slots, num_channels+1, H, W].

    #     out = out.view(bs, num_slots, num_channels + 1, height, width)
    #     recons = out[:, :, :num_channels, :, :]  # [B, num_slots, num_channels, H, W]
    #     masks = out[:, :, -1:, :, :]
    #     masks = F.softmax(masks, dim=1)  # [B, num_slots, 1, H, W]
    #     recon_combined = torch.sum(recons * masks, dim=1)  # [B, num_channels, H, W]
    #     return recon_combined, recons, masks, slots

    @property
    def dtype(self):
        return self.slot_attention.dtype

    @property
    def device(self):
        return self.slot_attention.device

if __name__ == '__main__':
    depth = 16
    hidden_size = 1024
    num_heads = 16
    max_seq_len = 128
    patch_size = 16
    
    encoder_dict = dict(
        in_channels=1,
        max_seq_len=max_seq_len,
        patch_size=patch_size,
        depth=depth,
        hidden_size=hidden_size,
        num_heads=num_heads
    )
    
    resolution=patch_size
    sample_rate=16000
    n_fft=1024
    hop_length=512
    n_mels=256
    num_slots=7
    slot_size=128
    slot_mlp_size=256
    num_iterations=3
    
    decoder_dict = dict(
        in_channels=1,
        slot_size=slot_size,
        max_seq_len=max_seq_len,
        patch_size=patch_size,
        depth=depth,
        hidden_size=hidden_size,
        num_heads=num_heads
    )
    
    model = SADiffusion(
        resolution, sample_rate, n_fft, hop_length, n_mels, num_slots, slot_size, slot_mlp_size, num_iterations, encoder_dict, decoder_dict
    ).to('cuda')
    
    x = torch.randn(16, 1, 16000).to('cuda')
    out = model(x)
    
    model.eval()
    out = model(x)
    
    for k, v in out.items():
        print(k, v.shape)