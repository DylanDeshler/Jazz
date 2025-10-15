# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import repeat, pack
from einops.layers.torch import Rearrange

from tqdm import tqdm
from fm import FM, FMEulerSampler
from vector_quantize import VectorQuantize

@torch.compile
def modulate(x, shift, scale):
    if scale.ndim == 3:
        return x * (1 + scale) + shift
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

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


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

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

class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
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

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        if c.ndim == 3:
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class LightningDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
    
    def forward(self, x, c, freqs_cis=None, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        if c.ndim == 3:
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis=freqs_cis, attn_mask=attn_mask)
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis=freqs_cis, attn_mask=attn_mask)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class CrossDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, t, context, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(9, dim=-1)
        if t.ndim == 3:
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
            x = x + gate_mca * self.cross_attn(modulate(self.norm2(x), shift_mca, scale_mca), context)
            x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        else:
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
            x = x + gate_mca.unsqueeze(1) * self.cross_attn(modulate(self.norm2(x), shift_mca, scale_mca), context)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels, local_window=1):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, local_window * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class LightningFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, local_window=1):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, local_window * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, hidden_size, max_input_size, local_window):
        super().__init__()
        self.reshape = Rearrange("b (t1 t2) d -> b t1 (t2 d)", t1=max_input_size // local_window, t2=local_window)
        self.proj = nn.Linear(local_window * in_channels, hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        x = self.reshape(x)
        x = self.proj(x)
        x = self.norm(x)
        return x

class Optimizers:
    def __init__(self, optimizers):
        super().__init__()
        self.optimizers = optimizers
    
    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        max_input_size=250,
        in_channels=128,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        conditional=False,
        local_window=1,
        out_channels=None,
        **kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels * 2 if learn_sigma else out_channels
        self.num_heads = num_heads
        self.conditional = conditional
        self.max_input_size = max_input_size

        self.x_embedder = PatchEmbed(in_channels, hidden_size, max_input_size, local_window)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if conditional:
            self.y_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        # self.y_embedder = LabelEmbedder(n_classes, hidden_size, class_dropout_prob)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, max_input_size // local_window, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels, local_window=local_window)
        self.unpatchify = Rearrange("b t1 (t2 d) -> b (t1 t2) d", t2=local_window)
        self.initialize_weights()
    
    # def configure_muon(self, muon_lr, adam_lr, betas, weight_decay):
    #     from muon import MuonWithAuxAdam
    #     hidden_weights = [p for p in self.blocks.parameters() if p.ndim >= 2]
    #     hidden_gains_biases = [p for p in self.blocks.parameters() if p.ndim < 2]
    #     nonhidden_params = [*self.x_embedder.parameters(), *self.t_embedder.parameters(), *self.final_layer.parameters()]
    #     if self.conditional:
    #         nonhidden_params += [*self.y_embedder.parameters()]
        
    #     param_groups = [
    #         dict(params=hidden_weights, use_muon=True,
    #             lr=muon_lr, weight_decay=weight_decay),
    #         dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
    #             lr=adam_lr, betas=betas, weight_decay=weight_decay),
    #     ]
    #     optimizer = MuonWithAuxAdam(param_groups)
    #     return optimizer

    def configure_muon(self, muon_lr, adam_lr, betas, weight_decay, local_rank, world_size):
        from flash_muon import Muon
        hidden_weights = [p for p in self.blocks.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in self.blocks.parameters() if p.ndim < 2]
        nonhidden_params = [*self.x_embedder.parameters(), *self.t_embedder.parameters(), *self.final_layer.parameters()]
        if self.conditional:
            nonhidden_params += [*self.y_embedder.parameters()]
        
        return Optimizers([
            Muon(params=hidden_weights, lr=muon_lr, weight_decay=weight_decay, rank=local_rank, world_size=world_size),
            torch.optim.AdamW(params=hidden_gains_biases+nonhidden_params,
                lr=adam_lr, betas=betas, weight_decay=weight_decay)
        ])

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.data)
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.conditional:
            nn.init.normal_(self.y_embedder.weight, std=0.02)
            nn.init.constant_(self.y_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y=None, attn_mask=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed[:, :x.shape[1]]  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        if y is not None:
            y = self.y_embedder(y)
            t = repeat(t, "b d -> b t d", t=y.shape[1])
            c = t + y
        else:
            c = t                                # (N, D)
        for block in self.blocks:
            x = block(x, c, attn_mask=attn_mask)                      # (N, T, D)
        x = self.final_layer(x, c)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

class ClassEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, hidden_size) # layer dependent embedding akin to adaLN
        # self.norm = RMSNorm(hidden_size)
        self.class_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, num_classes, bias=True),
            nn.Sigmoid()
        )
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        # )
    
    def forward(self, x, c):
        # shift, scale, gate = self.adaLN_modulation(x.mean(1)).chunk(3, dim=-1)
        class_gate = self.class_gate(x.mean(1))
        
        embs = self.embedding.weight.unsqueeze(0).expand(c.shape[0], -1, -1)
        mask = (c * class_gate).unsqueeze(-1)
        masked_embs = embs * mask
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = masked_embs.sum(dim=1) / counts.squeeze(1)

        # pooled = gate.unsqueeze(1) * modulate(self.norm(c), shift, scale)
        return pooled

class ChannelDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        max_input_size=250,
        in_channels=128,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        local_window=1,
        out_channels=None,
        **kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels * 2 if learn_sigma else out_channels
        self.num_heads = num_heads
        self.max_input_size = max_input_size

        self.x_embedder = PatchEmbed(in_channels, hidden_size, max_input_size, local_window)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.c_embedder = ClassEmbedder(26, hidden_size) # 0 is null embedding
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, max_input_size // local_window, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels, local_window=local_window)
        self.unpatchify = Rearrange("b t1 (t2 d) -> b (t1 t2) d", t2=local_window)
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.data)
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.c_embedder.embedding.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, c, attn_mask=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = torch.cat([x, y], dim=-1)
        x = self.x_embedder(x) + self.pos_embed[:, :x.shape[1]]  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        c = self.c_embedder(c)
        for block in self.blocks:
            x = block(x, t + c, attn_mask=attn_mask)                      # (N, T, D)
        x = self.final_layer(x, t + c)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

class LightningDiT(nn.Module):
    def __init__(
        self,
        max_input_size=250,
        in_channels=128,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        local_window=1,
        out_channels=None,
        **kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels * 2 if learn_sigma else out_channels
        self.num_heads = num_heads
        self.max_input_size = max_input_size

        self.x_embedder = PatchEmbed(in_channels, hidden_size, max_input_size, local_window)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.c_embedders = nn.ModuleList([
            ClassEmbedder(26, hidden_size) for _ in range(depth + 1)
        ])
        self.final_layer = LightningFinalLayer(hidden_size, self.out_channels, local_window=local_window)
        self.unpatchify = Rearrange("b t1 (t2 d) -> b (t1 t2) d", t2=local_window)

        freqs_cis = precompute_freqs_cis(
            hidden_size // num_heads,
            max_input_size * 2,
            10000,
            False,
        )
        self.register_buffer('freqs_cis', freqs_cis)
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed:
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.data)
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.c_embedders:
            nn.init.normal_(block.embedding.weight, std=0.02)
            nn.init.constant_(block.class_gate[1].weight, 0)
            nn.init.constant_(block.class_gate[1].bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, c, attn_mask=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        
        x = self.x_embedder(x) + y
        t = self.t_embedder(t)
        B, L, C = x.shape

        for block, c_embedder in zip(self.blocks, self.c_embedders):
            x = block(x, t + c_embedder(x, c), freqs_cis=self.freqs_cis[:L], attn_mask=attn_mask)
        x = self.final_layer(x, t + self.c_embedders[-1](x, c))
        x = self.unpatchify(x)
        return x

class SimpleLightningDiT(nn.Module):
    def __init__(
        self,
        max_input_size=250,
        in_channels=128,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        **kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.num_heads = num_heads
        self.max_input_size = max_input_size

        self.x_embedder = nn.Sequential(nn.Linear(in_channels, hidden_size, bias=True), RMSNorm(hidden_size))
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = LightningFinalLayer(hidden_size, in_channels, local_window=1)

        freqs_cis = precompute_freqs_cis(
            hidden_size // num_heads,
            max_input_size * 2,
            10000,
            False,
        )
        self.register_buffer('freqs_cis', freqs_cis)
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed:
        nn.init.xavier_uniform_(self.x_embedder[0].weight.data)
        nn.init.constant_(self.x_embedder[0].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        
        x = self.x_embedder(x)
        t = self.t_embedder(t)
        B, L, C = x.shape

        for block in self.blocks:
            x = block(x, t, freqs_cis=self.freqs_cis[:L])
        x = self.final_layer(x, t)
        return x

class SimpleLightningDiTWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = SimpleLightningDiT(**kwargs)
        
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
    
    def forward(self, x, t=None):
        return self.diffusion.loss(self.net, x, t=t)
    
    def sample(self, shape, n_steps=50):
        return self.sampler.sample(self.net, shape, n_steps)

class Transformer(nn.Module):
    def __init__(
        self,
        max_input_size=250,
        in_channels=128,
        out_channels=128,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        local_window=1,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(in_channels, hidden_size, max_input_size, local_window)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_input_size // local_window, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.final_layer = nn.Linear(hidden_size, out_channels, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.data)
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x, attn_mask=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2                             # (N, D)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)                      # (N, T, D)
        x = self.final_norm(x)
        x = self.final_layer(x)               # (N, T, patch_size ** 2 * out_channels)
        return x

class ClassTransformer(nn.Module):
    def __init__(
        self,
        max_input_size=250,
        in_channels=128,
        out_channels=128,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        local_window=1,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(in_channels, hidden_size, max_input_size, local_window)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_input_size // local_window, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.c_embedders = nn.ModuleList([
            ClassEmbedder(26, hidden_size) for _ in range(depth + 1)
        ])
        self.final_layer = FinalLayer(hidden_size, out_channels, local_window=local_window)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.data)
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        for block in self.c_embedders:
            nn.init.normal_(block.embedding.weight, std=0.02)
            nn.init.constant_(block.class_gate[1].weight, 0)
            nn.init.constant_(block.class_gate[1].bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, c, attn_mask=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2                             # (N, D)
        for block, c_embedder in zip(self.blocks, self.c_embedders):
            x = block(x, c_embedder(x, c), attn_mask=attn_mask)                      # (N, T, D)
        x = self.final_layer(x, self.c_embedders[-1](x, c))               # (N, T, patch_size ** 2 * out_channels)
        return x

class RegisterClassTransformer(nn.Module):
    def __init__(
        self,
        max_input_size=250,
        in_channels=128,
        out_channels=128,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        local_window=1,
        n_registers=16,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.n_registers = n_registers

        self.x_embedder = PatchEmbed(in_channels, hidden_size, max_input_size, local_window)
        self.registers = nn.Parameter(torch.randn(1, n_registers, hidden_size))

        self.blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.c_embedders = nn.ModuleList([
            ClassEmbedder(26, hidden_size) for _ in range(depth + 1)
        ])
        self.final_layer = FinalLayer(hidden_size, out_channels, local_window=local_window)

        freqs_cis = precompute_freqs_cis(
            hidden_size // num_heads,
            (max_input_size + n_registers) * 2,
            10000,
            False,
        )
        self.register_buffer('freqs_cis', freqs_cis)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.trunc_normal_(self.registers, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.data)
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        for block in self.c_embedders:
            nn.init.normal_(block.embedding.weight, std=0.02)
            nn.init.constant_(block.class_gate[1].weight, 0)
            nn.init.constant_(block.class_gate[1].bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, c, attn_mask=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x)
        x = torch.cat([self.registers.expand(x.shape[0], -1, -1), x], dim=1)
        B, L, C = x.shape

        for block, c_embedder in zip(self.blocks, self.c_embedders):
            x = block(x, c_embedder(x, c), freqs_cis=self.freqs_cis[:L], attn_mask=attn_mask)
        x = self.final_layer(x, self.c_embedders[-1](x, c))
        x = x[:, :self.n_registers]
        return x

from vector_quantize_pytorch import FSQ
class CausalLAM(nn.Module):
    def __init__(self, 
        max_input_size=250,
        in_channels=128, 
        hidden_size=1152,
        depth=12,
        num_heads=12,
        local_codebook_size=64,
        codebook_dim=32,
        local_window=1,
        action_dropout=0,
        use_fsq=False,
        ):
        super().__init__()
        assert max_input_size % local_window == 0

        self.action_dropout = action_dropout
        self.use_fsq = use_fsq

        self.encoder = Transformer(max_input_size=max_input_size, depth=depth, in_channels=in_channels, hidden_size=hidden_size, num_heads=num_heads, local_window=1)
        self.decoder = DiTWrapper(max_input_size=max_input_size-1, depth=depth, in_channels=in_channels, hidden_size=hidden_size, num_heads=num_heads, conditional=True, local_window=1)

        self.max_input_size = max_input_size
        self.local_window = local_window

        # self.to_local_action_emb = nn.Identity()
        self.to_local_action_emb = nn.Sequential(
            Rearrange("b (t1 t2) d -> b t1 (t2 d)", t1=max_input_size // local_window, t2=local_window),
            nn.Linear(local_window * in_channels, in_channels),
            nn.LayerNorm(in_channels)
        )
        if self.use_fsq:
            self.local_vq = FSQ(levels=[5, 5], dim=in_channels)
        else:
            self.local_vq = VectorQuantize(
                dim=in_channels,
                codebook_size=local_codebook_size,
                learnable_codebook=True,
                ema_update=False,
                use_cosine_sim=True,
                commitment_weight=0.25,
                codebook_dim=codebook_dim,
            )

        self.null_tokens = nn.Embedding(1, in_channels)

        self.initialize_weights()
        self.encoder.initialize_weights()
        self.decoder.model.initialize_weights()

        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)

        t = self.max_input_size // self.local_window
        # self.causal_plus_1_mask = self.register_buffer('causal_plus_1_mask', torch.tril(torch.ones((t, t), dtype=torch.bool), diagonal=1))
        self.causal_plus_1_mask = self.register_buffer('causal_plus_1_mask', torch.tril(torch.ones((t, t), dtype=torch.bool), diagonal=0))
        self.causal_mask = self.register_buffer('causal_mask', torch.tril(torch.ones((t, t), dtype=torch.bool), diagonal=0))

        # self.causal_plus_1_mask = self.register_buffer('causal_plus_1_mask', self.block_causal_mask(diag=1))
        # self.causal_mask = self.register_buffer('causal_mask', self.block_causal_mask(diag=0))
    
    def configure_optimizer(self, learning_rate, betas):
        def seperate_weights_and_biases(module):
            weights = [p for p in module.parameters() if p.ndim >= 2]
            biases = [p for p in module.parameters() if p.ndim < 2]
            return weights, biases
        def cast(thing):
            if isinstance(thing, list):
                return thing
            if isinstance(thing, nn.Parameter):
                return [thing]
            else:
                return thing.parameters()
        from muon import MuonWithAuxAdam
        adam_groups = [self.null_tokens, self.encoder.x_embedder, self.encoder.pos_embed, self.decoder.model.x_embedder, self.decoder.model.t_embedder, self.decoder.model.y_embedder, self.decoder.model.pos_embed, self.decoder.model.final_layer]

        hidden_groups = [self.to_global_action_emb, self.to_local_action_emb, self.global_vq, self.local_vq, 
                       self.encoder.blocks, self.encoder.final_norm, self.encoder.final_layer,
                       self.decoder.model.blocks]
        hidden_groups = [seperate_weights_and_biases(m) for m in hidden_groups]
        muon_groups = [g[0] for g in hidden_groups if len(g[0]) > 0]
        hidden_biases = [g[1] for g in hidden_groups if len(g[1]) > 0]
        muon_groups = [dict(params=cast(g), lr=100 * learning_rate, use_muon=True) for g in muon_groups]
        adam_groups = [dict(params=cast(g), lr=learning_rate, betas=betas, use_muon=False) for g in adam_groups + hidden_biases]

        param_groups = [*adam_groups, *muon_groups]
        optimizer = MuonWithAuxAdam(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        return optimizer

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.null_tokens.weight, std=0.02)
    
    def block_causal_mask(self, diag=0):
        """
        Create a block-causal attention mask.

        Args:
            diag: how many blocks ahead are visible (0 = current only, 1 = allow next block)

        Returns:
            mask (seq_len, seq_len) bool tensor
                mask[i, j] = True if token i can attend to token j
        """
        idx = torch.arange(self.max_input_size)
        bi = idx // self.local_window  # block index of each position
        
        bi_i = bi[:, None]  # (seq_len, 1)
        bi_j = bi[None, :]  # (1, seq_len)
        
        # Allow attending to all tokens up to "diag" blocks ahead
        allowed_blocks = (bi_j <= bi_i + diag)
        return allowed_blocks

    @torch.no_grad()
    def encode_action_indices(self, x):
        z = self.encoder(x)

        local_tokens = self.to_local_action_emb(z)
        if self.use_fsq:
            local_tokens, local_indices = self.local_vq(local_tokens)
        else:
            local_tokens, local_indices, local_vq_loss = self.local_vq(local_tokens, mask=None)

        return local_indices
    
    def encode_actions(self, x, attn_mask=None):
        z = self.encoder(x, attn_mask=attn_mask)

        # action embeddings
        local_tokens = self.to_local_action_emb(z)
        if self.use_fsq:
            local_tokens, local_indices = self.local_vq(local_tokens)
            local_vq_loss = 0
        else:
            local_tokens, local_indices, local_vq_loss = self.local_vq(local_tokens, mask=None)
        if self.training:
            mask = torch.rand(z.shape[0], z.shape[1]) < self.action_dropout
            local_tokens[mask.long()] = self.null_tokens.weight[0].to(local_tokens.dtype)
        local_tokens = repeat(local_tokens, "b t1 d -> b (t1 t2) d", t2=self.local_window)

        return local_tokens, local_indices, local_vq_loss

    def forward(self, x, targets):
        local_tokens, _, local_vq_loss = self.encode_actions(x, attn_mask=self.causal_plus_1_mask)

        local_tokens = local_tokens[:, 1:]
        x = x[:, :-1]

        loss = self.diffusion.causal_loss(self.decoder.model, x, targets, net_kwargs={'y': local_tokens, 'attn_mask': self.causal_mask}) + local_vq_loss

        return loss
    
    def generate(self, x, local_tokens, keep_tokens, max_new_tokens, n_steps=50):
        for i in tqdm(range(keep_tokens, max_new_tokens), desc='Generating'):
            if i + 1 >= x.shape[1]:
                break

            # mask = torch.ones(*x.shape[:2]).to(x.device)
            mask = torch.from_numpy(np.concatenate([np.zeros((x.shape[0], i)), np.ones((x.shape[0], max_new_tokens - i))], axis=1)).long().to(x.device)
            
            logits = self.sampler.inpaint(self.decoder.model, x.clone(), mask, net_kwargs={'y': local_tokens, 'attn_mask': self.causal_mask}, n_steps=n_steps)
            x[:, i + 1] = logits[:, i]
        
        return x
    
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
    
    def lam_vs_random_actions(self, latents, keep_tokens, max_new_tokens, n_steps=50):
        assert latents.ndim == 3

        b, c, f, device = *latents.shape, latents.device
        # t = self.max_input_size // self.local_window
        t = self.max_input_size - 1
        
        local_tokens, local_action_indices, _ = self.encode_actions(latents, attn_mask=self.causal_plus_1_mask)

        local_action_indices = local_action_indices[:, 1:]
        local_tokens = local_tokens[:, 1:]
        latents = latents[:, :-1]

        noise = torch.randn(latents.shape, device=next(self.parameters()).device)

        # generate random actions
        random_local_actions_indices = self.generate_random_different_actions(local_action_indices, self.local_vq.codebook_size, device)
        if self.use_fsq:
            random_local_action_tokens = self.local_vq.indices_to_codes(random_local_actions_indices)
        else:
            random_local_action_tokens = self.local_vq.get_output_from_indices(random_local_actions_indices)
        random_local_action_tokens = repeat(random_local_action_tokens, "b t1 d -> b (t1 t2) d", t2=self.local_window)

        # decode actions
        recon_latents = self.generate(latents, local_tokens, keep_tokens, max_new_tokens, n_steps=n_steps)
        # recon_latents = self.sampler.sample(self.decoder.model, latents.shape, net_kwargs={'y': local_tokens, 'attn_mask': self.causal_mask}, uncond_net_kwargs={'y': repeat(self.null_tokens.weight[0].to(latents.dtype), "d -> b t d", b=latents.shape[0], t=t), 'attn_mask': self.causal_mask}, n_steps=n_steps, guidance=guidance, noise=noise)
        
        # decode random actions
        random_recon_latents = self.generate(latents, random_local_action_tokens, keep_tokens, max_new_tokens, n_steps=n_steps)
        # random_recon_latents = self.sampler.sample(self.decoder.model, latents.shape, net_kwargs={'y': random_local_action_tokens, 'attn_mask': self.causal_mask}, uncond_net_kwargs={'y': repeat(self.null_tokens.weight.sum(0).to(latents.dtype), "d -> b t d", b=latents.shape[0], t=t), 'attn_mask': self.causal_mask}, n_steps=n_steps, guidance=guidance, noise=noise)

        return {'latents': recon_latents, 'local_actions': local_action_indices}, {'latents': random_recon_latents, 'local_actions': random_local_actions_indices}
    
    def inpaint_lam_vs_random_actions(self, latents, mask, max_new_tokens, n_steps=50):
        assert latents.ndim == 3

        b, c, f, device = *latents.shape, latents.device
        # t = self.max_input_size // self.local_window
        t = self.max_input_size - 1
        
        local_tokens, local_action_indices, _ = self.encode_actions(latents, attn_mask=self.causal_plus_1_mask)

        local_action_indices = local_action_indices[:, 1:]
        local_tokens = local_tokens[:, 1:]
        latents = latents[:, :-1]

        noise = torch.randn(latents.shape, device=next(self.parameters()).device)

        # generate random actions
        random_local_actions_indices = self.generate_random_different_actions(local_action_indices, self.local_vq.codebook_size, device)
        if self.use_fsq:
            random_local_action_tokens = self.local_vq.indices_to_codes(random_local_actions_indices)
        else:
            random_local_action_tokens = self.local_vq.get_output_from_indices(random_local_actions_indices)
        random_local_action_tokens = repeat(random_local_action_tokens, "b t1 d -> b (t1 t2) d", t2=self.local_window)

        # decode actions
        recon_latents = self.generate(latents, local_tokens, max_new_tokens, n_steps=n_steps)
        # recon_latents = self.sampler.inpaint(self.decoder.model, latents, mask, net_kwargs={'y': local_tokens, 'attn_mask': self.causal_mask}, uncond_net_kwargs={'y': repeat(self.null_tokens.weight[0].to(latents.dtype), "d -> b t d", b=latents.shape[0], t=t), 'attn_mask': self.causal_mask}, n_steps=n_steps, guidance=guidance, noise=noise)
        
        # decode random actions
        random_recon_latents = self.generate(latents, local_tokens, max_new_tokens, n_steps=n_steps)
        # random_recon_latents = self.sampler.inpaint(self.decoder.model, latents, mask, net_kwargs={'y': random_local_action_tokens, 'attn_mask': self.causal_mask}, uncond_net_kwargs={'y': repeat(self.null_tokens.weight.sum(0).to(latents.dtype), "d -> b t d", b=latents.shape[0], t=t), 'attn_mask': self.causal_mask}, n_steps=n_steps, guidance=guidance, noise=noise)

        return recon_latents, random_recon_latents
    
    def sample_with_actions(self, shape, local_action_indices=None, n_step=50, guidance=1):
        assert local_action_indices is not None

        # t = self.max_input_size // self.local_window
        t = self.max_input_size - 1
        device = next(self.parameters()).device
        noise = torch.randn(shape, device=device)
        if self.use_fsq:
            local_tokens = repeat(self.local_vq.indices_to_codes(local_action_indices), "b d -> b t d", t=t)
        else:
            local_tokens = repeat(self.local_vq.get_output_from_indices(local_action_indices), "b d -> b t d", t=t)

        samples = self.generate()
        samples = self.sampler.sample(self.decoder.model, shape, n_steps=n_step, net_kwargs={'y': local_tokens, 'attn_mask': self.causal_mask}, uncond_net_kwargs={'y': repeat(self.null_tokens.weight.sum(0).to(next(self.parameters()).dtype), "d -> b t d", b=shape[0], t=t), 'attn_mask': self.causal_mask}, guidance=guidance, noise=noise)

        return samples

class CrossDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone and cross attention for conditioning.
    """
    def __init__(
        self,
        max_input_size=250,
        in_channels=128,
        out_channels=128,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        y_channels=None,
        **kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = out_channels * 2 if learn_sigma else out_channels
        self.num_heads = num_heads
        self.max_input_size = max_input_size

        self.x_embedder = PatchEmbed(in_channels, hidden_size, max_input_size, 1)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Sequential(nn.Linear(in_channels if y_channels is None else y_channels, hidden_size, bias=True), nn.LayerNorm(hidden_size, eps=1e-6))
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, max_input_size, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            CrossDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels, local_window=1)
        self.unpatchify = Rearrange("b t1 (t2 d) -> b (t1 t2) d", t2=1)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                # torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.data)
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.weight, std=0.02)
        # nn.init.constant_(self.y_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed[:, :x.shape[1]]  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y)
        
        for block in self.blocks:
            x = block(x, t, y)                      # (N, T, D)
        
        x = self.final_layer(x, t)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

class InpaintingLAM(nn.Module):

    def __init__(self, 
        max_input_size=250,
        in_channels=128, 
        encoder_hidden_size=1152,
        decoder_hidden_size=1152,
        encoder_depth=12,
        decoder_depth=12,
        encoder_num_heads=12,
        decoder_num_heads=12,
        levels=[5, 5],
        window_size=8,
        min_block_size=None,
        max_block_size=None,
        ):
        super().__init__()

        self.encoder = Transformer(max_input_size=max_input_size, depth=encoder_depth, in_channels=in_channels, hidden_size=encoder_hidden_size, num_heads=encoder_num_heads, local_window=1)
        self.decoder = CrossDiT(max_input_size=max_input_size, depth=decoder_depth, in_channels=in_channels, hidden_size=decoder_hidden_size, num_heads=decoder_num_heads)

        self.max_input_size = max_input_size
        self.window_size = window_size
        self.min_block_size = min_block_size if min_block_size is not None else window_size
        self.max_block_size = max_block_size if max_block_size is not None else max_input_size
        
        self.to_action_emb = nn.Sequential(
            Rearrange("b (t1 t2) d -> b t1 (t2 d)", t1=max_input_size // window_size, t2=window_size),
            nn.Linear(window_size * in_channels, in_channels),
            nn.LayerNorm(in_channels)
        )
        self.vq = FSQ(levels=levels, dim=in_channels)

        self.attn_mask = self.register_buffer('attn_mask', self.block_attention_mask())
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
        
        self.initialize_weights()
    
    def block_attention_mask(self) -> torch.Tensor:
        """
        Create a block-diagonal attention mask.
        Each block of `window_size` allows full attention internally,
        but no attention across blocks.

        Args:
            seq_len (int): total sequence length
            window_size (int): block size

        Returns:
            mask (torch.Tensor): (seq_len, seq_len) boolean mask
                                True = keep, False = mask
        """
        assert self.max_input_size % self.window_size == 0, "Sequence length must be divisible by window_size"
        n_blocks = self.max_input_size // self.window_size

        # Identity matrix of blocks (n_blocks x n_blocks)
        block_mask = torch.eye(n_blocks, dtype=torch.bool)

        # Expand to (seq_len, seq_len) by Kronecker product
        full_mask = torch.kron(block_mask, torch.ones((self.window_size, self.window_size), dtype=torch.bool))
        
        return full_mask
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        self.encoder.initialize_weights()
        self.decoder.initialize_weights()

    @torch.no_grad()
    def encode_action_indices(self, x):
        z = self.encoder(x, attn_mask=self.attn_mask)

        tokens = self.to_action_emb(z)
        tokens, indices = self.vq(tokens)

        return indices
    
    def encode_actions(self, x):
        z = self.encoder(x, attn_mask=self.attn_mask)

        # action embeddings
        tokens = self.to_action_emb(z)
        tokens, indices = self.vq(tokens)

        return tokens, indices

    def forward(self, x):
        B, L, _ = x.shape

        tokens, _ = self.encode_actions(x)
        
        mask = torch.zeros(B, L, 1, dtype=torch.bool, device=x.device)
        lens = np.random.randint(self.min_block_size, self.max_block_size + 1, (B,))

        for i in range(B):
            start = np.random.randint(0, L - lens[i] + 1)
            mask[i, start:start + lens[i]] = True

        return self.diffusion.mask_loss(self.decoder, x, mask.long(), net_kwargs={'y': tokens})
    
    def forward(self, x):
        tokens, _ = self.encode_actions(x)

        return self.diffusion.loss(self.decoder, x, net_kwargs={'y': tokens})
    
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
    
    def lam_vs_random_actions(self, latents, mask, n_steps=50):
        action_tokens, action_indices = self.encode_actions(latents)

        # generate random actions
        random_actions_indices = self.generate_random_different_actions(action_indices, self.vq.codebook_size, latents.device)
        random_action_tokens = self.vq.indices_to_codes(random_actions_indices)

        # decode actions
        recon_latents = self.sampler.inpaint(self.decoder, latents, mask.long(), n_steps=n_steps, net_kwargs={'y': action_tokens}, clean=True)
        # decode random actions
        random_recon_latents = self.sampler.inpaint(self.decoder, latents, mask.long(), n_steps=n_steps, net_kwargs={'y': random_action_tokens}, clean=True)

        return recon_latents, random_recon_latents

class MaskLAM(nn.Module):

    def __init__(self, 
        max_input_size=250,
        in_channels=128, 
        encoder_hidden_size=1152,
        decoder_hidden_size=1152,
        encoder_depth=12,
        decoder_depth=12,
        encoder_num_heads=12,
        decoder_num_heads=12,
        levels=[5, 5],
        window_size=8,
        min_block_size=None,
        max_block_size=None,
        ):
        super().__init__()

        self.encoder = Transformer(max_input_size=max_input_size, depth=encoder_depth, in_channels=in_channels, out_channels=decoder_hidden_size, hidden_size=encoder_hidden_size, num_heads=encoder_num_heads, local_window=1)
        self.decoder = CrossDiT(max_input_size=max_input_size, depth=decoder_depth, in_channels=in_channels, out_channels=in_channels, hidden_size=decoder_hidden_size, num_heads=decoder_num_heads, y_channels=decoder_hidden_size)

        self.max_input_size = max_input_size
        self.window_size = window_size
        self.min_block_size = min_block_size if min_block_size is not None else window_size
        self.max_block_size = max_block_size if max_block_size is not None else max_input_size
        
        self.to_action_emb = nn.Sequential(
            Rearrange("b (t1 t2) d -> b t1 (t2 d)", t1=max_input_size // window_size, t2=window_size),
            nn.Linear(window_size * decoder_hidden_size, decoder_hidden_size),
            nn.LayerNorm(decoder_hidden_size)
        )
        self.vq = FSQ(levels=levels, dim=decoder_hidden_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, in_channels))

        self.attn_mask = self.register_buffer('attn_mask', self.block_attention_mask())
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
        
        self.initialize_weights()
    
    def block_attention_mask(self) -> torch.Tensor:
        """
        Create a block-diagonal attention mask.
        Each block of `window_size` allows full attention internally,
        but no attention across blocks.

        Args:
            seq_len (int): total sequence length
            window_size (int): block size

        Returns:
            mask (torch.Tensor): (seq_len, seq_len) boolean mask
                                True = keep, False = mask
        """
        assert self.max_input_size % self.window_size == 0, "Sequence length must be divisible by window_size"
        n_blocks = self.max_input_size // self.window_size

        # Identity matrix of blocks (n_blocks x n_blocks)
        block_mask = torch.eye(n_blocks, dtype=torch.bool)

        # Expand to (seq_len, seq_len) by Kronecker product
        full_mask = torch.kron(block_mask, torch.ones((self.window_size, self.window_size), dtype=torch.bool))
        
        return full_mask
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.mask_token, std=0.02)
        self.encoder.initialize_weights()
        self.decoder.initialize_weights()

    @torch.no_grad()
    def encode_action_indices(self, x):
        z = self.encoder(x, attn_mask=self.attn_mask)

        tokens = self.to_action_emb(z)
        tokens, indices = self.vq(tokens)

        return indices
    
    def encode_actions(self, x):
        z = self.encoder(x, attn_mask=self.attn_mask)

        # action embeddings
        tokens = self.to_action_emb(z)
        tokens, indices = self.vq(tokens)

        return tokens, indices

    def forward(self, x):
        target = x.clone()
        B, L, _ = x.shape

        tokens, _ = self.encode_actions(x)
        
        mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        lens = np.random.randint(self.min_block_size, self.max_block_size + 1, (B,))

        for i in range(B):
            start = np.random.randint(0, L - lens[i] + 1)
            mask[i, start:start + lens[i]] = True
        
        x[mask] = self.mask_token

        return self.diffusion.mask_target_loss(self.decoder, x, target, net_kwargs={'y': tokens})
    
    # def forward(self, x):
    #     tokens, _ = self.encode_actions(x)

    #     return self.diffusion.loss(self.decoder, x, net_kwargs={'y': tokens})
    
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
    
    def lam_vs_random_actions(self, latents, mask, n_steps=50):
        action_tokens, action_indices = self.encode_actions(latents)

        # generate random actions
        random_actions_indices = self.generate_random_different_actions(action_indices, self.vq.codebook_size, latents.device)
        random_action_tokens = self.vq.indices_to_codes(random_actions_indices)

        latents[mask] = self.mask_token

        noise = torch.randn(latents.shape, device=latents.device)
        # decode actions
        recon_latents = self.sampler.inpaint(self.decoder, latents, mask.long(), n_steps=n_steps, net_kwargs={'y': action_tokens}, noise=noise)
        # decode random actions
        random_recon_latents = self.sampler.inpaint(self.decoder, latents, mask.long(), n_steps=n_steps, net_kwargs={'y': random_action_tokens}, noise=noise)

        return recon_latents, random_recon_latents

class ConcatMaskLAM(nn.Module):
    def __init__(self, 
        max_input_size=250,
        in_channels=128, 
        encoder_hidden_size=1152,
        decoder_hidden_size=1152,
        encoder_depth=12,
        decoder_depth=12,
        encoder_num_heads=12,
        decoder_num_heads=12,
        levels=[5, 5],
        window_size=8,
        min_block_size=None,
        max_block_size=None,
        ):
        super().__init__()

        self.encoder = ClassTransformer(max_input_size=max_input_size, depth=encoder_depth, in_channels=in_channels, out_channels=decoder_hidden_size, hidden_size=encoder_hidden_size, num_heads=encoder_num_heads, local_window=1)
        self.decoder = ChannelDiT(max_input_size=max_input_size, depth=decoder_depth, in_channels=2*in_channels, out_channels=in_channels, hidden_size=decoder_hidden_size, num_heads=decoder_num_heads)

        self.max_input_size = max_input_size
        self.window_size = window_size
        self.min_block_size = min_block_size if min_block_size is not None else window_size
        self.max_block_size = max_block_size if max_block_size is not None else max_input_size
        
        self.to_action_emb = nn.Sequential(
            Rearrange("b (t1 t2) d -> b t1 (t2 d)", t1=max_input_size // window_size, t2=window_size),
            nn.Linear(window_size * decoder_hidden_size, decoder_hidden_size),
            nn.LayerNorm(decoder_hidden_size)
        )
        self.vq = FSQ(levels=levels, dim=decoder_hidden_size)
        self.action_proj = nn.Linear(decoder_hidden_size, in_channels) # this bottleneck could pose a large problem...
        self.mask_token = nn.Parameter(torch.zeros(1, 1, in_channels))

        self.attn_mask = self.register_buffer('attn_mask', self.block_attention_mask())
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
        
        self.initialize_weights()
    
    def block_attention_mask(self) -> torch.Tensor:
        """
        Create a block-diagonal attention mask.
        Each block of `window_size` allows full attention internally,
        but no attention across blocks.

        Args:
            seq_len (int): total sequence length
            window_size (int): block size

        Returns:
            mask (torch.Tensor): (seq_len, seq_len) boolean mask
                                True = keep, False = mask
        """
        assert self.max_input_size % self.window_size == 0, "Sequence length must be divisible by window_size"
        n_blocks = self.max_input_size // self.window_size

        # Identity matrix of blocks (n_blocks x n_blocks)
        block_mask = torch.eye(n_blocks, dtype=torch.bool)

        # Expand to (seq_len, seq_len) by Kronecker product
        full_mask = torch.kron(block_mask, torch.ones((self.window_size, self.window_size), dtype=torch.bool))
        
        return full_mask
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.mask_token, std=0.02)
        self.encoder.initialize_weights()
        self.decoder.initialize_weights()

    # @torch.no_grad()
    # def encode_action_indices(self, x):
    #     z = self.encoder(x, attn_mask=self.attn_mask)

    #     tokens = self.to_action_emb(z)
    #     tokens, indices = self.vq(tokens)

    #     return indices
    
    def encode_actions(self, x, labels):
        z = self.encoder(x, labels, attn_mask=self.attn_mask)

        # action embeddings
        tokens = self.to_action_emb(z)
        tokens, indices = self.vq(tokens)
        tokens = self.action_proj(tokens)
        tokens = repeat(tokens, "b t1 d -> b (t1 t2) d", t2=self.window_size)
        indices = repeat(indices, "b t1 -> b (t1 t2)", t2=self.window_size)

        return tokens, indices

    def forward(self, x, labels):
        target = x.clone()
        B, L, _ = x.shape

        tokens, _ = self.encode_actions(x, labels)
        
        mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        lens = np.random.randint(self.min_block_size, self.max_block_size + 1, (B,))

        for i in range(B):
            start = np.random.randint(0, L - lens[i] + 1)
            mask[i, start:start + lens[i]] = True

        x[mask] = self.mask_token

        return self.diffusion.concat_loss(self.decoder, x, target, mask.long(), net_kwargs={'y': tokens, 'c': labels})
    
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
    
    def lam_vs_random_actions(self, latents, labels, mask, n_steps=50):
        action_tokens, action_indices = self.encode_actions(latents, labels)

        # generate random actions
        random_actions_indices = self.generate_random_different_actions(action_indices, self.vq.codebook_size, latents.device)
        random_action_tokens = self.vq.indices_to_codes(random_actions_indices)
        random_action_tokens = self.action_proj(random_action_tokens)

        latents[mask] = self.mask_token
        noise = torch.randn(latents.shape, device=latents.device)

        # decode actions
        recon_latents = self.sampler.inpaint(self.decoder, latents, mask.long(), n_steps=n_steps, noise=noise, net_kwargs={'y': action_tokens, 'c': labels})
        # decode random actions
        random_recon_latents = self.sampler.inpaint(self.decoder, latents, mask.long(), n_steps=n_steps, noise=noise, net_kwargs={'y': random_action_tokens, 'c': labels})

        return recon_latents, random_recon_latents

class InstrumentConcatMaskLAM(nn.Module):
    def __init__(self, 
        max_input_size=250,
        in_channels=128, 
        encoder_hidden_size=1152,
        decoder_hidden_size=1152,
        encoder_depth=12,
        decoder_depth=12,
        encoder_num_heads=12,
        decoder_num_heads=12,
        levels=[5, 5],
        window_size=8,
        min_block_size=None,
        max_block_size=None,
        ):
        super().__init__()

        self.encoder = RegisterClassTransformer(max_input_size=max_input_size, depth=encoder_depth, in_channels=in_channels, out_channels=decoder_hidden_size, hidden_size=encoder_hidden_size, num_heads=encoder_num_heads, local_window=1, n_registers=max_input_size // window_size)
        self.decoder = LightningDiT(max_input_size=max_input_size, depth=decoder_depth, in_channels=in_channels, out_channels=in_channels, hidden_size=decoder_hidden_size, num_heads=decoder_num_heads)

        self.max_input_size = max_input_size
        self.window_size = window_size
        self.min_block_size = min_block_size if min_block_size is not None else window_size
        self.max_block_size = max_block_size if max_block_size is not None else max_input_size
        
        self.vq = FSQ(levels=levels, dim=decoder_hidden_size)
        self.mask_token = nn.Parameter(torch.randn(1, 1, in_channels))

        # self.attn_mask = self.register_buffer('attn_mask', self.block_attention_mask())
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
        
        self.initialize_weights()
    
    def block_attention_mask(self) -> torch.Tensor:
        """
        Create a block-diagonal attention mask.
        Each block of `window_size` allows full attention internally,
        but no attention across blocks.

        Args:
            seq_len (int): total sequence length
            window_size (int): block size

        Returns:
            mask (torch.Tensor): (seq_len, seq_len) boolean mask
                                True = keep, False = mask
        """
        assert self.max_input_size % self.window_size == 0, "Sequence length must be divisible by window_size"
        n_blocks = self.max_input_size // self.window_size

        # Identity matrix of blocks (n_blocks x n_blocks)
        block_mask = torch.eye(n_blocks, dtype=torch.bool)

        # Expand to (seq_len, seq_len) by Kronecker product
        full_mask = torch.kron(block_mask, torch.ones((self.window_size, self.window_size), dtype=torch.bool))
        
        return full_mask
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.encoder.initialize_weights()
        self.decoder.initialize_weights()

    @torch.no_grad()
    def encode_actions(self, x, labels):
        tokens = self.encoder(x, labels)

        # action embeddings
        tokens, indices = self.vq(tokens)
        tokens = repeat(tokens, "b t1 d -> b (t1 t2) d", t2=self.window_size)
        indices = repeat(indices, "b t1 -> b (t1 t2)", t2=self.window_size)

        return tokens, indices

    def forward(self, x, labels):
        target = x.clone()
        B, L, _ = x.shape

        tokens, _ = self.encode_actions(x, labels)
        
        mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        lens = np.random.randint(self.min_block_size, self.max_block_size + 1, (B,))

        for i in range(B):
            start = np.random.randint(0, L - lens[i] + 1)
            mask[i, start:start + lens[i]] = True

        x[mask] = self.mask_token

        return self.diffusion.concat_loss(self.decoder, x, target, mask.long(), net_kwargs={'y': tokens, 'c': labels})
    
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
    
    def lam_vs_random_actions(self, latents, labels, mask, n_steps=50):
        action_tokens, action_indices = self.encode_actions(latents, labels)

        # generate random actions
        random_actions_indices = self.generate_random_different_actions(action_indices, self.vq.codebook_size, latents.device)
        random_action_tokens = self.vq.indices_to_codes(random_actions_indices)

        latents[mask] = self.mask_token
        noise = torch.randn(latents.shape, device=latents.device)

        # decode actions
        recon_latents = self.sampler.inpaint(self.decoder, latents, mask.long(), n_steps=n_steps, noise=noise, net_kwargs={'y': action_tokens, 'c': labels})
        # decode random actions
        random_recon_latents = self.sampler.inpaint(self.decoder, latents, mask.long(), n_steps=n_steps, noise=noise, net_kwargs={'y': random_action_tokens, 'c': labels})

        return recon_latents, random_recon_latents

from collections import defaultdict
stats = defaultdict(list)
class MaskedDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        max_input_size=250,
        in_channels=128,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        **kwargs,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads

        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Linear(in_channels, hidden_size, bias=True)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Embedding(max_input_size, hidden_size)
        self.mask_token = nn.Embedding(1, hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.xavier_uniform_(self.x_embedder.weight.data)
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize label embedding proj:
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        nn.init.constant_(self.y_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def mask_tokens(self, x, t, mask=None, force_mask=False):
        if mask is not None:
            x[mask.long()] = self.mask_token.weight[0].to(x.dtype)
            return x
        
        if self.training or force_mask:
            B, T = x.shape[:2]
            device = x.device
            N = 2
            min_len = 25
            max_len = 50 * 3

            # Random number of spans per sample
            num_spans = torch.randint(1, N + 1, (B,), device=device)  # [B]

            # Random lengths and starts
            span_lens = torch.randint(min_len, max_len + 1, (B, N), device=device)
            span_starts = torch.randint(0, T, (B, N), device=device)
            span_starts = torch.minimum(span_starts, T - span_lens)

            t_range = torch.arange(T, device=device).view(1, 1, T)

            starts = span_starts.unsqueeze(-1)  # [B, N, 1]
            lengths = span_lens.unsqueeze(-1)   # [B, N, 1]

            span_mask = (t_range >= starts) & (t_range < starts + lengths)  # [B, N, T]

            ids = torch.arange(N, device=device).view(1, N)
            active = ids < num_spans.view(B, 1)
            active = active.unsqueeze(-1)  # [B, N, 1]

            span_mask = span_mask & active
            mask = span_mask.any(dim=1)  # [B, T]

            p = torch.rand(x.shape[0])
            full_mask = (p < 0.15).long()
            no_mask = (p < 0.05).long()
            mask[full_mask] = 1
            mask[no_mask] = 0
            
            # stats['min'].append(mask.float().min(0)[0].mean().item())
            # stats['mean'].append(mask[~full_mask & ~no_mask].float().mean(0).mean().item())
            # stats['std'].append(mask[~full_mask & ~no_mask].float().std(0).mean().item())
            # stats['max'].append(mask.float().max(0)[0].mean().item())

            # for k, v in stats.items():
            #     print(k, np.mean(v))

            def exponential_decay(t, lam):
                t_scaled = t / 1000
                probs = (torch.exp(-lam * t_scaled) - torch.exp(torch.tensor(-lam))) / (1 - torch.exp(torch.tensor(-lam)))
                return probs

            probs = exponential_decay(t, 20)
            mask_row = torch.bernoulli(probs).bool()
            mask = mask & mask_row.unsqueeze(-1)
            x[mask] = self.mask_token.weight[0].to(x.dtype)
            return x
        
        return x

    def forward(self, x, t, y, mask=None, force_mask=False):
        """
        Forward pass of DiT.
        x: (N, L, C) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, L, C) tensor of class labels
        """
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        x = self.mask_tokens(x, t, mask=mask, force_mask=force_mask)
        x = x + self.pos_embed.weight.unsqueeze(0).expand(x.shape)
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y)
        t = repeat(t, "b d -> b t d", t=y.shape[1])
        c = t + y                            # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)               # (N, T, patch_size ** 2 * out_channels)
        return x

class MaskedLAM(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        max_input_size=250,
        hidden_size=1152,
        local_codebook_size=16,
        global_codebook_size=8,
        codebook_dim=32,
        ema_update=False,
        local_window=1,
        **kwargs,
    ):
        super().__init__()
        assert max_input_size % local_window == 0

        self.encoder = encoder
        self.decoder = decoder

        self.to_global_action_emb = nn.Sequential(
            Rearrange("b t d -> b (t d)"),
            nn.Linear(max_input_size * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.max_input_size = max_input_size
        self.local_window = local_window
        self.to_local_action_emb = nn.Sequential(
            Rearrange("b (t1 t2) d -> b t1 (t2 d)", t1=max_input_size // local_window, t2=local_window),
            nn.Linear(local_window * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        self.global_vq = VectorQuantize(
            dim=hidden_size,
            codebook_size=global_codebook_size,
            learnable_codebook=not ema_update,
            ema_update=ema_update,
            use_cosine_sim=True,
            commitment_weight=0.25,
            codebook_dim=codebook_dim,
            decay=0.999,
        )

        self.local_vq = VectorQuantize(
            dim=hidden_size,
            codebook_size=local_codebook_size,
            learnable_codebook=not ema_update,
            ema_update=ema_update,
            use_cosine_sim=True,
            commitment_weight=0.25,
            codebook_dim=codebook_dim,
            decay=0.999,
        )

        self.null_tokens = nn.Embedding(2, hidden_size)

        self.initialize_weights()
        self.encoder.initialize_weights()
        self.decoder.model.initialize_weights()

        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def configure_optimizer(self, learning_rate, betas):
        def seperate_weights_and_biases(module):
            weights = [p for p in module.parameters() if p.ndim >= 2]
            biases = [p for p in module.parameters() if p.ndim < 2]
            return weights, biases
        def cast(thing):
            if isinstance(thing, list):
                return thing
            if isinstance(thing, nn.Parameter):
                return [thing]
            else:
                return thing.parameters()
        from muon import MuonWithAuxAdam
        adam_groups = [self.null_tokens, self.encoder.x_embedder, self.encoder.pos_embed, self.decoder.model.x_embedder, self.decoder.model.t_embedder, self.decoder.model.y_embedder, self.decoder.model.pos_embed, self.decoder.model.final_layer]

        hidden_groups = [self.to_global_action_emb, self.to_local_action_emb, self.global_vq, self.local_vq, 
                       self.encoder.blocks, self.encoder.final_norm, self.encoder.final_layer,
                       self.decoder.model.blocks]
        hidden_groups = [seperate_weights_and_biases(m) for m in hidden_groups]
        muon_groups = [g[0] for g in hidden_groups if len(g[0]) > 0]
        hidden_biases = [g[1] for g in hidden_groups if len(g[1]) > 0]
        muon_groups = [dict(params=cast(g), lr=100 * learning_rate, use_muon=True) for g in muon_groups]
        adam_groups = [dict(params=cast(g), lr=learning_rate, betas=betas, use_muon=False) for g in adam_groups + hidden_biases]

        param_groups = [*adam_groups, *muon_groups]
        optimizer = MuonWithAuxAdam(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        return optimizer

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.null_tokens.weight, std=0.02)

    @torch.no_grad()
    def encode_action_indices(self, x):
        z = self.encoder(x)

        global_tokens = self.to_global_action_emb(z)
        global_tokens, global_indices, global_vq_loss = self.global_vq(global_tokens, mask=None)

        local_tokens = self.to_local_action_emb(z)
        local_tokens, local_indices, local_vq_loss = self.local_vq(local_tokens, mask=None)

        return global_indices, local_indices
    
    def encode_actions(self, x):
        z = self.encoder(x)

        # action embeddings
        global_tokens = self.to_global_action_emb(z)
        global_tokens, global_indices, global_vq_loss = self.global_vq(global_tokens, mask=None)
        if self.training:
            mask = torch.rand(x.shape[0]) < 0.1
            global_tokens[mask.long()] = self.null_tokens.weight[0].to(global_tokens.dtype)
        global_tokens = repeat(global_tokens, "b d -> b t d", t=x.shape[1])

        local_tokens = self.to_local_action_emb(z)
        local_tokens, local_indices, local_vq_loss = self.local_vq(local_tokens, mask=None)
        if self.training:
            mask =  torch.rand(x.shape[0], x.shape[1]) < 0.1
            local_tokens[mask.long()] = self.null_tokens.weight[1].to(local_tokens.dtype)
        local_tokens = repeat(local_tokens, "b t1 d -> b (t1 t2) d", t2=self.local_window)

        return (global_tokens, local_tokens), (global_indices, local_indices), (global_vq_loss, local_vq_loss)

    def forward(self, x, force_mask=False):
        (global_tokens, local_tokens), _, (global_vq_loss, local_vq_loss) = self.encode_actions(x)

        loss = self.diffusion.loss(self.decoder.model, x, net_kwargs={'y': global_tokens + local_tokens, 'force_mask': force_mask}) + global_vq_loss + local_vq_loss

        return loss
    
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
    
    def lam_vs_random_actions(self, latents, n_steps=50, guidance=1):
        assert latents.ndim == 3

        b, t, c, device = *latents.shape, latents.device

        (global_tokens, local_tokens), (global_action_indices, local_action_indices), _ = self.encode_actions(latents)

        noise = torch.randn(latents.shape, device=next(self.parameters()).device)

        # generate random actions
        random_global_actions_indices = self.generate_random_different_actions(global_action_indices, self.global_vq.codebook_size, device)
        random_global_action_tokens = self.global_vq.get_output_from_indices(random_global_actions_indices)
        random_global_action_tokens = repeat(random_global_action_tokens, "b d -> b t d", t=latents.shape[1])

        random_local_actions_indices = self.generate_random_different_actions(local_action_indices, self.local_vq.codebook_size, device)
        random_local_action_tokens = self.local_vq.get_output_from_indices(random_local_actions_indices)
        random_local_action_tokens = repeat(random_local_action_tokens, "b t1 d -> b (t1 t2) d", t2=self.local_window)

        # decode actions
        recon_latents = self.sampler.masked_sample(self.decoder.model, latents.shape, net_kwargs={'y': global_tokens + local_tokens}, uncond_net_kwargs={'y': repeat(self.null_tokens.weight.sum(0).to(latents.dtype), "d -> b t d", b=latents.shape[0], t=latents.shape[1])}, n_steps=n_steps, guidance=guidance, noise=noise)
        
        # decode random actions
        random_recon_latents = self.sampler.masked_sample(self.decoder.model, latents.shape, net_kwargs={'y': random_global_action_tokens + random_local_action_tokens}, uncond_net_kwargs={'y': repeat(self.null_tokens.weight.sum(0).to(latents.dtype), "d -> b t d", b=latents.shape[0], t=latents.shape[1])}, n_steps=n_steps, guidance=guidance, noise=noise)

        return {'latents': recon_latents, 'global_actions': global_action_indices, 'local_actions': local_action_indices}, {'latents': random_recon_latents, 'global_actions': random_global_actions_indices, 'local_actions': random_local_actions_indices}
    
    def inpaint(self, latents, mask, n_steps=50, guidance=1):
        (global_tokens, local_tokens), _, _ = self.encode_actions(latents)

        noise = torch.randn(latents.shape, device=next(self.parameters()).device)
        inpaints = self.sampler.masked_inpaint(self.decoder.model, latents, mask, net_kwargs={'y': global_tokens + local_tokens}, uncond_net_kwargs={'y': repeat(self.null_tokens.weight.sum(0).to(latents.dtype), "d -> b t d", b=latents.shape[0], t=latents.shape[1])}, n_steps=n_steps, guidance=guidance, noise=noise)

        return inpaints
    
    def sample_with_actions(self, shape, global_action_indices=None, local_action_indices=None, n_step=50, guidance=1):
        assert global_action_indices is not None or local_action_indices is not None

        noise = torch.randn(shape, device=next(self.parameters()).device)
        if global_action_indices is not None:
            global_tokens = repeat(self.global_vq.get_output_from_indices(global_action_indices), "b d -> b t d", t=shape[1])
        else:
            global_tokens = repeat(self.null_tokens.weight[0], "d -> b t d", b=shape[0], t=shape[1])
        
        if local_action_indices is not None:
            local_tokens = repeat(self.local_vq.get_output_from_indices(local_action_indices), "b d -> b t d", t=shape[1])
        else:
            local_tokens = repeat(self.null_tokens.weight[1], "d -> b t d", b=shape[0], t=shape[1])

        samples = self.sampler.masked_sample(self.decoder.model, shape, n_steps=n_step, net_kwargs={'y': global_tokens + local_tokens}, uncond_net_kwargs={'y': repeat(self.null_tokens.weight.sum(0).to(next(self.parameters()).dtype), "d -> b t d", b=shape[0], t=shape[1])}, guidance=guidance, noise=noise)

        return samples

class LAM(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        encoder,
        decoder,
        max_input_size=250,
        hidden_size=1152,
        local_codebook_size=16,
        global_codebook_size=8,
        codebook_dim=32,
        ema_update=False,
        local_window=1,
        **kwargs,
    ):
        super().__init__()
        assert max_input_size % local_window == 0

        self.encoder = encoder
        self.decoder = decoder

        self.to_global_action_emb = nn.Sequential(
            Rearrange("b t d -> b (t d)"),
            nn.Linear(max_input_size * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.max_input_size = max_input_size
        self.local_window = local_window
        self.to_local_action_emb = nn.Sequential(
            Rearrange("b (t1 t2) d -> b t1 (t2 d)", t1=max_input_size // local_window, t2=local_window),
            nn.Linear(local_window * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        self.global_vq = VectorQuantize(
            dim=hidden_size,
            codebook_size=global_codebook_size,
            learnable_codebook=not ema_update,
            ema_update=ema_update,
            use_cosine_sim=True,
            commitment_weight=0.25,
            codebook_dim=codebook_dim,
            decay=0.999,
        )

        self.local_vq = VectorQuantize(
            dim=hidden_size,
            codebook_size=local_codebook_size,
            learnable_codebook=not ema_update,
            ema_update=ema_update,
            use_cosine_sim=True,
            commitment_weight=0.25,
            codebook_dim=codebook_dim,
            decay=0.999,
        )

        self.null_tokens = nn.Embedding(2, hidden_size)

        self.initialize_weights()
        self.encoder.initialize_weights()
        self.decoder.model.initialize_weights()

        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def configure_optimizer(self, learning_rate, betas):
        def seperate_weights_and_biases(module):
            weights = [p for p in module.parameters() if p.ndim >= 2]
            biases = [p for p in module.parameters() if p.ndim < 2]
            return weights, biases
        def cast(thing):
            if isinstance(thing, list):
                return thing
            if isinstance(thing, nn.Parameter):
                return [thing]
            else:
                return thing.parameters()
        from muon import MuonWithAuxAdam
        adam_groups = [self.null_tokens, self.encoder.x_embedder, self.encoder.pos_embed, self.decoder.model.x_embedder, self.decoder.model.t_embedder, self.decoder.model.y_embedder, self.decoder.model.pos_embed, self.decoder.model.final_layer]

        hidden_groups = [self.to_global_action_emb, self.to_local_action_emb, self.global_vq, self.local_vq, 
                       self.encoder.blocks, self.encoder.final_norm, self.encoder.final_layer,
                       self.decoder.model.blocks]
        hidden_groups = [seperate_weights_and_biases(m) for m in hidden_groups]
        muon_groups = [g[0] for g in hidden_groups if len(g[0]) > 0]
        hidden_biases = [g[1] for g in hidden_groups if len(g[1]) > 0]
        muon_groups = [dict(params=cast(g), lr=100 * learning_rate, use_muon=True) for g in muon_groups]
        adam_groups = [dict(params=cast(g), lr=learning_rate, betas=betas, use_muon=False) for g in adam_groups + hidden_biases]

        param_groups = [*adam_groups, *muon_groups]
        optimizer = MuonWithAuxAdam(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

        return optimizer

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.null_tokens.weight, std=0.02)

    @torch.no_grad()
    def encode_action_indices(self, x):
        z = self.encoder(x)

        global_tokens = self.to_global_action_emb(z)
        global_tokens, global_indices, global_vq_loss = self.global_vq(global_tokens, mask=None)

        local_tokens = self.to_local_action_emb(z)
        local_tokens, local_indices, local_vq_loss = self.local_vq(local_tokens, mask=None)

        return global_indices, local_indices
    
    def encode_actions(self, x):
        z = self.encoder(x)

        # action embeddings
        global_tokens = self.to_global_action_emb(z)
        global_tokens, global_indices, global_vq_loss = self.global_vq(global_tokens, mask=None)
        if self.training:
            mask = torch.rand(x.shape[0]) < 0.1
            global_tokens[mask.long()] = self.null_tokens.weight[0].to(global_tokens.dtype)
        global_tokens = repeat(global_tokens, "b d -> b t d", t=x.shape[1])

        local_tokens = self.to_local_action_emb(z)
        local_tokens, local_indices, local_vq_loss = self.local_vq(local_tokens, mask=None)
        if self.training:
            mask =  torch.rand(x.shape[0], x.shape[1]) < 0.1
            local_tokens[mask.long()] = self.null_tokens.weight[1].to(local_tokens.dtype)
        local_tokens = repeat(local_tokens, "b t1 d -> b (t1 t2) d", t2=self.local_window)

        return (global_tokens, local_tokens), (global_indices, local_indices), (global_vq_loss, local_vq_loss)

    def forward(self, x):
        (global_tokens, local_tokens), _, (global_vq_loss, local_vq_loss) = self.encode_actions(x)

        loss = self.diffusion.loss(self.decoder.model, x, net_kwargs={'y': global_tokens + local_tokens}) + global_vq_loss + local_vq_loss

        return loss
    
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
    
    def lam_vs_random_actions(self, latents, n_steps=50, guidance=1):
        assert latents.ndim == 3

        b, c, f, device = *latents.shape, latents.device

        (global_tokens, local_tokens), (global_action_indices, local_action_indices), _ = self.encode_actions(latents)

        noise = torch.randn(latents.shape, device=next(self.parameters()).device)

        # generate random actions
        random_global_actions_indices = self.generate_random_different_actions(global_action_indices, self.global_vq.codebook_size, device)
        random_global_action_tokens = self.global_vq.get_output_from_indices(random_global_actions_indices)
        random_global_action_tokens = repeat(random_global_action_tokens, "b d -> b t d", t=latents.shape[1])

        random_local_actions_indices = self.generate_random_different_actions(local_action_indices, self.local_vq.codebook_size, device)
        random_local_action_tokens = self.local_vq.get_output_from_indices(random_local_actions_indices)
        random_local_action_tokens = repeat(random_local_action_tokens, "b t1 d -> b (t1 t2) d", t2=self.local_window)

        # decode actions
        recon_latents = self.sampler.sample(self.decoder.model, latents.shape, net_kwargs={'y': global_tokens + local_tokens}, uncond_net_kwargs={'y': repeat(self.null_tokens.weight.sum(0).to(latents.dtype), "d -> b t d", b=latents.shape[0], t=latents.shape[1])}, n_steps=n_steps, guidance=guidance, noise=noise)
        
        # decode random actions
        random_recon_latents = self.sampler.sample(self.decoder.model, latents.shape, net_kwargs={'y': random_global_action_tokens + random_local_action_tokens}, uncond_net_kwargs={'y': repeat(self.null_tokens.weight.sum(0).to(latents.dtype), "d -> b t d", b=latents.shape[0], t=latents.shape[1])}, n_steps=n_steps, guidance=guidance, noise=noise)

        return {'latents': recon_latents, 'global_actions': global_action_indices, 'local_actions': local_action_indices}, {'latents': random_recon_latents, 'global_actions': random_global_actions_indices, 'local_actions': random_local_actions_indices}
    
    def inpaint(self, latents, mask, n_steps=50, guidance=1):
        (global_tokens, local_tokens), _, _ = self.encode_actions(latents)

        inpaints = self.sampler.inpaint(self.decoder.model, latents, mask, net_kwargs={'y': global_tokens + local_tokens}, uncond_net_kwargs={'y': repeat(self.null_tokens.weight.sum(0).to(latents.dtype), "d -> b t d", b=latents.shape[0], t=latents.shape[1])}, n_steps=n_steps, guidance=guidance)

        return inpaints
    
    def sample_with_actions(self, shape, global_action_indices=None, local_action_indices=None, n_step=50, guidance=1):
        assert global_action_indices is not None or local_action_indices is not None

        noise = torch.randn(shape, device=next(self.parameters()).device)
        if global_action_indices is not None:
            global_tokens = repeat(self.global_vq.get_output_from_indices(global_action_indices), "b d -> b t d", t=shape[1])
        else:
            global_tokens = repeat(self.null_tokens.weight[0], "d -> b t d", b=shape[0], t=shape[1])
        
        if local_action_indices is not None:
            local_tokens = repeat(self.local_vq.get_output_from_indices(local_action_indices), "b d -> b t d", t=shape[1])
        else:
            local_tokens = repeat(self.null_tokens.weight[1], "d -> b t d", b=shape[0], t=shape[1])

        samples = self.sampler.sample(self.decoder.model, shape, n_steps=n_step, net_kwargs={'y': global_tokens + local_tokens}, uncond_net_kwargs={'y': repeat(self.null_tokens.weight.sum(0).to(next(self.parameters()).dtype), "d -> b t d", b=shape[0], t=shape[1])}, guidance=guidance, noise=noise)

        return samples
    
    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

class DiTWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = DiT(*args, **kwargs)

        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x):
        loss = self.diffusion.loss(self.model, x)
        return loss
    
    def sample(self, shape, n_steps=50):
        return self.sampler.sample(self.model, shape, n_steps)
    
    def inpaint(self, z, mask, n_steps=50):
        return self.sampler.inpaint(self.model, z, mask, n_steps)

class CrossDiTWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = CrossDiT(*args, **kwargs)

        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x):
        loss = self.diffusion.loss(self.model, x)
        return loss
    
    def sample(self, shape, n_steps=50):
        return self.sampler.sample(self.model, shape, n_steps)
    
    def inpaint(self, z, mask, n_steps=50):
        return self.sampler.inpaint(self.model, z, mask, n_steps)

class MaskedDiTWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = MaskedDiT(*args, **kwargs)

        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x):
        loss = self.diffusion.loss(self.model, x)
        return loss
    
    def sample(self, shape, n_steps=50):
        return self.sampler.sample(self.model, shape, n_steps)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=np.float32)

    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiTWrapper(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiTWrapper(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiTWrapper(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiTWrapper(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def Transformer_L_2(**kwargs):
    return Transformer(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def LAM_L_2(**kwargs):
    return LAM(encoder=Transformer_B_2(**kwargs), decoder=DiT_L_2(conditional=True, **kwargs), depth=24, hidden_size=128, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiTWrapper(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiTWrapper(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiTWrapper(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_L(**kwargs):
    return DiTWrapper(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def DiT_M(**kwargs):
    return DiTWrapper(depth=20, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def DiT_B(**kwargs):
    return DiTWrapper(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def MaskedDiT_B_2(**kwargs):
    return MaskedDiTWrapper(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def Transformer_B_2(**kwargs):
    return Transformer(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def LAM_B_2(**kwargs):
    return LAM(encoder=Transformer_B_2(**kwargs), decoder=DiT_B_2(conditional=True, **kwargs), depth=12, hidden_size=128, patch_size=2, num_heads=12, **kwargs)

def MaskedLAM_B_2(**kwargs):
    return MaskedLAM(encoder=Transformer_B_2(**kwargs), decoder=MaskedDiT_B_2(conditional=True, **kwargs), depth=12, hidden_size=128, patch_size=2, num_heads=12, **kwargs)

def CausalLAM_L(**kwargs):
    return CausalLAM(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def CausalLAM_M(**kwargs):
    return CausalLAM(depth=20, hidden_size=768, num_heads=12, **kwargs)

def CausalLAM_B(**kwargs):
    return CausalLAM(depth=12, hidden_size=768, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiTWrapper(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiTWrapper(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiTWrapper(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiTWrapper(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiTWrapper(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def MaskLAM_M(**kwargs):
    return MaskLAM(encoder_depth=12, encoder_hidden_size=768, encoder_num_heads=12, decoder_depth=20, decoder_hidden_size=768, decoder_num_heads=12, **kwargs)

def ConcatMaskLAM_M(**kwargs):
    return ConcatMaskLAM(encoder_depth=12, encoder_hidden_size=768, encoder_num_heads=12, decoder_depth=20, decoder_hidden_size=768, decoder_num_heads=12, **kwargs)

def InstrumentMaskLAM_M(**kwargs):
    return InstrumentConcatMaskLAM(encoder_depth=12, encoder_hidden_size=768, encoder_num_heads=12, decoder_depth=20, decoder_hidden_size=768, decoder_num_heads=12, **kwargs)

def InpaintingLAM_L(**kwargs):
    return InpaintingLAM(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def InpaintingLAM_M(**kwargs):
    return InpaintingLAM(encoder_depth=12, encoder_hidden_size=768, encoder_num_heads=12, decoder_depth=20, decoder_hidden_size=768, decoder_num_heads=12, **kwargs)

def InpaintingLAM_B(**kwargs):
    return InpaintingLAM(depth=12, hidden_size=768, num_heads=12, **kwargs)

def LightningDiT_M(**kwargs):
    return SimpleLightningDiTWrapper(depth=20, hidden_size=1024, num_heads=16, **kwargs)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}