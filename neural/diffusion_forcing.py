import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint

import math
from typing import Optional, List, Callable
from scipy.optimize import linear_sum_assignment

# class FM:
    
#     def __init__(self, sigma_min=1e-5, timescale=1.0):
#         self.sigma_min = sigma_min
#         self.prediction_type = None
#         self.timescale = timescale
    
#     def alpha(self, t):
#         return 1.0 - t
    
#     def sigma(self, t):
#         return self.sigma_min + t * (1.0 - self.sigma_min)
    
#     def A(self, t):
#         return 1.0
    
#     def B(self, t):
#         return -(1.0 - self.sigma_min)
    
#     def get_betas(self, n_timesteps):
#         return torch.zeros(n_timesteps) # Not VP and not supported
    
#     def add_noise(self, x, t, noise=None):
#         noise = torch.randn_like(x) if noise is None else noise
#         s = [x.shape[0], x.shape[1], x.shape[2], 1]
#         x_t = self.alpha(t).view(*s) * x + self.sigma(t).view(*s) * noise
#         return x_t, noise
    
#     def loss(self, net, x, t=None, net_kwargs=None, return_loss_unreduced=False, return_all=False):
#         B, T, N, C = x.shape
        
#         if net_kwargs is None:
#             net_kwargs = {}
        
#         if t is None:
#             t = torch.rand(B, T, device=x.device)
#             # t = torch.sigmoid(torch.randn(B, T, device=x.device)) # for logit normal
#             repeat_t = t.unsqueeze(2).repeat(1, 1, N)
#         x_t, noise = self.add_noise(x, repeat_t)
        
#         pred = net(x_t, t=t * self.timescale, **net_kwargs)
        
#         target = self.A(repeat_t) * x + self.B(repeat_t) * noise # -dxt/dt
#         if return_loss_unreduced:
#             loss = ((pred.float() - target.float()) ** 2).mean(dim=[1, 2])
#             if return_all:
#                 return loss, t, x_t, pred
#             else:
#                 return loss, t
#         else:
#             loss = ((pred.float() - target.float()) ** 2).mean()
#             if return_all:
#                 return loss, x_t, pred
#             else:
#                 return loss
    
#     def get_prediction(
#         self,
#         net,
#         x_t,
#         t,
#         net_kwargs=None,
#         uncond_net_kwargs=None,
#         guidance=1.0,
#     ):
#         if net_kwargs is None:
#             net_kwargs = {}
        
#         if guidance != 1.0:
#             assert uncond_net_kwargs is not None
#             uncond_pred = net(x_t, t=t * self.timescale, **uncond_net_kwargs)
#             cond_pred = net(x_t, t=t * self.timescale, **net_kwargs)
#             pred = uncond_pred + guidance * (cond_pred - uncond_pred)
        
#         # if guidance != 1.0:
#         #     assert uncond_net_kwargs is not None
            
#         #     x_t = torch.cat([x_t, x_t], dim=0)
#         #     t = torch.cat([t, t], dim=0)
#         #     combined_kwargs = {}
#         #     # we assume the keys match
#         #     for k, v in net_kwargs.items():
#         #         combined_kwargs[k] = torch.cat([v, uncond_net_kwargs[k]], dim=0)
#         #     combined_pred = net(x_t, t=t * self.timescale, **combined_kwargs)
#         #     pred, uncond_pred = combined_pred.chunk(2, dim=0)
#         #     pred = uncond_pred + guidance * (pred - uncond_pred)
#         else:
#             pred = net(x_t, t=t * self.timescale, **net_kwargs)
            
#         return pred
    
#     def convert_sample_prediction(self, x_t, t, pred):
#         M = torch.tensor([
#             [self.alpha(t), self.sigma(t)],
#             [self.A(t), self.B(t)],
#         ], dtype=torch.float64)
#         M_inv = torch.linalg.inv(M)
#         sample_pred = M_inv[0, 0].item() * x_t + M_inv[0, 1].item() * pred
#         return sample_pred

# class FMEulerSampler:

#     def __init__(self, diffusion):
#         self.diffusion = diffusion

#     def sample(
#         self,
#         net,
#         shape,
#         n_steps,
#         net_kwargs=None,
#         uncond_net_kwargs=None,
#         guidance=1.0,
#         noise=None,
#     ):
#         """
#         Implements simple uniform noise sampling for bidirectional generation
#         """
#         device = next(net.parameters()).device
#         x_t = torch.randn(shape, device=device) if noise is None else noise
#         t_steps = torch.linspace(1, 0, n_steps + 1, device=device)

#         with torch.no_grad():
#             for i in range(n_steps):
#                 t = t_steps[i].repeat(x_t.shape[0], x_t.shape[1])
#                 neg_v = self.diffusion.get_prediction(
#                     net,
#                     x_t,
#                     t,
#                     net_kwargs=net_kwargs,
#                     uncond_net_kwargs=uncond_net_kwargs,
#                     guidance=guidance,
#                 )
#                 x_t = x_t + neg_v * (t_steps[i] - t_steps[i + 1])
#         return x_t

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

    @torch.compiler.disable
    def get_ot_noise(self, x: torch.Tensor, noise: torch.Tensor):
        B = x.shape[0]
        
        x_flat = x.view(B, -1).detach()
        noise_flat = noise.view(B, -1).detach()
        
        # cost_matrix = torch.cdist(x_flat, noise_flat, p=2)
        cost_matrix = torch.cdist(noise_flat, x_flat, p=2)
        cost_matrix_np = cost_matrix.cpu().numpy()
        
        _, col_ind = linear_sum_assignment(cost_matrix_np)
        
        col_ind_tensor = torch.from_numpy(col_ind).to(device=x.device, dtype=torch.long)
        
        return noise[col_ind_tensor]
    
    def loss(self, net, x, t=None, net_kwargs=None, return_loss_unreduced=False, return_all=False):
        B, T, N, C = x.shape
        
        if net_kwargs is None:
            net_kwargs = {}
        
        if t is None:
            # uniform
            # t = torch.rand(B, T, device=x.device)
            # logit normal
            t = torch.sigmoid(torch.randn(B, T, device=x.device))
            repeat_t = t.unsqueeze(2).repeat(1, 1, N)
        
        noise = torch.randn_like(x)
        noise = self.get_ot_noise(x, noise)
        
        x_t, noise = self.add_noise(x, repeat_t, noise=noise)
        
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
    
    @staticmethod
    def _concat_kwargs(kwarg_list, dim=0):
        """
        Recursively concatenates tensors in a list of identical-structure dictionaries.
        Safely handles nested dictionaries like 'unconditional_mask'.
        """
        combined = {}
        for k in kwarg_list[0].keys():
            val = kwarg_list[0][k]
            if isinstance(val, dict):
                # Recurse for nested dicts (e.g., the unconditional_mask)
                combined[k] = FM._concat_kwargs([kw[k] for kw in kwarg_list], dim=dim)
            elif isinstance(val, torch.Tensor):
                # Concatenate tensors along the batch dimension
                combined[k] = torch.cat([kw[k] for kw in kwarg_list], dim=dim)
            else:
                # For non-tensors (e.g., bool flags or strings), assume they are 
                # constant across the batch and just copy the first one.
                combined[k] = val
        return combined
    
    def get_prediction(
        self,
        net,
        x_t,
        t,
        net_kwargs=None,
        uncond_net_kwargs=None,
        guidance=1.0,
        memory_efficient=False,
        rescale_phi=0.0,
        cfg_mode="independent",
    ):
        if net_kwargs is None:
            net_kwargs = {}
            
        # Normalize inputs to lists
        if isinstance(net_kwargs, dict):
            net_kwargs_list = [net_kwargs]
            guidance_list = [guidance]
        else:
            net_kwargs_list = net_kwargs
            guidance_list = guidance if isinstance(guidance, list) else [guidance] * len(net_kwargs_list)

        is_cfg = any(g != 1.0 for g in guidance_list) or len(net_kwargs_list) > 1

        if not is_cfg:
            # Standard single pass (no CFG)
            return net(x_t, t=t * self.timescale, **net_kwargs_list[0])
            
        assert uncond_net_kwargs is not None, "uncond_net_kwargs must be provided when using guidance."

        # ==========================================
        # MODE 1: JOINT CFG (Single combined pass)
        # ==========================================
        if cfg_mode == "joint":
            # 1. Merge all isolated conditions into one master kwargs dictionary
            joint_kwargs = {}
            for kw in net_kwargs_list:
                joint_kwargs.update(kw)
                
            # 2. Pick a single guidance scale (defaults to the first one in the list)
            g = guidance_list[0]
            
            if not memory_efficient:
                batched_x_t = torch.cat([x_t] * 2, dim=0)
                batched_t = torch.cat([t] * 2, dim=0)
                
                combined_kwargs = self._concat_kwargs([uncond_net_kwargs, joint_kwargs], dim=0)
                combined_pred = net(batched_x_t, t=batched_t * self.timescale, **combined_kwargs)
                
                uncond_pred, joint_pred = combined_pred.chunk(2, dim=0)
            else:
                uncond_pred = net(x_t, t=t * self.timescale, **uncond_net_kwargs)
                joint_pred = net(x_t, t=t * self.timescale, **joint_kwargs)
                
            # Standard CFG Formula
            pred = uncond_pred + g * (joint_pred - uncond_pred)
            reference_pred = joint_pred # The reference is just the unscaled joint prediction
            
        # ==========================================
        # MODE 2: INDEPENDENT CFG (Compositional)
        # ==========================================
        elif cfg_mode == "independent":
            if not memory_efficient:
                n_passes = 1 + len(net_kwargs_list)
                batched_x_t = torch.cat([x_t] * n_passes, dim=0)
                batched_t = torch.cat([t] * n_passes, dim=0)
                
                list_to_cat = [uncond_net_kwargs] + net_kwargs_list
                combined_kwargs = self._concat_kwargs(list_to_cat, dim=0)
                
                combined_pred = net(batched_x_t, t=batched_t * self.timescale, **combined_kwargs)
                preds = combined_pred.chunk(n_passes, dim=0)
                
                uncond_pred = preds[0]
                cond_preds = preds[1:]
                
                pred = uncond_pred.clone()
                reference_pred = uncond_pred.clone() 
                
                for g, cp in zip(guidance_list, cond_preds):
                    delta = cp - uncond_pred
                    pred += g * delta
                    reference_pred += delta 
            else:
                uncond_pred = net(x_t, t=t * self.timescale, **uncond_net_kwargs)
                pred = uncond_pred.clone()
                reference_pred = uncond_pred.clone()
                
                for kwargs, g in zip(net_kwargs_list, guidance_list):
                    if g == 0.0:
                        continue 
                    cp = net(x_t, t=t * self.timescale, **kwargs)
                    delta = cp - uncond_pred
                    pred += g * delta
                    reference_pred += delta 
        else:
            raise ValueError(f"Unknown cfg_mode: {cfg_mode}")

        # ==========================================
        # GUIDANCE RESCALING (Applies to both modes)
        # ==========================================
        if rescale_phi > 0.0:
            dims_to_reduce = tuple(range(1, pred.ndim))
            
            std_cfg = pred.std(dim=dims_to_reduce, keepdim=True)
            std_ref = reference_pred.std(dim=dims_to_reduce, keepdim=True)
            
            factor = std_ref / (std_cfg + 1e-8)
            pred_rescaled = pred * factor
            
            pred = rescale_phi * pred_rescaled + (1.0 - rescale_phi) * pred
            
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
        memory_efficient=False,
        rescale_phi=0,
        cfg_mode="independent",
        t_dist="uniform",
    ):
        """
        Implements simple uniform noise sampling for bidirectional generation
        Supports Compositional CFG by passing lists to net_kwargs and guidance.
        """
        assert t_dist in ['uniform', 'logit'], f't_dist must be uniform or logit but got {t_dist}'
        device = next(net.parameters()).device
        x_t = torch.randn(shape, device=device) if noise is None else noise
        
        if t_dist == 'uniform':
            t_steps = torch.linspace(1, 0, n_steps + 1, device=device)
        elif t_dist == 'logit':
            u = torch.linspace(1.0 - 1e-5, 1e-5, n_steps + 1, device=device)
            z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
            t_steps = torch.sigmoid(z)
            t_steps[0] = 1.0
            t_steps[-1] = 0.0

        with torch.no_grad():
            for i in range(n_steps):
                t = t_steps[i].repeat(x_t.shape[0], x_t.shape[1])
                neg_v = self.diffusion.get_prediction(
                    net,
                    x_t,
                    t,
                    net_kwargs=net_kwargs,
                    uncond_net_kwargs=uncond_net_kwargs,
                    guidance=guidance,
                    memory_efficient=memory_efficient,
                    rescale_phi=rescale_phi,
                    cfg_mode=cfg_mode
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

    # @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=False, proj_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=True)
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size ** 0.5,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
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
        x = x + self.drop_path(gate_msa * rearrange(self.attn(rearrange(modulate(self.norm1(x), shift_msa, scale_msa), 'b t n c -> b (t n) c'), freqs_cis=freqs_cis, attn_mask=attn_mask), 'b (t n) c -> b t n c', t=T, n=N))
        x = x + self.drop_path(gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        x = rearrange(x, 'b t n c -> b (t n) c')
        return x

class DiTAirBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=False, proj_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=True)
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size ** 0.5,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x, t, freqs_cis=None, attn_mask=None):
        B, N, C = x.shape
        B, T, C = t.shape
        
        print('shapes: ', x.shape, t.shape)
        
        biases = self.scale_shift_table[None] + t.reshape(x.size(0), T, 6, -1)
        print(biases.shape)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = [chunk.squeeze(2) for chunk in biases.chunk(6, dim=-2)]
        print(shift_msa.shape, scale_msa.shape)
        
        modulate_out = torch.cat([x[:, :N-T], modulate(self.norm1(x[:, :-T]), shift_msa, scale_msa)], dim=1)
        attn_out = self.attn(modulate_out, freqs_cis=freqs_cis, attn_mask=attn_mask)
        x = x + self.drop_path(gate_msa * attn_out)
        modulate_out = torch.cat([x[:, :N-T], modulate(self.norm2(x[:, :-T]), shift_mlp, scale_mlp)], dim=1)
        x = x + self.drop_path(gate_mlp * self.mlp(modulate_out))
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

def token_drop(labels, null_token, training, p_uncond=0.1, p_full=0.3, p_ind_low=0.1, p_ind_high=0.6):
    """
    Partitions the batch into three mutually exclusive training modes:
    1. Unconditional (Drop All)
    2. Full Conditional (Keep All)
    3. Partial Conditional (Drop Individual Tokens)
    
    Args:
        labels: (B, ...) Input tensor
        null_token: (1, C) Learnable null vector
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
    
    sample_specific_drop_rates = torch.rand(batch_rand_shape, device=device) * (p_ind_high - p_ind_low) + p_ind_low
    token_noise = torch.rand(labels.shape[:-1], device=device)
    mask_token_drop = token_noise < sample_specific_drop_rates
    
    final_mask = mask_drop_all.unsqueeze(-1) | (mask_partial_mode.unsqueeze(-1) & mask_token_drop.unsqueeze(-1))
    null_token = null_token.to(labels.dtype)
    
    return torch.where(final_mask, null_token, labels)

# def multi_token_drop(
#     signals: dict, 
#     null_tokens: dict, 
#     training: bool, 
#     p_joint_uncond=0.1, 
#     p_joint_full=0.2, 
#     p_ind_uncond=0.1, 
#     p_ind_low=0.1, 
#     p_ind_high=0.5
# ):
#     """
#     Applies hierarchical CFG dropping across multiple conditioning signals.
    
#     Hierarchy:
#     1. Joint Uncond (p_joint_uncond): Drops ALL signals for the batch item.
#     2. Joint Full (p_joint_full): Keeps ALL signals perfectly intact.
#     3. Independent Mode: For remaining batch items, each signal independently decides:
#        a) Independent Uncond (p_ind_uncond): Drop this specific signal entirely.
#        b) Partial Token Drop: Drop individual tokens within this signal sequence.
       
#     Args:
#         signals: Dict of input tensors, e.g., {'chroma': (B, T, C), 'bpm': (B, T, C)}
#         null_tokens: Dict of learnable null vectors, e.g., {'chroma': (1, C)}
#         training: Boolean flag
#     """
#     if not training:
#         return signals
    
#     # Extract batch size and device from the first signal
#     first_key = list(signals.keys())[0]
#     B = signals[first_key].shape[0]
#     device = signals[first_key].device
    
#     # For a (B, T, C) tensor, this creates a shape of (B, 1)
#     batch_rand_shape = (B,) + (1,) * (signals[first_key].ndim - 2)
    
#     # --- LEVEL 1: SYNCHRONIZED JOINT MASKS ---
#     # These masks are shared across ALL signals to maintain the joint distribution
#     batch_rand = torch.rand(batch_rand_shape, device=device)
    
#     mask_joint_uncond = batch_rand < p_joint_uncond
#     mask_joint_full = batch_rand >= (1.0 - p_joint_full)
#     mask_independent_mode = ~(mask_joint_uncond | mask_joint_full)
    
#     output_signals = {}
    
#     # --- LEVEL 2: INDEPENDENT MASKS ---
#     for key, labels in signals.items():
#         null_t = null_tokens[key].to(labels.dtype)
        
#         # 1. Independent Unconditional Drop (Drop the entire sequence for this specific signal)
#         ind_rand = torch.rand(batch_rand_shape, device=device)
#         mask_ind_uncond = mask_independent_mode & (ind_rand < p_ind_uncond)
        
#         # 2. Token-Level Drop (Only happens if we are in independent mode AND didn't drop the whole signal)
#         mask_token_mode = mask_independent_mode & ~mask_ind_uncond
        
#         # Randomize drop sparsity per batch item
#         sample_specific_drop_rates = torch.rand(batch_rand_shape, device=device) * (p_ind_high - p_ind_low) + p_ind_low
        
#         # Generate noise for every token in the sequence -> (B, T)
#         token_noise = torch.rand(labels.shape[:-1], device=device)
        
#         # Evaluate which tokens to drop
#         # sample_specific_drop_rates broadcasts from (B, 1) to (B, T) naturally
#         # mask_token_drop = token_noise < sample_specific_drop_rates.squeeze(-1) if sample_specific_drop_rates.dim() > 1 else token_noise < sample_specific_drop_rates
#         mask_token_drop = token_noise < sample_specific_drop_rates
        
#         # --- COMBINE ALL MASKS ---
#         # Final mask shape needs to be (B, T, 1) to broadcast over the channel dimension
#         final_mask = mask_joint_uncond.unsqueeze(-1) | \
#                      mask_ind_uncond.unsqueeze(-1) | \
#                      (mask_token_mode.unsqueeze(-1) & mask_token_drop.unsqueeze(-1))
        
#         # Apply the mask: Replace dropped tokens with the specific null token for this signal
#         output_signals[key] = torch.where(final_mask, null_t, labels)
        
#     return output_signals

def multi_token_drop(
    signals: dict, 
    null_tokens: dict, 
    training: bool, 
    p_joint_uncond=0.10, 
    p_joint_full=0.40, 
    p_one_hot=0.30,
    p_ind_uncond=0.20, 
    p_ind_low=0.05, 
    p_ind_high=0.30
):
    """
    Applies hierarchical CFG dropping optimized for Compositional CFG inference.
    
    Hierarchy (Batched Partitioning):
    1. Joint Uncond (10%): Drops ALL signals.
    2. Joint Full (40%): Keeps ALL signals pristine.
    3. One-Hot Mode (30%): Keeps EXACTLY 1 signal, drops the rest.
    4. Independent Mode (20%): Binomial dropping per signal, plus partial sequence drops.
    """
    if not training:
        return signals
    
    first_key = list(signals.keys())[0]
    B = signals[first_key].shape[0]
    device = signals[first_key].device
    num_signals = len(signals)
    
    # --- LEVEL 0: BATCH PARTITIONING ---
    # Generate a single random float per batch item to route it to one of the 4 modes
    batch_rand = torch.rand((B,), device=device)
    
    limit_1 = p_joint_uncond
    limit_2 = limit_1 + p_joint_full
    limit_3 = limit_2 + p_one_hot
    
    mask_joint_uncond = batch_rand < limit_1
    mask_joint_full   = (batch_rand >= limit_1) & (batch_rand < limit_2)
    mask_one_hot      = (batch_rand >= limit_2) & (batch_rand < limit_3)
    mask_independent  = batch_rand >= limit_3
    
    # Pre-calculate the "kept" index for the One-Hot slice of the batch
    # Each batch item in this mode will randomly select one index (0 to num_signals-1) to keep
    one_hot_keep_idx = torch.randint(0, num_signals, (B,), device=device)
    
    output_signals = {}
    
    # --- PROCESS EACH SIGNAL ---
    for idx, (key, labels) in enumerate(signals.items()):
        null_t = null_tokens[key].to(labels.dtype).to(device)
        T = labels.shape[1]
        
        # 1. One-Hot Drop Logic
        # Drop the signal if we are in one-hot mode AND it was not the randomly selected index
        is_not_the_kept_signal = (idx != one_hot_keep_idx)
        mask_drop_one_hot = mask_one_hot & is_not_the_kept_signal
        
        # 2. Independent Unconditional Drop Logic
        # Calculate a 20% drop chance, applied ONLY if routed to independent mode
        ind_rand = torch.rand((B,), device=device)
        mask_ind_uncond = mask_independent & (ind_rand < p_ind_uncond)
        
        # Combine all sequence-level dropping scenarios into one 1D mask: Shape (B,)
        full_drop_mask = mask_joint_uncond | mask_drop_one_hot | mask_ind_uncond
        
        # 3. Token-Level Drop Logic
        # Only applies to batch items in independent mode where the signal SURVIVED the un-cond drop
        mask_token_mode = mask_independent & ~mask_ind_uncond
        
        # Determine how severe the sequence masking is per batch item
        sample_drop_rates = torch.rand((B,), device=device) * (p_ind_high - p_ind_low) + p_ind_low
        
        # Generate token-level noise and evaluate: Shape (B, T)
        token_noise = torch.rand((B, T), device=device)
        # Unsqueeze sample_drop_rates to (B, 1) so it broadcasts smoothly across the T dimension
        mask_token_drop = token_noise < sample_drop_rates.unsqueeze(1)
        
        # Apply the token mode gate
        mask_token_drop = mask_token_mode.unsqueeze(1) & mask_token_drop
        
        # --- COMBINE MASKS & APPLY ---
        # Broadcast the 1D full drop mask to 2D, then combine with token drops: Shape (B, T)
        final_mask = full_drop_mask.unsqueeze(1) | mask_token_drop
        
        # Safely align dimensions for torch.where
        # Expands (B, T) into (B, T, 1) or (B, T, 1, 1) based on target label shape
        while final_mask.ndim < labels.ndim:
            final_mask = final_mask.unsqueeze(-1)
            
        # Swap the dropped tokens for the learned null vectors
        output_signals[key] = torch.where(final_mask, null_t, labels)
        
    return output_signals

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
        bias: bool = True,
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
            padding=kernel_size//2,
            bias=bias
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
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride, 
            dilation=dilation,
            num_groups=num_groups,
            bias=bias
        )

        self.block2 = ConvBlock1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            num_groups=num_groups,
            bias=BpmRmsChromaStyleConditionalModernDiT_smedium
        )

        if in_channels != out_channels or stride != 1:
            self.to_out = nn.Conv1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=1, 
                stride=stride,
                bias=bias
            )
        else:
            self.to_out = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.to_out(x)

class Patcher(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        
        self.block = ResnetBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=patch_size,
            num_groups=1,
            bias=bias
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
                 use_null_token=False,
                 ):
        super().__init__()
        self.spatial_window = spatial_window
        self.use_null_token = use_null_token
        max_input_size = spatial_window * n_chunks
        
        self.t_embedder = TimestepEmbedder(hidden_size, bias=True, swiglu=True)
        self.bpm_embedder = TimestepEmbedder(hidden_size, bias=True, swiglu=True, max_period=1000)
        self.x_embedder = Patcher(in_channels, hidden_size)
        
        self.fuse_conditioning = SwiGLUMlp(hidden_size + style_dim, int(2 / 3 * mlp_ratio * hidden_size), hidden_size, bias=True)
        
        if self.use_null_token:
            self.null_token = nn.Parameter(torch.randn(style_dim) / style_dim ** 0.5)
        
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6, bias=True),
        )
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.final_layer_scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size ** 0.5,
        )
        self.fc = nn.Linear(hidden_size, in_channels, bias=False)
        
        self.initialize_weights()
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
    
    def forward(self, x, t, bpm, actions, unconditional_mask=None):
        B, T, N, C = x.shape
        
        x = rearrange(x, 'b t n c -> (b t) c n')
        x = self.x_embedder(x)
        x = rearrange(x, '(b t) c n -> b t n c', b=B, t=T)
        bpm = self.bpm_embedder(bpm.flatten()).view(B, T, 1, -1)
        
        x = x + bpm
        x = rearrange(x, 'b t n c -> b (t n) c')
        
        t = self.t_embedder(t.flatten()).view(B, T, -1)
        
        if self.use_null_token:
            actions = token_drop(actions, self.null_token.unsqueeze(0), self.training, p_uncond=0.1, p_full=0.8, p_ind_low=0.1, p_ind_high=0.5)
            if unconditional_mask is not None:
                actions = torch.where(unconditional_mask, self.null_token.unsqueeze(0), actions)
        
        t = torch.cat([t, actions], dim=-1)
        t = self.fuse_conditioning(t)
        t0 = self.t_block(t)
        
        freqs_cis = self.freqs_cis[:x.shape[1]]
        for block in self.blocks:
            x = block(x, t0, freqs_cis=freqs_cis)
        
        # SAM Audio does not use a non-linearity on t here
        shift, scale = (self.final_layer_scale_shift_table[None, None] + F.silu(t[:, :, None])).chunk(
            2, dim=2
        )
        x = rearrange(x, 'b (t n) c -> b t n c', t=T, n=N)
        x = modulate(self.norm(x), shift.expand(-1, -1, N, -1), scale.expand(-1, -1, N, -1))
        x = self.fc(x)
        
        return x

class ModernDiTWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = ModernDiT(**kwargs)
        
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x, bpm, actions, t=None):
        return self.diffusion.loss(self.net, x, t=t, net_kwargs={'actions': actions, 'bpm': bpm})
    
    def generate(self, x, bpm, actions, unconditional_mask=None, n_steps=50, uncond_net_kwargs=None, guidance=1.0):
        return self.sampler.sample(self.net, x.shape, n_steps=n_steps, net_kwargs={'actions': actions, 'bpm': bpm, 'unconditional_mask': unconditional_mask}, uncond_net_kwargs=uncond_net_kwargs, guidance=guidance)

class UnconditionalModernDiT(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 spatial_window,
                 n_chunks,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 gradient_checkpointing=False,
                 patch_size=1,
                 **kwargs,
                 ):
        super().__init__()
        self.spatial_window = spatial_window
        self.gradient_checkpointing = gradient_checkpointing
        max_input_size = spatial_window * n_chunks
        self.patch_size = patch_size
        
        self.t_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True)
        self.x_embedder = Patcher(in_channels, hidden_size, patch_size=patch_size)
        
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
        self.fc = nn.Linear(hidden_size, in_channels * patch_size, bias=False)
        
        self.initialize_weights()
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
    
    def forward(self, x, t):
        B, T, N, C = x.shape
        
        x = rearrange(x, 'b t n c -> (b t) c n')
        x = self.x_embedder(x)
        x = rearrange(x, '(b t) c n -> b (t n) c', b=B, t=T)
        
        t = self.t_embedder(t.flatten()).view(B, T, -1)
        t0 = self.t_block(t)
        
        freqs_cis = self.freqs_cis[:x.shape[1]]
        
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, t0, freqs_cis=freqs_cis, use_reentrant=False)
            else:
                x = block(x, t0, freqs_cis=freqs_cis)
        
        # SAM Audio does not use a non-linearity on t here
        shift, scale = (self.final_layer_scale_shift_table[None, None] + F.silu(t[:, :, None])).chunk(
            2, dim=2
        )
        x = rearrange(x, 'b (t n) c -> b t n c', t=T, n=N // self.patch_size)
        x = modulate(self.norm(x), shift.expand(-1, -1, N // self.patch_size, -1), scale.expand(-1, -1, N // self.patch_size, -1))
        x = self.fc(x)
        x = rearrange(x, 'b t n (p c) -> b t (n p) c', p=self.patch_size, c=C)
        
        return x

class UnconditionalModernDiTWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = UnconditionalModernDiT(**kwargs)
        
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x, t=None):
        return self.diffusion.loss(self.net, x, t=t)
    
    def generate(self, shape, n_steps=50, noise=None):
        return self.sampler.sample(self.net, shape, n_steps=n_steps, noise=noise)

class StyleConditionalModernDiT(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 spatial_window,
                 n_chunks,
                 style_dim,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 gradient_checkpointing=False,
                 patch_size=1,
                 use_null_token=False,
                 **kwargs,
                 ):
        super().__init__()
        self.spatial_window = spatial_window
        self.gradient_checkpointing = gradient_checkpointing
        max_input_size = spatial_window * n_chunks
        self.patch_size = patch_size
        self.use_null_token = use_null_token
        
        self.t_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True)
        self.x_embedder = Patcher(in_channels, hidden_size, patch_size=patch_size)
        self.c_embedder = nn.Linear(style_dim, hidden_size, bias=True)
        self.fuse_conditioning = SwiGLUMlp(hidden_size + hidden_size, int(2 / 3 * mlp_ratio * hidden_size), hidden_size, bias=False)
        
        if self.use_null_token:
            self.null_token = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
        
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
        self.fc = nn.Linear(hidden_size, in_channels * patch_size, bias=False)
        
        self.initialize_weights()
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
    
    def forward(self, x, t, c, unconditional_mask=None):
        B, T, N, C = x.shape
        
        x = rearrange(x, 'b t n c -> (b t) c n')
        x = self.x_embedder(x)
        x = rearrange(x, '(b t) c n -> b (t n) c', b=B, t=T)
        
        t = self.t_embedder(t.flatten()).view(B, T, -1)
        c = self.c_embedder(c)
        
        if self.use_null_token:
            c = token_drop(c, self.null_token.unsqueeze(0), self.training, p_uncond=0.1, p_full=0.8, p_ind_low=0.1, p_ind_high=0.5)
            if unconditional_mask is not None:
                c = torch.where(unconditional_mask, self.null_token.unsqueeze(0), c)
        
        t = torch.cat([t, c], dim=-1)
        t = self.fuse_conditioning(t)
        t0 = self.t_block(t)
        
        freqs_cis = self.freqs_cis[:x.shape[1]]
        
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, t0, freqs_cis=freqs_cis, use_reentrant=False)
            else:
                x = block(x, t0, freqs_cis=freqs_cis)
        
        # SAM Audio does not use a non-linearity on t here
        shift, scale = (self.final_layer_scale_shift_table[None, None] + F.silu(t[:, :, None])).chunk(
            2, dim=2
        )
        x = rearrange(x, 'b (t n) c -> b t n c', t=T, n=N // self.patch_size)
        x = modulate(self.norm(x), shift.expand(-1, -1, N // self.patch_size, -1), scale.expand(-1, -1, N // self.patch_size, -1))
        x = self.fc(x)
        x = rearrange(x, 'b t n (p c) -> b t (n p) c', p=self.patch_size, c=C)
        
        return x

class StyleConditionalModernDiTWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = StyleConditionalModernDiT(**kwargs)
        
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x, c, t=None):
        return self.diffusion.loss(self.net, x, t=t, net_kwargs={'c': c})
    
    def generate(self, shape, net_kwargs=None, uncond_net_kwargs=None, n_steps=50, guidance=1.0, noise=None):
        return self.sampler.sample(self.net, shape, n_steps=n_steps, net_kwargs=net_kwargs, uncond_net_kwargs=uncond_net_kwargs, guidance=guidance, noise=noise)

class BpmRmsChromaStyleConditionalModernDiT(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 spatial_window,
                 n_chunks,
                 style_dim,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 gradient_checkpointing=False,
                 patch_size=1,
                 use_null_token=False,
                 **kwargs,
                 ):
        super().__init__()
        self.spatial_window = spatial_window
        self.gradient_checkpointing = gradient_checkpointing
        max_input_size = spatial_window * n_chunks
        self.patch_size = patch_size
        self.use_null_token = use_null_token
        
        self.t_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True)
        self.x_embedder = Patcher(in_channels, hidden_size, patch_size=patch_size)
        self.style_embedder = nn.Linear(style_dim, hidden_size, bias=True)
        self.chroma_embedder = nn.Linear(12, hidden_size, bias=True)
        self.rms_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True, max_period=10)
        self.bpm_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True, max_period=10)
        self.measure_embedder = nn.Embedding(n_chunks, hidden_size)
        
        self.fuse_conditioning = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), hidden_size, bias=False)
        
        if self.use_null_token:
            self.null_style = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
            self.null_chroma = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
            self.null_rms = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
            self.null_bpm = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
        
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
        self.fc = nn.Linear(hidden_size, in_channels * patch_size, bias=False)
        
        self.initialize_weights()
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
    
    def forward(self, x, t, bpm, rms, chroma, style, unconditional_mask=None):
        B, T, N, C = x.shape
        
        x = rearrange(x, 'b t n c -> (b t) c n')
        x = self.x_embedder(x)
        
        x = rearrange(x, '(b t) c n -> b t n c', b=B, t=T)
        measure_ids = torch.arange(T, device=x.device)
        measure_embs = self.measure_embedder(measure_ids).unsqueeze(1)
        x = x + measure_embs
        x = rearrange(x, 'b t n c -> b (t n) c', b=B, t=T)
        
        rms = (rms - 0.09749302) / 0.047287412
        bpm = (bpm - 187.5) / (226.4151001 - 144.57830811)  # IQR
        
        t = self.t_embedder(t.flatten()).view(B, T, -1)
        style = self.style_embedder(style)
        chroma = self.chroma_embedder(chroma)
        rms = self.rms_embedder(rms.flatten()).view(B, T, -1)
        bpm = self.bpm_embedder(bpm.flatten()).view(B, T, -1)
        
        if self.use_null_token:
            signals = {'style': style, 'chroma': chroma, 'bpm': bpm, 'rms': rms}
            null_tokens = {'style': self.null_style, 'chroma': self.null_chroma, 'bpm': self.null_bpm, 'rms': self.null_rms}
            signals = multi_token_drop(signals, null_tokens, self.training, p_ind_uncond=0.1, p_joint_full=0.8, p_ind_low=0.1, p_ind_high=0.5)
            style = signals['style']
            chroma = signals['chroma']
            bpm = signals['bpm']
            rms = signals['rms']
            
            if unconditional_mask is not None:
                style = torch.where(unconditional_mask['style'], self.null_style, style)
                chroma = torch.where(unconditional_mask['chroma'], self.null_chroma, chroma)
                bpm = torch.where(unconditional_mask['bpm'], self.null_bpm, bpm)
                rms = torch.where(unconditional_mask['rms'], self.null_rms, rms)
        
        t = t + style + chroma + rms + bpm
        t = self.fuse_conditioning(t)
        t0 = self.t_block(t)
        
        freqs_cis = self.freqs_cis[:x.shape[1]]
        
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, t0, freqs_cis=freqs_cis, use_reentrant=False)
            else:
                x = block(x, t0, freqs_cis=freqs_cis)
        
        # SAM Audio does not use a non-linearity on t here
        shift, scale = (self.final_layer_scale_shift_table[None, None] + F.silu(t[:, :, None])).chunk(
            2, dim=2
        )
        x = rearrange(x, 'b (t n) c -> b t n c', t=T, n=N // self.patch_size)
        x = modulate(self.norm(x), shift.expand(-1, -1, N // self.patch_size, -1), scale.expand(-1, -1, N // self.patch_size, -1))
        x = self.fc(x)
        x = rearrange(x, 'b t n (p c) -> b t (n p) c', p=self.patch_size, c=C)
        
        return x

class BpmRmsChromaStyleConditionalModernDiTWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = BpmRmsChromaStyleConditionalModernDiT(**kwargs)
        
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x, bpm, rms, chroma, style, t=None):
        return self.diffusion.loss(self.net, x, t=t, net_kwargs={'style': style, 'chroma': chroma, 'bpm': bpm, 'rms': rms})
    
    def generate(self, shape, net_kwargs=None, uncond_net_kwargs=None, n_steps=50, guidance=1.0, noise=None, memory_efficient=False):
        return self.sampler.sample(self.net, shape, n_steps=n_steps, net_kwargs=net_kwargs, uncond_net_kwargs=uncond_net_kwargs, guidance=guidance, noise=noise, memory_efficient=memory_efficient)

class MetaConditionalModernDiT(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 spatial_window,
                 n_chunks,
                 style_dim,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 gradient_checkpointing=False,
                 patch_size=1,
                 use_null_token=False,
                 **kwargs,
                 ):
        super().__init__()
        self.spatial_window = spatial_window
        self.gradient_checkpointing = gradient_checkpointing
        max_input_size = spatial_window * n_chunks
        self.patch_size = patch_size
        self.use_null_token = use_null_token
        
        self.t_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True)
        self.x_embedder = Patcher(in_channels, hidden_size, patch_size=patch_size)
        self.style_embedder = nn.Linear(style_dim, hidden_size, bias=True)
        self.chroma_embedder = nn.Linear(12, hidden_size, bias=True)
        self.rms_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True)#, max_period=20)
        self.bpm_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True)#, max_period=20)
        # self.rms_embedder = nn.Sequential(nn.Linear(1, hidden_size, bias=True), SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=False))
        # self.bpm_embedder = nn.Sequential(nn.Linear(1, hidden_size, bias=True), SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=False))
        self.mfcc_embedder = nn.Linear(12, hidden_size, bias=True)
        self.density_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True)#, max_period=20)
        self.zcr_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True)#, max_period=20)
        # self.density_embedder = nn.Sequential(nn.Linear(1, hidden_size, bias=True), SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=False))
        # self.zcr_embedder = nn.Sequential(nn.Linear(1, hidden_size, bias=True), SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=False))
        self.measure_embedder = nn.Embedding(n_chunks, hidden_size)
        
        self.fuse_conditioning = SwiGLUMlp(hidden_size * 2, int(2 / 3 * mlp_ratio * hidden_size * 2), hidden_size, bias=False)
        
        if self.use_null_token:
            self.null_style = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
            self.null_chroma = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
            self.null_rms_low = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
            self.null_rms_mid = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
            self.null_rms_high = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
            self.null_bpm = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
            self.null_mfcc = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
            self.null_density = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
            self.null_zcr = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
        
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
        self.fc = nn.Linear(hidden_size, in_channels * patch_size, bias=False)
        
        self.initialize_weights()
        self.register_buffer('freqs_cis',  precompute_freqs_cis(hidden_size // num_heads, max_input_size))
        self.register_buffer('mcff_mean', torch.tensor([
            113.30053, -17.395779, 27.279049, -11.116686, 3.1354604, -9.138969,
            -2.866072, -7.1674404, -1.6265253, -5.047512, -1.7705443, -5.0958815
        ]))
        self.register_buffer('mfcc_std', torch.tensor([
            38.435783, 28.687775, 18.932358, 14.646409, 13.498735, 10.035576,
            9.510887, 8.25433, 8.212691, 7.155225, 7.3324447, 6.5340915
        ]))
    
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
    
    def forward(self, x, t, bpm, rms_low, rms_mid, rms_high, density, zcr, mfcc, chroma, style, unconditional_mask=None):
        B, T, N, C = x.shape
        
        x = rearrange(x, 'b t n c -> (b t) c n')
        x = self.x_embedder(x)
        
        x = rearrange(x, '(b t) c n -> b t n c', b=B, t=T)
        measure_ids = torch.arange(T, device=x.device)
        measure_embs = self.measure_embedder(measure_ids).unsqueeze(1)
        x = x + measure_embs
        x = rearrange(x, 'b t n c -> b (t n) c', b=B, t=T)
        
        # rms_low = (rms_low - 6.2784767) / 4.1725345
        # rms_mid = (rms_mid - 3.2565875) / 1.7880434
        # rms_high = (rms_high - 0.26109472) / 0.3474748
        # density = (density - 2.5229013) / 1.230155
        # zcr = (zcr - 0.10766766) / 0.048143145
        # bpm = (bpm - 187.5) / (226.4151001 - 144.57830811)  # IQR
        
        mfcc = (mfcc - self.mcff_mean) / self.mfcc_std
        
        t = self.t_embedder(t.flatten()).view(B, T, -1)
        style = self.style_embedder(style)
        chroma = self.chroma_embedder(chroma)
        mfcc = self.mfcc_embedder(mfcc)
        
        rms_low = self.rms_embedder(rms_low.flatten()).view(B, T, -1)
        rms_mid = self.rms_embedder(rms_mid.flatten()).view(B, T, -1)
        rms_high = self.rms_embedder(rms_high.flatten()).view(B, T, -1)
        bpm = self.bpm_embedder(bpm.flatten()).view(B, T, -1)
        density = self.density_embedder(density.flatten()).view(B, T, -1)
        zcr = self.zcr_embedder(zcr.flatten()).view(B, T, -1)
        
        # rms_low = self.rms_embedder(rms_low.unsqueeze(-1))
        # rms_mid = self.rms_embedder(rms_mid.unsqueeze(-1))
        # rms_high = self.rms_embedder(rms_high.unsqueeze(-1))
        # bpm = self.bpm_embedder(bpm.unsqueeze(-1))
        # density = self.density_embedder(density.unsqueeze(-1))
        # zcr = self.zcr_embedder(zcr.unsqueeze(-1))
        
        if self.use_null_token:
            signals = {
                'style': style,
                'chroma': chroma,
                'bpm': bpm,
                'rms_low': rms_low,
                'rms_mid': rms_mid,
                'rms_high': rms_high,
                'mfcc': mfcc,
                'density': density,
                'zcr': zcr
            }
            null_tokens = {
                'style': self.null_style, 
                'chroma': self.null_chroma, 
                'bpm': self.null_bpm, 
                'rms_low': self.null_rms_low, 
                'rms_mid': self.null_rms_mid, 
                'rms_high': self.null_rms_high, 
                'mfcc': self.null_mfcc, 
                'density': self.null_density, 
                'zcr': self.null_zcr
            }
            signals = multi_token_drop(
                signals, 
                null_tokens, 
                self.training,
                p_joint_uncond=0.1, 
                p_joint_full=0.5,
                p_one_hot=0.3, 
                p_ind_uncond=0.1, 
                p_ind_low=0.05, 
                p_ind_high=0.3
            )
            style = signals['style']
            chroma = signals['chroma']
            bpm = signals['bpm']
            rms_low = signals['rms_low']
            rms_mid = signals['rms_mid']
            rms_high = signals['rms_high']
            mfcc = signals['mfcc']
            density = signals['density']
            zcr = signals['zcr']
            
            if unconditional_mask is not None:
                style = torch.where(unconditional_mask['style'], self.null_style, style)
                chroma = torch.where(unconditional_mask['chroma'], self.null_chroma, chroma)
                bpm = torch.where(unconditional_mask['bpm'], self.null_bpm, bpm)
                rms_low = torch.where(unconditional_mask['rms_low'], self.null_rms_low, rms_low)
                rms_mid = torch.where(unconditional_mask['rms_mid'], self.null_rms_mid, rms_mid)
                rms_high = torch.where(unconditional_mask['rms_high'], self.null_rms_high, rms_high)
                mfcc = torch.where(unconditional_mask['mfcc'], self.null_mfcc, mfcc)
                density = torch.where(unconditional_mask['density'], self.null_density, density)
                zcr = torch.where(unconditional_mask['zcr'], self.null_zcr, zcr)
        
        c = style + chroma + rms_low + rms_mid + rms_high + mfcc + density + zcr + bpm
        t = torch.cat([t, c], dim=-1)
        t = self.fuse_conditioning(t)
        t0 = self.t_block(t)
        
        freqs_cis = self.freqs_cis[:x.shape[1]]
        
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, t0, freqs_cis=freqs_cis, use_reentrant=False)
            else:
                x = block(x, t0, freqs_cis=freqs_cis)
        
        # SAM Audio does not use a non-linearity on t here
        shift, scale = (self.final_layer_scale_shift_table[None, None] + F.silu(t[:, :, None])).chunk(
            2, dim=2
        )
        x = rearrange(x, 'b (t n) c -> b t n c', t=T, n=N // self.patch_size)
        x = modulate(self.norm(x), shift.expand(-1, -1, N // self.patch_size, -1), scale.expand(-1, -1, N // self.patch_size, -1))
        x = self.fc(x)
        x = rearrange(x, 'b t n (p c) -> b t (n p) c', p=self.patch_size, c=C)
        
        return x

class MetaConditionalModernDiTWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = MetaConditionalModernDiT(**kwargs)
        
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x, bpm, rms_low, rms_mid, rms_high, density, zcr, mfcc, chroma, style, t=None):
        return self.diffusion.loss(
            self.net, 
            x, 
            t=t, 
            net_kwargs={
                'style': style, 
                'chroma': chroma, 
                'bpm': bpm, 
                'rms_low': rms_low,
                'rms_mid': rms_mid,
                'rms_high': rms_high,
                'density': density,
                'zcr': zcr,
                'mfcc': mfcc,
            }
        )
    
    def generate(self, shape, net_kwargs=None, uncond_net_kwargs=None, n_steps=50, guidance=1.0, noise=None, memory_efficient=True, rescale_phi=0, cfg_mode="independent"):
        return self.sampler.sample(
            self.net, 
            shape, 
            n_steps=n_steps, 
            net_kwargs=net_kwargs, 
            uncond_net_kwargs=uncond_net_kwargs, 
            guidance=guidance, 
            noise=noise, 
            memory_efficient=memory_efficient,
            rescale_phi=rescale_phi,
            cfg_mode=cfg_mode
        )

class MetaConditionalModernDiTV2(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 spatial_window,
                 n_chunks,
                 style_dim,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 gradient_checkpointing=False,
                 patch_size=1,
                 use_null_token=False,
                 stage=1,
                 drop_path_rate=0.1,
                 **kwargs,
                 ):
        super().__init__()
        self.spatial_window = spatial_window
        self.gradient_checkpointing = gradient_checkpointing
        max_input_size = spatial_window * n_chunks
        self.patch_size = patch_size
        self.use_null_token = use_null_token
        
        self.t_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True)
        self.x_embedder = Patcher(in_channels, hidden_size, patch_size=patch_size, bias=True)
        self.local_embedder = Patcher(16, hidden_size, patch_size=patch_size, bias=True)
        self.style_embedder = nn.Linear(style_dim, hidden_size, bias=True)
        self.bpm_embedder = nn.Embedding(350, hidden_size)
        
        if self.use_null_token:
            self.null_style = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
            self.null_bpm = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
        
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6, bias=True),
        )
        
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=dp_rates[i]) for i in range(depth)
        ])

        self.norm = RMSNorm(hidden_size)
        self.final_layer_scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size ** 0.5,
        )
        self.fc = nn.Linear(hidden_size, in_channels * patch_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(in_channels * patch_size))
        
        self.register_buffer('freqs_cis',  precompute_freqs_cis(hidden_size // num_heads, max_input_size))
        self.register_buffer('chroma_mean', torch.tensor([
            0.45533183, 0.39680213, 0.44615716, 0.42044115, 0.40855545, 0.45450154, 0.3971631, 0.496346, 0.44164586, 0.4416672, 0.44793198, 0.39493898
        ]))
        self.register_buffer('chroma_std', torch.tensor([
            0.18241853, 0.16477719, 0.18014704, 0.18011539, 0.1677363, 0.18919244, 0.16196373, 0.19185093, 0.18003348, 0.1768027, 0.18706752, 0.1618064
        ]))
        self.register_buffer('rms_mean', torch.tensor([3.2653894]))
        self.register_buffer('rms_std', torch.tensor([3.597796]))
        self.register_buffer('density_mean', torch.tensor([2.5229013]))
        self.register_buffer('density_std', torch.tensor([1.230155]))
        self.register_buffer('zcr_mean', torch.tensor([0.10766766]))
        self.register_buffer('zcr_std', torch.tensor([0.048143145]))
        self.register_buffer('flatness_mean', torch.tensor([0.011151944]))
        self.register_buffer('flatness_std', torch.tensor([0.018700112]))
        
        self.initialize_weights()
        self.set_training_stage(stage)
    
    def set_training_stage(self, stage):
        assert stage in [1, 2], f'Stage must be 1 or 2 but got {stage}'
        
        self.stage = stage
        if self.stage == 1:
            for param in self.local_embedder.parameters():
                param.requires_grad = False
        elif self.stage == 2:
            for param in self.local_embedder.parameters():
                param.requires_grad = True
    
    def create_optimizer_groups(self, weight_decay=1e-2, base_lr=1e-4, new_lr=1e-3):
        base_decay = []
        base_no_decay = []
        new_decay = []
        new_no_decay = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            new_layer = 'local_embedder' in name
            no_decay = param.ndim < 2 or name == 'bpm_embedder.weight'
            
            if new_layer:
                if no_decay:
                    new_no_decay.append(param)
                else:
                    new_decay.append(param)
            else:
                if no_decay:
                    base_no_decay.append(param)
                else:
                    base_decay.append(param)
                
        
        optim_groups = [
            {"params": base_decay, "weight_decay": weight_decay, "lr": base_lr},
            {"params": base_no_decay, "weight_decay": 0.0, "lr": base_lr},
            {"params": new_decay, "weight_decay": weight_decay, "lr": new_lr},
            {"params": new_no_decay, "weight_decay": 0.0, "lr": new_lr},
        ]
        return optim_groups
    
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
        # zero out un-trained weights
        nn.init.zeros_(self.local_embedder.block.block1.project.weight)
        nn.init.zeros_(self.local_embedder.block.block2.project.weight)
        nn.init.zeros_(self.local_embedder.block.to_out.weight)
        nn.init.zeros_(self.local_embedder.block.block1.project.bias)
        nn.init.zeros_(self.local_embedder.block.block2.project.bias)
        nn.init.zeros_(self.local_embedder.block.to_out.bias)
    
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
    
    def forward(self, x, t, bpm, rms, density, zcr, flatness, chroma, style, unconditional_mask=None):
        assert self.stage in [1, 2], f'Stage must be 1 or 2 but got {self.stage}'
        B, T, N, C = x.shape
        
        rms = (rms - self.rms_mean) / self.rms_std
        density = (density - self.density_mean) / self.density_std
        zcr = (zcr - self.zcr_mean) / self.zcr_std
        flatness = (flatness - self.flatness_mean) / self.flatness_std
        chroma = (chroma - self.chroma_mean) / self.chroma_std
        
        t = self.t_embedder(t.flatten()).view(B, T, -1)
        style = self.style_embedder(style)
        bpm = self.bpm_embedder(torch.clamp(torch.round(bpm), min=0, max=349).long())
        
        if self.use_null_token:
            signals = {
                'style': style,
                'chroma': chroma,
                'bpm': bpm,
                'rms': rms,
                'density': density,
                'zcr': zcr,
                'flatness': flatness,
            }
            null_tokens = {
                'style': self.null_style, 
                'bpm': self.null_bpm, 
                'chroma': torch.zeros_like(chroma[0, 0]), 
                'rms': torch.zeros_like(rms[0, 0]), 
                'density': torch.zeros_like(density[0, 0]), 
                'zcr': torch.zeros_like(zcr[0, 0]),
                'flatness': torch.zeros_like(flatness[0, 0]),
            }
            if self.stage == 1:
                signals = multi_token_drop(
                    signals, 
                    null_tokens, 
                    self.training,
                    p_joint_uncond=0.1, 
                    p_joint_full=0.9,
                    p_one_hot=0, 
                    p_ind_uncond=0, 
                    p_ind_low=0, 
                    p_ind_high=0
                )
            elif self.stage == 2:
                # probabilities taken from Composer https://arxiv.org/pdf/2302.09778
                signals = multi_token_drop(
                    signals, 
                    null_tokens, 
                    self.training,
                    p_joint_uncond=0.1, 
                    p_joint_full=0.1,
                    p_one_hot=0, 
                    p_ind_uncond=0.5, 
                    p_ind_low=0, 
                    p_ind_high=0
                )
            
            style = signals['style']
            chroma = signals['chroma']
            bpm = signals['bpm']
            rms = signals['rms']
            density = signals['density']
            zcr = signals['zcr']
            flatness = signals['flatness']
            
            if unconditional_mask is not None:
                scalar_zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                
                style = torch.where(unconditional_mask['style'], null_tokens['style'], style)
                bpm = torch.where(unconditional_mask['bpm'], null_tokens['bpm'], bpm)
                
                chroma = torch.where(unconditional_mask['chroma'], scalar_zero, chroma)
                rms = torch.where(unconditional_mask['rms'].squeeze(), scalar_zero, rms)
                density = torch.where(unconditional_mask['density'].squeeze(), scalar_zero, density)
                zcr = torch.where(unconditional_mask['zcr'].squeeze(), scalar_zero, zcr)
                flatness = torch.where(unconditional_mask['flatness'].squeeze(), scalar_zero, flatness)
        
        x = rearrange(x, 'b t n c -> (b t) c n')
        x = self.x_embedder(x)
        
        if self.stage == 2:
            c = torch.cat([chroma, rms.unsqueeze(-1), density.unsqueeze(-1), zcr.unsqueeze(-1), flatness.unsqueeze(-1)], dim=-1)
            c = c.unsqueeze(2).repeat(1, 1, N, 1)
            c = rearrange(c, 'b t n c -> (b t) c n')
            c = self.local_embedder(c)
            x = x + c
        x = rearrange(x, '(b t) c n -> b (t n) c', b=B, t=T)
        
        t = t + style + bpm
        t0 = self.t_block(t)
        
        freqs_cis = self.freqs_cis[:x.shape[1]]
        
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, t0, freqs_cis=freqs_cis, use_reentrant=False)
            else:
                x = block(x, t0, freqs_cis=freqs_cis)
        
        # SAM Audio does not use a non-linearity on t here
        shift, scale = (self.final_layer_scale_shift_table[None, None] + F.silu(t[:, :, None])).chunk(
            2, dim=2
        )
        x = rearrange(x, 'b (t n) c -> b t n c', t=T, n=N // self.patch_size)
        x = modulate(self.norm(x), shift.expand(-1, -1, N // self.patch_size, -1), scale.expand(-1, -1, N // self.patch_size, -1))
        x = self.fc(x) + self.bias
        x = rearrange(x, 'b t n (p c) -> b t (n p) c', p=self.patch_size, c=C)
        
        return x

class MetaConditionalModernDiTV2Wrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = MetaConditionalModernDiTV2(**kwargs)
        
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x, bpm, rms, density, zcr, flatness, chroma, style, t=None):
        return self.diffusion.loss(
            self.net, 
            x, 
            t=t, 
            net_kwargs={
                'style': style, 
                'chroma': chroma, 
                'bpm': bpm, 
                'rms': rms,
                'density': density,
                'zcr': zcr,
                'flatness': flatness,
            }
        )
    
    def generate(self, shape, net_kwargs=None, uncond_net_kwargs=None, n_steps=50, guidance=1.0, noise=None, memory_efficient=True, rescale_phi=0, cfg_mode="independent", t_dist="uniform"):
        return self.sampler.sample(
            self.net, 
            shape, 
            n_steps=n_steps, 
            net_kwargs=net_kwargs, 
            uncond_net_kwargs=uncond_net_kwargs, 
            guidance=guidance, 
            noise=noise, 
            memory_efficient=memory_efficient,
            rescale_phi=rescale_phi,
            cfg_mode=cfg_mode,
            t_dist=t_dist
        )

class _GradientBalancerFunction(torch.autograd.Function):
    """
    The hidden autograd engine that intercepts the gradient flowing from 
    a specific head down into the shared trunk, scaling it on the fly.
    """
    @staticmethod
    def forward(ctx, features, total_buffer, fix_buffer, task_weight, total_weight, ema_decay, total_norm, epsilon):
        # Save variables for the backward pass
        ctx.total_buffer = total_buffer
        ctx.fix_buffer = fix_buffer
        ctx.task_weight = task_weight
        ctx.total_weight = total_weight
        ctx.ema_decay = ema_decay
        ctx.total_norm = total_norm
        ctx.epsilon = epsilon
        
        # Pass features through untouched during the forward pass
        return features.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # 1. Compute per-batch-item norm (EnCodec style)
        dims = tuple(range(1, grad_output.dim()))
        norm = grad_output.norm(dim=dims).mean()
        batch_size = grad_output.shape[0]
        
        # 2. Update EMA buffers IN-PLACE 
        # (Using pure tensor ops: no float(), no .item() -> zero graph breaks!)
        ctx.total_buffer.mul_(ctx.ema_decay).add_(norm * batch_size)
        ctx.fix_buffer.mul_(ctx.ema_decay).add_(batch_size)
        
        # 3. Calculate the smoothed average norm
        avg_norm = ctx.total_buffer / ctx.fix_buffer
        
        # 4. Calculate the EnCodec scaling factor
        ratio = ctx.task_weight / ctx.total_weight
        scale = ratio * ctx.total_norm / (ctx.epsilon + avg_norm)
        
        # 5. Scale the gradient before it passes down into the trunk
        grad_input = grad_output * scale
        
        # Return gradients for the inputs (None for the hyperparameter arguments)
        return grad_input, None, None, None, None, None, None, None


class GradientBalancer(nn.Module):
    """
    A drop-in module that wraps the EnCodec balancing math into an automatic layer.
    """
    def __init__(self, weights: dict, ema_decay=0.999, total_norm=1.0, epsilon=1e-12):
        super().__init__()
        self.weights = weights
        self.ema_decay = ema_decay
        self.total_norm = total_norm
        self.epsilon = epsilon
        self.total_weight = sum(weights.values())
        
        # Register EMA trackers as PyTorch buffers so they live on the GPU 
        # and are saved in your model's state_dict
        for task in weights.keys():
            self.register_buffer(f'total_{task}', torch.tensor(0.0))
            self.register_buffer(f'fix_{task}', torch.tensor(0.0))

    def forward(self, features, task_name):
        """
        Pass the trunk features through this function before sending them to a head.
        """
        # Fetch the specific EMA buffers for this task
        total_buffer = getattr(self, f'total_{task_name}')
        fix_buffer = getattr(self, f'fix_{task_name}')
        task_weight = self.weights[task_name]
        
        return _GradientBalancerFunction.apply(
            features, 
            total_buffer, 
            fix_buffer, 
            task_weight, 
            self.total_weight, 
            self.ema_decay, 
            self.total_norm, 
            self.epsilon
        )

class PerceiverTokenPooler(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.d_model = d_model
        
        # 1. The Single Latent Query Token (Learned Parameter)
        # We initialize it as (1, 1, d_model) so it easily broadcasts across batches
        self.latents = nn.Parameter(torch.randn(d_model) / d_model ** 0.5)
        
        # 2. Multi-Head Cross-Attention Layer
        # batch_first=True expects input shapes to be (batch, seq_len, features)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=0, 
            batch_first=True
        )
        
        # 3. Standard Post-Attention Processing (LayerNorm + FeedForward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.mlp = SwiGLUMlp(d_model, int(2 / 3 * mlp_ratio * d_model), bias=True)

    def forward(self, signals: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            signals: A list of Tensors, where each tensor represents a processed signal.
                     Each tensor in the list must have shape (batch_size, seq_len_i, d_model).
                     Note: seq_len_i can vary between different signals!
        Returns:
            pooled_token: A tensor of shape (batch_size, 1, d_model) representing 
                          the single fused token for your DiT.
        """
        B, T, C = signals[0].shape
        M = len(signals)
        
        print('signals')
        for signal in signals:
            print(signal.shape)
            
        stacked = torch.stack(signals, dim=1).permute(0, 2, 1, 3)
        
        # Step 1: Concatenate all signals along the sequence dimension (dim=1)
        # This creates a "bag of features" of shape (batch_size, total_M_tokens, d_model)
        kv_sequence = self.norm1(stacked.reshape(B * T, M, C))
        
        # Step 2: Prepare the single Query token for the batch
        # Expand the learned latent token from (1, 1, d_model) to (batch_size, 1, d_model)
        q = self.norm2(self.latents.unsqueeze(0).unsqueeze(0).expand(B * T, -1, -1))
        
        # Step 3: Perform Cross-Attention
        # Query comes from our learned latent. Keys/Values come from the combined signals.
        # attn_output shape: (batch_size, 1, d_model)
        attn_output, _ = self.cross_attn(
            query=q, 
            key=kv_sequence, 
            value=kv_sequence
        )
        
        # Step 4: Residual connection and LayerNorm
        x = q + attn_output
        
        # Step 5: Feed-Forward Network block
        x = x + self.mlp(self.norm3(x))
        x = x.reshape(B, T, C)
        
        return x

class MetaConditionalModernDiTV2Composer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_size,
                 spatial_window,
                 n_chunks,
                 style_dim,
                 n_text_tokens,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4,
                 gradient_checkpointing=False,
                 use_null_token=False,
                 patch_size=1,
                 drop_path_rate=0.1,
                 signal_dim = {},
                 weights = {},
                 **kwargs,
                 ):
        super().__init__()
        self.spatial_window = spatial_window
        self.gradient_checkpointing = gradient_checkpointing
        self.n_chunks = n_chunks
        max_input_size = spatial_window * n_chunks + n_text_tokens
        self.patch_size = patch_size
        self.signal_dim = signal_dim
        self.use_null_token = use_null_token

        self.balancer = GradientBalancer(weights=weights)
        
        self.t_embedder = TimestepEmbedder(hidden_size, bias=False, swiglu=True)
        self.local_embedder = Patcher(16, hidden_size, patch_size=patch_size, bias=True)
        self.style_embedder = nn.Linear(style_dim, hidden_size, bias=True)
        self.bpm_embedder = nn.Embedding(350, hidden_size)
        self.text_embedder = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, hidden_size, bias=True))
        
        self.pooler = PerceiverTokenPooler(hidden_size, num_heads, mlp_ratio)
        
        self.text_embed = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
        self.audio_embed = nn.Parameter(torch.randn(hidden_size) / hidden_size ** 0.5)
        
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6, bias=True),
        )
        
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.ModuleList([
            DiTAirBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=dp_rates[i]) for i in range(depth)
        ])

        self.norm = nn.ModuleDict({
            name: RMSNorm(hidden_size) for name in signal_dim.keys()
        })
        self.final_layer_scale_shift_table = nn.ParameterDict({
            name: nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5,) for name in signal_dim.keys()
        })
        self.fc = nn.ModuleDict({
            name: nn.Linear(hidden_size, dim * patch_size, bias=False) for name, dim in signal_dim.items()
        })
        self.bias = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(dim * patch_size)) for name, dim in signal_dim.items()
        })
        
        self.register_buffer('freqs_cis',  precompute_freqs_cis(hidden_size // num_heads, max_input_size))
        self.initialize_weights()
    
    def create_optimizer_groups(self, weight_decay=1e-2, lr=1e-4):
        decay = []
        no_decay = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if param.ndim < 2:
                no_decay.append(param)
            else:
                decay.append(param)
        
        optim_groups = [
            {"params": decay, "weight_decay": weight_decay, "lr": lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        ]
        return optim_groups
    
    def initialize_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        for name in self.signal_dim.keys():
            nn.init.zeros_(self.fc[name].weight)
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
    
    def forward(self, x, t, text, unconditional_mask=None):
        x = x.squeeze(2)
        
        print(x.shape)
        style = x[..., :128]
        chroma = x[..., 128:128+12]
        rms = x[..., [128+12]]
        density = x[..., [128+13]]
        zcr = x[..., [128+14]]
        flatness = x[..., [128+15]]
        bpm = x[..., 128+16]
        
        # text null token is embedding of empty string
        if self.use_null_token:
            signals = {
                'text': text,
            }
            null_tokens = {
                'text': self.null_text,
            }
            signals = multi_token_drop(
                signals, 
                null_tokens, 
                self.training,
                p_joint_uncond=0.1, 
                p_joint_full=0.9,
                p_one_hot=0, 
                p_ind_uncond=0, 
                p_ind_low=0, 
                p_ind_high=0
            )
            
            text = signals['text']
            
            if unconditional_mask is not None:
                text = torch.where(unconditional_mask['text'], null_tokens['text'], text)
        
        # prepend x with text
        print(chroma.shape, rms.shape, density.shape)
        x = torch.cat([chroma, rms, density, zcr, flatness], dim=-1)
        print(x.shape)
        x = self.local_embedder(x.transpose(1, 2)).transpose(1, 2)
        bpm = self.bpm_embedder(torch.clamp(torch.round(bpm), min=0, max=349).long())
        style = self.style_embedder(style)
        x = self.pooler([x, bpm, style]) + self.audio_embed
        print(x.shape)
        
        print('text: ', text.shape)
        text = self.text_embedder(text) + self.text_embed
        x = torch.cat([text, x], dim=1)
        print(x.shape)
        
        B, T = t.shape
        t = self.t_embedder(t.flatten()).view(B, T, -1)
        
        t = t + text.mean(dim=1, keepdims=True) # mean pool text for global embedder
        t0 = self.t_block(t)
        print('pool: ', text.mean(dim=1, keepdims=True).shape)
        
        freqs_cis = self.freqs_cis[:x.shape[1]]
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, t0, freqs_cis=freqs_cis, use_reentrant=False)
            else:
                x = block(x, t0, freqs_cis=freqs_cis)
        
        out = {}
        for name in self.signal_dim.keys():
            features = self.balancer(x[:-self.n_chunks], name)
            print(features.shape)
            # SAM Audio does not use a non-linearity on t here
            shift, scale = (self.final_layer_scale_shift_table[name][None] + F.silu(t[:, None])).chunk(
                2, dim=2
            )
            features = rearrange(features, 'b (t n) c -> b t n c', t=T, n=1 // self.patch_size)
            features = modulate(self.norm[name](features), shift.expand(-1, -1, 1 // self.patch_size, -1), scale.expand(-1, -1, 1 // self.patch_size, -1))
            features = self.fc[name](features) + self.bias[name]
            features = rearrange(features, 'b t n (p c) -> b t (n p) c', p=self.patch_size, c=C)
            out[name] = features
        
        out = torch.cat(list(out.values()), dim=-1)
        return out

class MetaConditionalModernDiTV2ComposerWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = MetaConditionalModernDiTV2Composer(**kwargs)
        
        self.diffusion = FM(timescale=1000.0)
        self.sampler = FMEulerSampler(self.diffusion)
    
    def forward(self, x, text, t=None):
        return self.diffusion.loss(
            self.net, 
            x, 
            t=t, 
            net_kwargs={'text': text}
        )
    
    def generate(self, shape, net_kwargs=None, uncond_net_kwargs=None, n_steps=50, guidance=1.0, noise=None, memory_efficient=True, rescale_phi=0, cfg_mode="independent", t_dist="uniform"):
        return self.sampler.sample(
            self.net, 
            shape, 
            n_steps=n_steps, 
            net_kwargs=net_kwargs, 
            uncond_net_kwargs=uncond_net_kwargs, 
            guidance=guidance, 
            noise=noise, 
            memory_efficient=memory_efficient,
            rescale_phi=rescale_phi,
            cfg_mode=cfg_mode,
            t_dist=t_dist
        )

def ModernDiT_large(**kwargs):
    return ModernDiTWrapper(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def ModernDiT_medium(**kwargs):
    return ModernDiTWrapper(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def ModernDiT_small(**kwargs):
    return ModernDiTWrapper(depth=16, hidden_size=1024, num_heads=16, **kwargs)

def ModernDiT_tiny(**kwargs):
    return ModernDiTWrapper(depth=16, hidden_size=768, num_heads=12, **kwargs)

def UnconditionalModernDiT_large(**kwargs):
    return UnconditionalModernDiTWrapper(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def UnconditionalModernDiT_medium(**kwargs):
    return UnconditionalModernDiTWrapper(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def UnconditionalModernDiT_smedium(**kwargs):
    return UnconditionalModernDiTWrapper(depth=20, hidden_size=768, num_heads=12, **kwargs)

def UnconditionalModernDiT_small(**kwargs):
    return UnconditionalModernDiTWrapper(depth=16, hidden_size=1024, num_heads=16, **kwargs)

def UnconditionalModernDiT_tiny(**kwargs):
    return UnconditionalModernDiTWrapper(depth=16, hidden_size=768, num_heads=12, **kwargs)

def StyleConditionalModernDiT_large(**kwargs):
    return StyleConditionalModernDiTWrapper(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def StyleConditionalModernDiT_medium(**kwargs):
    return StyleConditionalModernDiTWrapper(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def StyleConditionalModernDiT_smedium(**kwargs):
    return StyleConditionalModernDiTWrapper(depth=20, hidden_size=768, num_heads=12, **kwargs)

def StyleConditionalModernDiT_small(**kwargs):
    return StyleConditionalModernDiTWrapper(depth=16, hidden_size=1024, num_heads=16, **kwargs)

def StyleConditionalModernDiT_tiny(**kwargs):
    return StyleConditionalModernDiTWrapper(depth=16, hidden_size=768, num_heads=12, **kwargs)

def BpmRmsChromaStyleConditionalModernDiT_large(**kwargs):
    return BpmRmsChromaStyleConditionalModernDiTWrapper(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def BpmRmsChromaStyleConditionalModernDiT_medium(**kwargs):
    return BpmRmsChromaStyleConditionalModernDiTWrapper(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def BpmRmsChromaStyleConditionalModernDiT_smedium(**kwargs):
    return BpmRmsChromaStyleConditionalModernDiTWrapper(depth=20, hidden_size=768, num_heads=12, **kwargs)

def BpmRmsChromaStyleConditionalModernDiT_small(**kwargs):
    return BpmRmsChromaStyleConditionalModernDiTWrapper(depth=16, hidden_size=1024, num_heads=16, **kwargs)

def BpmRmsChromaStyleConditionalModernDiT_tiny(**kwargs):
    return BpmRmsChromaStyleConditionalModernDiTWrapper(depth=16, hidden_size=768, num_heads=12, **kwargs)

def MetaConditionalModernDiT_large(**kwargs):
    return MetaConditionalModernDiTWrapper(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def MetaConditionalModernDiT_medium(**kwargs):
    return MetaConditionalModernDiTWrapper(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def MetaConditionalModernDiT_smedium(**kwargs):
    return MetaConditionalModernDiTWrapper(depth=20, hidden_size=768, num_heads=12, **kwargs)

def MetaConditionalModernDiT_small(**kwargs):
    return MetaConditionalModernDiTWrapper(depth=16, hidden_size=1024, num_heads=16, **kwargs)

def MetaConditionalModernDiT_tiny(**kwargs):
    return MetaConditionalModernDiTWrapper(depth=16, hidden_size=768, num_heads=12, **kwargs)

def MetaConditionalModernDiTV2_large(**kwargs):
    return MetaConditionalModernDiTV2Wrapper(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def MetaConditionalModernDiTV2_medium(**kwargs):
    return MetaConditionalModernDiTV2Wrapper(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def MetaConditionalModernDiTV2_smedium(**kwargs):
    return MetaConditionalModernDiTV2Wrapper(depth=20, hidden_size=768, num_heads=12, **kwargs)

def MetaConditionalModernDiTV2_small(**kwargs):
    return MetaConditionalModernDiTV2Wrapper(depth=16, hidden_size=1024, num_heads=16, **kwargs)

def MetaConditionalModernDiTV2_tiny(**kwargs):
    return MetaConditionalModernDiTV2Wrapper(depth=16, hidden_size=768, num_heads=12, **kwargs)

def MetaConditionalModernDiTV2Composer_large(**kwargs):
    return MetaConditionalModernDiTV2ComposerWrapper(depth=28, hidden_size=1152, num_heads=16, **kwargs)

def MetaConditionalModernDiTV2Composer_medium(**kwargs):
    return MetaConditionalModernDiTV2ComposerWrapper(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def MetaConditionalModernDiTV2Composer_smedium(**kwargs):
    return MetaConditionalModernDiTV2ComposerWrapper(depth=20, hidden_size=768, num_heads=12, **kwargs)

def MetaConditionalModernDiTV2Composer_small(**kwargs):
    return MetaConditionalModernDiTV2ComposerWrapper(depth=16, hidden_size=1024, num_heads=16, **kwargs)

def MetaConditionalModernDiTV2Composer_tiny(**kwargs):
    return MetaConditionalModernDiTV2ComposerWrapper(depth=16, hidden_size=768, num_heads=12, **kwargs)