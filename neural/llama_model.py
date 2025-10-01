import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# from vocos import MelSpectrogramFeatures, ISTFTHead

@dataclass
class ModelArgs:
    dim: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    vocab_size: int = 8 * 1024
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    dropout: float = 0.0
    rate: int = 16000
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 128
    patch_height: int = None
    patch_width: int = None

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from einops import rearrange

def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

class DitheredFSQ(nn.Module):
    def __init__(
        self,
        levels: List[int],
        dither_inference: bool = False,
        num_codebooks: int = 1,
        noise_dropout: float = 0.5,
        scale: float = 1.0,
    ):
        super().__init__()
        self.levels = levels

        _levels = torch.tensor(levels, dtype=torch.int64)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int64)
        self.register_buffer("_basis", _basis, persistent = False)

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        self.codebook_size = _levels.prod().item()

        self.num_codebooks = num_codebooks

        self.dim = codebook_dim * num_codebooks

        self.dither_inference = dither_inference

        self.scale = scale

        half_l = self.scale * 2 / (self._levels - 1)
        self.register_buffer("half_l", half_l, persistent = False)

        self.allowed_dtypes = (torch.float32, torch.float64)

        self.noise_dropout = noise_dropout

    def quantize(self, z, skip_tanh: bool = False):
        if not skip_tanh: z = torch.tanh(z)

        if not self.training:
            quantized = self._scale_and_shift_inverse(round_ste(self._scale_and_shift(z)))
        else:
            quantized = z
            mask = torch.bernoulli(torch.full([z.shape[0],1,1,1], self.noise_dropout, device = z.device)).bool().expand_as(z)
            quantized = torch.where(mask, quantized, self._scale_and_shift_inverse(round_ste(self._scale_and_shift(quantized))))
            mask = torch.bernoulli(torch.full([z.shape[0],1,1,1], self.noise_dropout, device = z.device)).bool().expand_as(z)
            quantized = torch.where(mask, quantized, z + (torch.rand_like(z) - 0.5) * self.half_l)

        return quantized

    def _scale_and_shift(self, z):
        level_indices = (z + 1 * self.scale) / self.half_l
        return level_indices
    
    def _scale_and_shift_inverse(self, level_indices):
        z = level_indices * self.half_l - 1 * self.scale
        return z

    def _indices_to_codes(self, indices):
        level_indices = self._indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def _codes_to_indices(self, zhat):
        zhat = self._scale_and_shift(zhat)
        zhat = zhat.round().to(torch.int64)
        out = (zhat * self._basis).sum(dim=-1)
        return out

    def _indices_to_level_indices(self, indices):
        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        # Expects input of batch x sequence x num_codebooks
        assert indices.shape[-1] == self.num_codebooks, f'expected last dimension of {self.num_codebooks} but found last dimension of {indices.shape[-1]}'
        codes = self._indices_to_codes(indices.to(torch.int64))
        codes = rearrange(codes, '... c d -> ... (c d)')
        return codes

    @torch.amp.autocast(device_type="cuda", enabled = False)
    def forward(self, z, skip_tanh: bool = False):

        orig_dtype = z.dtype

        assert z.shape[-1] == self.dim, f'expected dimension of {self.num_codebooks * self.dim} but found dimension of {z.shape[-1]}'

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # make sure allowed dtype before quantizing

        if z.dtype not in self.allowed_dtypes:
            z = z.to(torch.float64)

        codes = self.quantize(z, skip_tanh=skip_tanh)
        indices = self._codes_to_indices(codes)
        codes = rearrange(codes, 'b n c d -> b n (c d)')

        # cast codes back to original dtype

        if codes.dtype != orig_dtype:
            codes = codes.type(orig_dtype)

        # return quantized output and indices

        return codes, indices

class ResidualFSQBottleneck(nn.Module):
    def __init__(self, stages: List[Tuple[List[int], float]]):
        super().__init__()

        # 1st for single_tokens, others - residuals.
        self.quantizers = nn.ModuleList([
            DitheredFSQ(levels=levels, scale=scale).eval().requires_grad_(False)
            for (levels, scale) in stages])

        self.n_codebooks = len(stages)
        self.codebook_size = sum(map(len, stages)) * self.n_codebooks
        print(self.n_codebooks, self.codebook_size)

    def encode(self, x):
        input_dtype = x.dtype
        z = torch.tanh(x.to(torch.float32))
        z = rearrange(z, "b c n -> b n c")

        r = z
        res_ids = []
        for quantizer in self.quantizers:
            q, ids = quantizer(r, skip_tanh=True)
            r = r - q.to(torch.float32)
            res_ids.append(ids)

        return res_ids

    def decode(self, res_ids):
        z = sum([
            q.indices_to_codes(res_ids[i])
            for (i, q) in enumerate(self.quantizers)
        ])
        return rearrange(z, "b n c -> b c n")


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        
        self.in_proj = nn.Sequential(weight_norm(nn.Conv1d(input_dim, codebook_dim, kernel_size=1)))
        self.out_proj = nn.Sequential(weight_norm(nn.Conv1d(codebook_dim, input_dim, kernel_size=1)))
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(
                residual
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            raise ValueError('Must use flash attention')

    def forward(
        self,
        x: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x):
        h = x + self.attention.forward(self.attention_norm(x))
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

MEANS, STDS = [], []
MEAN, STD = -2.240806818008423, 3.4039316177368164
class PatchEmbed(nn.Module):
    def __init__(self, patch_width, patch_height, dim):
        super().__init__()
        self.patch_width = patch_width
        self.patch_height = patch_height

        self.head = MelSpectrogramFeatures(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=200, padding='center', power=2, to_db=True)
        self.linear = nn.Linear(patch_height * patch_width, dim)
    
    def forward(self, x):
        x = self.head(x) / 40

        # x = (x - MEAN) / STD
        # print(x.mean(), x.std())
        # x = (x - x.mean((1, 2), keepdim=True)) / (x.std((1, 2), keepdim=True) + 1e-11)
        # MEANS.append(x.mean((1, 2)).cpu().detach().numpy())
        # STDS.append(x.std((1, 2)).cpu().detach().numpy())
        # print(f'MEAN: {np.mean(MEANS)} +- {np.std(MEANS)}, STD: {np.mean(STDS)} +- {np.std(STDS)}')
        
        B, H, W = x.shape

        patches = x.clone().reshape(B, H // self.patch_height, self.patch_height, W // self.patch_width, self.patch_width)
        patches = patches.permute(0, 1, 3, 2, 4)
        patches = patches.reshape(B, (H // self.patch_height) * (W // self.patch_width), self.patch_height * self.patch_width)

        patches = self.linear(patches)

        return x, patches

class UnPatchEmbed(nn.Module):
    def __init__(self, patch_width, patch_height, dim):
        super().__init__()
        self.patch_width = patch_width
        self.patch_height = patch_height

        # self.head = ISTFTHead(dim=dim, n_fft=1024, hop_length=256, padding='center')
        self.linear = nn.Linear(dim, patch_height * patch_width)

    def forward(self, x, H, W):
        B, T, C = x.shape

        x = self.linear(x)

        x = x.reshape(B, H // self.patch_height, W // self.patch_width, self.patch_height, self.patch_width)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(B, H, W)
        
        return x

def create_patches(x, patch_height, patch_width):
    B, H, W = x.shape
    assert H % patch_height == 0 and W % patch_width == 0

    patches = x.reshape(B, H // patch_height, patch_height, W // patch_width, patch_width)
    patches = patches.permute(0, 1, 3, 2, 4)
    patches = patches.reshape(B, (H // patch_height) * (W // patch_width), patch_height * patch_width)

    return patches

def revert_patches(x, patch_height, patch_width, H, W):
    B, T, C = x.shape

    x = x.reshape(B, H // patch_height, W // patch_width, patch_height, patch_width)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.reshape(B, H, W)

    return x

def create_patches_column_major(x, patch_height, patch_width):
    B, H, W = x.shape
    assert H % patch_height == 0 and W % patch_width == 0

    patches = x.reshape(B, (H // patch_height), patch_height, (W // patch_width), patch_width)
    patches = patches.permute(0, 3, 1, 2, 4)
    patches = patches.reshape(B, (W // patch_width) * (H // patch_height), patch_height * patch_width)

    return patches

def revert_patches_column_major(x, patch_height, patch_width, H, W):
    B, T, C = x.shape

    x = x.reshape(B, W // patch_width, H // patch_height, patch_height, patch_width)
    x = x.permute(0, 2, 3, 1, 4)
    x = x.reshape(B, H, W)

    return x

def min_max_norm(batch: torch.Tensor, eps: float = 1e-8):
    batch_min = batch.reshape(batch.size(0), -1).min(dim=1, keepdim=True)[0].reshape(-1, 1, 1)
    batch_max = batch.reshape(batch.size(0), -1).max(dim=1, keepdim=True)[0].reshape(-1, 1, 1)
    return (batch - batch_min) / (batch_max - batch_min + eps), (batch_min, batch_max)

def inv_min_max_norm(batch: torch.Tensor, batch_min: torch.Tensor, batch_max: torch.Tensor, eps: float = 1e-8):
    return batch * (batch_max - batch_min + eps) + batch_min

def mean_std_norm(batch: torch.Tensor, eps: float = 1e-8):
    batch_mean = batch.mean((1, 2), keepdim=True)
    batch_std = batch.std((1, 2), keepdim=True)
    return (batch - batch_mean) / (batch_std + eps), (batch_mean, batch_std)

def inv_mean_std_norm(batch: torch.Tensor, batch_mean: torch.Tensor, batch_std: torch.Tensor, eps: float = 1e-8):
    return batch * (batch_std + eps) + batch_mean

def safe_log(x: torch.Tensor, clip_val: float = 1e-7) -> torch.Tensor:
    """
    Computes the element-wise logarithm of the input tensor with clipping to avoid near-zero values.

    Args:
        x (Tensor): Input tensor.
        clip_val (float, optional): Minimum value to clip the input tensor. Defaults to 1e-7.

    Returns:
        Tensor: Element-wise logarithm of the input tensor with clipping applied.
    """
    return torch.log(torch.clip(x, min=clip_val))

'''
https://arxiv.org/pdf/1910.11910
https://arxiv.org/pdf/2503.16989v1
'''
class STFT(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.n_fft = params.n_fft
        self.hop_length = params.hop_length
        self.win_length = params.win_length

        self.patch_height = params.patch_height
        self.patch_width = params.patch_width

        self.mag_proj = nn.Linear(params.patch_height * params.patch_width, params.dim)
        self.phase_proj = nn.Linear(params.patch_height * params.patch_width, params.dim)

        self.window = torch.hann_window(params.win_length)
    
    def forward(self, x):
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window.to(x.device), return_complex=True, center=True)
        mag, phase = torch.abs(x), torch.angle(x)

        # # global normalization
        # mag = (safe_log(mag) + 17) / 23

        # sample normalization
        mag, mag_scale = min_max_norm(safe_log(mag))

        phase = torch.from_numpy(np.unwrap(phase.cpu().detach().numpy(), axis=-1)).to(x.device)
        phase = torch.diff(phase, dim=-1, prepend=torch.zeros(*phase.shape[:2], 1).to(x.device))   # temporal differencing

        targets = mag.clone(), phase.clone()

        # # pad
        # target_T, target_F = 204, 204
        # mag = F.pad(mag, (0, target_T - mag.shape[-1], 0, target_F - mag.shape[-2]))
        # phase = F.pad(phase, (0, target_T - phase.shape[-1], 0, target_F - phase.shape[-2]))

        # drop DC band
        mag = mag[:, 1:, 1:]
        phase = phase[:, 1:, 1:]

        mag = create_patches_column_major(mag, self.patch_height, self.patch_width)
        phase = create_patches_column_major(phase, self.patch_height, self.patch_width)
        
        mag = self.mag_proj(mag)
        phase = self.phase_proj(phase)

        h = mag + phase
        return targets, mag_scale, h

# should be better to invert the transform but qualitatively doesnt seem to be the case...
def instantaneous_frequency_to_phase(x):
    return x
    phase = torch.cumsum(x, dim=-1)
    print(phase.shape)
    # return phase

    group_delay = (phase[:, 1:] - phase[:, :-1]) % 2 * torch.pi
    average_group_delay = torch.mean(group_delay, dim=1, keepdim=True)
    print(average_group_delay.shape)

    phase[:, [0]] = average_group_delay
    return phase

    # inital_phase = 

    # phase = torch.cat([initial_phase, phase], dim=-1)
    # return phase


'''
https://arxiv.org/pdf/2306.06546
For the reconstruction loss, we use distance between log-mel
spectrograms with window lengths [32,64,128,256,512,1024,2048], with corresponding number
of mels for each of [5,10,20,40,80,160,320]. The hop length is 1/4 of the window length

MultiScaleSTFTLoss.window_lengths: [2048, 512]
MelSpectrogramLoss.n_mels: [5, 10, 20, 40, 80, 160, 320]
MelSpectrogramLoss.window_lengths: [32, 64, 128, 256, 512, 1024, 2048]
MelSpectrogramLoss.mel_fmin: [0, 0, 0, 0, 0, 0, 0]
MelSpectrogramLoss.mel_fmax: [null, null, null, null, null, null, null]
MelSpectrogramLoss.pow: 1.0
MelSpectrogramLoss.clamp_eps: 1.0e-5
MelSpectrogramLoss.mag_weight: 0.0
'''
class MultiScaleMelLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=power,
        )

import torchaudio
class STFTHead(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.patch_height = params.patch_height
        self.patch_width = params.patch_width

        self.mag_mlp = nn.Sequential(
            nn.Linear(params.dim, 4 * params.dim, bias=True),
            nn.GELU(),
            nn.Linear(4 * params.dim, params.patch_height * params.patch_width, bias=True),
        )
        self.phase_mlp = nn.Sequential(
            nn.Linear(params.dim, 4 * params.dim, bias=True),
            nn.GELU(),
            nn.Linear(4 * params.dim, params.patch_height * params.patch_width, bias=True),
        )

        self.mel_transform = torchaudio.transforms.MelScale(n_mels=100, sample_rate=16000, n_stft=257)
        
    def forward(self, x, H, W, scales, targets=None):
        mag = self.mag_mlp(x)
        phase = self.phase_mlp(x)

        mag = revert_patches_column_major(mag, self.patch_height, self.patch_width, H, W)
        phase = revert_patches_column_major(phase, self.patch_height, self.patch_width, H, W)

        # # revert padding
        # mag = mag[:, :201, :201]
        # phase = phase[:, :201, :201]
        # add DC band as 0
        mag = F.pad(mag, (0, 1, 0, 1))
        phase = F.pad(phase, (0, 1, 0, 1))

        mag = torch.clamp(mag, 0, 1)

        if targets is not None:
            target_mag, target_phase = targets
            mag_loss = F.mse_loss(mag, target_mag)  # hinge loss might be best...
            mel_loss = 0#1 * F.mse_loss(self.mel_transform(mag), self.mel_transform(target_mag))

            # # global normalization
            # unnorm_mag = torch.exp(target_mag * 23 - 17)
            # unnorm_mag = torch.clip(unnorm_mag, max=1e2)s

            # invert log for this or no?
            unnorm_mag = torch.clip(torch.exp(inv_min_max_norm(target_mag, *scales)), max=1e2)

            weight = torch.clamp(torch.abs(unnorm_mag) / (1e-7 + unnorm_mag.flatten(1).max(1)[0].unsqueeze(-1).unsqueeze(-1)), 0.1, 1)
            phase_loss = (1 - torch.cos(phase - target_phase)) * weight
            phase_loss =  0.002 * phase_loss.mean()  # ideal weighting may vary by training iteration 0.001 - 0.01 or 1 at end
            # print(mag_loss.item(), phase_loss.item())
            loss = mag_loss + mel_loss + phase_loss
        else:
            loss = None

        # # global normalization
        # mag = torch.exp(mag * 23 - 17)
        # mag = torch.clip(mag, max=1e2)

        # sample normalization
        mag = torch.clip(torch.exp(inv_min_max_norm(mag, *scales)), max=1e2)

        phase = instantaneous_frequency_to_phase(phase)
        x = mag * (torch.cos(phase) + 1j * torch.sin(phase))
        return x, loss

from vector_quantize_pytorch import ResidualVQ, ResidualFSQ, FSQ
class VQTransformer4(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        
        self.tok_embeddings = STFT(params)
        self.pos_embeddings = nn.Embedding(params.max_seq_len, params.dim)  # could do x pos + y pos but should be able to learn that...
        self.dropout = nn.Dropout(params.dropout)
        self.encoder_layers = torch.nn.ModuleList()
        for layer_id in range(8):
            self.encoder_layers.append(TransformerBlock(layer_id, params))
        self.vq = ResidualVQ(
            dim = params.dim,
            num_quantizers = 12,
            codebook_size = 8192,
            codebook_dim = 64,
            stochastic_sample_codes = True,
            sample_codebook_temp = 0.1,
            shared_codebook = True,
            rotation_trick = True,
            threshold_ema_dead_code = 2,
            kmeans_init = True,
            kmeans_iters = 200,
            decay = 0.95,
            commitment_weight = 0.25,

            # accept_image_fmap = True,                   # set this true to be able to pass in an image feature map
            # orthogonal_reg_weight = 10,                 # in paper, they recommended a value of 10
            # orthogonal_reg_max_codes = 128,             # this would randomly sample from the codebook for the orthogonal regularization loss, for limiting memory usage
            # orthogonal_reg_active_codes_only = True
        )
        # self.vq = ResidualVectorQuantize(
        #     input_dim=params.dim,
        #     n_codebooks=8,
        #     codebook_size=8192,
        #     codebook_dim=32,
        #     quantizer_dropout=0
        # )
        # self.vq = ResidualFSQ(
        #     dim = params.dim,
        #     num_quantizers = 8,
        #     levels = [8, 8, 8, 6, 5],
        # )
        # self.vq_in = nn.Sequential(RMSNorm(params.dim, eps=params.norm_eps), nn.Linear(params.dim, 6, bias=False))
        # self.vq = ResidualFSQBottleneck([
        #     ([3, 3, 3, 3, 3, 3], 1.0),
        #     ([3, 3, 3, 3, 3, 3], 0.5),
        #     ([3, 3, 3, 3, 3, 3], 0.25),
        #     ([3, 3, 3, 3, 3, 3], 0.125),
        # ])
        # self.vq_out = nn.Sequential(RMSNorm(6, eps=params.norm_eps), nn.Linear(6, params.dim, bias=False))
        self.proj = nn.Linear(params.dim, 512, bias=False)
        params.dim = 512
        params.n_heads = 8
        self.decoder_layers = torch.nn.ModuleList()
        for layer_id in range(8):
            self.decoder_layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = STFTHead(params)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * (3 + 6)))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        H, W = 256, 100

        targets, scales, h = self.tok_embeddings(tokens)
        B, T, C = h.shape
        h = h + self.pos_embeddings(torch.arange(T, device=h.device))
        h = self.dropout(h)

        for layer in self.encoder_layers:
            h = layer(h)
        
        if isinstance(self.vq, ResidualFSQ) or isinstance(self.vq, FSQ):
            h, indices = self.vq(h)
        elif isinstance(self.vq, ResidualFSQBottleneck):
            h = self.vq_in(h).transpose(1, 2)
            h = self.vq.encode(h)
            h = self.vq.decode(h)
            h = self.vq_out(h.transpose(1, 2))
        elif isinstance(self.vq, ResidualVectorQuantize):
            h, codes, latents, commitment_loss, codebook_loss = self.vq(h.transpose(1, 2))
            h = h.transpose(1, 2)
        else:
            h, indices, commit_loss = self.vq(h)

        h = self.proj(h)
        for layer in self.decoder_layers:
            h = layer(h)

        h = self.norm(h)

        logits, loss = self.output(h, H, W, scales, targets)
        # self.last_loss = 45 * loss + codebook_loss + 0.25 * commitment_loss
        self.last_loss = loss

        return logits

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        H, W = 256, 100

        targets, scales, h = self.tok_embeddings(tokens)
        B, T, C = h.shape
        h = h + self.pos_embeddings(torch.arange(T, device=h.device))
        h = self.dropout(h)

        for layer in self.encoder_layers:
            h = layer(h)
        
        h, indices, commit_loss = self.vq(h)

        return indices, scales
    
    def get_codes_from_indices(self, indices):
        from einops import rearrange, repeat, reduce, pack, unpack

        from einx import get_at

        def exists(val):
            return val is not None

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices, ps = pack([indices], 'b * q')

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.vq.num_quantizers:
            # assert self.quantize_dropout > 0., 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.vq.num_quantizers - quantize_dim), value = -1)

        # take care of quantizer dropout

        mask = indices == -1.
        indices = indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        if not self.vq.implicit_neural_codebook and self.vq.uniform_codebook_size:

            all_codes = get_at('q [c] d, b n q -> q b n d', self.vq.codebooks, indices)

        else:
            # else if using implicit neural codebook, or non uniform codebook sizes, codes will need to be derived layer by layer

            code_transform_mlps = (None, *self.vq.mlps)

            all_codes = []
            quantized_out = 0.

            for codes, indices, maybe_transform_mlp in zip(self.vq.codebooks, indices.unbind(dim = -1), code_transform_mlps):

                if exists(maybe_transform_mlp):
                    codes = maybe_transform_mlp(codes, condition = quantized_out)
                    layer_codes = get_at('b n [c] d, b n -> b n d', codes, indices)
                else:
                    layer_codes = get_at('[c] d, b n -> b n d', codes, indices)

                all_codes.append(layer_codes)
                quantized_out += layer_codes

            all_codes = torch.stack(all_codes)

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.)

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        all_codes, = unpack(all_codes, ps, 'q b * d')

        return all_codes
    
    def get_output_from_indices(self, indices):
        from einops import rearrange, repeat, reduce, pack, unpack

        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.vq.project_out(codes_summed)

    def decode(self, codes: torch.Tensor, scales) -> torch.Tensor:
        H, W = 256, 100
        
        # h = self.vq.get_output_from_indices(codes)
        h = self.get_output_from_indices(codes)

        h = self.proj(h)
        for layer in self.decoder_layers:
            h = layer(h)

        h = self.norm(h)

        logits, loss = self.output(h, H, W, scales)

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

class Transformer4(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        
        self.tok_embeddings = STFT(params)
        self.pos_embeddings = nn.Embedding(params.max_seq_len, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = STFTHead(params)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        H, W = 256, 200

        targets, h = self.tok_embeddings(tokens)
        B, T, C = h.shape
        h = h + self.pos_embeddings(torch.arange(T, device=h.device))
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)

        logits, loss = self.output(h, H, W, targets)
        self.last_loss = loss

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.pos_embeddings = nn.Embedding(params.max_seq_len, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = UnPatchEmbed(9, 20, params.dim)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        H, W = 200, 126

        tokens, h = self.tok_embeddings(tokens)
        B, T, C = h.shape
        h = h + self.pos_embeddings(torch.arange(T, device=tokens.device))
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)

        logits = self.output(h, H, W)
        self.last_loss = F.mse_loss(logits, tokens)

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
