from __future__ import annotations

import random
from math import ceil
from functools import partial

import torch
from torch import nn, tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.amp import autocast
import torch.distributed as dist

from vector_quantize_pytorch.finite_scalar_quantization import FSQ

from einops import rearrange, repeat, reduce, pack, unpack

from einx import get_at

# helper functions

def exists(val):
    return val is not None

def first(l):
    return l[0]

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# distributed helpers

def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def get_maybe_sync_seed(device, max_size = 10_000):
    rand_int = torch.randint(0, max_size, (), device = device)

    if is_distributed():
        dist.all_reduce(rand_int)

    return rand_int.item()

# main class

class InvertibleLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        
        self.register_buffer('current_mean', None, persistent=False)
        self.register_buffer('current_std', None, persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        
        self.current_mean = x.mean(dim=2, keepdim=True)  # (B, C, 1)
        variance = x.var(dim=2, keepdim=True, unbiased=False)  # (B, C, 1)
        self.current_std = torch.sqrt(variance + self.eps)
        
        normalized = (x - self.current_mean) / self.current_std
        
        weight = self.weight.view(1, C, 1)  # (1, C, 1)
        bias = self.bias.view(1, C, 1)      # (1, C, 1)
        
        return weight * normalized + bias
    
    def inverse(self, normalized_x: torch.Tensor) -> torch.Tensor:
        if self.current_mean is None or self.current_std is None:
            raise RuntimeError("mean or std are None!")
        
        B, C, L = normalized_x.shape
        
        weight = self.weight.view(1, C, 1)
        bias = self.bias.view(1, C, 1)
        
        denormalized = (normalized_x - bias) / weight
        return denormalized * self.current_std + self.current_mean

# class ResidualFSQ(nn.Module):
#     def __init__(self, levels, num_quantizers):
#         super().__init__()
        
#         self.vqs = [FSQ(levels=levels) for _ in range(num_quantizers)]

class ResidualFSQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        levels: list[int],
        num_quantizers,
        dim = None,
        is_channel_first = False,
        quantize_dropout = False,
        quantize_dropout_cutoff_index = 0,
        quantize_dropout_multiple_of = 1,
        soft_clamp_input_value: float | list[float] | torch.Tensor | None = None,
        bound_hard_clamp = True,
        **kwargs
    ):
        super().__init__()
        codebook_dim = len(levels)
        dim = default(dim, codebook_dim)

        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.is_channel_first = is_channel_first
        self.num_quantizers = num_quantizers

        # layers

        self.levels = levels
        self.layers = nn.ModuleList([])

        levels_tensor = tensor(levels)
        assert (levels_tensor > 1).all()

        self.norms = nn.ModuleList([])

        for ind in range(num_quantizers):
            self.norms.append(InvertibleLayerNorm(codebook_dim))

            fsq = FSQ(
                levels = levels,
                dim = codebook_dim,
                preserve_symmetry = True,
                bound_hard_clamp = bound_hard_clamp,
                **kwargs
            )

            self.layers.append(fsq)

        assert all([not fsq.has_projections for fsq in self.layers])

        self.codebook_size = self.layers[0].codebook_size

        self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of  # encodec paper proposes structured dropout, believe this was set to 4

        # soft clamping the input value

        if bound_hard_clamp:
            assert not exists(soft_clamp_input_value)
            soft_clamp_input_value = 1 + (1 / (levels_tensor - 1))

        if isinstance(soft_clamp_input_value, (list, float)):
            soft_clamp_input_value = tensor(soft_clamp_input_value)

        self.register_buffer('soft_clamp_input_value', soft_clamp_input_value, persistent = False)

    @property
    def codebooks(self):
        codebooks = [layer.implicit_codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks

    def get_codes_from_indices(self, indices):

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices, ps = pack([indices], 'b * q')

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0., 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # take care of quantizer dropout

        mask = indices == -1
        indices = indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        all_codes = get_at('q [c] d, b n q -> q b n d', self.codebooks, indices)

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.)

        # scale the codes

        scales = rearrange(self.scales, 'q d -> q 1 1 d')
        all_codes = all_codes * scales

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        all_codes, = unpack(all_codes, ps, 'q b * d')

        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def forward(
        self,
        x,
        return_all_codes = False,
        rand_quantize_dropout_fixed_seed = None
    ):
        num_quant, quant_dropout_multiple_of, device = self.num_quantizers, self.quantize_dropout_multiple_of, x.device

        # handle channel first

        if self.is_channel_first:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack([x], 'b * d')

        # maybe project in

        x = self.project_in(x)

        # maybe softclamp input before residual layers

        if exists(self.soft_clamp_input_value):
            clamp_value = self.soft_clamp_input_value
            x = (x / clamp_value).tanh() * clamp_value

        # ready some variables to be accumulated

        quantized_out = 0.
        residual = x

        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout and torch.is_grad_enabled()

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices

        if should_quantize_dropout:

            # check if seed is manually passed in

            if not exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)

            rand = random.Random(rand_quantize_dropout_fixed_seed)

            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1

            null_indices = torch.full(x.shape[:2], -1., device = device, dtype = torch.long)

        # go through the layers

        with autocast('cuda', enabled = False):
            for quantizer_index, (layer, norm) in enumerate(zip(self.layers, self.norms)):

                if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                    all_indices.append(null_indices)
                    continue
                
                normalized_residual = norm(residual)
                quantized_normalized, indices = layer(normalized_residual)
                quantized_true = norm.inverse(quantized_normalized)

                residual = residual - quantized_true.detach()
                quantized_out = quantized_out + quantized_true

                all_indices.append(indices)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # stack all indices

        all_indices = torch.stack(all_indices, dim = -1)

        # channel first out

        if self.is_channel_first:
            quantized_out, = unpack(quantized_out, ps, 'b * d')
            all_indices, = unpack(all_indices, ps, 'b * d')

            quantized_out = rearrange(quantized_out, 'b ... d -> b d ...')
            all_indices = rearrange(all_indices, 'b ... d -> b d ...')

        # return

        ret = (quantized_out, all_indices)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers

        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)
