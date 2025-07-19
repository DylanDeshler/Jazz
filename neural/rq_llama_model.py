import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, reduce, repeat

def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

def exists(val):
    return val is not None

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

@dataclass
class ModelArgs:
    dim: int = 1024
    n_spatial_layers: int = 12
    n_depth_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    vocab_size: int = 8 * 1024
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    depth_seq_len: int = 8
    dropout: float = 0.0


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
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

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
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
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

class RQTransformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        
        self.spatial_start_token = nn.Parameter(torch.randn(params.dim))
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.spatial_pos_emb = nn.Embedding(params.max_seq_len, params.dim)
        self.depth_pos_emb = nn.Embedding(params.depth_seq_len, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        
        self.spatial_layers = torch.nn.ModuleList()
        for layer_id in range(params.n_spatial_layers):
            self.spatial_layers.append(TransformerBlock(layer_id, params))
        self.spatial_norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.depth_layers = torch.nn.ModuleList()
        for layer_id in range(params.n_depth_layers):
            self.depth_layers.append(TransformerBlock(layer_id, params))
        self.depth_norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        # self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * (params.n_depth_layers + params.n_spatial_layers)))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None
        self.pad_id = -1

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_empty(self, batch_size):
        # take care of special case
        # where you sample from input of 0 (start token only)

        spatial_tokens = repeat(self.spatial_start_token, 'd -> b 1 d', b = batch_size)
        for layer in self.spatial_layers:
            spatial_tokens = layer(spatial_tokens)
        depth_tokens = self.spatial_norm(spatial_tokens)

        for layer in self.depth_layers:
            depth_tokens = layer(depth_tokens)
        depth_tokens = self.depth_norm(depth_tokens)

        return self.output(depth_tokens)

    def forward(self, ids, return_loss=False):
        assert ids.ndim in {2, 3}
        flattened_dim = ids.ndim == 2
        ids_orig_ndim = ids.ndim

        if ids.numel() == 0:
            return self.forward_empty(ids.shape[0])

        if flattened_dim:
            # allow for ids to be given in the shape of (batch, seq)
            # in which case it will be auto-padded to the next nearest multiple of depth seq len
            seq_len = ids.shape[-1]
            padding = remainder_to_mult(seq_len, self.params.depth_seq_len)
            ids = F.pad(ids, (0, padding), value = self.pad_id)
            ids = rearrange(ids, 'b (s d) -> b s d', d = self.params.depth_seq_len)
        else:
            seq_len = ids.shape[1] * ids.shape[2]

        b, space, depth, device = *ids.shape, ids.device
        assert space <= (self.params.max_seq_len + 1), 'spatial dimension is greater than the max_spatial_seq_len set'
        assert depth == self.params.depth_seq_len, 'depth dimension must be equal to depth_seq_len'

        # get token embeddings
        
        tokens = self.tok_embeddings(ids)

        spatial_pos = self.spatial_pos_emb(torch.arange(space, device = device))
        depth_pos = self.depth_pos_emb(torch.arange(depth, device = device))

        tokens_with_depth_pos = tokens + depth_pos

        # spatial tokens is tokens with depth pos reduced along depth dimension + spatial positions

        spatial_tokens = reduce(tokens_with_depth_pos, 'b s d f -> b s f', 'sum') + spatial_pos

        spatial_tokens = torch.cat((
            repeat(self.spatial_start_token, 'f -> b 1 f', b = b),
            spatial_tokens
        ), dim = -2)        

        for layer in self.spatial_layers:
            spatial_tokens = layer(spatial_tokens)
        spatial_tokens = self.spatial_norm(spatial_tokens)

        spatial_tokens = rearrange(spatial_tokens, 'b s f -> b s 1 f')

        # spatial tokens become the start tokens of the depth dimension

        tokens_with_depth_pos = F.pad(tokens_with_depth_pos, (0, 0, 0, 0, 0, 1), value = 0.)

        depth_tokens = torch.cat((spatial_tokens, tokens_with_depth_pos), dim = -2)

        depth_tokens = rearrange(depth_tokens, '... n d -> (...) n d')

        for layer in self.depth_layers:
            depth_tokens = layer(depth_tokens)
        depth_tokens = self.depth_norm(depth_tokens)

        depth_tokens = rearrange(depth_tokens, '(b s) d f -> b s d f', b = b)

        logits = self.output(depth_tokens)
        logits = rearrange(logits, 'b ... f -> b (...) f')
        logits = logits[:, :(seq_len + 1)]

        if not return_loss:
            logits = logits[:, 1:]

            if flattened_dim:
                return rearrange(logits, 'b ... n -> b (...) n')

            return logits

        logits = logits[:, :-1]
        
        preds = rearrange(logits, 'b ... c -> b c (...)')
        labels = rearrange(ids, 'b s d -> b (s d)')

        loss = F.cross_entropy(preds, labels, ignore_index = self.pad_id)
        
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
        L, H, Q, T = cfg.n_spatial_layers + cfg.n_depth_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, prime = None, filter_thres = 0.9, temperature = 1., default_batch_size = 1, max_seq_len = None):
        if max_seq_len is not None:
            total_seq_len = max_seq_len * self.params.depth_seq_len
        else:
            total_seq_len = self.params.depth_seq_len * self.params.max_seq_len
        device = next(self.parameters()).device

        if not exists(prime):
            prime = torch.empty((default_batch_size, 0), dtype = torch.long, device = device)

        seq = prime

        for _ in range(total_seq_len - seq.shape[-1]):
            logits = self.forward(seq)[:, -1]
            logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(logits, dim = -1, temperature = temperature)
            seq = torch.cat((seq, rearrange(sampled, 'b -> b 1')), dim = -1)

        return rearrange(seq, 'b (s d) -> b s d', d = self.params.depth_seq_len)
    
    # def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    #     """
    #     Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    #     the sequence max_new_tokens times, feeding the predictions back into the model each time.
    #     Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    #     Also note this is a super inefficient version of sampling with no key/value cache.
    #     """
    #     for _ in range(max_new_tokens):
    #         # if the sequence context is growing too long we must crop it at block_size
    #         idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
    #         # forward the model to get the logits for the index in the sequence
    #         logits = self(idx_cond)
    #         logits = logits[:, -1, :] # crop to just the final time step
    #         if temperature == 0.0:
    #             # "sample" the single most likely index
    #             _, idx_next = torch.topk(logits, k=1, dim=-1)
    #         else:
    #             # pluck the logits at the final step and scale by desired temperature
    #             logits = logits / temperature
    #             # optionally crop the logits to only the top k options
    #             if top_k is not None:
    #                 v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    #                 logits[logits < v[:, [-1]]] = -float('Inf')
    #             # apply softmax to convert logits to (normalized) probabilities
    #             probs = F.softmax(logits, dim=-1)
    #             idx_next = torch.multinomial(probs, num_samples=1)
    #         # append sampled index to the running sequence and continue
    #         idx = torch.cat((idx, idx_next), dim=1)

    #     return idx
