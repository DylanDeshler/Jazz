import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List, Callable

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
            q_mask = None,
            kv_mask = None,
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
                attn_mask=kv_mask.unsqueeze(1).unsqueeze(1) if kv_mask is not None else None,
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
        if q_mask is not None:
            x = x * ~q_mask.unsqueeze(-1).repeat(1, 1, C)
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

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, proj_bias=False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.norm3 = RMSNorm(dim)
        self.mlp = SwiGLUMlp(dim, int(2 / 3 * mlp_ratio * dim), bias=proj_bias)
    
    def forward(self, x, context, q_mask=None, kv_mask=None):
        x = x + self.attn(self.norm1(x), self.norm2(context), q_mask=q_mask, kv_mask=kv_mask)
        x = x + self.mlp(self.norm3(x))
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, proj_bias=False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.sa = Attention(dim, num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        self.ca = CrossAttention(dim, num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.norm4 = RMSNorm(dim)
        self.mlp = SwiGLUMlp(dim, int(2 / 3 * mlp_ratio * dim), bias=proj_bias)
    
    def forward(self, x, context, q_mask=None, kv_mask=None):
        x = x + self.sa(self.norm1(x))
        x = x + self.ca(self.norm2(x), self.norm3(context), q_mask=q_mask, kv_mask=kv_mask)
        x = x + self.mlp(self.norm4(x))
        return x

class SequenceEncoder(nn.Module):
    def __init__(self, in_dim, n_queries, max_seq_len, hidden_dim, num_heads, depth, mlp_ratio=4., qkv_bias=False, proj_bias=False):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, n_queries, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([Block(hidden_dim, num_heads, mlp_ratio, qkv_bias, proj_bias) for _ in range(depth)])
        
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, in_dim, bias=True)

        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        nn.init.zeros_(self.out_proj.weight)
        # zero out c_proj weights in all blocks
        for block in self.blocks:
            nn.init.zeros_(block.mlp.w3.weight)
            nn.init.zeros_(block.sa.proj.weight)
            nn.init.zeros_(block.ca.proj.weight)
    
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
    
    def forward(self, x, mask=None):
        B, _, T = x.shape
        
        x = x.transpose(1, 2)
        x = self.in_norm(x)
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        x = x.transpose(1, 2)
        
        queries = self.queries.repeat(B, 1, 1)
        x = x + self.pos_embed.repeat(B, 1, 1)[:, :T]
        for block in self.blocks:
            queries = block(queries, x, kv_mask=mask)
        
        queries = self.out_norm(queries)
        queries = self.out_proj(queries)
        queries = queries.transpose(1, 2)
        
        return queries
    
class SequenceDecoder(nn.Module):
    def __init__(self, in_dim, max_seq_len, hidden_dim, num_heads, depth, mlp_ratio=4., qkv_bias=False, proj_bias=False):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=7, padding=3)
        
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([Block(hidden_dim, num_heads, mlp_ratio, qkv_bias, proj_bias) for _ in range(depth)])
        
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, in_dim, bias=True)
                
        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        nn.init.zeros_(self.out_proj.weight)
        # zero out c_proj weights in all blocks
        for block in self.blocks:
            nn.init.zeros_(block.mlp.w3.weight)
            nn.init.zeros_(block.sa.proj.weight)
            nn.init.zeros_(block.ca.proj.weight)
    
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
    
    def forward(self, x, shape, mask=None):
        B, _, T = shape
        
        x = x.transpose(1, 2)
        x = self.in_norm(x)
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        x = x.transpose(1, 2)
        
        queries = self.pos_embed.repeat(B, 1, 1)[:, :T]
        x = x + self.pos_embed.repeat(B, 1, 1)[:, :x.shape[1]]
        for block in self.blocks:
            queries = block(queries, x, q_mask=mask)
        
        queries = self.out_norm(queries)
        queries = self.out_proj(queries)
        queries = queries.transpose(1, 2)
        
        return queries

class InvertibleAdapter(nn.Module):
    def __init__(self, in_dim, n_queries, max_seq_len, hidden_dim, num_heads, enocder_depth, decoder_depth, mlp_ratio=4., qkv_bias=False, proj_bias=False):
        super().__init__()
        self.encoder = SequenceEncoder(
            in_dim, n_queries, max_seq_len, hidden_dim, num_heads, enocder_depth, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias
        )
        self.decoder = SequenceDecoder(
            in_dim, max_seq_len, hidden_dim, num_heads, decoder_depth, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias
        )
    
    def encode(self, x, mask=None):
        return self.encoder(x, mask=mask), x.shape
    
    def decode(self, x, shape, mask=None):
        return self.decoder(x, shape, mask=mask)
    
    def forward(self, x, mask=None):
        x, shape = self.encode(x, mask=mask)
        return self.decode(x, shape, mask=mask)

if __name__ == '__main__':
    in_dim = 16
    n_queries = 48
    max_seq_len = 512
    hidden_dim = 512
    num_heads = 8
    enocder_depth = 1
    decoder_depth = 2
    device = 'cuda'
    
    model = InvertibleAdapter(in_dim, n_queries, max_seq_len, hidden_dim, num_heads, enocder_depth, decoder_depth).to(device)
    
    x = torch.randn(32, 16, 48).to(device)
    y = model(x)
    print(x.shape, y.shape)
    
    x = torch.randn(32, 16, 97).to(device)
    mask = torch.zeros((32, 97))
    mask[:59] = 1
    mask = mask.bool().to('cuda')
    y = model(x, mask)
    print(x.shape, y.shape)
    print(y[:, :, 90].mean())