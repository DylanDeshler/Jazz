"""
CLAP-style audio<->text contrastive model for the jazz dataset.

Design notes
------------
Both modalities are aligned in a shared embedding space via a symmetric InfoNCE
(CLIP) loss. Unlike vanilla CLAP we do NOT reuse any pretrained CLAP weights
(our data - 1800s-1900s jazz - is heavily OOD for AudioSet-trained encoders).
Instead both "towers" consume *already-extracted, frozen* features:

  * audio: per-measure 128-d vectors from the in-domain `contrast.py` encoder
           (`forward_features`), i.e. `..._style_*.bin` memmaps. Shape [B, M, 128].
  * text:  T5-v1.1-xxl encoder `last_hidden_state` token sequences, i.e. the
           `caption_embeddings_*` memmaps. Shape [B, 256, 1024].

Each tower is a small trainable transformer (1D RoPE) + learned attention pool +
projection into the shared space. This is "locked-tuning" (LiT) taken to its
cheap limit: frozen precomputed towers, trainable heads only.

To upgrade to a fine-tunable raw-audio tower later, swap `audio_tower` for a
wrapper around `contrast.Transformer` (feeding raw wav) - the loss / training
loop are unchanged.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse the primitives from the contrastive model so both codebases stay in sync
from contrast import RMSNorm, SwiGLUMlp, CrossAttention, precompute_freqs_cis, apply_rotary_emb


class SelfAttention1D(nn.Module):
    """Self-attention with 1D RoPE and optional key-padding mask."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if freqs_cis is not None:
            q = apply_rotary_emb(q.transpose(1, 2), freqs_cis).transpose(1, 2)
            k = apply_rotary_emb(k.transpose(1, 2), freqs_cis).transpose(1, 2)
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = SelfAttention1D(hidden_size, num_heads=num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=False)

    def forward(self, x, freqs_cis=None, attn_mask=None):
        x = x + self.attn(self.norm1(x), freqs_cis=freqs_cis, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionPool(nn.Module):
    """Pool a variable-length sequence into a single vector via a learned query."""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.query, std=0.02)
        self.pool = CrossAttention(hidden_size, num_heads, qkv_bias=False, proj_bias=False)

    def forward(self, x, key_padding=None):
        # x: [B, N, C]; key_padding: [B, N] bool, True == valid token
        x = self.norm(x)
        q = self.query.expand(x.shape[0], -1, -1)
        attn_mask = key_padding[:, None, None, :] if key_padding is not None else None  # True == attend
        return self.pool(q, x, attn_mask=attn_mask).squeeze(1)


class Tower(nn.Module):
    """in_proj -> transformer -> attention pool -> projection -> L2 normalize."""

    def __init__(self, in_dim, hidden_size, proj_dim, depth, num_heads, mlp_ratio=4.0, use_rope=True):
        super().__init__()
        self.use_rope = use_rope
        self.head_dim = hidden_size // num_heads
        self.in_proj = nn.Linear(in_dim, hidden_size, bias=False)
        self.blocks = nn.ModuleList([Block(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.pool = AttentionPool(hidden_size, num_heads)
        self.out_norm = RMSNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, proj_dim, bias=False)

    def forward(self, x, key_padding=None):
        B, N, C = x.shape
        x = self.in_proj(x)
        freqs_cis = precompute_freqs_cis(self.head_dim, N).to(x.device) if self.use_rope else None
        attn_mask = key_padding[:, None, None, :] if key_padding is not None else None
        for blk in self.blocks:
            x = blk(x, freqs_cis=freqs_cis, attn_mask=attn_mask)
        x = self.pool(x, key_padding=key_padding)
        x = self.out_norm(x)
        x = self.out_proj(x)
        return F.normalize(x, dim=-1)


class CLAP(nn.Module):
    def __init__(
        self,
        audio_dim=128,
        text_dim=1024,
        hidden_size=512,
        proj_dim=512,
        audio_depth=4,
        text_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        init_temperature=0.07,
    ):
        super().__init__()
        self.audio_tower = Tower(audio_dim, hidden_size, proj_dim, audio_depth, num_heads, mlp_ratio)
        self.text_tower = Tower(text_dim, hidden_size, proj_dim, text_depth, num_heads, mlp_ratio)
        # learnable log temperature (CLIP-style), init at 1/0.07
        self.log_temperature = nn.Parameter(torch.log(torch.ones(1) / init_temperature))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813 (matches contrast.py)
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_audio(self, audio, audio_mask=None):
        return self.audio_tower(audio, key_padding=audio_mask)

    def encode_text(self, text, text_mask=None):
        return self.text_tower(text, key_padding=text_mask)

    def forward(self, text, audio, text_mask=None, audio_mask=None, song_ids=None):
        a = self.encode_audio(audio, audio_mask)   # [B, D]
        t = self.encode_text(text, text_mask)       # [B, D]

        logit_scale = torch.exp(self.log_temperature).clamp(max=100)
        logits = logit_scale * (a @ t.T)            # rows: audio, cols: text
        B = logits.shape[0]
        targets = torch.arange(B, device=logits.device)

        # Different captions of the *same song* can co-occur in a batch (the text
        # rows are globally shuffled). Those are false negatives - mask them out
        # of the off-diagonal so only the true (audio_i, text_i) pair is positive.
        if song_ids is not None:
            same = song_ids[:, None] == song_ids[None, :]
            eye = torch.eye(B, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(same & ~eye, float('-inf'))

        loss_a2t = F.cross_entropy(logits, targets)
        loss_t2a = F.cross_entropy(logits.T, targets)
        loss = 0.5 * (loss_a2t + loss_t2a)

        with torch.no_grad():
            acc_a2t = (logits.argmax(dim=1) == targets).float().mean()
            acc_t2a = (logits.argmax(dim=0) == targets).float().mean()

        return {
            'loss': loss,
            'logits': logits,
            'sim': a @ t.T,
            'acc': 0.5 * (acc_a2t + acc_t2a),
            'acc_a2t': acc_a2t,
            'acc_t2a': acc_t2a,
            'audio_features': a,
            'text_features': t,
        }


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLAP().to(device)
    from torchinfo import summary
    summary(model)

    B, M, T = 8, 64, 256
    text = torch.randn(B, T, 1024, device=device)
    audio = torch.randn(B, M, 128, device=device)
    text_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    audio_mask = torch.ones(B, M, dtype=torch.bool, device=device)
    song_ids = torch.arange(B, device=device)

    out = model(text, audio, text_mask=text_mask, audio_mask=audio_mask, song_ids=song_ids)
    print({k: (v.item() if v.ndim == 0 else tuple(v.shape)) for k, v in out.items()})
