"""
CLAP-style audio<->text contrastive model for the jazz dataset.

Design notes
------------
Both modalities are aligned in a shared embedding space via a symmetric InfoNCE
(CLIP) loss. The TEXT tower consumes *frozen precomputed* T5-v1.1-xxl token
features (small trainable head). The AUDIO tower is a FULL raw-audio encoder
trained end-to-end on raw waveforms.

Why raw audio (and not the frozen contrast per-measure style vectors)?
  The contrast.py vectors were trained for *instance discrimination* -- they are
  near-unique per-song fingerprints. Aligning those to per-song captions lets
  CLAP memorize a fingerprint->caption lookup that cannot generalize to val
  songs (severe overfitting). Training a raw-audio tower restores the crop +
  SpecAugment stochasticity that made the original contrastive model
  un-overfittable: the only crop-invariant signal that also predicts the caption
  is song-level *style*, which is exactly what we want the embedding to encode.

Everything messy (the raw-audio backbone, its non-detaching embedding path, the
DDP gradient-preserving all-gather, and a raw-wav batch loader) lives here so
contrast.py / train_contrast.py stay untouched. We only *import* primitives from
contrast.py; we never modify it.

Audio tower init (two schemes, see `CLAP(audio_init=...)`):
  * 'scratch'  : random init from `audio_cfg`.
  * 'contrast' : reuse the pretrained contrast.py backbone (mel/patch/blocks/
                 pool/mlp), re-initializing only the final projection to
                 `proj_dim`. Call `model.load_audio_backbone(ckpt)` after build.

Towers: (audio) mel -> patch -> 2D-RoPE transformer -> attn pool -> proj -> L2
        (text)  in_proj -> 1D-RoPE transformer -> attn pool -> proj -> L2
"""

import os
import glob as _glob
import math
import random
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# reuse the primitives from the contrastive model so both codebases stay in sync
# (importing does NOT modify contrast.py)
from contrast import (
    RMSNorm, SwiGLUMlp, CrossAttention, SelfAttention,
    SelfAttentionBlock, ToMel, SpecAugment,
    precompute_freqs_cis, apply_rotary_emb, precompute_freqs_cis_2d,
)
from torch.utils.checkpoint import checkpoint

try:
    import soundfile as sf
except Exception:  # pragma: no cover - only needed for the raw loader
    sf = None
from einops import rearrange


# =============================================================================
# DDP: gradient-preserving all-gather (SimCLR/MoCo-v3 style)
# =============================================================================
class GatherLayer(torch.autograd.Function):
    """all_gather that lets gradients flow back to the local shard.

    Plain dist.all_gather detaches the gathered copies; here backward sums the
    per-rank grads and returns this rank's slice, so contrastive logits computed
    over the *global* batch still train the *local* samples correctly.
    """

    @staticmethod
    def forward(ctx, x):
        out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x.contiguous())
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        g = torch.stack(grads, dim=0)
        dist.all_reduce(g)                      # sum grads across ranks
        return g[dist.get_rank()]


def _ddp_active():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def all_gather_grad(x):
    """Gather `x` across ranks with grad; returns [world*B, ...]. No-op if 1 rank."""
    if not _ddp_active():
        return x
    return torch.cat(GatherLayer.apply(x), dim=0)


def all_gather_nograd(x):
    """Gather an int/label tensor across ranks (no grad). No-op if 1 rank."""
    if not _ddp_active():
        return x
    out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=0)


# =============================================================================
# TEXT tower (unchanged): small trainable head over frozen T5 token features
# =============================================================================
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


class DropPath(nn.Module):
    """Stochastic depth: randomly zero the whole residual branch per-sample.

    In train mode each sample's branch output is dropped with prob `drop_prob`
    and the survivors are rescaled by 1/(1-drop_prob) so the expectation is
    preserved. A no-op at eval or when drop_prob==0 (no params, adds no ckpt keys).
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # per-sample mask, broadcast over all non-batch dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = SelfAttention1D(hidden_size, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=False)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, freqs_cis=None, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cis=freqs_cis, attn_mask=attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AudioBlock(nn.Module):
    """2D-RoPE transformer block for the audio tower, with stochastic depth.

    Reimplements contrast.SelfAttentionBlock here (same submodule names
    norm1/attn/norm2/mlp, same gradient-checkpointing) so a pretrained contrast
    backbone still loads, but adds a DropPath on each residual branch. DropPath
    has no parameters, so it introduces no new checkpoint keys.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0.0, use_checkpoint=False):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=False)
        self.drop_path = DropPath(drop_path)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x, freqs_cis=None, is_causal=False):
        fc = freqs_cis[:x.shape[1]] if freqs_cis is not None else None
        x = x + self.drop_path(self.attn(self.norm1(x), is_causal=is_causal, freqs_cis=fc))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x, freqs_cis=None, is_causal=False):
        # Activation checkpointing recomputes the whole block forward during
        # backward -- a memory/speed trade only worth paying near the memory
        # ceiling. At small per-GPU batch it just doubles the fwd cost, so it is
        # OFF by default and enabled explicitly (see AudioTower(use_checkpoint=)).
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(self._forward_impl, x, freqs_cis, is_causal, use_reentrant=False)
        return self._forward_impl(x, freqs_cis, is_causal)


class TextTower(nn.Module):
    """in_proj -> transformer -> pool -> mlp -> fc -> L2 normalize.

    The output head (pool_norm/pool/mlp_norm/mlp/fc_norm/fc) mirrors AudioTower's
    exactly so both modalities share the same pooling+projection structure; the
    only difference is that text carries a key-padding mask (T5 zero-padding),
    which is applied to both the mean query and the cross-attention keys.
    """

    def __init__(self, in_dim, hidden_size, proj_dim, depth, num_heads, mlp_ratio=4.0,
                 use_rope=True, max_seq_len=512, drop=0.0, drop_path=0.0, in_drop=0.0):
        super().__init__()
        self.use_rope = use_rope
        self.head_dim = hidden_size // num_heads
        # Dropout on the FROZEN T5 features, before in_proj. `drop` above only
        # reaches attention (attn_drop/proj_drop); the tower's input is otherwise
        # perfectly deterministic -- the same 18 vectors per song every epoch --
        # which is the surface the text side memorizes. This is the one knob that
        # makes that input stochastic. Param-free, so it adds no checkpoint keys.
        self.in_drop = nn.Dropout(in_drop)
        self.in_proj = nn.Linear(in_dim, hidden_size, bias=False)
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio, drop=drop, drop_path=drop_path)
            for _ in range(depth)
        ])
        # output head -- identical structure to AudioTower
        self.pool_norm = RMSNorm(hidden_size)
        self.pool = CrossAttention(hidden_size, num_heads, qkv_bias=False, proj_bias=False)
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=False)
        self.fc_norm = RMSNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, proj_dim, bias=False)
        if use_rope:
            self.register_buffer('freqs_cis', precompute_freqs_cis(self.head_dim, max_seq_len), persistent=False)

    def forward(self, x, key_padding=None):
        B, N, C = x.shape
        x = self.in_proj(self.in_drop(x))
        freqs_cis = self.freqs_cis[:N] if self.use_rope else None
        attn_mask = key_padding[:, None, None, :] if key_padding is not None else None
        for blk in self.blocks:
            x = blk(x, freqs_cis=freqs_cis, attn_mask=attn_mask)

        x = self.pool_norm(x)
        # masked mean query over valid tokens (matches AudioTower's x.mean(1))
        if key_padding is not None:
            m = key_padding[:, :, None].to(x.dtype)          # [B, N, 1], 1 == valid
            q = (x * m).sum(1, keepdims=True) / m.sum(1, keepdims=True).clamp(min=1.0)
        else:
            q = x.mean(1, keepdims=True)
        x = self.pool(q, x, attn_mask=attn_mask).squeeze(1)
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.fc_norm(x)
        x = self.fc(x)
        return F.normalize(x, dim=-1)


# =============================================================================
# AUDIO tower: full raw-waveform encoder (mirrors contrast.Transformer backbone
# param names so a pretrained contrast checkpoint loads cleanly)
# =============================================================================
class AudioTower(nn.Module):
    def __init__(self,
                 proj_dim,
                 in_channels=1,
                 hidden_size=768,
                 patch_size=16,
                 sample_rate=16000,
                 n_fft=1024,
                 hop_length=512,
                 n_mels=192,
                 time_length=32,
                 frequency_length=12,
                 num_heads=12,
                 depth=12,
                 mlp_ratio=4.0,
                 drop_path=0.0,
                 use_checkpoint=False):
        super().__init__()
        self.patch_size = patch_size
        self.head_dim = hidden_size // num_heads

        # NOTE: names match contrast.Transformer so load_state_dict can reuse a
        # pretrained backbone (everything except `fc`, which changes to proj_dim).
        # AudioBlock mirrors contrast.SelfAttentionBlock (same submodule names) but
        # adds stochastic depth; DropPath has no params so the ckpt still loads.
        self.to_mel = ToMel(sample_rate, n_fft, hop_length, n_mels)
        self.augment = SpecAugment(time_length, frequency_length)
        self.x_embedder = nn.Linear(in_channels * patch_size * patch_size, hidden_size, bias=False)
        self.blocks = nn.ModuleList([
            AudioBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path,
                       use_checkpoint=use_checkpoint)
            for _ in range(depth)
        ])
        self.pool_norm = RMSNorm(hidden_size)
        self.pool = CrossAttention(hidden_size, num_heads, qkv_bias=False, proj_bias=False)
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(2 / 3 * mlp_ratio * hidden_size), bias=False)
        self.fc_norm = RMSNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, proj_dim, bias=False)  # -> shared CLAP space

    @torch.compiler.disable
    def _compute_mel(self, x):
        return self.to_mel(x)

    @torch.compiler.disable
    def _compute_freqs(self, H, W, device):
        # precompute_freqs_cis_2d uses torch.polar (a complex op) which inductor
        # cannot lower -- tracing it into the compiled graph every step blows up
        # compile time. Fence it out (like _compute_mel). H/W are constant across
        # steps, so this is trivially cheap and never triggers a recompile.
        return precompute_freqs_cis_2d(
            dim=self.head_dim, height=H // self.patch_size, width=W // self.patch_size
        ).to(device)

    def forward(self, x):
        # x: raw waveform [B, 1, T]
        x = self._compute_mel(x)
        if self.training:
            x = self.augment(x)

        # per-sample instance normalization (matches contrast.py)
        mu = x.mean((-1, -2), keepdims=True)
        std = x.std((-1, -2), keepdims=True)
        x = (x - mu) / (std + 1e-6)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
        x = self.x_embedder(x)

        freqs_cis = self._compute_freqs(H, W, x.device)
        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis)

        x = self.pool_norm(x)
        x = self.pool(x.mean(1, keepdims=True), x).squeeze(1)
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.fc_norm(x)
        x = self.fc(x)
        # NON-detaching (unlike contrast.forward_features): grads must flow.
        return F.normalize(x, dim=-1)


# =============================================================================
# CLAP
# =============================================================================
class CLAP(nn.Module):
    def __init__(
        self,
        audio_cfg,                 # audio-specific front-end args (mel/patch); see below
        text_dim=1024,
        # shared transformer size -- BOTH towers use these (only depth differs)
        hidden_size=768,
        num_heads=12,
        mlp_ratio=4.0,
        proj_dim=512,
        audio_depth=12,            # audio tower depth (symmetric with text_depth)
        text_depth=4,
        text_drop=0.1,             # attn/proj dropout inside the text blocks
        text_in_drop=0.0,          # dropout on the frozen T5 features (see TextTower)
        drop_path=0.1,             # stochastic depth rate, applied to BOTH towers
        audio_checkpoint=False,    # activation-checkpoint the audio blocks (memory<->speed)
        init_temperature=0.07,
        audio_init='scratch',      # 'scratch' | 'contrast' (informational; weights loaded separately)
        n_text_tokens=256,
    ):
        super().__init__()
        self.audio_init = audio_init
        # audio_cfg holds ONLY audio front-end params (mel/patch). Depth and the
        # transformer body size (hidden_size/num_heads/mlp_ratio) are passed
        # explicitly: body size is SHARED with the text tower so both live in the
        # same representational space, and depth is a first-class arg (like
        # text_depth) so it can never go missing regardless of audio_init.
        self.audio_cfg = dict(audio_cfg)
        self.audio_cfg.pop('depth', None)   # tolerate/ignore a stray depth in cfg
        self.audio_tower = AudioTower(
            proj_dim=proj_dim, hidden_size=hidden_size, num_heads=num_heads,
            mlp_ratio=mlp_ratio, depth=audio_depth, drop_path=drop_path,
            use_checkpoint=audio_checkpoint, **self.audio_cfg,
        )
        self.text_tower = TextTower(
            text_dim, hidden_size, proj_dim, text_depth, num_heads, mlp_ratio,
            max_seq_len=n_text_tokens, drop=text_drop, drop_path=drop_path,
            in_drop=text_in_drop,
        )
        self.log_temperature = nn.Parameter(torch.log(torch.ones(1) / init_temperature))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ---- init scheme: reuse pretrained contrast backbone -----------------
    @torch.no_grad()
    def load_audio_backbone(self, contrast_ckpt_path, map_location='cpu', verbose=True):
        """Load the pretrained contrast.py backbone into the audio tower,
        skipping any weight whose shape doesn't match (notably `fc`, which goes
        from proj_size to proj_dim). Returns (loaded, skipped) key lists."""
        ckpt = torch.load(contrast_ckpt_path, map_location=map_location)
        sd = ckpt['model']
        unwanted = '_orig_mod.'
        sd = {(k[len(unwanted):] if k.startswith(unwanted) else k): v for k, v in sd.items()}

        tgt = self.audio_tower.state_dict()
        loaded, skipped = [], []
        for k, v in tgt.items():
            if k in sd and sd[k].shape == v.shape:
                tgt[k] = sd[k]
                loaded.append(k)
            else:
                skipped.append(k)
        self.audio_tower.load_state_dict(tgt)
        if verbose:
            print(f"[load_audio_backbone] loaded {len(loaded)} tensors, "
                  f"re-init {len(skipped)}: {skipped}")
        return loaded, skipped

    # ---- encoders --------------------------------------------------------
    def encode_audio(self, audio, audio_mask=None):
        # audio: raw waveform [B, 1, T]; audio_mask ignored (one crop -> one vec).
        return self.audio_tower(audio)

    def encode_text(self, text, text_mask=None):
        return self.text_tower(text, key_padding=text_mask)

    # ---- loss ------------------------------------------------------------
    def forward(self, text, audio, text_mask=None, audio_mask=None, song_ids=None):
        a = self.encode_audio(audio)                 # [b, D] (grad)
        t = self.encode_text(text, text_mask)        # [b, D] (grad)

        scale = torch.exp(self.log_temperature).clamp(max=100)

        # gather keys across all ranks so every local query sees world*b negatives
        a_all = all_gather_grad(a)                    # [B, D]
        t_all = all_gather_grad(t)                    # [B, D]
        b = a.shape[0]
        offset = dist.get_rank() * b if _ddp_active() else 0
        B = a_all.shape[0]

        logits_a2t = scale * (a @ t_all.T)           # [b, B]
        logits_t2a = scale * (t @ a_all.T)           # [b, B]
        targets = torch.arange(b, device=a.device) + offset

        if song_ids is not None:
            ids_all = all_gather_nograd(song_ids)    # [B]
            same = song_ids[:, None] == ids_all[None, :]         # [b, B]
            pos = F.one_hot(targets, num_classes=B).bool()       # the true pair
            fn_mask = same & ~pos                                 # same-song false negs
            logits_a2t = logits_a2t.masked_fill(fn_mask, float('-inf'))
            logits_t2a = logits_t2a.masked_fill(fn_mask, float('-inf'))

        loss_a2t = F.cross_entropy(logits_a2t, targets)
        loss_t2a = F.cross_entropy(logits_t2a, targets)
        loss = 0.5 * (loss_a2t + loss_t2a)

        with torch.no_grad():
            acc_a2t = (logits_a2t.argmax(dim=1) == targets).float().mean()
            acc_t2a = (logits_t2a.argmax(dim=1) == targets).float().mean()

        return {
            'loss': loss,
            'logits': logits_a2t,
            'sim': a @ t.T,
            'acc': 0.5 * (acc_a2t + acc_t2a),
            'acc_a2t': acc_a2t,
            'acc_t2a': acc_t2a,
            'audio_features': a,
            'text_features': t,
        }


# =============================================================================
# raw-wav batch loader (adapted from train_contrast.py; lives here to keep the
# training script's data plumbing minimal)
# =============================================================================
_wav_index = {}     # basename(no-ext) -> path
_frames_cache = {}  # path -> frame count


def build_wav_index(wav_glob='/data/wavs/*'):
    """Index available wavs by basename-without-extension for path resolution."""
    _wav_index.clear()
    for p in _glob.glob(wav_glob):
        key = os.path.basename(p).split('.')[0]
        _wav_index[key] = p
    print(f"[clap] indexed {len(_wav_index)} wavs from {wav_glob}")
    return _wav_index


def _resolve(file_path):
    """Map a caption/audio-map `file_path` to an actual wav path."""
    if os.path.isabs(file_path) and os.path.exists(file_path):
        return file_path
    key = os.path.basename(file_path).split('.')[0]
    if key in _wav_index:
        return _wav_index[key]
    raise KeyError(f"could not resolve wav for '{file_path}' "
                   f"(call build_wav_index() first, or check the glob)")


def _frames(path):
    if path not in _frames_cache:
        _frames_cache[path] = sf.info(path).frames
    return _frames_cache[path]


# libsndfile releases the GIL inside sf.read, so a thread pool genuinely
# overlaps the (seek + decode) of a whole batch instead of doing it serially.
_read_pool = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 8)))


def _read_one_crop(fp, n_samples):
    path = _resolve(fp)
    nf = _frames(path)
    start = random.randint(0, max(0, nf - n_samples))
    wav, _ = sf.read(path, start=start, frames=min(n_samples, nf), dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if len(wav) < n_samples:
        wav = np.pad(wav, (0, n_samples - len(wav)))
    return wav


def load_wav_crops(file_paths, n_samples, device='cuda', pin=True, to_device=True):
    """Read one random `n_samples` crop per file -> [B, 1, n_samples] float32.

    Reads the whole batch in parallel (thread pool; sf.read drops the GIL).
    Mirrors train_contrast.py's random-crop loading, but ONE view per song
    (the CLAP positive is (audio_i, caption_i), not two-crop instance disc.).
    With to_device=False the tensor stays on CPU (pinned) for background prefetch.
    """
    xs = list(_read_pool.map(lambda fp: _read_one_crop(fp, n_samples), file_paths))
    x = torch.from_numpy(np.asarray(xs, dtype=np.float32)).unsqueeze(1)
    if pin:
        x = x.pin_memory()
    if to_device:
        x = x.to(device, non_blocking=True)
    return x


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    audio_cfg = dict(
        in_channels=1, patch_size=16, sample_rate=16000,
        n_fft=1024, hop_length=512, n_mels=192, time_length=32,
        frequency_length=12,
    )
    model = CLAP(
        audio_cfg=audio_cfg, hidden_size=768, num_heads=12, mlp_ratio=4.0,
        proj_dim=512, audio_depth=12, text_depth=4,
    ).to(device)
    from torchinfo import summary
    summary(model)

    B, T = 4, 163830
    text = torch.randn(B, 256, 1024, device=device)
    audio = torch.randn(B, 1, T, device=device)
    text_mask = torch.ones(B, 256, dtype=torch.bool, device=device)
    song_ids = torch.arange(B, device=device)
    out = model(text, audio, text_mask=text_mask, song_ids=song_ids)
    print({k: (v.item() if v.ndim == 0 else tuple(v.shape)) for k, v in out.items()})
