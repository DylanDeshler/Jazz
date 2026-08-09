"""
Train a CLAP-style audio<->text model on the jazz dataset.

Both towers consume precomputed, frozen features (see clap.py):
  * text : T5-v1.1-xxl encoder last_hidden_state  -> caption_embeddings_* memmap
  * audio: contrast.py per-measure style vectors  -> ..._style_* memmap

The messy caption-row -> song -> measure-range mapping is reused verbatim from
train_modern_composer_fancy_dit_measures.py (map_to_slices / audio_*_map).

Single GPU:
    $ python train_clap.py
DDP (4 gpus):
    $ torchrun --standalone --nproc_per_node=4 train_clap.py
"""

import os
import time
import math
import json
import pickle
from contextlib import nullcontext
from tqdm import tqdm
from torchinfo import summary

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf

from clap import CLAP as net
from clap import build_wav_index, load_wav_crops

# -----------------------------------------------------------------------------
# I/O
out_dir = 'clap_pre'
eval_interval = 2000
log_interval = 100
eval_iters = 100
eval_only = False
profile = False  # if True, print a per-stage timing breakdown each log_interval
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume'
# wandb
wandb_log = True
wandb_project = 'clap_jazz'
wandb_run_name = str(time.time())
# data
gradient_accumulation_steps = 1
batch_size = 512          # per-GPU batch; DDP all-gather makes negatives = world*batch
n_samples = 16383 * 10    # raw-audio crop length (~10s @ 16kHz) fed to the audio tower
text_dim = 1024           # T5-v1.1-xxl hidden dim
n_text_tokens = 256
wav_glob = '/data/wavs/*' # raw wavs, indexed by basename for path resolution
# audio tower init: 'scratch' (random) or 'contrast' (reuse pretrained backbone)
audio_init = 'contrast'
contrast_ckpt = 'contrast_learntmep_instance_10s/ckpt.pt'   # used when audio_init=='contrast'
# retrieval-audio dump (item 4): save the wavs of top-k retrieved songs to listen to
sample_rate = 16000       # raw wavs are 16 kHz
retrieval_clip_seconds = 15   # seconds of audio to write per retrieved song
retrieval_n_queries = 6       # number of text queries to dump audio for
retrieval_topk = 3            # top-k retrieved songs per query
# model
# shared transformer body: BOTH towers use these, only depth differs.
# NOTE: for audio_init=='contrast' these MUST match the contrast.py backbone
# (hidden 768 / heads 12 / mlp_ratio 4) or the pretrained weights won't load.
hidden_size = 768         # shared audio+text hidden dim
num_heads = 12            # shared audio+text heads
mlp_ratio = 4.0           # shared audio+text mlp ratio
proj_dim = 512            # shared CLAP space
audio_depth = 12          # audio tower depth
text_depth = 4            # text tower depth
drop_path = 0.1           # stochastic depth rate, applied to BOTH towers (regularization)
# audio front-end (mirrors contrast.py)
patch_size = 16
n_fft = 1024
hop_length = 512
n_mels = 192
# SpecAugment mask WIDTHS are absolute (frames/mels), but the 10s window has ~5x
# more time frames than the 2s window these were tuned for (~320 vs ~64), so the
# original time_length=32 masked ~5x less of the clip -> weak regularization ->
# overfitting. Scale time_length by the window ratio to restore ~50% expected
# time coverage (torchaudio draws each mask width in [0, time_length]).
# freq masking is unchanged (mel dim is still 192, so coverage is unaffected).
time_length = 160
frequency_length = 12
# optimizer
learning_rate = 5e-4
max_iters = 100000
weight_decay = 0.2        # CLIP-style heavier wd on a small trainable head
beta1 = 0.9
beta2 = 0.98
grad_clip = 1.0
# lr schedule
decay_lr = False
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = learning_rate / 10
# DDP / system
backend = 'nccl'
device = 'cuda:3'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Metadata: caption-row -> song -> measure-range  (reused from the composer script)
# -----------------------------------------------------------------------------
with open('/data/binaries/caption_embeddings_expanded_shuffled_split_metadata.pkl', 'rb') as f:
    text_meta = pickle.load(f)

with open('/data/binaries/low_large_24576_subset_chroma_rms_density_zcr_flatness_train_map.json', 'r') as f:
    audio_train_map = json.load(f)
with open('/data/binaries/low_large_24576_subset_chroma_rms_density_zcr_flatness_val_map.json', 'r') as f:
    audio_val_map = json.load(f)

with open('/home/dylandeshler/Jazz/preprocess/final_llm_captions_expanded.jsonl', 'r', encoding='utf-8') as f:
    raw_captions = [json.loads(line) for line in f]

train_raw_paths = [cap.get('file_path', '') for cap in raw_captions if cap.get('file_path', '') in audio_train_map]
val_raw_paths = [cap.get('file_path', '') for cap in raw_captions if cap.get('file_path', '') in audio_val_map]

# parallel lists of the caption dicts, for recovering readable text during eval
train_raw_captions = [cap for cap in raw_captions if cap.get('file_path', '') in audio_train_map]
val_raw_captions = [cap for cap in raw_captions if cap.get('file_path', '') in audio_val_map]

train_shuffled_indices = text_meta['train']['shuffled_indices']
train_orig_sub_shape = text_meta['train']['orig_sub_shape']
val_shuffled_indices = text_meta['val']['shuffled_indices']
val_orig_sub_shape = text_meta['val']['orig_sub_shape']

# row counts for the shuffled text memmaps: n_songs * NUM_TIERS(3) * NUM_VARS(6)
N_TRAIN_ROWS = len(train_raw_paths) * 3 * 6
N_VAL_ROWS = len(val_raw_paths) * 3 * 6
print(f'Text rows -> train: {N_TRAIN_ROWS}, val: {N_VAL_ROWS}')

# index raw wavs by basename so get_batch can resolve song file_path -> wav path
build_wav_index(wav_glob)


def map_rows(idx, split, count):
    """For a run of `count` shuffled text rows starting at `idx`, return
    (bounds[count,2] audio measure ranges, song_ids[count], tiers[count]).
    tier: 0=short, 1=medium, 2=long caption."""
    if split == 'train':
        shuffled_indices, orig_sub_shape = train_shuffled_indices, train_orig_sub_shape
        song_paths_list, audio_map = train_raw_paths, audio_train_map
    else:
        shuffled_indices, orig_sub_shape = val_shuffled_indices, val_orig_sub_shape
        song_paths_list, audio_map = val_raw_paths, audio_val_map

    bounds, song_ids, tiers = [], [], []
    for i in range(idx, idx + count):
        orig_flat_idx = shuffled_indices[i]
        song_idx, tier_j, var_k = np.unravel_index(orig_flat_idx, orig_sub_shape)
        bounds.append(audio_map[song_paths_list[song_idx]])
        song_ids.append(int(song_idx))
        tiers.append(int(tier_j))
    return np.array(bounds), np.array(song_ids), np.array(tiers)


def get_text_from_index(row_idx, split='train'):
    """Recover the literal caption string that produced shuffled text row `row_idx`."""
    if split == 'train':
        shuffled_indices, orig_sub_shape = train_shuffled_indices, train_orig_sub_shape
        caption_source_list = train_raw_captions
    else:
        shuffled_indices, orig_sub_shape = val_shuffled_indices, val_orig_sub_shape
        caption_source_list = val_raw_captions

    orig_flat_idx = shuffled_indices[row_idx]
    song_idx, tier_j, var_k = np.unravel_index(orig_flat_idx, orig_sub_shape)
    song_data = caption_source_list[song_idx].get('llm_output', {})
    if isinstance(song_data, list) or not song_data:
        song_data = {}
    if tier_j == 0:
        caption_list = song_data.get('short_caption', [])
    elif tier_j == 1:
        caption_list = song_data.get('medium_caption', [])
    else:
        caption_list = song_data.get('long_caption', [])
    NUM_VARS = 6
    caption_list = (caption_list + [''] * NUM_VARS)[:NUM_VARS]
    return caption_list[var_k]


_batch_times = {}  # populated by get_batch when profile=True


def get_batch(split='train', batch_size=batch_size, return_start=False):
    if split == 'train':
        text_mm = np.memmap('/data/binaries/caption_embeddings_expanded_shuffled_train.bin',
                            dtype=np.float16, mode='r', shape=(N_TRAIN_ROWS, n_text_tokens, text_dim))
        song_paths_list = train_raw_paths
        n_rows = N_TRAIN_ROWS
    else:
        text_mm = np.memmap('/data/binaries/caption_embeddings_expanded_shuffled_val.bin',
                            dtype=np.float16, mode='r', shape=(N_VAL_ROWS, n_text_tokens, text_dim))
        song_paths_list = val_raw_paths
        n_rows = N_VAL_ROWS

    start = np.random.randint(n_rows - batch_size)
    bounds, song_ids, tiers = map_rows(start, split, batch_size)

    _t = time.time() if profile else None

    # --- text: contiguous slab of shuffled rows -> [B, 256, 1024] ---
    # Keep fp16 and transfer as-is: autocast casts it for the in_proj matmul, so a
    # CPU fp32 upcast just doubles the copied bytes (536MB vs 268MB) and burns CPU.
    text = torch.from_numpy(np.ascontiguousarray(text_mm[start:start + batch_size].copy()))
    if profile:
        _batch_times['text_read'] = (time.time() - _t) * 1000; _t = time.time()

    # --- audio: one random raw-wav crop per song (full audio tower) ---
    # song_ids are song indices into song_paths_list; resolve to file paths and
    # read a random ~10s crop each. augmentation lives inside the audio tower.
    file_paths = [song_paths_list[int(s)] for s in song_ids]
    audio = load_wav_crops(file_paths, n_samples, device=device)   # [B, 1, n_samples]
    audio_mask = None                                              # one crop -> one vector
    if profile:
        _batch_times['audio_read'] = (time.time() - _t) * 1000; _t = time.time()

    song_ids = torch.from_numpy(song_ids)
    tiers = torch.from_numpy(tiers)

    text = text.pin_memory().to(device, non_blocking=True)
    # mask on-device: abs().sum over 134M elems is ~free on GPU, costly on CPU.
    # T5 padded to max_length with zeros; valid token == any nonzero feature.
    text_mask = (text.abs().sum(-1) > 0)
    text_mask[:, 0] = True   # guard: never fully-empty (would nan pooling softmax)
    song_ids = song_ids.pin_memory().to(device, non_blocking=True)
    tiers = tiers.pin_memory().to(device, non_blocking=True)
    if profile:
        torch.cuda.synchronize()
        _batch_times['text_h2d'] = (time.time() - _t) * 1000
    if return_start:
        return (text, audio, text_mask, audio_mask, song_ids), start, tiers
    return text, audio, text_mask, audio_mask, song_ids


# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

# audio_cfg carries ONLY audio front-end params (mel/patch). Depth and the
# shared body size (hidden_size/num_heads/mlp_ratio) are passed to CLAP directly.
audio_cfg = dict(
    in_channels=1, patch_size=patch_size,
    sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    time_length=time_length, frequency_length=frequency_length,
)
model_args = dict(
    audio_cfg=audio_cfg, text_dim=text_dim,
    hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
    proj_dim=proj_dim, audio_depth=audio_depth, text_depth=text_depth,
    drop_path=drop_path, n_text_tokens=n_text_tokens, audio_init=audio_init,
)

if init_from == 'scratch':
    print(f"Initializing a new CLAP model from scratch (audio_init={audio_init})")
    model = net(**model_args)
    if audio_init == 'contrast':
        model.load_audio_backbone(contrast_ckpt, map_location='cpu')
    tokens_trained = 0
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    model = net(**model_args)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    tokens_trained = checkpoint['tokens']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)
summary(model)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

if compile and 'cuda' in device:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# number of (audio, text) pairs pooled into the retrieval gallery each eval.
# larger == harder / more meaningful retrieval metrics.
retrieval_pool = 1024


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        accs = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters), desc=f'Estimatiing loss for {split}'):
            batch = get_batch(split, batch_size=batch_size)
            with ctx:
                res = model(*batch)
            losses[k] = res['loss'].item()
            accs[k] = res['acc'].item()
        out[f'{split}/loss'] = losses.mean()
        out[f'{split}/acc'] = accs.mean()
    out.update(retrieval_metrics('val'))
    model.train()
    return out


def _recall_metrics(sim, same, tag):
    """Given sim [Q, G] and same [Q, G] positive mask, compute R@k + median rank."""
    metrics = {}
    order = sim.argsort(dim=1, descending=True)     # [Q, G] ranked gallery indices
    hit = same.gather(1, order)                     # bool: is each ranked item a positive
    first = hit.float().argmax(dim=1) + 1           # rank (1-indexed) of first positive
    for k in (1, 5, 10):
        metrics[f'{tag}/R@{k}'] = hit[:, :k].any(dim=1).float().mean()
    metrics[f'{tag}/medrank'] = first.median().float()
    return metrics


@torch.no_grad()
def retrieval_metrics(split='val'):
    """Retrieval + collapse diagnostics over a pooled gallery of `retrieval_pool` pairs.

    Because captions are per-song, an audio's positive is any text of the SAME
    song (and vice-versa), so a rank counts as correct if any same-song item is
    retrieved by rank k. Also breaks retrieval down per caption tier
    (0=short, 1=medium, 2=long) and logs embedding-collapse diagnostics + a
    positive-vs-negative similarity histogram.
    """
    a_all, t_all, ids_all, tier_all = [], [], [], []
    collected = 0
    while collected < retrieval_pool:
        (text, audio, text_mask, audio_mask, song_ids), _, tiers = get_batch(split, batch_size=batch_size, return_start=True)
        enc_a = model.module.encode_audio if ddp else model.encode_audio
        enc_t = model.module.encode_text if ddp else model.encode_text
        with ctx:
            a = enc_a(audio, audio_mask)
            t = enc_t(text, text_mask)
        a_all.append(a.float()); t_all.append(t.float())
        ids_all.append(song_ids); tier_all.append(tiers)
        collected += a.shape[0]

    a = torch.cat(a_all)[:retrieval_pool]
    t = torch.cat(t_all)[:retrieval_pool]
    ids = torch.cat(ids_all)[:retrieval_pool]
    tiers = torch.cat(tier_all)[:retrieval_pool]

    sim = a @ t.T                                   # rows: audio, cols: text
    same = ids[:, None] == ids[None, :]            # [n, n] same-song positives

    metrics = {}
    metrics.update(_recall_metrics(sim, same, f'{split}/a2t'))
    metrics.update(_recall_metrics(sim.T, same.T, f'{split}/t2a'))

    # --- (3) per-tier retrieval: which caption granularity actually aligns ---
    # text->audio, restricted to queries of a given tier (gallery stays full).
    tier_names = {0: 'short', 1: 'medium', 2: 'long'}
    for tj, tname in tier_names.items():
        q = tiers == tj
        if q.sum() < 2:
            continue
        m = _recall_metrics(sim.T[q], same.T[q], f'{split}/t2a_{tname}')
        # only keep R@1/R@10 to avoid flooding the log
        metrics[f'{split}/R@1_{tname}'] = m[f'{split}/t2a_{tname}/R@1']
        metrics[f'{split}/R@10_{tname}'] = m[f'{split}/t2a_{tname}/R@10']

    # --- (1) positive vs negative similarity separation ---
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    pos_vals = sim[eye]                             # matched (audio_i, text_i) pairs
    neg_vals = sim[same & ~eye] if (same & ~eye).any() else sim[~same]
    rand_neg = sim[~same]                           # strict cross-song negatives
    metrics[f'{split}/sim_pos_mean'] = pos_vals.mean()
    metrics[f'{split}/sim_neg_mean'] = rand_neg.mean()
    metrics[f'{split}/sim_gap'] = pos_vals.mean() - rand_neg.mean()

    # --- (2) collapse diagnostics ---
    # mean off-diagonal similarity creeping toward 1.0 => collapse
    metrics[f'{split}/offdiag_mean'] = rand_neg.mean()
    # effective rank of the embedding matrices (crashing toward 1 => collapse)
    metrics[f'{split}/erank_audio'] = _effective_rank(a)
    metrics[f'{split}/erank_text'] = _effective_rank(t)
    # per-dim std averaged (near 0 => collapse)
    metrics[f'{split}/emb_std_audio'] = a.std(dim=0).mean()
    metrics[f'{split}/emb_std_text'] = t.std(dim=0).mean()

    # stash arrays for the histogram plotter (numpy, small)
    retrieval_metrics.last_hist = (
        pos_vals.detach().cpu().numpy(),
        rand_neg.detach().cpu().numpy(),
    )
    return metrics


def _effective_rank(x):
    """Effective rank = exp(entropy of normalized singular values). Collapse -> 1."""
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x.float())
    p = s / (s.sum() + 1e-12)
    ent = -(p * (p + 1e-12).log()).sum()
    return torch.exp(ent)


@torch.no_grad()
def save_similarity_histogram(step, split='val'):
    """Plot positive vs negative cosine-similarity distributions (the key eyeball)."""
    hist = getattr(retrieval_metrics, 'last_hist', None)
    if hist is None:
        return
    pos_vals, neg_vals = hist
    batch_dir = os.path.join(out_dir, str(step))
    os.makedirs(batch_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(neg_vals, bins=60, density=True, alpha=0.55, color='tomato', label='negative (cross-song)')
    plt.hist(pos_vals, bins=60, density=True, alpha=0.55, color='teal', label='positive (matched pair)')
    plt.axvline(float(np.mean(neg_vals)), color='tomato', ls='--', lw=1)
    plt.axvline(float(np.mean(pos_vals)), color='teal', ls='--', lw=1)
    plt.title(f'audio-text similarity @ iter {step} (gap={np.mean(pos_vals)-np.mean(neg_vals):.3f})')
    plt.xlabel('cosine similarity'); plt.ylabel('density'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(batch_dir, 'sim_hist.png'))
    plt.close()


def _load_wav_clip(wav_path, seconds):
    """Read up to `seconds` of audio from a random offset. Returns None on failure."""
    try:
        info = sf.info(wav_path)
        want = int(seconds * info.samplerate)
        start = 0
        if info.frames > want:
            start = int(np.random.randint(0, info.frames - want))
        wav, sr = sf.read(wav_path, start=start, frames=want, dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)  # downmix to mono
        return wav, sr
    except Exception as e:
        print(f'[retrieval-audio] could not read {wav_path}: {e}')
        return None


@torch.no_grad()
def save_retrieval_samples(step, split='val', n_show=8, pool=512):
    """Dump text->audio retrievals: readable captions to a text file AND the
    actual wavs of the top-k retrieved songs so they can be listened to.

    Layout under {out_dir}/{step}/retrieval/:
        q0_QUERY.txt                 the query caption (+ ranked results summary)
        q0_rank1_song{id}_{HIT|miss}.wav
        q0_rank2_...wav              ...
    """
    batch_dir = os.path.join(out_dir, str(step))
    os.makedirs(batch_dir, exist_ok=True)
    audio_dir = os.path.join(batch_dir, 'retrieval')
    os.makedirs(audio_dir, exist_ok=True)
    raw_paths = train_raw_paths if split == 'train' else val_raw_paths

    model.eval()
    (text, audio, text_mask, audio_mask, song_ids), start, _ = get_batch(split, batch_size=min(pool, batch_size), return_start=True)
    with ctx:
        enc_a = model.module.encode_audio if ddp else model.encode_audio
        enc_t = model.module.encode_text if ddp else model.encode_text
        a = enc_a(audio, audio_mask).float()
        t = enc_t(text, text_mask).float()
    sim = t @ a.T                                    # text query -> audio gallery
    k = min(retrieval_topk, sim.shape[1])
    topv, topk = sim.topk(k, dim=1)

    captions = [get_text_from_index(start + i, split) for i in range(t.shape[0])]
    lines = []
    n_q = min(retrieval_n_queries, n_show, t.shape[0])
    for q in range(n_q):
        q_song = int(song_ids[q].item())
        correct = [int(song_ids[r].item() == q_song) for r in topk[q].tolist()]
        lines.append(f'[query {q}] song={q_song} :: {captions[q][:200]}')
        lines.append(f'    top-{k} audio song_ids={topk[q].tolist()} sims={[round(v,3) for v in topv[q].tolist()]} match={correct}')

        # write the query caption and the retrieved wavs
        with open(os.path.join(audio_dir, f'q{q}_QUERY.txt'), 'w') as f:
            f.write(f'{captions[q]}\n\nquery_song_id={q_song}\n')
        # ground-truth audio for the query's own song (the ideal retrieval)
        gt = _load_wav_clip(raw_paths[q_song], retrieval_clip_seconds)
        if gt is not None:
            sf.write(os.path.join(audio_dir, f'q{q}_GT_song{q_song}.wav'), gt[0], gt[1])
        for rank, gallery_idx in enumerate(topk[q].tolist(), start=1):
            r_song = int(song_ids[gallery_idx].item())
            clip = _load_wav_clip(raw_paths[r_song], retrieval_clip_seconds)
            if clip is None:
                continue
            wav, sr = clip
            tag = 'HIT' if r_song == q_song else 'miss'
            fname = f'q{q}_rank{rank}_song{r_song}_{tag}.wav'
            sf.write(os.path.join(audio_dir, fname), wav, sr)

    with open(os.path.join(batch_dir, 'retrieval.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    model.train()


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if not decay_lr:
        return learning_rate
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if wandb_log and master_process:
    import wandb
    if init_from == 'resume':
        wandb.init(project=wandb_project, name=wandb_run_name, id='rimcfdy2', resume='must', config=config)
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

batch = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
running_dt = -1.0  # EMA of step time (ms), smooths out page-cache I/O jitter

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

while True:
    # always go through get_lr: warmup must run regardless of decay_lr, else the
    # contrastive temperature/projection see full LR at step 0 and collapse to
    # uniform logits (loss pinned at ln(batch_size)). get_lr already returns a
    # constant post-warmup LR when decay_lr is False.
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    tokens_trained += batch_size * gradient_accumulation_steps

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        save_retrieval_samples(iter_num)
        save_similarity_histogram(iter_num)
        print(f"iter {iter_num}: train loss {losses['train/loss']:.4f} acc {losses['train/acc']:.3f} | "
              f"val loss {losses['val/loss']:.4f} acc {losses['val/acc']:.3f} | "
              f"val t2a R@1 {losses['val/t2a/R@1']:.3f} R@10 {losses['val/t2a/R@10']:.3f} "
              f"medrank {losses['val/t2a/medrank']:.0f} | gap {losses['val/sim_gap']:.3f} "
              f"erank(a/t) {losses['val/erank_audio']:.0f}/{losses['val/erank_text']:.0f}")
        tier_str = ' '.join(
            f"{tname[:3]} R@1 {losses[f'val/R@1_{tname}']:.3f} R@10 {losses[f'val/R@10_{tname}']:.3f}"
            for tname in ('short', 'medium', 'long') if f'val/R@1_{tname}' in losses
        )
        if tier_str:
            print(f"    per-tier t2a: {tier_str}")
        if wandb_log:
            wandb.log({'iter': iter_num, 'lr': lr, 'tokens': tokens_trained,
                       'temperature': torch.exp(raw_model.log_temperature).clamp(max=100).item(),
                       **{k: v.item() for k, v in losses.items()}})
        if (losses['val/loss'] < best_val_loss or always_save_checkpoint) and iter_num > 0:
            best_val_loss = min(best_val_loss, losses['val/loss'].item())
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
                'tokens': tokens_trained,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if eval_only:
        break

    _tc = time.time() if profile else None
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            res = model(*batch)
            loss = res['loss'] / gradient_accumulation_steps
        if profile:
            torch.cuda.synchronize(); _fwd = (time.time() - _tc) * 1000; _tc = time.time()
        batch = get_batch('train')
        if profile:
            _fetch = (time.time() - _tc) * 1000; _tc = time.time()
        scaler.scale(loss).backward()
        if profile:
            torch.cuda.synchronize(); _bwd = (time.time() - _tc) * 1000; _tc = time.time()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    # rolling (EMA) step time; skip the first few iters so compile / warmup
    # don't skew the average.
    if local_iter_num >= 5:
        running_dt = dt if running_dt == -1.0 else 0.9 * running_dt + 0.1 * dt
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        avg_ms = (running_dt if running_dt > 0 else dt) * 1000
        print(f"iter {iter_num}: loss {lossf:.4f}, acc {res['acc'].item():.3f}, time {avg_ms:.2f}ms (avg)")
        if profile:
            print(f"    [profile] fwd {_fwd:.0f} bwd {_bwd:.0f} | next-batch fetch {_fetch:.0f} "
                  f"(text_read {_batch_times.get('text_read',0):.0f} "
                  f"audio_read {_batch_times.get('audio_read',0):.0f} "
                  f"text_h2d {_batch_times.get('text_h2d',0):.0f}) ms")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
