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

from clap import CLAP as net

# -----------------------------------------------------------------------------
# I/O
out_dir = 'clap_jazz_lit'
eval_interval = 2000
log_interval = 100
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume'
# wandb
wandb_log = False
wandb_project = 'clap_jazz'
wandb_run_name = str(time.time())
# data
gradient_accumulation_steps = 1
batch_size = 256          # large batches matter for contrastive learning
n_measures = 64           # audio measures pooled per song (temporal context)
style_dim = 128           # contrast.py feature dim
text_dim = 1024           # T5-v1.1-xxl hidden dim
n_text_tokens = 256
# model
hidden_size = 512
proj_dim = 512
audio_depth = 4
text_depth = 4
num_heads = 8
# optimizer
learning_rate = 5e-4
max_iters = 100000
weight_decay = 0.2        # CLIP-style heavier wd on a small trainable head
beta1 = 0.9
beta2 = 0.98
grad_clip = 1.0
# lr schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = learning_rate / 10
# DDP / system
backend = 'nccl'
device = 'cuda:0'
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


def map_rows(idx, split, count):
    """For a run of `count` shuffled text rows starting at `idx`, return
    (bounds[count,2] audio measure ranges, song_ids[count])."""
    if split == 'train':
        shuffled_indices, orig_sub_shape = train_shuffled_indices, train_orig_sub_shape
        song_paths_list, audio_map = train_raw_paths, audio_train_map
    else:
        shuffled_indices, orig_sub_shape = val_shuffled_indices, val_orig_sub_shape
        song_paths_list, audio_map = val_raw_paths, audio_val_map

    bounds, song_ids = [], []
    for i in range(idx, idx + count):
        orig_flat_idx = shuffled_indices[i]
        song_idx, tier_j, var_k = np.unravel_index(orig_flat_idx, orig_sub_shape)
        bounds.append(audio_map[song_paths_list[song_idx]])
        song_ids.append(int(song_idx))
    return np.array(bounds), np.array(song_ids)


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


def get_batch(split='train', batch_size=batch_size, return_start=False):
    if split == 'train':
        text_mm = np.memmap('/data/binaries/caption_embeddings_expanded_shuffled_train.bin',
                            dtype=np.float16, mode='r', shape=(N_TRAIN_ROWS, n_text_tokens, text_dim))
        style_mm = np.memmap('/data/binaries/contrast_learntmep_instance_10s_style_train.bin',
                             dtype=np.float32, mode='r', shape=(4490789, style_dim))
        n_rows = N_TRAIN_ROWS
    else:
        text_mm = np.memmap('/data/binaries/caption_embeddings_expanded_shuffled_val.bin',
                            dtype=np.float16, mode='r', shape=(N_VAL_ROWS, n_text_tokens, text_dim))
        style_mm = np.memmap('/data/binaries/contrast_learntmep_instance_10s_style_val.bin',
                             dtype=np.float32, mode='r', shape=(99131, style_dim))
        n_rows = N_VAL_ROWS

    start = np.random.randint(n_rows - batch_size)
    bounds, song_ids = map_rows(start, split, batch_size)

    # --- text: contiguous slab of shuffled rows -> [B, 256, 1024] ---
    text = torch.from_numpy(text_mm[start:start + batch_size].astype(np.float32))
    # T5 padded to max_length with zeros; valid token == any nonzero feature
    text_mask = (text.abs().sum(-1) > 0)
    # guard: never fully-empty (would nan the pooling softmax)
    text_mask[:, 0] = True

    # --- audio: sample a window of measures within each song ---
    song_starts = bounds[:, 0]
    song_stops = bounds[:, 1] - 1
    song_len = np.maximum(song_stops - song_starts, 1)
    highs = np.maximum(song_stops - n_measures, song_starts + 1)
    offsets = np.floor(np.random.rand(batch_size) * (highs - song_starts)).astype(int)
    starts = song_starts + offsets
    idx_matrix = starts[:, None] + np.arange(n_measures)          # [B, M]
    valid = idx_matrix <= song_stops[:, None]                     # measures that exist
    idx_matrix = np.minimum(idx_matrix, song_stops[:, None])      # clamp for gather
    audio = torch.from_numpy(style_mm[idx_matrix])               # [B, M, 128]
    audio_mask = torch.from_numpy(valid)
    audio_mask[:, 0] = True

    song_ids = torch.from_numpy(song_ids)

    text = text.pin_memory().to(device, non_blocking=True)
    text_mask = text_mask.pin_memory().to(device, non_blocking=True)
    audio = audio.pin_memory().to(device, non_blocking=True)
    audio_mask = audio_mask.pin_memory().to(device, non_blocking=True)
    song_ids = song_ids.pin_memory().to(device, non_blocking=True)
    if return_start:
        return (text, audio, text_mask, audio_mask, song_ids), start
    return text, audio, text_mask, audio_mask, song_ids


# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

model_args = dict(
    audio_dim=style_dim, text_dim=text_dim, hidden_size=hidden_size, proj_dim=proj_dim,
    audio_depth=audio_depth, text_depth=text_depth, num_heads=num_heads,
)

if init_from == 'scratch':
    print("Initializing a new CLAP model from scratch")
    model = net(**model_args)
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
        for k in range(eval_iters):
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


@torch.no_grad()
def retrieval_metrics(split='val'):
    """Recall@k and median rank over a pooled gallery of `retrieval_pool` pairs.

    Because captions are per-song, an audio's positive is any text of the SAME
    song (and vice-versa), so we score a rank as correct if any same-song item
    is retrieved by rank k.
    """
    a_all, t_all, ids_all = [], [], []
    collected = 0
    while collected < retrieval_pool:
        text, audio, text_mask, audio_mask, song_ids = get_batch(split, batch_size=batch_size)
        with ctx:
            a = model.encode_audio(audio, audio_mask) if not ddp else model.module.encode_audio(audio, audio_mask)
            t = model.encode_text(text, text_mask) if not ddp else model.module.encode_text(text, text_mask)
        a_all.append(a.float())
        t_all.append(t.float())
        ids_all.append(song_ids)
        collected += a.shape[0]

    a = torch.cat(a_all)[:retrieval_pool]
    t = torch.cat(t_all)[:retrieval_pool]
    ids = torch.cat(ids_all)[:retrieval_pool]
    n = a.shape[0]

    sim = a @ t.T                                   # rows: audio, cols: text
    same = ids[:, None] == ids[None, :]             # [n, n] same-song positives

    metrics = {}
    for name, s, pos in [('a2t', sim, same), ('t2a', sim.T, same.T)]:
        order = s.argsort(dim=1, descending=True)   # [n, n] ranked col indices
        hit = pos.gather(1, order)                  # bool, is each ranked item a positive
        # rank (1-indexed) of the first positive for each query
        first = hit.float().argmax(dim=1) + 1
        for k in (1, 5, 10):
            metrics[f'{split}/R@{k}_{name}'] = hit[:, :k].any(dim=1).float().mean()
        metrics[f'{split}/medrank_{name}'] = first.median().float()
    return metrics


@torch.no_grad()
def save_retrieval_samples(step, split='val', n_show=8, pool=512):
    """Dump a few text->audio retrievals with readable captions to a text file."""
    batch_dir = os.path.join(out_dir, str(step))
    os.makedirs(batch_dir, exist_ok=True)

    model.eval()
    (text, audio, text_mask, audio_mask, song_ids), start = get_batch(split, batch_size=min(pool, batch_size), return_start=True)
    with ctx:
        enc_a = model.module.encode_audio if ddp else model.encode_audio
        enc_t = model.module.encode_text if ddp else model.encode_text
        a = enc_a(audio, audio_mask).float()
        t = enc_t(text, text_mask).float()
    sim = t @ a.T                                    # text query -> audio gallery
    topk = sim.topk(min(3, sim.shape[1]), dim=1).indices

    captions = [get_text_from_index(start + i, split) for i in range(t.shape[0])]
    lines = []
    for q in range(min(n_show, t.shape[0])):
        correct = [int(song_ids[r].item() == song_ids[q].item()) for r in topk[q].tolist()]
        lines.append(f'[query {q}] song={song_ids[q].item()} :: {captions[q][:160]}')
        lines.append(f'    top-3 audio song_ids={topk[q].tolist()} match={correct}')
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
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

batch = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    tokens_trained += batch_size * gradient_accumulation_steps

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        save_retrieval_samples(iter_num)
        print(f"iter {iter_num}: train loss {losses['train/loss']:.4f} acc {losses['train/acc']:.3f} | "
              f"val loss {losses['val/loss']:.4f} acc {losses['val/acc']:.3f} | "
              f"val R@1 {losses['val/R@1_t2a']:.3f} R@10 {losses['val/R@10_t2a']:.3f} "
              f"medrank {losses['val/medrank_t2a']:.0f}")
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

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            res = model(*batch)
            loss = res['loss'] / gradient_accumulation_steps
        batch = get_batch('train')
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, acc {res['acc'].item():.3f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
