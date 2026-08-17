
"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import json
import time
import math
import copy
from datetime import timedelta
from contextlib import nullcontext
from tqdm import tqdm
from torchinfo import summary

from scipy.signal import medfilt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from einops import rearrange

from diffusion_forcing import MetaConditionalModernDiTV2_smedium as net, zero_init_local_embedder
from dito import DiToV5 as Tokenizer
from adapter import InvertibleAdapter
import soundfile as sf

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
stage = 2
out_dir = f'Stage{stage}_MetaConditionalModernDiTV2_smedium_24576_subset_adapter_longtrain_24chunks_nulltokens_clap'
eval_interval = 5000
sample_interval = 5000
log_interval = 100
save_interval = 5000
eval_iters = 600
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = out_dir
wandb_run_name = str(time.time())
# data
dataset = ''
gradient_accumulation_steps = 4
batch_size = 32
TARGET_SIG = 4
TARGET_BPM = 60 * TARGET_SIG / (24576 / 16000)
# model
patch_size = 2
gradient_checkpointing = False
spatial_window = 64
n_chunks = 24
max_seq_len = spatial_window * n_chunks
vae_embed_dim = 16
n_style_embeddings = 256
style_dim = 512
use_null_token = True
respect_song_boundaries = True
# measures per consistency-decoder call during sampling. the full batch is
# n_samples * n_chunks = 240, which needs a ~5 GiB contiguous activation block
# and OOMs against a fragmented post-training allocator. 0 disables chunking.
decode_chunk_size = 60
cut_seconds = 1
drop_path_rate = 0.1
# adamw optimizer
learning_rate = 1e-4 # max learning rate
max_iters = 1000000 # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = False # whether to decay the learning rate
warmup_iters = 5000 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = learning_rate / 10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # eval is split across ranks and sampling is sharded by variant, but rank 0
    # still carries the checkpoint writes on its own. eval_iters=600 over two
    # splits plus the sampling passes can exceed the default 30-minute collective
    # timeout, at which point the NCCL watchdog aborts the whole job. Give it a
    # lot of headroom.
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    # bind the process group to this rank's device explicitly. without device_id,
    # device-less collectives (barrier) guess from the ambient context and warn,
    # and the nccl communicator is built lazily on first use instead of here.
    init_process_group(backend=backend, timeout=timedelta(hours=4), device_id=torch.device(device))
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0, f'world size and accumulation steps are not divisible!'
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_rank = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

_bpm_checked = set()


def _check_bpm_alignment(split, bpms, n_rows):
    """The bpm bin comes from generate_bpm_dataset.py, a SEPARATE pass over its own
    path list that silently skips measures with duration <= 0. Every other array
    here derives from the ..._map.json. If the two disagree on length they are
    misaligned, and every row past the first divergence carries another song's
    tempo -- silent label corruption, no crash. Fail fast instead of training on it."""
    if split in _bpm_checked:
        return
    _bpm_checked.add(split)
    assert len(bpms) == n_rows, (
        f"[{split}] bpm bin has {len(bpms)} rows but the measure arrays have {n_rows}. "
        f"These were built by different generators and are misaligned -- BPM labels "
        f"would be wrong for most of the corpus. Regenerate the bpm bin against the "
        f"same map, or set respect_song_boundaries aside and fix this first."
    )


def get_batch(split='train', batch_size=batch_size, return_idx=False):
    if split == 'train':
        data = np.memmap('/data/binaries/low_large_24576_subset_adapter_longtrain_v2_64_train.bin', dtype=np.float32, mode='r', shape=(4490789, spatial_window, vae_embed_dim))
        style = np.memmap('/data/binaries/clap_nopre_clap_style_train.bin', dtype=np.float16, mode='r', shape=(4490789, style_dim))
        meta = np.memmap('/data/binaries/low_large_24576_subset_chroma_rms_density_zcr_flatness_train.bin', dtype=np.float32, mode='r', shape=(4490789, 16))
        bpms = np.memmap('/data/binaries/low_large_24576_subset_adapter_longtrain_v2_64_bpm_train.bin', dtype=np.float32, mode='r')
    else:
        data = np.memmap('/data/binaries/low_large_24576_subset_adapter_longtrain_v2_64_val.bin', dtype=np.float32, mode='r', shape=(99131, spatial_window, vae_embed_dim))
        style = np.memmap('/data/binaries/clap_nopre_clap_style_val.bin', dtype=np.float16, mode='r', shape=(99131, style_dim))
        meta = np.memmap('/data/binaries/low_large_24576_subset_chroma_rms_density_zcr_flatness_val.bin', dtype=np.float32, mode='r', shape=(99131, 16))
        bpms = np.memmap('/data/binaries/low_large_24576_subset_adapter_longtrain_v2_64_bpm_val.bin', dtype=np.float32, mode='r')
    
    _check_bpm_alignment(split, bpms, data.shape[0])

    if respect_song_boundaries:
        starts = valid_starts(split, n_chunks)
        # draw through the TORCH rng, not numpy's: torch.manual_seed(1337 + ddp_rank)
        # is what gives each rank a different data stream. numpy's global rng is
        # not seeded here, so using it would put ranks' sampling outside our control.
        idxs = torch.from_numpy(starts[torch.randint(len(starts), (batch_size,)).numpy()])
    else:
        idxs = torch.randint(len(data) - n_chunks, (batch_size,))

    x = torch.from_numpy(np.stack([data[idx:idx+n_chunks] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    # style bin is stored fp16 by generate_clap_dataset.py; widen to match the rest of the batch
    style = torch.from_numpy(np.stack([style[idx:idx+n_chunks] for idx in idxs], axis=0).astype(np.float32)).pin_memory().to(device, non_blocking=True)
    chroma = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, :12] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    rms = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 12] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    density = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 13] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    zcr = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 14] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    flatness = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 15] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    bpm = torch.from_numpy(np.stack([bpms[idx:idx+n_chunks] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)

    if return_idx:
        return x, bpm, rms, density, zcr, flatness, chroma, style, idxs.numpy()
    return x, bpm, rms, density, zcr, flatness, chroma, style

# -----------------------------------------------------------------------------
# measure row -> song -> caption
#
# The style bin holds one AUDIO-tower embedding per measure, so a row has no text
# attached to it directly. generate_clap_dataset.py writes the join we need:
#   song_measure_start/stop[song]  -- the contiguous measure block each song owns
#   song_paths[song]               -- that song's wav path, which keys the caption
# so a measure row maps to a song by locating the block that contains it.
# -----------------------------------------------------------------------------
CLAP_INDEX = {
    'train': '/data/binaries/clap_nopre_clap_index_train.npz',
    'val':   '/data/binaries/clap_nopre_clap_index_val.npz',
}
CAPTIONS_JSONL = '/home/dylandeshler/Jazz/preprocess/final_llm_captions_expanded.jsonl'

_clap_index = {}
_captions_by_path = None


def _load_clap_index(split):
    if split not in _clap_index:
        z = np.load(CLAP_INDEX[split], allow_pickle=False)
        start = z['song_measure_start'].astype(np.int64)
        stop = z['song_measure_stop'].astype(np.int64)
        # blocks are not guaranteed to be listed in ascending row order, so sort
        # before searchsorted -- an unsorted haystack returns silent nonsense
        order = np.argsort(start)
        _clap_index[split] = {
            'start': start[order], 'stop': stop[order],
            'paths': z['song_paths'][order],
        }
    return _clap_index[split]

CAPTION_TIERS = ('short_caption', 'medium_caption', 'long_caption')
NUM_VARS = 6

def _captions():
    """file_path -> {tier: [variation, ...]}.

    augment_captions.py writes {'file_path': ..., 'llm_output': {tier: [...]}}:
    the text is NESTED under llm_output, and each tier is a LIST -- the original
    caption at index 0, then NUM_VARS-1 LLM-selected rewrites of it. A top-level
    rec['short_caption'] therefore does not exist and silently reads as empty.
    Some records carry a malformed llm_output (a bare list, or nothing); every
    other consumer in the repo treats those as "no captions", so do the same.
    Keyed by path rather than by position so it cannot silently mis-join if the
    jsonl and the song list ever diverge."""
    global _captions_by_path
    if _captions_by_path is None:
        _captions_by_path = {}
        with open(CAPTIONS_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                path = rec.get('file_path')
                if not path:
                    continue
                text = rec.get('llm_output', {})
                if isinstance(text, list) or not text:
                    text = {}
                _captions_by_path[path] = {
                    tier: (list(text.get(tier, [])) + [''] * NUM_VARS)[:NUM_VARS]
                    for tier in CAPTION_TIERS
                }
    return _captions_by_path

_valid_start_cache = {}

def valid_starts(split, n_meas):
    """Every measure row where an n_meas window stays inside one song.

    get_batch's plain randint can start a window near the end of a song and run
    it into the next one, splicing two unrelated recordings into a single
    training example -- with conditioning (style/bpm/chroma) taken from both.
    Sampling from this array instead makes that impossible. Built once per
    split and cached; ~36 MB for the train split.
    """
    key = (split, int(n_meas))
    if key in _valid_start_cache:
        return _valid_start_cache[key]
    ix = _load_clap_index(split)
    segs, dropped, dropped_rows = [], 0, 0
    for s, e in zip(ix['start'], ix['stop']):
        if e - s >= n_meas:
            segs.append(np.arange(s, e - n_meas + 1, dtype=np.int64))
        else:
            dropped += 1
            dropped_rows += int(e - s)
    if not segs:
        raise ValueError(f'[{split}] no song has {n_meas} measures; cannot build windows')
    starts = np.concatenate(segs)
    total = int(ix['stop'].max())
    if master_process:
        print(f'[{split}] boundary-safe starts: {len(starts)} of {total} rows '
              f'({len(starts) / total * 100:.1f}%); {dropped} songs shorter than '
              f'{n_meas} measures excluded ({dropped_rows} rows)')
    _valid_start_cache[key] = starts
    return starts

def measures_to_songs(idxs, split, n_meas=1):
    """For each measure row, the owning song and its captions.

    n_meas is the window length: get_batch slices n_chunks consecutive measures
    and nothing stops that window from running off the end of a song, so we
    report whether it straddles rather than pretending the first song owns it.
    """
    ix = _load_clap_index(split)
    caps = _captions()
    misses = 0
    out = []
    for idx in np.atleast_1d(idxs).astype(np.int64):
        s = int(np.searchsorted(ix['start'], idx, side='right') - 1)
        if s < 0 or idx >= ix['stop'][s]:
            out.append({'song': None, 'note': f'measure {int(idx)} falls in no song block'})
            continue
        last = int(np.searchsorted(ix['start'], idx + n_meas - 1, side='right') - 1)
        path = str(ix['paths'][s])
        rec = caps.get(path)
        if rec is None:
            misses += 1
            rec = {}
        # variation 0 is the caption the LLM actually wrote; 1..5 are rewrites of
        # it that the CLAP text tower also trained on
        out.append({
            'song': path,
            'measure_row': int(idx),
            'measure_in_song': int(idx - ix['start'][s]),
            'song_measures': int(ix['stop'][s] - ix['start'][s]),
            'straddles_song_boundary': bool(last != s or idx + n_meas > ix['stop'][s]),
            **{tier: rec.get(tier, [''])[0] for tier in CAPTION_TIERS},
        })
    if misses:
        # the index and the jsonl are both keyed by file_path, so a miss means
        # they were built from different caption files -- say so rather than
        # writing blank captions and looking like the songs have none
        print(f'measures_to_songs: {misses} of {len(out)} songs had no caption '
              f'record in {CAPTIONS_JSONL}')
    return out

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

ckpt_path = os.path.join('tokenizer_low_large_24576_subset_longtrain', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
tokenizer_args = checkpoint['model_args']

tokenizer = Tokenizer(**tokenizer_args).to(device)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
tokenizer.load_state_dict(state_dict)
tokenizer.eval()
del state_dict
encoder_ratios = math.prod(tokenizer.encoder.ratios)

ckpt_path = os.path.join('tokenizer_adapter_low_large_24576_subset_longtrain_v2', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
adapter_args = checkpoint['model_args']

adapter = InvertibleAdapter(**adapter_args).to(device)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
adapter.load_state_dict(state_dict)
adapter.eval()
del state_dict
max_adapter_len = adapter.max_seq_len

model_args = dict(in_channels=vae_embed_dim, style_dim=style_dim, n_chunks=n_chunks, spatial_window=spatial_window, use_null_token=use_null_token, gradient_checkpointing=gradient_checkpointing, patch_size=patch_size, stage=stage, drop_path_rate=drop_path_rate)

class EMAModel:
    def __init__(self, model, decay=0.9999, step_offset=0):
        self.decay = decay
        # steps of averaging this EMA already carries. The (1+s)/(10+s) ramp below
        # exists so a from-scratch EMA isn't pinned to its random init; at a stage
        # boundary that ramp is actively harmful, because step 0 gives decay 0.1
        # and the loaded stage-1 EMA is 90% overwritten by the raw model on the
        # very first update. Carrying stage 1's step count forward keeps it at the
        # full 0.9999 ceiling, which is what the averaging history actually earns.
        self.step_offset = step_offset
        self.ema_model = copy.deepcopy(model).eval()
        self.ema_model.requires_grad_(False)
        self.ema_model = torch.compile(self.ema_model)

    @torch.no_grad()
    def update(self, model, step):
        step = step + self.step_offset
        current_decay = min(self.decay, (1 + step) / (10 + step))

        for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
            if model_param.requires_grad:
                ema_param.data.mul_(current_decay).add_(model_param.data, alpha=1.0 - current_decay)

def _zero_local_embedder(m):
    """Re-zero the stage-2 adapter after loading stage-1 weights over it.

    ModernDiT.initialize_weights already did this at construction, but the
    load_state_dict above can overwrite it, so redo it here. Shares the model's
    own helper rather than re-listing layers, so the two can't drift -- zeroing
    block1.project as well would silently freeze both kernel-3 convs forever.
    """
    m = getattr(m, '_orig_mod', m)
    zero_init_local_embedder(m.net.local_embedder)


def load_stage1_weights(module, state_dict, what, skip='local_embedder'):
    """Load a stage-1 checkpoint into `module`, reporting what did NOT land.

    The trap this exists to close: torch.compile registers the real model as a
    '_orig_mod' child, so a COMPILED module's own keys carry that prefix while
    an uncompiled one's do not. Stripping the prefix and loading into a compiled
    module (which is what `ema.ema_model` is) matches nothing at all, and with
    strict=False that is completely silent -- you keep whatever weights the
    module was constructed with. Unwrap to _orig_mod so both targets take the
    same stripped keys, then assert on the leftovers instead of trusting it.
    """
    target = getattr(module, '_orig_mod', module)      # undo torch.compile
    # .replace, not .startswith: compiling a submodule puts the prefix MID-key
    sd = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    sd = {k: v for k, v in sd.items() if skip not in k}
    missing, unexpected = target.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if skip not in k]
    assert not unexpected, (
        f'[{what}] checkpoint carries {len(unexpected)} keys the model has no slot '
        f'for, e.g. {unexpected[:3]}. These weights were NOT loaded.')
    assert not real_missing, (
        f'[{what}] model has {len(real_missing)} params the checkpoint did not '
        f'fill, e.g. {real_missing[:3]}. These are still at random init.')
    if master_process:
        print(f'[{what}] loaded {len(sd)} tensors, {len(missing)} left at init ({skip})')


if init_from == 'scratch':
    if stage == 2:
        stage1_ckpt = torch.load(os.path.join(out_dir.replace('Stage2', 'Stage1'), 'ckpt.pt'), map_location=device)

        model = net(**model_args)
        load_stage1_weights(model, stage1_ckpt['model'], 'stage2 <- stage1 model')
        _zero_local_embedder(model)

        # deepcopy(model) would seed the EMA with the RAW stage-1 weights; the
        # point of loading stage1_ckpt['ema'] is that the averaged weights are
        # better, and estimate_loss reports the EMA, so getting this wrong shows
        # up directly as a worse stage-2 starting loss.
        ema = EMAModel(model, step_offset=stage1_ckpt.get('iter_num', 0))
        load_stage1_weights(ema.ema_model, stage1_ckpt['ema'], 'stage2 <- stage1 ema')
        _zero_local_embedder(ema.ema_model)
        if master_process:
            print(f'stage2: EMA resumes at step_offset={ema.step_offset} '
                  f'(decay {min(ema.decay, (1 + ema.step_offset) / (10 + ema.step_offset)):.6f})')
    elif stage == 1:
        # init a new model from scratch
        print("Initializing a new model from scratch")
        model = net(**model_args)
        ema = EMAModel(model)
    tokens_trained = 0
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']
    model_args['gradient_checkpointing'] = gradient_checkpointing

    model = net(**model_args)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    ema = EMAModel(model)
    state_dict = checkpoint['ema']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    # unwanted_prefix = '_orig_mod.'
    # for k,v in list(state_dict.items()):
    #     if k.startswith(unwanted_prefix):
    #         state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    ema.ema_model.load_state_dict(state_dict)
    
    iter_num = checkpoint['iter_num']
    tokens_trained = checkpoint['tokens']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = net.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

model.to(device)
ema.ema_model.to(device)
if master_process:
    summary(model)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# compile the model
if compile and 'cuda' in device:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
    tokenizer = torch.compile(tokenizer)
    adapter = torch.compile(adapter)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    """Evaluate on EVERY rank, then average.

    MUST be entered by all ranks -- it contains a collective. Each rank walks
    eval_iters // world_size batches from its own (differently seeded) stream, so
    the union is still ~eval_iters batches but the node finishes in a quarter of
    the wall-clock instead of rank 0 grinding while the others block.

    Batch size is deliberately left at batch_size * gradient_accumulation_steps:
    that matches the training shape, so the compiled ema forward is not retraced.
    """
    n_local = math.ceil(eval_iters / ddp_world_size)
    out = {}
    for split in ['train', 'val']:
        # accumulate on-device; .item() per step would sync the GPU every batch
        losses = torch.zeros(n_local, device=device)
        for k in tqdm(range(n_local), desc=f'eval {split}', disable=not master_process):
            X = get_batch(split, batch_size=batch_size * gradient_accumulation_steps)
            with ctx:
                loss = ema.ema_model(*X)
            losses[k] = loss.detach().float()
        mean = losses.mean()
        if ddp:
            dist.all_reduce(mean, op=dist.ReduceOp.AVG)
        out[split] = mean.item()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if not decay_lr:
        return learning_rate
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def smooth_bpm_predictions(bpm_tensor: torch.Tensor, method: str = 'median', window_size: int = 3) -> torch.Tensor:
    """
    Smooths the instantaneous BPM predictions across the chunk dimension.
    bpm_tensor: shape (Batch, Chunks)
    method: One of median, global, moving_average
    """
    if method == 'global':
        # Collapse the sequence to the mean tempo per batch item
        mean_bpm = bpm_tensor.mean(dim=1, keepdim=True)
        return mean_bpm.expand_as(bpm_tensor)
    
    bpm_np = bpm_tensor.float().cpu().detach().numpy()
    smoothed = np.zeros_like(bpm_np)
    
    for i in range(bpm_np.shape[0]):
        if method == 'median':
            # medfilt requires odd window sizes
            smoothed[i] = medfilt(bpm_np[i], kernel_size=window_size)
            
        elif method == 'moving_average':
            kernel = np.ones(window_size) / window_size
            # Pad edges so the sequence length stays 15
            padded = np.pad(bpm_np[i], (window_size//2, window_size//2), mode='edge')
            smoothed[i] = np.convolve(padded, kernel, mode='valid')
            
    return torch.from_numpy(smoothed).to(bpm_tensor.device)

def crossfade_segments(segment_a, segment_b, sample_rate, crossfade_ms=15):
    """
    Crossfades two 1D numpy audio arrays to prevent boundary clicks.
    
    Args:
        segment_a: numpy array of the first measure.
        segment_b: numpy array of the second measure.
        sample_rate: The sample rate of the audio (e.g., 44100 or 24000).
        crossfade_ms: Duration of the crossfade in milliseconds.
    """
    # Convert milliseconds to exact sample count
    crossfade_samples = int(sample_rate * (crossfade_ms / 1000.0))
    
    # Safety check: if segments are too short, just concatenate
    if len(segment_a) < crossfade_samples or len(segment_b) < crossfade_samples:
        return np.concatenate((segment_a, segment_b))
        
    # Create linear fade curves (can also use np.cos for equal-power crossfades)
    fade_out = np.linspace(1.0, 0.0, crossfade_samples)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples)
    
    # Apply fades to the overlapping edges
    overlap_a = segment_a[-crossfade_samples:] * fade_out
    overlap_b = segment_b[:crossfade_samples] * fade_in
    
    # Sum the overlapped audio
    mixed_overlap = overlap_a + overlap_b
    
    # Stitch the untouched beginnings/ends with the mixed overlap
    stitched_audio = np.concatenate((
        segment_a[:-crossfade_samples],
        mixed_overlap,
        segment_b[crossfade_samples:]
    ))
    
    return stitched_audio

def _decode_measures(y, shape, mask, max_len, n_steps, decoder_noise=None):
    """adapter.decode + tokenizer.decode over `shape[0]` measures, in slices.

    The whole sample batch is n_samples * n_chunks = 240 measures, and the
    consistency decoder's activations for that in one go want a single ~5 GiB
    contiguous block. After a few thousand training steps the caching allocator
    is fragmented around training's shapes and cannot hand one out even with
    plenty of total headroom, so this decodes in slices and concatenates. The
    decoder is per-item, so slicing changes nothing about the result.
    """
    n = shape[0]
    chunk = decode_chunk_size if decode_chunk_size > 0 else n
    outs = []
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        z = adapter.decode(y[i:j], (j - i, shape[1], shape[2]), mask=mask[i:j])
        outs.append(tokenizer.decode(
            z, shape=(1, max_len), n_steps=n_steps,
            noise=decoder_noise[i:j, :, :max_len] if decoder_noise is not None else None,
        ))
        del z
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


def predict_measures(gen_shape, net_kwargs, uncond_net_kwargs, n_steps, guidance=1, gen_noise=None, decoder_noise=None, method='median', window_size=3, memory_efficient=False, rescale_phi=0, cfg_mode="independent", t_dist="uniform"):
    with ctx:
        y = ema.ema_model.generate(gen_shape, net_kwargs=net_kwargs, uncond_net_kwargs=uncond_net_kwargs, n_steps=n_steps, guidance=guidance, noise=gen_noise, memory_efficient=memory_efficient, rescale_phi=rescale_phi, cfg_mode=cfg_mode, t_dist=t_dist)
    
    if isinstance(net_kwargs, list):
        bpm = net_kwargs[0]['bpm']
    else:
        bpm = net_kwargs['bpm']
    
    seconds_per_beat = 60.0 / bpm
    measure_duration_sec = seconds_per_beat * TARGET_SIG
    
    target_samples = (measure_duration_sec * 16000).long()
    max_len = min(target_samples.max().item(), encoder_ratios * (max_adapter_len - 1))
    max_len = encoder_ratios * math.ceil(max_len / encoder_ratios)
    max_latent_len = max_len // encoder_ratios
    
    indices = torch.arange(max_latent_len, device=device).view(1, 1, -1)
    lengths = ((target_samples + encoder_ratios - 1) // encoder_ratios).unsqueeze(-1)
    mask = indices < lengths
    mask = mask.view(gen_shape[0] * n_chunks, max_latent_len)
    shape = (gen_shape[0] * n_chunks, vae_embed_dim, max_latent_len)

    with ctx:
        y = rearrange(y, 'b t n c -> (b t) c n')
        y = _decode_measures(y, shape, mask, max_len, n_steps, decoder_noise=decoder_noise)

    target_samples = target_samples.flatten().cpu().detach().numpy()
    y = y.squeeze().cpu().detach().numpy()
    
    target_samples = target_samples.reshape(gen_shape[0], n_chunks)
    # out = [np.concatenate([y_[:min(int(samples), max_len)] for y_, samples in zip(y[i*n_chunks:(i+1)*n_chunks], target_samples[i])], axis=0).astype(np.float32) for i in range(gen_shape[0])]
    
    out = []
    for i in range(gen_shape[0]):
        temp = y[i*n_chunks][:min(int(target_samples[i][0]), max_len)]
        for j in range(1, n_chunks):
            temp = crossfade_segments(temp, y[i*n_chunks+j][:min(int(target_samples[i][j]), max_len)], sample_rate=16000, crossfade_ms=20)
        out.append(temp.astype(np.float32))
    
    return out

def decode_latents(y, bpm, n_steps, decoder_noise=None):
    seconds_per_beat = 60.0 / bpm
    measure_duration_sec = seconds_per_beat * TARGET_SIG
    
    target_samples = (measure_duration_sec * 16000).long()
    max_len = min(target_samples.max().item(), encoder_ratios * (max_adapter_len - 1))
    max_len = encoder_ratios * math.ceil(max_len / encoder_ratios)
    max_latent_len = max_len // encoder_ratios
    
    indices = torch.arange(max_latent_len, device=device).view(1, 1, -1)
    lengths = ((target_samples + encoder_ratios - 1) // encoder_ratios).unsqueeze(-1)
    mask = indices < lengths
    mask = mask.view(bpm.shape[0] * n_chunks, max_latent_len)
    shape = (bpm.shape[0] * n_chunks, vae_embed_dim, max_latent_len)

    with ctx:
        y = rearrange(y, 'b t n c -> (b t) c n')
        y = _decode_measures(y, shape, mask, max_len, n_steps, decoder_noise=decoder_noise)

    target_samples = target_samples.flatten().cpu().detach().numpy()
    y = y.squeeze().cpu().detach().numpy()
    
    target_samples = target_samples.reshape(bpm.shape[0], n_chunks)
    # out = [np.concatenate([y_[:min(int(samples), max_len)] for y_, samples in zip(y[i*n_chunks:(i+1)*n_chunks], target_samples[i])], axis=0).astype(np.float32) for i in range(bpm.shape[0])]
    
    out = []
    for i in range(bpm.shape[0]):
        temp = y[i*n_chunks][:min(int(target_samples[i][0]), max_len)]
        for j in range(1, n_chunks):
            temp = crossfade_segments(temp, y[i*n_chunks+j][:min(int(target_samples[i][j]), max_len)], sample_rate=16000, crossfade_ms=20)
        out.append(temp.astype(np.float32))
    
    return out

# the four generation passes save_samples makes, with a rough relative cost used
# only to balance the static assignment below. joint-cfg runs cond+uncond through
# one generate, so it costs ~2x a plain pass; gt skips diffusion generation
# entirely and only pays the adapter/tokenizer decode.
_SAMPLE_VARIANTS = (('cfg', 2.0), ('cond', 1.0), ('bpm_only', 1.0), ('gt', 0.3))


def _variant_owners():
    """Greedy longest-processing-time split of the sampling passes across ranks.
    Purely a function of world size, so every rank derives the same assignment
    without communicating. At world_size=1 rank 0 owns all four."""
    loads = [0.0] * ddp_world_size
    owners = {}
    for name, cost in sorted(_SAMPLE_VARIANTS, key=lambda kv: -kv[1]):
        r = min(range(ddp_world_size), key=lambda i: loads[i])
        owners[name] = r
        loads[r] += cost
    return owners


@torch.no_grad()
def save_samples(step):
    batch_dir = os.path.join(out_dir, str(step))
    os.makedirs(batch_dir, exist_ok=True)

    t_dist = 'logit'
    cfg_mode = 'joint'
    n_steps = 32
    n_samples = 10
    owners = _variant_owners()
    x, bpm, rms, density, zcr, flatness, chroma, style, idxs = get_batch('val', batch_size=n_samples, return_idx=True)

    gen_noise = torch.randn(x.shape).to(device)
    decoder_noise = torch.randn(n_samples * n_chunks, 1, encoder_ratios * (max_adapter_len - 1)).to(device)

    if ddp:
        # every rank must condition on the SAME batch and the SAME noise, otherwise
        # the four variants describe four different songs and are not comparable.
        # torch.manual_seed(1337 + ddp_rank) gives each rank its own draw, so rank
        # 0's wins and everyone else's is overwritten in place.
        for t in (x, bpm, rms, density, zcr, flatness, chroma, style, gen_noise, decoder_noise):
            dist.broadcast(t, src=0)

    # captions are text-only and only rank 0 writes them, so don't make every rank
    # pay for the jsonl load. idxs is rank 0's draw, matching the broadcast batch.
    songs = []
    if master_process:
        try:
            songs = measures_to_songs(idxs, 'val', n_meas=n_chunks)
        except (OSError, KeyError, ValueError) as e:
            print(f'save_samples: could not resolve captions ({e}); writing audio without text')
            songs = [{} for _ in range(n_samples)]

    unconditional_mask = {
        'bpm': torch.ones(*bpm.shape, 1).to(device).bool(),
        'rms': torch.ones(*rms.shape, 1).to(device).bool(),
        'density': torch.ones(*density.shape, 1).to(device).bool(),
        'zcr': torch.ones(*zcr.shape, 1).to(device).bool(),
        'flatness': torch.ones(*flatness.shape, 1).to(device).bool(),
        'chroma': torch.ones(*chroma.shape[:-1], 1).to(device).bool(),
        'style': torch.ones(*style.shape[:-1], 1).to(device).bool(),
    }
    net_kwargs = {
        'bpm': bpm,
        'rms': rms,
        'density': density,
        'zcr': zcr,
        'flatness': flatness,
        'chroma': chroma,
        'style': style,
    }
    uncond_net_kwargs = net_kwargs | {'unconditional_mask': unconditional_mask}
    
    if cfg_mode == 'joint':
        joint_conditional_mask = {k: ~v for k, v in unconditional_mask.items()}
        cfg_net_kwargs = [net_kwargs | {'unconditional_mask': joint_conditional_mask}]
    elif cfg_mode == 'independent':
        cfg_net_kwargs = []
        for k, v in unconditional_mask.items():
            temp_mask = unconditional_mask.copy()
            temp_mask[k] = ~v
            cfg_net_kwargs.append(net_kwargs | {'unconditional_mask': temp_mask})
    
    cfg_guidances = [3] * len(unconditional_mask)

    bpm_only_mask = {k: (torch.zeros_like(v) if k == 'bpm' else v) for k, v in unconditional_mask.items()}
    bpm_only_net_kwargs = net_kwargs | {'unconditional_mask': bpm_only_mask}

    # each rank runs only the passes it owns. the variant name doubles as the wav
    # suffix, so ranks write disjoint files into the shared batch_dir and nothing
    # has to be gathered back (the audio is ragged, one array per sample).
    audio = {}
    # training leaves ~13 GiB reserved-but-unallocated in the caching allocator,
    # carved into training-shaped blocks. sampling wants a few large contiguous
    # ones, so hand the cached segments back before asking. costs a slow re-warm
    # over the first few steps after the eval, which is cheap at this interval.
    torch.cuda.empty_cache()
    if owners['cfg'] == ddp_rank:
        audio['cfg'] = predict_measures(x.shape, cfg_net_kwargs, uncond_net_kwargs, n_steps, guidance=cfg_guidances, gen_noise=gen_noise, decoder_noise=decoder_noise, method='median', window_size=3, memory_efficient=False, rescale_phi=0, cfg_mode=cfg_mode, t_dist=t_dist)
    if owners['cond'] == ddp_rank:
        audio['cond'] = predict_measures(x.shape, net_kwargs, uncond_net_kwargs, n_steps, guidance=1.0, gen_noise=gen_noise, decoder_noise=decoder_noise, method='median', window_size=3, t_dist=t_dist)
    if owners['bpm_only'] == ddp_rank:
        audio['bpm_only'] = predict_measures(x.shape, bpm_only_net_kwargs, uncond_net_kwargs, n_steps, guidance=1.0, gen_noise=gen_noise, decoder_noise=decoder_noise, method='median', window_size=3, t_dist=t_dist)
    if owners['gt'] == ddp_rank:
        audio['gt'] = decode_latents(x, bpm, n_steps, decoder_noise=decoder_noise)

    for name, waves in audio.items():
        for i in range(n_samples):
            sf.write(os.path.join(batch_dir, f'{i}_{name}.wav'), waves[i].flatten(), 16000)

    if not master_process:
        return

    for i in range(n_samples):
        s = songs[i] if i < len(songs) else {}
        np.savez(
            os.path.join(batch_dir, f'{i}_cond.npz'),
            bpm=bpm[i].detach().cpu().numpy(),
            rms=rms[i].detach().cpu().numpy(),
            density=density[i].detach().cpu().numpy(),
            zcr=zcr[i].detach().cpu().numpy(),
            flatness=flatness[i].detach().cpu().numpy(),
            chroma=chroma[i].detach().cpu().numpy(),
            style=style[i].detach().cpu().numpy(),
            **{k: str(v) for k, v in s.items()},
        )
        # the text the style vector *should* correspond to -- these embeddings come
        # from the audio tower, so this is the caption of the source song, not a
        # prompt the model was given
        with open(os.path.join(batch_dir, f'{i}_caption.txt'), 'w', encoding='utf-8') as f:
            f.write(f"song:   {s.get('song')}\n")
            f.write(f"measure {s.get('measure_in_song')} of {s.get('song_measures')} "
                    f"(row {s.get('measure_row')}, window {n_chunks} measures)\n")
            if s.get('straddles_song_boundary'):
                f.write("WARNING: this window crosses a song boundary; the caption "
                        "only describes the song it starts in\n")
            f.write(f"\nshort:  {s.get('short_caption')}\n")
            f.write(f"\nmedium: {s.get('medium_caption')}\n")
            f.write(f"\nlong:   {s.get('long_caption')}\n")

# logging
if wandb_log and master_process:
    import wandb
    if init_from == 'resume':
        wandb.init(project=wandb_project, name=wandb_run_name, id='9bohl8v3', resume='must', config=config)
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# optimizer
optimizer = torch.optim.AdamW(raw_model.net.create_optimizer_groups(weight_decay=weight_decay, base_lr=learning_rate, new_lr=learning_rate), betas=(beta1, beta2))
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # lr_scale = lr / learning_rate 
    # for param_group in optimizer.param_groups:
    #     if 'initial_lr' not in param_group:
    #         param_group['initial_lr'] = param_group['lr']

    #     param_group['lr'] = param_group['initial_lr'] * lr_scale
    
    tokens_trained += batch_size * gradient_accumulation_steps * max_seq_len

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        # EVERY rank enters estimate_loss -- it ends in an all_reduce, so gating it
        # on master_process would deadlock the collective.
        losses = estimate_loss()

        # sampling is sharded by variant across ranks (see _variant_owners), and it
        # starts with a broadcast, so EVERY rank has to enter save_samples. the
        # branch is on iter_num, which is identical everywhere, so it stays collective.
        if iter_num % sample_interval == 0:
            with ctx:
                save_samples(iter_num)

        if master_process:
            # checkpointing stays rank-0 only
            print(f"iter {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}")

            if wandb_log and not (init_from == 'resume' and local_iter_num == 0):
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                    "tokens": tokens_trained,
                })
            if iter_num > 0 and losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'val_loss': best_val_loss,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'tokens': tokens_trained,
                    'ema': ema.ema_model.state_dict(),
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                print(f"saving new best checkpoint to {out_dir}")
            if iter_num > 0 and always_save_checkpoint:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'val_loss': losses['val'],
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'tokens': tokens_trained,
                    'ema': ema.ema_model.state_dict(),
                }
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))

        # resync: ranks that finished their sampling pass early wait here while the
        # rest finish and rank 0 writes checkpoints, instead of racing into the next
        # backward and blocking inside its all-reduce (which is far harder to
        # diagnose from a stack dump). also guarantees every wav is on disk here.
        if ddp:
            dist.barrier()

    # eval_only is a config constant, identical on every rank, so this break is
    # collective -- no rank is left waiting on a partner that already exited
    if eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            loss = model(*X)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    ema.update(model, iter_num)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = 0#raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.6f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()