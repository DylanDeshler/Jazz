
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
import time
import math
import copy
import json
import pickle
from contextlib import nullcontext
from tqdm import tqdm
from torchinfo import summary

from scipy.signal import medfilt
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from einops import rearrange

from diffusion_forcing import MetaConditionalModernDiTV2Composer_smedium as net, MetaConditionalModernDiTV2_smedium as dit
from dito import DiToV5 as Tokenizer
from adapter import InvertibleAdapter
import soundfile as sf

import torch

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'MetaConditionalModernDiTV2Composer_smedium_24576_subset_adapter_longtrain_24chunks'
eval_interval = 5000
sample_interval = 5000
log_interval = 100
save_interval = 5000
eval_iters = 600
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = out_dir
wandb_run_name = str(time.time())
# data
dataset = ''
gradient_accumulation_steps = 1
batch_size = 128
TARGET_SIG = 4
TARGET_BPM = 60 * TARGET_SIG / (24576 / 16000)
# model
patch_size = 1
gradient_checkpointing = False
spatial_window = 1
n_chunks = 192
max_seq_len = spatial_window * n_chunks
vae_embed_dim = 16
n_style_embeddings = 256
n_text_tokens = 256
style_dim = 128
signal_dim = {'style': style_dim, 'chroma': 12, 'rms': 1, 'density': 1, 'zcr': 1, 'flatness': 1, 'bpm': 768}
weights = {k: 1 for k in signal_dim.keys()}
use_null_token = True
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
device = 'cuda:1' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
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
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
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

with open('/data/binaries/low_large_24576_subset_chroma_rms_density_zcr_flatness_train_map.json', 'r', encoding='utf-8') as f:
    train_map = json.load(f)
with open('/data/binaries/low_large_24576_subset_chroma_rms_density_zcr_flatness_val_map.json', 'r', encoding='utf-8') as f:
    val_map = json.load(f)
train_paths = list(train_map.keys())[:21030]
val_paths = list(val_map.keys())[:21030]
print(f'Found {len(train_paths)} train paths and {len(val_paths)} val paths')

chroma_mean = torch.tensor([
    0.45533183, 0.39680213, 0.44615716, 0.42044115, 0.40855545, 0.45450154, 0.3971631, 0.496346, 0.44164586, 0.4416672, 0.44793198, 0.39493898
]).to(device)
chroma_std = torch.tensor([
    0.18241853, 0.16477719, 0.18014704, 0.18011539, 0.1677363, 0.18919244, 0.16196373, 0.19185093, 0.18003348, 0.1768027, 0.18706752, 0.1618064
]).to(device)
rms_mean = torch.tensor([3.2653894]).to(device)
rms_std = torch.tensor([3.597796]).to(device)
density_mean = torch.tensor([2.5229013]).to(device)
density_std = torch.tensor([1.230155]).to(device)
zcr_mean = torch.tensor([0.10766766]).to(device)
zcr_std = torch.tensor([0.048143145]).to(device)
flatness_mean = torch.tensor([0.011151944]).to(device)
flatness_std = torch.tensor([0.018700112]).to(device)

def get_batch(split='train', batch_size=batch_size):
    data = np.memmap('/data/binaries/caption_embeddings.bin', dtype=np.float32, mode='r', shape=(21030, 3, 256, 1024))
    caption_idxs = np.random.randint(data.shape[1], size=batch_size)
    if split == 'train':
        song_idxs = np.random.randint(len(train_paths), size=batch_size)
        bounds = np.array([train_map[train_paths[i]] for i in song_idxs])
        style = np.memmap('/data/binaries/contrast_learntmep_instance_10s_style_train.bin', dtype=np.float32, mode='r', shape=(4490789, style_dim))
        meta = np.memmap('/data/binaries/low_large_24576_subset_chroma_rms_density_zcr_flatness_train.bin', dtype=np.float32, mode='r', shape=(4490789, 16))
        bpms = np.memmap('/data/binaries/low_large_24576_subset_adapter_longtrain_v2_64_bpm_train.bin', dtype=np.float32, mode='r')
    else:
        song_idxs = np.random.randint(len(val_paths), size=batch_size)
        bounds = np.array([val_map[val_paths[i]] for i in song_idxs])
        style = np.memmap('/data/binaries/contrast_learntmep_instance_10s_style_val.bin', dtype=np.float32, mode='r', shape=(99131, style_dim))
        meta = np.memmap('/data/binaries/low_large_24576_subset_chroma_rms_density_zcr_flatness_val.bin', dtype=np.float32, mode='r', shape=(99131, 16))
        bpms = np.memmap('/data/binaries/low_large_24576_subset_adapter_longtrain_v2_64_bpm_val.bin', dtype=np.float32, mode='r')
    
    song_starts = bounds[:, 0]
    song_stops  = bounds[:, 1]

    highs = np.maximum(song_stops - n_chunks - 1, song_starts + 1)
    random_offsets = np.floor(np.random.rand(batch_size) * (highs - song_starts)).astype(int)
    starts = song_starts + random_offsets
    stops = np.minimum(starts + n_chunks, song_stops)
    idx_matrix = starts[:, None] + np.arange(n_chunks)
    idx_matrix = np.minimum(idx_matrix, stops[:, None])
    
    text = torch.from_numpy(np.stack([data[i, j, :n_text_tokens] for i, j in zip(song_idxs, caption_idxs)], axis=0)).pin_memory().to(device, non_blocking=True)
    
    style = torch.from_numpy(style[idx_matrix]).pin_memory().to(device, non_blocking=True)
    chroma = torch.from_numpy(meta[idx_matrix, :12]).pin_memory().to(device, non_blocking=True)
    rms = torch.from_numpy(meta[idx_matrix, 12]).pin_memory().to(device, non_blocking=True)
    density = torch.from_numpy(meta[idx_matrix, 13]).pin_memory().to(device, non_blocking=True)
    zcr = torch.from_numpy(meta[idx_matrix, 14]).pin_memory().to(device, non_blocking=True)
    flatness = torch.from_numpy(meta[idx_matrix, 15]).pin_memory().to(device, non_blocking=True)
    bpm = torch.from_numpy(bpms[idx_matrix]).pin_memory().to(device, non_blocking=True)
    
    rms = (rms - rms_mean) / rms_std
    density = (density - density_mean) / density_std
    zcr = (zcr - zcr_mean) / zcr_std
    flatness = (flatness - flatness_mean) / flatness_std
    chroma = (chroma - chroma_mean) / chroma_std
    bpm = dit.net.bpm_embedder(torch.clamp(torch.round(bpm), min=0, max=349).long())
    
    x = torch.cat([style, chroma, rms.unsqueeze(-1), density.unsqueeze(-1), zcr.unsqueeze(-1), flatness.unsqueeze(-1), bpm], dim=-1).unsqueeze(2)

    return x, text

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

ckpt_path = os.path.join('Stage2_MetaConditionalModernDiTV2_smedium_24576_subset_adapter_longtrain_24chunks', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
dit_args = checkpoint['model_args']

dit = dit(**dit_args).to(device)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
dit.load_state_dict(state_dict)
dit.eval()
del state_dict

model_args = dict(in_channels=vae_embed_dim, style_dim=style_dim, n_chunks=n_chunks, n_text_tokens=n_text_tokens, spatial_window=spatial_window, signal_dim=signal_dim, use_null_token=use_null_token, gradient_checkpointing=gradient_checkpointing, patch_size=patch_size, drop_path_rate=drop_path_rate, weights=weights)

class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        self.ema_model.requires_grad_(False)
        self.ema_model = torch.compile(self.ema_model)
        
    @torch.no_grad()
    def update(self, model, step):
        current_decay = min(self.decay, (1 + step) / (10 + step))
        
        for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
            if model_param.requires_grad:
                ema_param.data.mul_(current_decay).add_(model_param.data, alpha=1.0 - current_decay)

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = net(**model_args)
    # model.net.bpm_embedder.load_state_dict(dit.net.bpm_embedder.state_dict())
    # model.net.bpm_embedder.requires_grad_(False)
    model.net.null_text = torch.from_numpy(np.memmap('/data/binaries/caption_embeddings.bin', dtype=np.float32, mode='r', shape=(21030, 3, 256, 1024))[-1, -1].copy())
    tokens_trained = 0
    
    ema = EMAModel(model)
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
    out = {}
    for i, split in enumerate(['train', 'val']):
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters)):
            X = get_batch(split, batch_size=batch_size * gradient_accumulation_steps)
            with ctx:
                loss = ema.ema_model(*X)
            losses[k] = loss.item()
        out[split] = losses.mean()
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
        y = adapter.decode(y, shape, mask=mask)
        y = tokenizer.decode(y, shape=(1, max_len), n_steps=n_steps, noise=decoder_noise[:, :, :max_len] if decoder_noise is not None else None)
    
    target_samples = target_samples.flatten().cpu().detach().numpy()
    y = y.squeeze().cpu().detach().numpy()
    
    target_samples = target_samples.reshape(gen_shape[0], n_chunks)
    
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
        y = adapter.decode(y, shape, mask=mask)
        y = tokenizer.decode(y, shape=(1, max_len), n_steps=n_steps, noise=decoder_noise[:, :, :max_len] if decoder_noise is not None else None)
    
    target_samples = target_samples.flatten().cpu().detach().numpy()
    y = y.squeeze().cpu().detach().numpy()
    
    target_samples = target_samples.reshape(bpm.shape[0], n_chunks)
    
    out = []
    for i in range(bpm.shape[0]):
        temp = y[i*n_chunks][:min(int(target_samples[i][0]), max_len)]
        for j in range(1, n_chunks):
            temp = crossfade_segments(temp, y[i*n_chunks+j][:min(int(target_samples[i][j]), max_len)], sample_rate=16000, crossfade_ms=20)
        out.append(temp.astype(np.float32))
    
    return out

@torch.no_grad()
def save_samples(step):
    batch_dir = os.path.join(out_dir, str(step))
    os.makedirs(batch_dir, exist_ok=True)
    
    t_dist = 'logit'
    cfg_mode = 'joint'
    n_steps = 100
    n_samples = 10
    x, bpm, rms, density, zcr, flatness, chroma, style = get_batch('val', batch_size=n_samples)
    
    gen_noise = torch.randn(x.shape).to(device)
    decoder_noise = torch.randn(n_samples * n_chunks, 1, encoder_ratios * (max_adapter_len - 1)).to(device)
    
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
    
    y_cfg = predict_measures(x.shape, cfg_net_kwargs, uncond_net_kwargs, n_steps, guidance=cfg_guidances, gen_noise=gen_noise, decoder_noise=decoder_noise, method='median', window_size=3, memory_efficient=False, rescale_phi=0, cfg_mode=cfg_mode, t_dist=t_dist)
    y = predict_measures(x.shape, net_kwargs, uncond_net_kwargs, n_steps, guidance=1.0, gen_noise=gen_noise, decoder_noise=decoder_noise, method='median', window_size=3, t_dist=t_dist)
    y_gt = decode_latents(x, bpm, n_steps, decoder_noise=decoder_noise)
    
    for i in range(n_samples):
        sf.write(os.path.join(batch_dir, f'{i}.wav'), y[i].flatten(), 16000)
        sf.write(os.path.join(batch_dir, f'{i}_cfg.wav'), y_cfg[i].flatten(), 16000)
        sf.write(os.path.join(batch_dir, f'{i}_gt.wav'), y_gt[i].flatten(), 16000)

# logging
if wandb_log and master_process:
    import wandb
    if init_from == 'resume':
        wandb.init(project=wandb_project, name=wandb_run_name, id='ows76kwp', resume='must', config=config)
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# optimizer
optimizer = torch.optim.AdamW(model.net.create_optimizer_groups(weight_decay=weight_decay, lr=learning_rate), betas=(beta1, beta2))
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # lr_scale = lr / learning_rate 
    # for param_group in optimizer.param_groups:
    #     if 'initial_lr' not in param_group:
    #         param_group['initial_lr'] = param_group['lr']

    #     param_group['lr'] = param_group['initial_lr'] * lr_scale
    
    tokens_trained += batch_size * gradient_accumulation_steps * max_seq_len

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        # if iter_num % sample_interval == 0 and master_process:
        #     with ctx:
        #         save_samples(iter_num)
            
        losses = estimate_loss()
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