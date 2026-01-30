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
import pickle
from contextlib import nullcontext
from tqdm import tqdm
from torchinfo import summary

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from einops import rearrange

from style import IDM_S as net
from dito import DiToV5 as Tokenizer
import soundfile as sf
from scipy import signal

import pyrubberband as pyrb
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'Style_fix_256_adaln_1measures_bpm_S_nobias_poolfirst_norm_nohistory_1head_top5'
eval_interval = 5000
sample_interval = 5000
log_interval = 100
save_interval = 5000
eval_iters = 400
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = out_dir
wandb_run_name = str(time.time())
# data
dataset = ''
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 768 # * 5 * 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
# model
cut_seconds = 1
spatial_window = 48
n_encoder_chunks = 0
n_decoder_chunks = 1
n_chunks = n_encoder_chunks + n_decoder_chunks
max_seq_len = spatial_window * n_chunks
vae_embed_dim = 16
n_style_embeddings = 256
# adamw optimizer
learning_rate = 1e-4 # max learning rate
max_iters = 1000000 # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = False # whether to decay the learning rate
warmup_iters = 15000 # how many steps to warm up for
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

# poor man's data loader
def get_batch(split='train', batch_size=batch_size):
    # TODO: sample within songs (this can go over song boundaries)
    data = np.memmap('/home/ubuntu/Data/low_measures_large.bin', dtype=np.float16, mode='r', shape=(4403211, 48, vae_embed_dim))
    meta = np.memmap('/home/ubuntu/Data/measures_meta.bin', dtype=np.float32, mode='r', shape=(4403211, 2))
    if split == 'train':
        idxs = torch.randint(int(len(data) * 0.98) - n_chunks, (batch_size,))
    else:
        idxs = torch.randint(int(len(data) * 0.98), len(data) - n_chunks, (batch_size,))
    x = torch.from_numpy(np.stack([data[idx:idx+n_chunks] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    ratio = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 0] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    bpm = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 1] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    return x, ratio, bpm

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

ckpt_path = os.path.join('tokenizer_low_measures_large', 'ckpt.pt')
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

model_args = dict(in_channels=vae_embed_dim, spatial_window=spatial_window, n_encoder_chunks=n_encoder_chunks, n_decoder_chunks=n_decoder_chunks, n_style_embeddings=n_style_embeddings)

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = net(**model_args)
    tokens_trained = 0
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']

    model = net(**model_args)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
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
summary(model)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# compile the model
if compile and 'cuda' in device:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
    tokenizer = torch.compile(tokenizer)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for i, split in enumerate(['train', 'val']):
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters)):
            X, ratio, bpm = get_batch(split, batch_size=batch_size * gradient_accumulation_steps)
            with ctx:
                loss, _ = model(X, bpm)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def estimate_style_entropy():
    out1, out2, out3 = {}, {}, {}
    model.eval()
    for i, split in enumerate(['train', 'val']):
        entropies = torch.zeros(eval_iters)
        batch_entropies = torch.zeros(eval_iters)
        usages = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters)):
            X, ratio, bpm = get_batch(split, batch_size=batch_size * gradient_accumulation_steps)
            with ctx:
                entropy, batch_entropy, usage = model.action_model.style_entropy(X, bpm)
            entropies[k] = entropy
            batch_entropies[k] = batch_entropy
            usages[k] = usage
        out1[split] = entropies.mean()
        out2[split] = batch_entropies.mean()
        out3[split] = usages.mean()
    model.train()
    return out1, out2, out3

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def restore_measure(audio, stretch_ratio, sr=16000):
    """
    Restores a time-warped measure to its original duration.
    
    Args:
        audio (np.array): The fixed-length audio (from VAE or .npy file).
                          Can be shape (1, 24576) or (24576,).
        stretch_ratio (float): The ratio saved in your metadata 
                               (Original Length / Target Length).
        sr (int): Sampling rate (default 16000).
        
    Returns:
        np.array: The restored audio array at original duration.
    """
    
    if audio.dtype == np.float16:
        audio = audio.astype(np.float32)
    restore_rate = 1.0 / stretch_ratio
    
    y_restored = pyrb.time_stretch(audio, sr, restore_rate)
    return y_restored

def create_auto_grid(n_plots, figsize=(12, 8)):
    """
    Automatically creates a grid of subplots closest to a square shape.
    Returns the figure and a flattened list of axes.
    """
    cols = math.ceil(math.sqrt(n_plots))
    rows = math.ceil(n_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes_flat = axes.flatten()
    
    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].set_axis_off()
        
    return fig, axes_flat[:n_plots]

@torch.no_grad()
def generate_lam_actions(step):
    batch_dir = os.path.join(out_dir, str(step), 'actions')
    os.makedirs(batch_dir, exist_ok=True)
    
    model.eval()
    x, ratio, bpm = get_batch('val')
    x, ratio, bpm = x[[0]], ratio[[0]], bpm[[0]]
    
    n_samples = 32
    x, ratio, bpm = x.repeat(n_samples, 1, 1, 1), ratio.repeat(n_samples, 1), bpm.repeat(n_samples, 1)
    
    B, T, N, D = x.shape
    
    action_weights = torch.nn.functional.one_hot(torch.randint(n_style_embeddings, size=(n_samples,)), n_style_embeddings).float().pin_memory().to(device, non_blocking=True)
    
    with ctx:
        noise = torch.randn(x[:, -n_decoder_chunks:].shape, device=x.device)
        recon = model.generate_from_actions(x.clone(), bpm, action_weights, n_steps=50, noise=noise)
        
        x = tokenizer.decode(x.view(B * T, N, D).permute(0, 2, 1), shape=(1, 24576 * cut_seconds), n_steps=50).view(B, T, 1, 24576 * cut_seconds)
        recon = tokenizer.decode(recon.view(B * n_decoder_chunks, N, D).permute(0, 2, 1), shape=(1, 24576 * cut_seconds), n_steps=50).view(B, n_decoder_chunks, 1, 24576 * cut_seconds)
    
    x = x.cpu().detach().float().numpy().squeeze(-2)
    recon = recon.cpu().detach().float().numpy().squeeze(-2)
    
    wavs = []
    for i in range(x.shape[0]):
        og, y, r = x[i], recon[i], ratio[i].cpu().detach().numpy()
        tail_r = r[-n_decoder_chunks:]
        
        og_wav = np.concatenate([restore_measure(og[j], r[j].item()) for j in range(n_chunks)])
        recon_wav = np.concatenate([restore_measure(y[j], tail_r[j].item()) for j in range(n_decoder_chunks)])
        
        if n_encoder_chunks > 0:
            base = np.concatenate([restore_measure(og[j], r[j].item()) for j in range(n_encoder_chunks)])
            recon_wav = np.concatenate([base, recon_wav])
        
        sf.write(os.path.join(batch_dir, f'real.wav'), og_wav, 16000)
        sf.write(os.path.join(batch_dir, f'{i}.wav'), recon_wav, 16000)
        
        if i == 0:
            wavs.append(og_wav)
        wavs.append(recon_wav)
    
    T = len(og_wav) / 16000
    
    vmin, vmax = [], []
    for wav in wavs:
        frequencies, times, Sxx = signal.spectrogram(wav, 16000)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        vmin.append(Sxx_log.min())
        vmax.append(Sxx_log.max())

    vmin = min(vmin)
    vmax = max(vmax)
    
    pcm = None
    fig, axes = create_auto_grid(n_samples + 1)
    for i, ax in enumerate(axes):
        frequencies, times, Sxx = signal.spectrogram(wavs[i], 16000)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        
        pcm = ax.pcolormesh(times, frequencies, Sxx_log, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
        
        if i == 0:
            ax.set_title('Real')
        else:
            ax.set_title(f'{i}')

    fig.colorbar(pcm, ax=axes, label='Intensity [dB]')
    fig.supxlabel('Time [s]')
    fig.supylabel('Frequency [Hz]')

    plt.savefig(os.path.join(batch_dir, 'wavs.png'))
    plt.close('all')

@torch.no_grad()
def generate_lam_vs_random_actions(step):
    batch_dir = os.path.join(out_dir, str(step))
    os.makedirs(batch_dir, exist_ok=True)
    
    model.eval()
    x, ratio, bpm = get_batch('val')
    x, ratio, bpm = x[:20], ratio[:20], bpm[:20]

    B, T, N, D = x.shape

    with ctx:
        noise = torch.randn(x[:, -n_decoder_chunks:].shape, device=x.device)
        recon, random_recon = model.lam_vs_random_actions(x.clone(), bpm, n_steps=50, noise=noise)
    
        x = tokenizer.decode(x.view(B * T, N, D).permute(0, 2, 1), shape=(1, 24576 * cut_seconds), n_steps=50).view(B, T, 1, 24576 * cut_seconds)
        recon = tokenizer.decode(recon.view(B * n_decoder_chunks, N, D).permute(0, 2, 1), shape=(1, 24576 * cut_seconds), n_steps=50).view(B, n_decoder_chunks, 1, 24576 * cut_seconds)
        random_recon = tokenizer.decode(random_recon.view(B * n_decoder_chunks, N, D).permute(0, 2, 1), shape=(1, 24576 * cut_seconds), n_steps=50).view(B, n_decoder_chunks, 1, 24576 * cut_seconds)
    
    x = x.cpu().detach().float().numpy().squeeze(-2)
    recon = recon.cpu().detach().float().numpy().squeeze(-2)
    random_recon = random_recon.cpu().detach().float().numpy().squeeze(-2)
    
    for i in range(20):
        og, y, random_y, r = x[i], recon[i], random_recon[i], ratio[i].cpu().detach().numpy()
        tail_r = r[-n_decoder_chunks:]
        
        og_wav = np.concatenate([restore_measure(og[j], r[j].item()) for j in range(n_chunks)])
        recon_wav = np.concatenate([restore_measure(y[j], tail_r[j].item()) for j in range(n_decoder_chunks)])
        random_wav = np.concatenate([restore_measure(random_y[j], tail_r[j].item()) for j in range(n_decoder_chunks)])
        
        if n_encoder_chunks > 0:
            base = np.concatenate([restore_measure(og[j], r[j].item()) for j in range(n_encoder_chunks)])
            recon_wav = np.concatenate([base, recon_wav])
            random_wav = np.concatenate([base, random_wav])
        
        # save .wavs
        sf.write(os.path.join(batch_dir, f'{i}_real.wav'), og_wav, 16000)
        sf.write(os.path.join(batch_dir, f'{i}_recon.wav'), recon_wav, 16000)
        
        T = len(og_wav) / 16000
        t = np.linspace(0, T, len(og_wav), endpoint=False)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
        
        # Real
        axes.ravel()[0].plot(t, og_wav)
        axes.ravel()[0].set_title('Real Waveform')
        axes.ravel()[0].set_xlabel('Time [s]')
        axes.ravel()[0].set_ylabel('Amplitude')
        axes.ravel()[0].set_xlim(0, T)
        
        frequencies, times, Sxx = signal.spectrogram(og_wav, 16000)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)

        pcm = axes.ravel()[2].pcolormesh(times, frequencies, Sxx_log, shading='gouraud', cmap='viridis')
        axes.ravel()[2].set_title('Real Spectrogram')
        axes.ravel()[2].set_xlabel('Time [s]')
        axes.ravel()[2].set_ylabel('Frequency [Hz]')
        fig.colorbar(pcm, ax=axes.ravel()[2], label='Intensity [dB]')
        
        # Reconstruction
        axes.ravel()[1].plot(t, recon_wav)
        axes.ravel()[1].set_title('Reconstruction Waveform')
        axes.ravel()[1].set_xlabel('Time [s]')
        axes.ravel()[1].set_ylabel('Amplitude')
        axes.ravel()[1].set_xlim(0, T)
        
        frequencies, times, Sxx = signal.spectrogram(recon_wav, 16000)
        Sxx_log = 10 * np.log10(Sxx + 1e-10)

        pcm = axes.ravel()[3].pcolormesh(times, frequencies, Sxx_log, shading='gouraud', cmap='viridis')
        axes.ravel()[3].set_title('Reconstruction Spectrogram')
        axes.ravel()[3].set_xlabel('Time [s]')
        axes.ravel()[3].set_ylabel('Frequency [Hz]')
        fig.colorbar(pcm, ax=axes.ravel()[3], label='Intensity [dB]')

        plt.savefig(os.path.join(batch_dir, f'{i}_wavs.png'))
        plt.close('all')

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, ratio, bpm = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    tokens_trained += batch_size * gradient_accumulation_steps * max_seq_len

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        entropy, batch_entropy, usage = estimate_style_entropy()
        losses = estimate_loss()
        print(f"iter {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}")
        print(f"iter {iter_num}: train entropy {entropy['train']:.6f}, val entropy {entropy['val']:.6f}, train usage {usage['train']:.6f}, val usage {usage['val']:.6f}, train batch entropy {batch_entropy['train']:.6f}, val batch entropy {batch_entropy['val']:.6f}")
        if iter_num % sample_interval == 0 and master_process:
            model.eval()
            with ctx:
                generate_lam_actions(iter_num)
                generate_lam_vs_random_actions(iter_num)
            model.train()
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "train/entropy": entropy['train'],
                "val/entropy": entropy['val'],
                "train/usage": usage['train'],
                "val/usage": usage['val'],
                "train/batch_entropy": batch_entropy['train'],
                "val/batch_entropy": batch_entropy['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "tokens": tokens_trained,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
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
    
    if iter_num == 0 and eval_only:
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
            loss, _ = model(X, bpm)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, ratio, bpm = get_batch('train')
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