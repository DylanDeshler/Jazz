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

from slot_diffusion import SADiffusion as net
import soundfile as sf

import pyrubberband as pyrb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'slot_diffusion'
eval_interval = 5000
sample_interval = 5000
log_interval = 100
save_interval = 5000
eval_iters = 600
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = out_dir
wandb_run_name = str(time.time())
# data
dataset = ''
gradient_accumulation_steps = 1
batch_size = 128
# model
n_samples = 16000
depth = 16
hidden_size = 1024
num_heads = 16
max_seq_len = 256
patch_size = 8

encoder_dict = dict(
    in_channels=1,
    max_seq_len=max_seq_len,
    patch_size=patch_size,
    depth=depth,
    hidden_size=hidden_size,
    num_heads=num_heads
)

resolution=patch_size
sample_rate=16000
n_fft=1024
hop_length=512
n_mels=192
num_slots=16
slot_size=128
slot_mlp_size=256
num_iterations=3

decoder_dict = dict(
    in_channels=1,
    slot_size=slot_size,
    max_seq_len=max_seq_len,
    patch_size=patch_size,
    depth=depth,
    hidden_size=hidden_size,
    num_heads=num_heads
)
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
compile = False # use PyTorch 2.0 to compile the model to be faster
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
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size
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

def get_batch(split='train', batch_size=batch_size):
    data = np.memmap("/home/dylan.d/research/music/Jazz/wavs_16khz.bin", dtype=np.float32, mode='r')
    if split == 'train':
        idxs = torch.randint(int(len(data) * 0.98) - n_samples, (batch_size,))
    else:
        idxs = torch.randint(int(len(data) * 0.98), len(data) - n_samples, (batch_size,))
        
    x = torch.from_numpy(np.stack([data[idx:idx+n_samples] for idx in idxs], axis=0).astype(np.float32)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
    # x = torch.randn_like(x)
    return x

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

model_args = dict(
    encoder_dict=encoder_dict,
    resolution=patch_size,
    sample_rate=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
    num_slots=num_slots,
    slot_size=slot_size,
    slot_mlp_size=slot_mlp_size,
    num_iterations=num_iterations,
    decoder_dict=decoder_dict,
)

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
    model = torch.compile(model)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    # model.eval()
    for i, split in enumerate(['train', 'val']):
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters)):
            X = get_batch(split, batch_size=batch_size * gradient_accumulation_steps)
            with ctx:
                loss = model(X)['loss']
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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

def save_samples(iter_num):
    batch_dir = os.path.join(out_dir, str(iter_num))
    os.makedirs(batch_dir, exist_ok=True)
    
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().squeeze()
        return x
    
    X = get_batch('valid', 32)
    out = model(X)
    
    gt = model.to_mel(X)
    masks = out['masks'].argmax(dim=1)
    samples = out['samples']
    
    print(gt.shape, masks.shape, samples.shape)
    
    gts = to_numpy(gt)
    recons = to_numpy(samples)
    mask_nps = to_numpy(masks)
    
    for gt, recon, mask_np in zip(gts, recons, mask_nps):
        fig, axes = plt.subplots(1, 3, figsize=(16, 12))
        
        plot_kwargs = {'origin': 'lower', 'aspect': 'auto', 'interpolation': 'nearest'}
        
        vmin = min(gt.min(), recon.min())
        vmax = max(gt.max(), recon.max())

        # --- Panel 1: Ground Truth ---
        im1 = axes[0].imshow(gt, cmap='magma', vmin=vmin, vmax=vmax, **plot_kwargs)
        axes[0].set_title("Ground Truth")
        axes[0].set_ylabel("Mel Bins")
        axes[0].set_xlabel("Time Frames")
        plt.colorbar(im1, ax=axes[0], format='%+2.0f dB')

        # --- Panel 2: Reconstruction ---
        im2 = axes[1].imshow(recon, cmap='magma', vmin=vmin, vmax=vmax, **plot_kwargs)
        axes[1].set_title("Reconstruction")
        axes[1].set_xlabel("Time Frames")
        axes[1].set_yticks([]) # Hide Y-axis labels for cleanliness
        plt.colorbar(im2, ax=axes[1], format='%+2.0f dB')

        # --- Panel 3: Reconstruction + Masks ---
        axes[2].imshow(recon, cmap='gray', vmin=vmin, vmax=vmax, **plot_kwargs)
        
        # Define a distinct color palette for masks (Red, Blue, Green, Yellow...)
        # Using matplotlib's 'tab10' qualitative colormap
        colors = plt.cm.tab10.colors 
        legend_patches = []

        # Overlay each mask layer
        for i in range(mask_np.shape[0]):
            current_mask = mask_np[i]
            
            # Only process if mask is not empty
            if np.sum(current_mask) > 0:
                # Create an RGBA image for this mask
                # H, W = height, width
                H, W = current_mask.shape
                color_rgba = np.zeros((H, W, 4))
                
                # Assign the specific color to this mask index
                c = colors[i % len(colors)]
                
                # Paint the color (R, G, B)
                color_rgba[..., 0] = c[0]
                color_rgba[..., 1] = c[1]
                color_rgba[..., 2] = c[2]
                
                # Set Alpha (Transparency): 0.0 where no mask, 0.4 where mask exists
                color_rgba[..., 3] = np.where(current_mask > 0, 0.4, 0.0)
                
                # Overlay it
                axes[2].imshow(color_rgba, origin='lower', aspect='auto')
                
                # Add to legend
                legend_patches.append(mpatches.Patch(color=c, label=f'Mask {i+1}', alpha=0.6))

        axes[2].set_title("Recon + Mask Overlay")
        axes[2].set_xlabel("Time Frames")
        axes[2].set_yticks([])
        
        # Add a legend if we have masks
        if legend_patches:
            axes[2].legend(handles=legend_patches, loc='upper right', framealpha=0.9)

        plt.tight_layout()
        
        plt.savefig(os.path.join(batch_dir, f'{j}.png'), dpi=150)
            

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X = get_batch('train') # fetch the very first batch
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
        # losses = estimate_loss()
        # print(f"iter {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}")
        
        if iter_num % sample_interval == 0 and master_process:
            model.eval()
            with ctx:
                save_samples(iter_num)
            model.train()
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
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
            loss = model(X)['loss']
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