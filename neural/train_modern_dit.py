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
import copy

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from einops import rearrange

from dito import DiToV5 as Tokenizer
import soundfile as sf

import argparse
parser = argparse.ArgumentParser(description="Process a specific level argument.")
valid_levels = [f"L{i}" for i in range(0, 6)]

parser.add_argument(
    '--level', 
    type=str, 
    required=True, 
    choices=valid_levels,
    help="Specify the level. Must be one of L1 through L5."
)
parser.add_argument(
    '--axis',
    type=str,
    required=True,
    choices=['width', 'depth'],
    help="Specify width or depth scaling experiment."
)
parser.add_argument(
    '--n_chunks',
    type=int,
    required=True,
    help="Specify the number of chunks defining the context length."
)
parser.add_argument(
    '--device', 
    type=str, 
    required=True,
    help="Specify the device. Like cuda:1."
)

args = parser.parse_args()

from diffusion_forcing import (
    UnconditionalModernDiT_smedium_W0, UnconditionalModernDiT_smedium_W1, UnconditionalModernDiT_smedium_W2, UnconditionalModernDiT_smedium_W3, UnconditionalModernDiT_smedium_W4, UnconditionalModernDiT_smedium_W5,
    UnconditionalModernDiT_smedium_D0, UnconditionalModernDiT_smedium_D1, UnconditionalModernDiT_smedium_D2, UnconditionalModernDiT_smedium_D3, UnconditionalModernDiT_smedium_D4, UnconditionalModernDiT_smedium_D5
)

net_map = {
    'W0': UnconditionalModernDiT_smedium_W0,
    'W1': UnconditionalModernDiT_smedium_W1,
    'W2': UnconditionalModernDiT_smedium_W2,
    'W3': UnconditionalModernDiT_smedium_W3,
    'W4': UnconditionalModernDiT_smedium_W4,
    'W5': UnconditionalModernDiT_smedium_W5,
    
    'D0': UnconditionalModernDiT_smedium_D0,
    'D1': UnconditionalModernDiT_smedium_D1,
    'D2': UnconditionalModernDiT_smedium_D2,
    'D3': UnconditionalModernDiT_smedium_D3,
    'D4': UnconditionalModernDiT_smedium_D4,
    'D5': UnconditionalModernDiT_smedium_D5
}
level = f'{args.axis[0].upper()}{args.level[1]}'
net = net_map[level]
n_chunks = args.n_chunks
print(f"Begining training for {level} on {args.device}")

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = f'UnconditionalModernDiT_smedium_{level}_24576_subset_longtrain_{n_chunks}chunks'
eval_interval = 5000
sample_interval = 10000
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
gradient_accumulation_steps = 1
batch_size = 64
if int(level[1]) > 4:
    gradient_accumulation_steps *= 2
    batch_size = batch_size // 2
    print(f'Big model {level} accumulating gradients!')
# model
patch_size = 2
gradient_checkpointing = False
# set to match with measures using n_chunks=24, spatial_window=64
spatial_window = 48
max_seq_len = spatial_window * n_chunks
vae_embed_dim = 16
n_style_embeddings = 256
style_dim = 128
use_null_token = True
cut_seconds = 1
# adamw optimizer
learning_rate = 1e-4 # max learning rate
max_iters = 100000 # total number of training iterations
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
device = args.device # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
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

def get_batch(split='train', batch_size=batch_size):
    if split == 'train':
        data = np.memmap('/data/binaries/low_large_24576_longtrain_train.bin', dtype=np.float32, mode='r', shape=(230917632, vae_embed_dim))
    else:
        data = np.memmap('/data/binaries/low_large_24576_longtrain_val.bin', dtype=np.float32, mode='r', shape=(4845696, vae_embed_dim))
    
    idxs = torch.randint(len(data) - n_chunks * spatial_window, (batch_size,))    
    x = torch.from_numpy(np.stack([data[idx:idx+n_chunks * spatial_window] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    x = x.view(batch_size, n_chunks, spatial_window, vae_embed_dim)

    return x

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

model_args = dict(in_channels=vae_embed_dim, style_dim=style_dim, n_chunks=n_chunks, spatial_window=spatial_window, use_null_token=use_null_token, gradient_checkpointing=gradient_checkpointing, patch_size=patch_size)

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
    tokens_trained = 0
    
    ema = EMAModel(model)
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
            X = get_batch(split, batch_size=batch_size * gradient_accumulation_steps)
            with ctx:
                loss = ema.ema_model(X)
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

@torch.no_grad()
def save_samples(step):
    """
    Generates samples with the same actions as the input audio
    """
    batch_dir = os.path.join(out_dir, str(step))
    os.makedirs(batch_dir, exist_ok=True)
    
    n_samples = 5
    n_steps = 100
    t_dist = 'logit'
    cfg_mode = 'joint'
    x = get_batch('val')[:n_samples]

    B, T, N, D = x.shape
    
    gen_noise = torch.randn(*x.shape).to(device)
    decoder_noise = torch.randn(n_samples * n_chunks, 1, 24576).to(device)
    
    with ctx:
        y = ema.ema_model.generate(x.shape, n_steps=n_steps, noise=gen_noise, memory_efficient=False, cfg_mode=cfg_mode, t_dist=t_dist)
        y = tokenizer.decode(y.view(B * T, N, D).permute(0, 2, 1), shape=(1, 24576 * cut_seconds), n_steps=n_steps, noise=decoder_noise).view(B, T, 1, 24576 * cut_seconds)
    
    y = y.cpu().detach().float().numpy().squeeze(-2)

    for i in range(n_samples):
        sf.write(os.path.join(batch_dir, f'{i}.wav'), y[i].flatten(), 16000)

# logging
if wandb_log and master_process:
    import wandb
    if init_from == 'scratch':
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    elif init_from == 'resume':
        wandb.init(project=wandb_project, name=wandb_run_name, config=config, id='to9c2olw', resume='must')

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
        if iter_num % sample_interval == 0 and master_process:
            save_samples(iter_num)
                
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
                'val_loss': best_val_loss,
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
            loss = model(X)
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