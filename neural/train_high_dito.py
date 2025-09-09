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
import shutil
import pickle
from contextlib import nullcontext
from tqdm import tqdm
from torchinfo import summary

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from dito import DiToV2 as Transformer

import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import glob

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'tokenizer_high3'
eval_interval = 1000
log_interval = 100
save_interval = eval_interval * 10
eval_iters = 100
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = out_dir #'zinc20++'
wandb_run_name = 'llama' + str(time.time())
# data
dataset = ''
gradient_accumulation_steps = 2 # used to simulate larger batch sizes
batch_size = 16 # if gradient_accumulation_steps > 1, this is the micro-batch size
# model
rate = 16000
n_samples = rate * 2 # 5
# adamw optimizer
learning_rate = 1e-4 # max learning rate
max_iters = 1000000 # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = False # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = learning_rate / 10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'gloo' # 'nccl', 'gloo', etc.
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
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), os.path.join(out_dir, os.path.basename(os.path.abspath(__file__))))
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
paths = glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean/*.wav')
print(len(paths))

def get_batch(split='train'):
    if split == 'train':
        idxs = torch.randint(int(len(paths) * 0.98), (batch_size,))
        samples = [paths[idx] for idx in idxs]
        batch = []
        for sample in samples:
            x, sr = librosa.load(sample, sr=None)
            assert sr == rate

            start = np.random.randint(len(x) - n_samples)
            batch.append(x[start:start + n_samples])
        batch = torch.from_numpy(np.stack(batch, axis=0)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
        return batch
    
    else:
        idxs = torch.randint(int(len(paths) * 0.98), len(paths), (batch_size,))
        samples = [paths[idx] for idx in idxs]
        batch = []
        for sample in samples:
            x, sr = librosa.load(sample, sr=None)
            assert sr == rate

            start = np.random.randint(len(x) - n_samples)
            batch.append(x[start:start + n_samples])
        batch = torch.from_numpy(np.stack(batch, axis=0)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
        return batch

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

def _extract_state_dict(checkpoint):
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict

def average_checkpoints(paths):
    sd0 = _extract_state_dict(torch.load(paths[0], map_location='cpu'))
    dtypes0 = {k: v.dtype for k, v in sd0.items()}
    acc = {}

    # Float tensors: accumulate in fp32; non-floats: copy from first
    for k, v in sd0.items():
        acc[k] = v.float().clone() if torch.is_floating_point(v) else v.clone()

    # Add remaining checkpoints
    for p in paths[1:]:
        sdi = _extract_state_dict(torch.load(p, map_location='cpu'))
        for k, v in sdi.items():
            if torch.is_floating_point(v):
                acc[k].add_(v.float())
            # non-floats ignored (kept from first)

    n = float(len(paths))
    # Finalize: average floats and cast back to original dtype
    for k in acc:
        if torch.is_floating_point(acc[k]):
            acc[k].div_(n)
            acc[k] = acc[k].to(dtypes0[k])
    return acc

model_args = dict(z_shape=(256, 50), n_residual_layers=3, lstm=0, transformer=1, dimension=256, n_filters=32, ratios=[8, 5, 5, 2, 2, 2], channels=[128, 256, 512, 640, 768], dilation_base=3, down_proj=1024)
# model_args = dict(z_shape=(256, 50), n_residual_layers=2, lstm=0, transformer=1, dimension=256, n_filters=64, ratios=[8, 5, 5, 4, 2], c0=192, c1=384, c2=768)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = Transformer(**model_args)
    tokens_trained = 0
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']

    model = Transformer(**model_args)
    if eval_only:
        model.load_state_dict(average_checkpoints([os.path.join(out_dir, f'ckpt_{n}.pt') for n in [50000, 60000, 70000]]))
    else:
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
    model = Transformer.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

model.to(device)
summary(model)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
# optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile and 'cuda' in device:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters * gradient_accumulation_steps)
        for k in tqdm(range(eval_iters * gradient_accumulation_steps)):
            X = get_batch(split)
            with ctx:
                loss = model(X)
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


def save_samples(xs, ys, step):
    batch_dir = os.path.join(out_dir, str(step))
    os.makedirs(batch_dir, exist_ok=True)

    for i in range(min(8, xs.shape[0])):
        x, y = xs[i].squeeze(), ys[i].squeeze()

        # save .wavs
        sf.write(os.path.join(batch_dir, f'{i}_real.wav'), x, rate)
        sf.write(os.path.join(batch_dir, f'{i}_recon.wav'), y, rate)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
step1 = 3001
step2 = 5001
step3 = 8001
step4 = 15001

if eval_only:
    gradient_accumulation_steps *= 2
    model.eval()
    losses = estimate_loss()
    print(f"step {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}")
    import sys
    sys.exit()

X = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if iter_num == step1 or local_iter_num == 0 and iter_num >= step1:
        batch_size = 32
    if iter_num == step2 or local_iter_num == 0 and iter_num >= step2:
        gradient_accumulation_steps *= 2
    # if iter_num == step3 or local_iter_num == 0 and iter_num >= step3:
    #     gradient_accumulation_steps *= 2
    # if iter_num == step4 or local_iter_num == 0 and iter_num >= step4:
    #     gradient_accumulation_steps *= 2

    tokens_trained += batch_size * gradient_accumulation_steps

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        X = get_batch('test')
        model.eval()
        with ctx:
            logits = raw_model.reconstruct(X, n_steps=50)
        model.train()
        save_samples(X.cpu().detach().float().numpy(), logits.cpu().detach().float().numpy(), iter_num)
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}")
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

                if iter_num % save_interval == 0:
                    torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))

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