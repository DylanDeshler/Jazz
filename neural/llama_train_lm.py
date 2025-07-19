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

from llama_model import ModelArgs, VQTransformer4 as Transformer
checkpoint = torch.load('/Users/dylan.d/Documents/research/music/tokenizer7/ckpt.pt')
tokenizer_args = ModelArgs(**checkpoint['model_args'])
tokenizer = Transformer(tokenizer_args)
tokenizer.load_state_dict(checkpoint['model'])
tokenizer.eval()

rate = tokenizer_args.rate
n_fft = tokenizer_args.n_fft
win_length = tokenizer_args.win_length
hop_length = tokenizer_args.hop_length

from rq_llama_model import RQTransformer as Transformer, ModelArgs

import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import glob

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'rqtransformer'
eval_interval = 1000
log_interval = 10
eval_iters = 300
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = out_dir #'zinc20++'
wandb_run_name = 'llama' + str(time.time())
# data
dataset = ''
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 24# * 5 * 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 200
# model
depth_seq_len = 4
n_spatial_layer = 8
n_depth_layer = 3
n_head = 8
n_embd = 512
multiple_of = 128
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 2 * 1e-3 # max learning rate
max_iters = 200000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = learning_rate / 10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'mps' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
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
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
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

META_TOKEN_MAP = {
    'EOS': 8192,
    'PRE': 8193,
    'MID': 8194,
    'SUF': 8195
}

# poor man's data loader
def spm(x):
    
    META_TOKEN_MAP['PRE'] + META_TOKEN_MAP['SUF'] + suf + META_TOKEN_MAP['MID'] + pre + mid + META_TOKEN_MAP['EOS']

def psm(x):
    pass

def fim(x, prob=0.5):
    p = np.random.rand()
    if p < prob / 2:
        return spm(x)
    elif p < prob:
        return psm(x)
    return x

def get_batch(split='train'):
    data = np.memmap('/Users/dylan.d/Documents/research/music/jazz.bin', mode='r', dtype=np.uint16, shape=(6534430, 40, 12))
    if split == 'train':
        ix = torch.randint(int(len(data) * 0.98) - block_size // 40 - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(np.concatenate(data[i:i+block_size // 40, :, :depth_seq_len], axis=0).astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(np.concatenate(data[i+1:i+1+block_size // 40, :, :depth_seq_len], axis=0).astype(np.int64)) for i in ix]).to(device)

        return x, y
    
    else:
        ix = torch.randint(int(len(data) * 0.98) - block_size // 40 - 1, len(data) - block_size // 40 - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(np.concatenate(data[i:i+block_size // 40, :, :depth_seq_len], axis=0).astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(np.concatenate(data[i+1:i+1+block_size // 40, :, :depth_seq_len], axis=0).astype(np.int64)) for i in ix]).to(device)

        return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_vocab_size = 8192
# meta_path = os.path.join(data_dir, 'meta.pkl')
# meta_vocab_size = None
# if os.path.exists(meta_path):
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
#     meta_vocab_size = meta['vocab_size']
#     print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_spatial_layers=n_spatial_layer, n_depth_layers=n_depth_layer, n_heads=n_head, dim=n_embd, max_seq_len=block_size,
                  vocab_size=meta_vocab_size, dropout=dropout, depth_seq_len=depth_seq_len) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    tokens_trained = 0
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = checkpoint['model_args']

    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    # for k in ['n_layers', 'n_heads', 'dim', 'max_seq_len']:#, 'max_seq_len']:
    #     model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
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
# crop down the model block size if desired, using model surgery
if block_size < model.params.max_seq_len:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)
summary(model)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
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
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters)):
            X, Y = get_batch(split)
            with ctx:
                logits = model(X, return_loss=True)
                loss = raw_model.last_loss
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
def save_samples(iter_num):
    model.eval()

    X, Y = get_batch('val')
    X = X[:4, :80]
    X = X.view(X.shape[0], X.shape[1] * X.shape[2])
    with ctx:
        samples = model.generate(X, max_seq_len=160).cpu().detach()
        samples = tokenizer.decode(samples)
    
    for i in range(4):
        # sample = librosa.griffinlim(tokenizer.decode(model.generate(X, max_seq_len=160).cpu().detach()).cpu().detach().numpy(), n_iter=1000, hop_length=hop_length, win_length=win_length, n_fft=n_fft, init=None)
        sample = librosa.griffinlim(samples[i].cpu().detach().numpy(), n_iter=1000, hop_length=hop_length, win_length=win_length, n_fft=n_fft, init=None)

        # save .wavs
        sf.write(os.path.join(out_dir, f'{iter_num}_{i}.wav'), sample, rate)

        x = librosa.stft(sample, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        # Magnitude
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        mag, phase = np.abs(x), np.angle(x)
        img1 = librosa.display.specshow(librosa.amplitude_to_db(mag, ref=np.max),
                                sr=rate, hop_length=hop_length, y_axis='linear', x_axis='time',
                                cmap='magma')
        plt.title('Log-magnitude Spectrogram')
        plt.colorbar(img1, format="%+2.0f dB")

        freqs = librosa.fft_frequencies(sr=rate, n_fft=n_fft)
        times = librosa.times_like(x)

        # Phase
        plt.subplot(3, 1, 2)
        mag, phase = np.abs(x), np.angle(x)
        unwrapped_phase = np.unwrap(phase, axis=1)
        unwrapped_phase = np.diff(unwrapped_phase, prepend=0) * np.clip(np.abs(mag) / np.max(mag), 0.1, 1)
        librosa.display.specshow(unwrapped_phase, sr=rate, hop_length=hop_length,
                                x_axis='time', y_axis='linear', cmap='twilight_shifted')
        plt.title('Weighted Unwrapped Phase Spectrogram')
        plt.colorbar(label='Phase (radians)')

        phase_exp = 2*np.pi*np.multiply.outer(freqs,times)

        # Rainbowgram
        plt.subplot(3, 1, 3)
        mag, phase = np.abs(x), np.angle(x)
        img = librosa.display.specshow(np.diff(np.unwrap(np.angle(phase)-phase_exp, axis=1), axis=1, prepend=0),
                                cmap='hsv',
                                alpha=librosa.amplitude_to_db(mag, ref=np.max)/80 + 1,
                                y_axis='linear',
                                x_axis='time')
        # plt.facecolor('#000')
        cbar = plt.colorbar(img, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.ax.set(yticklabels=['-π', '-π/2', "0", 'π/2', 'π'])
        plt.title('Rainbowgram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        plt.savefig(os.path.join(out_dir, f'{iter_num}_{i}.png'))
        plt.close('all')

    model.train()

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
step1 = 2001
step2 = 8001
step3 = 20001
step4 = 38001
step5 = 50001

X, Y = get_batch('train') # fetch the very first batch
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
        # batch_size *= 4
    if iter_num == step3 or local_iter_num == 0 and iter_num >= step3:
        gradient_accumulation_steps *= 2
        # batch_size *= 4
    if iter_num == step4 or local_iter_num == 0 and iter_num >= step4:
        gradient_accumulation_steps *= 2
    if iter_num == step5 or local_iter_num == 0 and iter_num >= step5:
        gradient_accumulation_steps *= 2
    

    tokens_trained += block_size * batch_size * gradient_accumulation_steps

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        # save_samples(iter_num)    # breaks everything for some reason...
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
            logits = model(X, return_loss=True)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
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
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.6f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()