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

from dito import DiToTrainer as Transformer

import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import glob

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'tokenizer9'
eval_interval = 2000
log_interval = 100
eval_iters = 100
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = out_dir #'zinc20++'
wandb_run_name = 'llama' + str(time.time())
# data
dataset = ''
gradient_accumulation_steps = 2 # used to simulate larger batch sizes
batch_size = 12# * 5 * 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
# model
rate = 16000
n_samples = rate
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
        batch = torch.from_numpy(np.stack(batch, axis=0)).unsqueeze(1).to(device)
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
        batch = torch.from_numpy(np.stack(batch, axis=0)).unsqueeze(1).to(device)
        return batch

print(f"Loading from {out_dir}")
# resume training from a checkpoint.
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']

model = Transformer(**model_args)
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

model.to(device)
model.eval()

# compile the model
if compile and 'cuda' in device:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

def save_samples(xs, ys, samples, step):
    batch_dir = os.path.join(out_dir, str(step))
    os.makedirs(batch_dir, exist_ok=True)

    for i in range(8):
        x, y, sample = xs[i].squeeze(), ys[i].squeeze(), samples[i].squeeze()

        # save .wavs
        sf.write(os.path.join(batch_dir, f'{i}_real.wav'), x, rate)
        sf.write(os.path.join(batch_dir, f'{i}_recon.wav'), y, rate)
        sf.write(os.path.join(batch_dir, f'{i}_sample.wav'), sample, rate)

X = get_batch('test')
with torch.no_grad():
    with ctx:
        # logits, loss = model(X)
        logits = model.reconstruct(X, (batch_size, 1, n_samples), n_steps=50)
        samples = model.sample((batch_size, 1, n_samples), n_steps=50)

print('X: ', X.min().item(), X.mean().item(), X.std().item(), X.max().item())
print('Reconstruction: ', logits.min().item(), logits.mean().item(), logits.std().item(), logits.max().item())
print('Sample: ', samples.min().item(), samples.mean().item(), samples.std().item(), samples.max().item())

# save_samples(X.cpu().detach().float().numpy(), logits.cpu().detach().float().numpy(), samples.cpu().detach().float().numpy(), iter_num)
