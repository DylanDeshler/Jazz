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

from lapa import LAM_B as net
from dito import DiToV4 as Tokenizer

import matplotlib.pyplot as plt
import soundfile as sf

from transformers import Wav2Vec2FeatureExtractor, AutoModel
import torch
import torchaudio
from sklearn.metrics.pairwise import paired_cosine_distances

emb_model_id = "m-a-p/MERT-v1-330M"
emb_model = AutoModel.from_pretrained(emb_model_id, trust_remote_code=True)
processor = Wav2Vec2FeatureExtractor.from_pretrained(emb_model_id, trust_remote_code=True)

resampler = torchaudio.transforms.Resample(16000, 24000)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'LAPA_B_FSQ'
eval_interval = 5000
sample_interval = 5000
log_interval = 100
save_interval = 5000
eval_iters = 200
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
batch_size = 192# * 5 * 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
# model
temporal_window = 2
spatial_window = 32
decoder_window = 32
cut_seconds = 1
cut_len = decoder_window * cut_seconds
max_seq_len = temporal_window * cut_len
vae_embed_dim = 16
levels = [8, 8]
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
def get_batch(split='train'):
    if split == 'train':
        data = np.memmap('/home/dylan.d/research/music/Jazz/latents/low_large_train.bin', dtype=np.float32, mode='r', shape=(204654816, vae_embed_dim))
        idxs = torch.randint(len(data) - max_seq_len, (batch_size,))
        x = torch.from_numpy(np.stack([np.stack([data[idx+i*spatial_window:idx+(i+1)*spatial_window] for i in range(temporal_window)], axis=0) for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
        return x
    
    else:
        data = np.memmap('/home/dylan.d/research/music/Jazz/latents/low_large_val.bin', dtype=np.float32, mode='r', shape=(4446944, vae_embed_dim))
        idxs = torch.randint(len(data) - max_seq_len, (batch_size,))
        x = torch.from_numpy(np.stack([np.stack([data[idx+i*spatial_window:idx+(i+1)*spatial_window] for i in range(temporal_window)], axis=0) for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
        return x

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

ckpt_path = os.path.join('tokenizer_low_large', 'ckpt.pt')
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

model_args = dict(in_channels=vae_embed_dim, levels=levels, spatial_window=spatial_window, temporal_window=temporal_window)
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
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for i, split in enumerate(['train', 'val']):
        losses = torch.zeros(eval_iters * gradient_accumulation_steps)
        for k in tqdm(range(eval_iters * gradient_accumulation_steps)):
            X = get_batch(split)
            with ctx:
                loss, _ = model(X)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def estimate_codebook_usage():
    out = {}
    model.eval()
    for i, split in enumerate(['train', 'val']):
        usage = torch.zeros(eval_iters * gradient_accumulation_steps)
        for k in tqdm(range(eval_iters * gradient_accumulation_steps)):
            X = get_batch(split)
            with ctx:
                _, indices = model(X)
                
                indices = indices.flatten()
                num_tokens = indices.numel()
            
                counts = torch.bincount(indices, minlength=math.prod(levels)).float()
                probs = counts / num_tokens
                
                # Add epsilon to avoid log(0)
                probs = probs + 1e-10
                entropy = -torch.sum(probs * torch.log(probs))
                perplexity = torch.exp(entropy).item()
            
            usage[k] = perplexity
        out[split] = usage.mean()
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
def generate_lam_vs_random_actions(step):
    batch_dir = os.path.join(out_dir, str(step))
    os.makedirs(batch_dir, exist_ok=True)
    
    x = get_batch('val')

    B, T, N, D = x.shape

    with ctx:
        recon, random_recon = raw_model.lam_vs_random_actions(x.clone(), n_steps=50)
    
    if decoder_window > spatial_window:
        raise NotImplementedError()
        t2 = decoder_window // spatial_window
        t1 = (temporal_window // t2) + (n_autoregressive_steps // t2)
        x = rearrange(x, 'b (t1 t2) n c -> b t1 (t2 n) c', t1=t1, t2=t2)
        recon = rearrange(recon, 'b (t1 t2) n c -> b t1 (t2 n) c', t1=t1, t2=t2)
        random_recon = rearrange(random_recon, 'b (t1 t2) n c -> b t1 (t2 n) c', t1=t1, t2=t2)
        
        B, T, N, D = x.shape
    
    out = tokenizer.decode(torch.cat([x[:, 1], recon, random_recon], dim=0).permute(0, 2, 1), shape=(1, 16384 * cut_seconds), n_steps=50)
    x, recon, random_recon = [res.cpu().detach().numpy().squeeze(1) for res in out.split(B, dim=0)]

    recon_psnr = psnr(x, recon)
    random_psnr = psnr(x, random_recon)

    for i in range(20):
        og, y, random_y = x[i], recon[i], random_recon[i]

        # save .wavs
        sf.write(os.path.join(batch_dir, f'{i}_real.wav'), og, 16000)
        sf.write(os.path.join(batch_dir, f'{i}_recon.wav'), y, 16000)
        sf.write(os.path.join(batch_dir, f'{i}_random_actions.wav'), random_y, 16000)
    
    x = resampler(torch.from_numpy(x))
    recon = resampler(torch.from_numpy(recon))
    random_recon = resampler(torch.from_numpy(random_recon))
    print(x.shape, recon.shape, random_recon.shape)

    x_inputs = processor(x, sampling_rate=24000, return_tensors="pt")
    recon_inputs = processor(recon, sampling_rate=24000, return_tensors="pt")
    random_recon_inputs = processor(random_recon, sampling_rate=24000, return_tensors="pt")
    with torch.no_grad():
        x_emb = emb_model(**x_inputs, output_hidden_states=True).last_hidden_state.mean(dim=1)
        recon_emb = emb_model(**recon_inputs, output_hidden_states=True).last_hidden_state.mean(dim=1)
        random_recon_emb = emb_model(**random_recon_inputs, output_hidden_states=True).last_hidden_state.mean(dim=1)

    recon_sim = 1 - paired_cosine_distances(x_emb, recon_emb)
    random_sim = 1 - paired_cosine_distances(x_emb, random_recon_emb)
    
    return {'PSNR': np.mean(recon_psnr - random_psnr).item(), 'Similarity': np.mean(recon_sim - random_sim).item()}

def psnr(y_true, y_pred, max_val=1.):
    mse = np.mean((y_true - y_pred) ** 2, axis=1)  # (B,)
    res = 10 * np.log10((max_val ** 2) / (mse + 1e-8))
    return res

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
        usage = estimate_codebook_usage()
        losses = estimate_loss()
        if iter_num % sample_interval == 0 and master_process:
            model.eval()
            with ctx:
                metrics = generate_lam_vs_random_actions(iter_num)
            model.train()
            print(f"iter {iter_num}: delta PSNR {metrics['PSNR']:.3f}, delta Similarity {metrics['Similarity']:.3f}")
        print(f"iter {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}, train perplexity: {usage['train']:.2f}, val perplexity {usage['val']:.2f}")
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

                if iter_num % save_interval == 0 and master_process == 0:
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
            loss, _ = model(X)
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