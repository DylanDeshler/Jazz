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

from dito import DiToV5 as Transformer

import pyrubberband as pyrb
import soundfile as sf
import librosa
import glob

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'tokenizer_low_measures_2std_subset'
eval_interval = 5000
log_interval = 100
eval_iters = 600
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = out_dir #'zinc20++'
wandb_run_name = 'llama' + str(time.time())
# data
dataset = ''
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 64 # if gradient_accumulation_steps > 1, this is the micro-batch size
# model
rate = 16000
n_samples = 24576
TARGET_SIG = 4
TARGET_BPM = 60 * TARGET_SIG / (n_samples / rate) # beats / (target samples / sample rate)
# adamw optimizer
learning_rate = 1e-4 # max learning rate
max_iters = 100000 # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 5000 # how many steps to warm up for
lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
min_lr = learning_rate / 10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
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

import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer

def parse_beat_file(beat_path):
    """
    Parses the beat_this output file.
    Expected format per line: <timestamp> <beat_number>
    
    Returns a list of dictionaries: {'time': float, 'beat': int}
    """
    beat_data = []
    with open(beat_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    ts = float(parts[0])
                    # specific beat number (1, 2, 3, 4)
                    # Default to 0 if not present
                    bn = 0 
                    if len(parts) >= 2:
                        try:
                            bn = int(float(parts[1]))
                        except ValueError:
                            pass
                    
                    beat_data.append({'time': ts, 'beat': bn})
                except ValueError:
                    continue
    
    return beat_data

def calculate_bpm(beat_path, index):
    beat_data = parse_beat_file(beat_path)
    
    downbeat_indices = [i for i, b in enumerate(beat_data) if b['beat'] == 1]
    
    start_idx = downbeat_indices[index]
    end_idx = downbeat_indices[index+1]
    
    t_start = beat_data[start_idx]['time']
    t_end = beat_data[end_idx]['time']
    
    frame_start = int(t_start * rate)
    frame_end = int(t_end * rate)
    
    duration_sec = (frame_end - frame_start) / rate
    instant_bpm = (TARGET_SIG / duration_sec) * 60
    
    return instant_bpm

bpm_bins = torch.arange(40, 300, 5, dtype=torch.float32)
year_bins = torch.arange(1900, 1980, 5, dtype=torch.float32)
bpm_sigma, year_sigma = 5, 2.5

cards = pickle.load(open('/home/dylan.d/research/music/Jazz/JazzSet.0.9.pkl', "rb"))
cards = [card for card in cards if card]

years = []
labels = []
people = []
instruments = []
artists = []
urls = []
for card in cards:
    urls.append(card['URLS'][0]['FILE'].split('/')[-2].split('.')[0])
    years.append(card['DATE']['YEAR'])
    labels.append(card['RECORD']['LABEL'])
    people.append(list(card['PERSONNEL']['PEOPLE'].keys()))
    instruments.append(list(card['PERSONNEL']['INSTRUMENTS'].keys()))
    artists.append(card['ARTIST'])

url_map = {url: i for i, url in enumerate(urls)}

# Record labels
record_label_names = set(list(pd.DataFrame(labels, columns=['LABEL']).value_counts(normalize=True).reset_index()['LABEL'].iloc[:25]))
record_label_names = [label if label in record_label_names else 'Other' for label in labels]
mlb = MultiLabelBinarizer().fit(record_label_names)
record_labels = mlb.transform(record_label_names)
record_labels = torch.from_numpy(record_labels)

# Instruments
instrument_map_df = pd.read_csv('/home/dylan.d/research/music/Jazz/instrument_mapping.csv')
instrument_map_df = instrument_map_df.apply(lambda col: col.astype(str).str.lower())
instrument_map = {row['Abbreviation']: row['Consolidated_Category'] for i, row in instrument_map_df.iterrows()}
instrument_categories = set(list(instrument_map.values()))

instrument_labels = defaultdict(list)
for instrument_list in instruments:
    categories = {instrument_map[l.lower()] for l in instrument_list if l in instrument_map}
    for cat in instrument_categories:
        if cat in categories:
            instrument_labels[cat].append(True)
        else:
            instrument_labels[cat].append(False)

props = pd.DataFrame(instrument_labels, columns=list(instrument_categories)).sum(0)
instrument_categories = {cat for cat in instrument_categories if props[cat] >= 1000}

instrument_labels = []
for instrument_list in instruments:
    categories = {instrument_map[l.lower()] for l in instrument_list if l in instrument_map}
    categories = categories.intersection(instrument_categories)
    instrument_labels.append(list(filter(None, categories)))

mlb = MultiLabelBinarizer().fit(instrument_labels)
instrument_labels = mlb.transform(instrument_labels)
instrument_labels = torch.from_numpy(instrument_labels)

import json
import glob
import librosa
paths = glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_measures/*.wav')
with open('/home/dylan.d/research/music/Jazz/valid_files_by_bpm.json', 'r') as f:
    beat_paths = json.load(f)
beat_paths = [os.path.join('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_beats', path) for path in beat_paths]
print(len(paths), len(beat_paths))
wavs = []

from sklearn.model_selection import StratifiedGroupKFold
kf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=0)

valid_idxs = []
for path in paths:
    url = path.split('/')[-1].split('.')[0]
    valid_idxs.append(url_map[url])

year_keys = [(year // 10) * 10 for year in years]
strat_key = [f'{year}_{record}' for i, (year, record) in enumerate(zip(year_keys, record_label_names)) if i in valid_idxs]
artists = [artist for i, artist in enumerate(artists) if i in valid_idxs]

df = pd.DataFrame(strat_key, columns=['key'])
key_counts = df.value_counts()
rare_keys = key_counts[key_counts < 2].index
df.loc[df['key'].isin(rare_keys), 'key'] = 'rare_combo'
train_idx, test_idx = next(kf.split(np.arange(len(paths))[:, np.newaxis], df['key'], artists))

import concurrent.futures
from multiprocessing import cpu_count
wavs = [None] * len(paths)

with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count() // 2) as executor:
    future_to_index = {
        executor.submit(lambda x: librosa.load(x, sr=rate)[0], path): i 
        for i, path in enumerate(paths)
    }
    
    for future in tqdm(concurrent.futures.as_completed(future_to_index), desc='Loading wav files', total=len(paths)):
        original_index = future_to_index[future]
        wav = future.result()
        wavs[original_index] = wav

durations = np.asarray([len(wav) for wav in wavs])
train_durations = durations[train_idx] / np.sum(durations[train_idx])
test_durations = durations[test_idx] / np.sum(durations[test_idx])

def get_batch(split='train'):
    if split == 'train':
        idxs = np.random.choice(train_idx, batch_size, p=train_durations).tolist()
    else:
        idxs = np.random.choice(test_idx, batch_size, p=test_durations).tolist()
    
    x = []
    for idx in idxs:
        wav = wavs[idx]
        
        start = np.random.randint((len(wav) // n_samples) - 1)
        x.append(wav[start * n_samples:(start + 1) * n_samples])
        
    x = torch.from_numpy(np.asarray(x).astype(np.float32)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
    return x

def get_meta_batch(split='train'):
    if split == 'train':
        idxs = np.random.choice(train_idx, batch_size, p=train_durations).tolist()
    else:
        idxs = np.random.choice(test_idx, batch_size, p=test_durations).tolist()
    
    x = []
    ratio = []
    for idx in idxs:
        wav = wavs[idx]
        beat_path = paths[idx].replace('jazz_data_16000_full_clean_measures', 'jazz_data_16000_full_clean_beats').replace('.wav', '.beats')
        
        start = np.random.randint((len(wav) // n_samples) - 1)
        instant_bpm = calculate_bpm(beat_path, start)
        
        ratio.append(TARGET_BPM / instant_bpm)
        x.append(wav[start * n_samples:(start + 1) * n_samples])
        
    x = torch.from_numpy(np.asarray(x).astype(np.float32)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
    ratio = torch.from_numpy(np.asarray(ratio).astype(np.float32)).pin_memory().to(device, non_blocking=True)
    return x, ratio

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

model_args = dict(z_shape=(16, 32), n_residual_layers=3, lstm=0, transformer=1, dimension=16, n_filters=32, ratios=[8, 4, 4, 2, 2], channels=[128, 256, 512, 512, 1024], dilation_base=2)
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
print('Encoder # params: ', sum(param.numel() for param in model.encoder.parameters()))
print('Decoder # params: ', sum(param.numel() for param in model.unet.parameters()))

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

def save_samples(step):
    batch_dir = os.path.join(out_dir, str(step))
    os.makedirs(batch_dir, exist_ok=True)
    
    X, ratios = get_meta_batch('test')
    model.eval()
    with ctx:
        Y = raw_model.reconstruct(X, n_steps=100)
    model.train()
    
    X = X.cpu().detach().float().numpy()
    Y = Y.cpu().detach().float().numpy()
    ratios = ratios.cpu().detach().numpy()

    for i in range(min(10, len(X))):
        x, y, ratio = X[i].squeeze(), Y[i].squeeze(), ratios[i].item()

        # save .wavs
        sf.write(os.path.join(batch_dir, f'{i}_real.wav'), restore_measure(x, ratio), rate)
        sf.write(os.path.join(batch_dir, f'{i}_recon.wav'), restore_measure(y, ratio), rate)

# logging
if wandb_log and master_process:
    import wandb
    if init_from == 'scratch':
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    elif init_from == 'resume':
        wandb.init(project=wandb_project, name=wandb_run_name, config=config, id='uqonnc7q', resume='must')

# training loop
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

    tokens_trained += batch_size * gradient_accumulation_steps

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        save_samples(iter_num)
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
            }
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
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
            }
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