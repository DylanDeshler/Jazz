import torch
import torch.nn.functional as F

import os
import csv
import math
import time
import numpy as np
import soundfile as sf
from torchinfo import summary
from fad import MultiTaskFAD as net

import pickle
import pandas as pd
from tqdm import tqdm
from contextlib import nullcontext
from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from sklearn.preprocessing import MultiLabelBinarizer
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'FAD'
eval_interval = 5000
sample_interval = 5000
log_interval = 100
save_interval = 5000
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = out_dir
wandb_run_name = str(time.time())
# data
dataset = ''
gradient_accumulation_steps = 1
batch_size = 256
# model
n_samples = 16383 * 5
sample_rate = 16000
n_fft = 1024
hop_length = 512
n_mels = 192
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
device = 'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

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

def create_gaussian_soft_labels(targets, bins, sigma):
    valid_mask = (~torch.isnan(targets)) & (targets > 0)
    targets = torch.where(valid_mask, targets, bins[0])
    
    targets = targets.unsqueeze(1)
    bins = bins.unsqueeze(0).to(targets.device)
    
    squared_distances = (bins - targets) ** 2
    gaussian_weights = torch.exp(-squared_distances / (2 * sigma ** 2))
    soft_labels = gaussian_weights / (gaussian_weights.sum(dim=1, keepdim=True) + 1e-8)
    
    return soft_labels, valid_mask

def read_beat_timestamps(tsv_path):
    """Reads the beat_this TSV file and extracts a list of timestamps."""
    timestamps = []
    try:
        with open(tsv_path, 'r') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            for row in reader:
                if row:
                    try:
                        timestamps.append(float(row[0]))
                    except ValueError:
                        continue
        return timestamps
    except Exception as e:
        print(f"Error reading TSV file: {e}")
        return []

def calculate_subset_bpm(timestamps, start_time, end_time):
    """Calculates the BPM for a specific time window given a list of beat timestamps."""
    subset_beats = [t for t in timestamps if start_time <= t <= end_time]
    
    if len(subset_beats) < 2:
        return 0.0
    
    num_intervals = len(subset_beats) - 1
    duration_of_intervals = subset_beats[-1] - subset_beats[0]
    
    if duration_of_intervals == 0:
        return 0.0
        
    bpm = (num_intervals / duration_of_intervals) * 60.0
    return bpm

import glob
import librosa
paths = glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean/*.wav')
wavs = []

from sklearn.model_selection import StratifiedGroupKFold
kf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=0)

year_bins = [(year // 10) * 10 for year in years]
strat_key = [f'{year}_{record}' for year, record in zip(years, record_label_names)]

df = pd.DataFrame(strat_key, columns=['key'])
key_counts = df.value_counts()
rare_keys = key_counts[key_counts < 2].index
df.loc[df['key'].isin(rare_keys), 'key'] = 'rare_combo'
print(len(paths), df.shape, len(artists))
train_idx, test_idx = next(kf.split(np.arange(len(paths))[:, np.newaxis], df['key'], artists))
print(train_idx.shape, test_idx.shape)

import concurrent.futures
from multiprocessing import cpu_count
with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count() // 2) as executor:
    future_to_path = [
        executor.submit(lambda x: librosa.load(x, sr=sample_rate)[0], path) for path in paths
    ]
    
    for future in tqdm(concurrent.futures.as_completed(future_to_path), desc='Loading wav files', total=len(paths)):
        wav = future.result()
        wavs.append(wav)

def get_batch(split='train', batch_size=batch_size):
    if split == 'train':
        idxs = np.random.randint(low=0, high=int(0.98 * len(paths)), size=batch_size)
    else:
        idxs = np.random.randint(low=int(0.98 * len(paths)), high=len(paths), size=batch_size)
    
    x = []
    bpm = []
    year = []
    label = []
    inst = []
    for idx in idxs:
        wav = wavs[idx]
        beat_path = paths[idx].replace('jazz_data_16000_full_clean', 'jazz_data_16000_full_clean_beats').replace('.wav', '.beats')
        url = paths[idx].split('/')[-1].split('.')[0]
        
        start = np.random.randint(len(wav) - n_samples)
        timestamps = read_beat_timestamps(beat_path)
        label.append(record_labels[url_map[url]])
        year.append(years[url_map[url]])
        inst.append(instrument_labels[url_map[url]])
        bpm.append(calculate_subset_bpm(timestamps, start / sample_rate, (start + n_samples) / sample_rate))
        x.append(wav[start:start+n_samples])
        
    x = torch.from_numpy(np.asarray(x).astype(np.float32)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
    bpm = torch.from_numpy(np.asarray(bpm).astype(np.float32)).pin_memory().to(device, non_blocking=True)
    bpm, bpm_mask = create_gaussian_soft_labels(bpm, bpm_bins, bpm_sigma)
    label = torch.from_numpy(np.asarray(label).astype(np.float32)).pin_memory().to(device, non_blocking=True)
    year = torch.from_numpy(np.asarray(year).astype(np.float32)).pin_memory().to(device, non_blocking=True)
    year, year_mask = create_gaussian_soft_labels(year, year_bins, year_sigma)
    inst = torch.from_numpy(np.asarray(inst).astype(np.float32)).pin_memory().to(device, non_blocking=True)
    
    targets = {
        'bpm': bpm,
        'year': year,
        'label': label,
        'inst': inst
    }
    masks = {
        'bpm': bpm_mask,
        'year': year_mask
    }
    return x, targets, masks

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

num_instruments = instrument_labels.shape[1]
num_labels = record_labels.shape[1]
num_bpm_bins = len(bpm_bins)
num_year_bins = len(year_bins)

model_args = dict(
    num_instruments=num_instruments,
    num_labels=num_labels,
    bpm_bins=num_bpm_bins,
    year_bins=num_year_bins,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
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

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for i, split in enumerate(['train', 'val']):
        losses = defaultdict(lambda: torch.zeros(eval_iters))
        for k in tqdm(range(eval_iters)):
            X, targets, masks = get_batch(split, batch_size=batch_size * gradient_accumulation_steps)
            with ctx:
                loss = model(X, targets, masks)['loss']
            for key, value in loss.items():
                losses[key][k] = value.item()
        out[split] = {key: value.mean() for key, value in losses.items()}
    model.train()
    return out

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


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, targets, masks = get_batch('train') # fetch the very first batch
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
    
    tokens_trained += batch_size * gradient_accumulation_steps

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"iter {iter_num}: train loss {losses['train']['total']:.6f}, val loss {losses['val']['total']:.6f}")
        
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "tokens": tokens_trained,
            }
            log_dict = log_dict | {f'train/{k}': v for k, v in losses['train'].items()}
            log_dict = log_dict | {f'val/{k}': v for k, v in losses['val'].items()}
            wandb.log(log_dict)
            
        best_val_loss = losses['val']['total']
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config,
            'tokens': tokens_trained,
        }
        if iter_num > 0 and losses['val']['total'] < best_val_loss:
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if iter_num > 0 and always_save_checkpoint:
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
            loss = model(X, targets, masks)['loss']['total']
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, targets, masks = get_batch('train')
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
