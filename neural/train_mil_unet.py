import torch
import torch.nn.functional as F
import torchaudio.transforms as T

import os
import math
import time
import numpy as np
import soundfile as sf
from torchinfo import summary
from mil import UNet as net

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.ndimage
import scipy.signal
import h5py

import pickle
import pandas as pd
from tqdm import tqdm
from contextlib import nullcontext
from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'MIL_unet'
eval_interval = 2500
sample_interval = 2500
log_interval = 100
save_interval = 2500
eval_iters = 300
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = out_dir
wandb_run_name = str(time.time())
# data
dataset = ''
gradient_accumulation_steps = 2
batch_size = 64
# model
n_samples = 16383 * 5
sample_rate = 16000
n_fft = 1024
hop_length = 256
n_mels = 192
time_length = 32
frequency_length = 64
# adamw optimizer
learning_rate = 4e-3 * math.sqrt(batch_size / 4096) # max learning rate
max_iters = 1000000 # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.999
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
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

bpm_bins = torch.arange(40, 300, 5, dtype=torch.float32)
year_bins = torch.arange(1900, 1980, 5, dtype=torch.float32)
bpm_sigma, year_sigma = 5, 2.5

cards = pickle.load(open('/home/ubuntu/Data/JazzSet.0.9.pkl', "rb"))
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
instrument_map_df = pd.read_csv('/home/ubuntu/Data/instrument_mapping.csv')
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

# sax_high             20546
# strings                342
# organ                    4
# acoustic_guitar      24386
# flute                   48
# percussion               0
# vocals                  10
# electric_guitar          0
# double_bass          29035
# harmonica_whistle        0
# drums                29444
# trumpet              24199
# horns_other            283
# mallets_harp          1931
# trombone             21745
# keys                 34117
# reeds                19629
# sax_low              23665

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

import glob
import json
paths = glob.glob('/home/ubuntu/Data/measures/*')
with open('/home/ubuntu/Data/valid_files_by_bpm.json', 'r') as f:
    beat_paths = json.load(f)
paths = [os.path.join('/home/ubuntu/Data/wavs', os.path.basename(path)) for path in paths if os.path.basename(path) in beat_paths]
print(len(paths))

# from sklearn.model_selection import StratifiedGroupKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
kf = MultilabelStratifiedKFold(n_splits=20, shuffle=True, random_state=0)

valid_idxs = []
for path in paths:
    url = path.split('/')[-1].split('.')[0]
    valid_idxs.append(url_map[url])

year_keys = [(year // 10) * 10 for year in years]
strat_key = [f'{year}_{record}' for i, (year, record) in enumerate(zip(year_keys, record_label_names)) if i in valid_idxs]
artists = [artist for i, artist in enumerate(artists) if i in valid_idxs]
train_idx, test_idx = next(kf.split(np.arange(len(paths))[:, np.newaxis], [inst for i, inst in enumerate(instrument_labels) if i in valid_idxs]))

durations = []
for path in tqdm(paths):
    durations.append(sf.info(path).duration)
durations = np.asarray(durations)
train_durations = durations[train_idx] / np.sum(durations[train_idx])
test_durations = durations[test_idx] / np.sum(durations[test_idx])
del durations

positive_thresholds = {
    0: 0.703,
    1: 0.652,
    2: 0.522,
    3: 0.845,
    4: 0.364,
    5: 0.629,
    6: 0.687,
    7: 0.746,
    8: 0.740,
    9: 0.740
}
negative_thresholds = {
    0: 0.303,
    1: 0.199,
    2: 0.291,
    3: 0.724,
    4: 0.152,
    5: 0.225,
    6: 0.238,
    7: 0.291,
    8: 0.280,
    9: 0.314
}
positive_thresholds = np.array([positive_thresholds[i] for i in range(len(mlb.classes_))])
negative_thresholds = np.array([negative_thresholds[i] for i in range(len(mlb.classes_))])

from concurrent.futures import ThreadPoolExecutor
import threading

# 1. Thread-local storage: 
# Opening an H5 file has overhead. We want each worker thread to open the 
# file EXACTLY ONCE and keep it open in the background for future batches.
thread_local = threading.local()

def get_h5_file(path):
    if not hasattr(thread_local, "h5_file"):
        # swmr=True (Single Writer Multiple Reader) prevents thread locking
        thread_local.h5_file = h5py.File(path, 'r', swmr=True, libver='latest')
    return thread_local.h5_file

# 2. The Worker Function (Runs in parallel)
def fetch_single_sample(idx, n_samples, pos_thresh, neg_thresh):
    h5_file = get_h5_file('/home/ubuntu/Data/MIL_labels.h5')
    dset = h5_file[str(idx)]
    
    # Calculate the random start BEFORE reading any data
    total_len = dset.shape[0]
    start = np.random.randint(0, total_len - n_samples)
    
    # DIRECT DISK SLICING: We only load the exact n_samples we need into RAM.
    # This prevents the disk from reading the whole 30 second song.
    chunk = dset[start : start + n_samples].astype(np.float32)
    wav = chunk[:, 0]
    labels = chunk[:, 1:]
    
    # VECTORIZED THRESHOLDING: 
    # Instead of a Python for-loop, we do this across the entire matrix at C-speed.
    # Initialize a new array with our default "-1" (Ignore) state
    new_labels = np.full_like(labels, -1.0)
    
    # NumPy broadcasting allows us to compare the 2D labels against the 1D thresholds instantly
    pos_mask = labels > pos_thresh
    neg_mask = labels < neg_thresh
    
    new_labels[pos_mask] = 1.0
    new_labels[neg_mask] = 0.0
    
    return wav, new_labels

# 3. The Global Thread Pool
# Create this ONCE outside of your function.
executor = ThreadPoolExecutor(max_workers=16)

def get_batch(split='train'):
    if split == 'train':
        idxs = np.random.choice(train_idx, batch_size, p=train_durations).tolist()
    else:
        idxs = np.random.choice(test_idx, batch_size, p=test_durations).tolist()
    
    # 4. Multithreaded Mapping
    # This fires off all `batch_size` requests simultaneously.
    # The main thread waits here until all workers finish grabbing their samples.
    results = list(executor.map(
        lambda idx: fetch_single_sample(idx, n_samples, positive_thresholds, negative_thresholds), 
        idxs
    ))
    
    # Unpack the results
    x = [res[0] for res in results]
    inst = [res[1] for res in results]
    
    # Convert to Tensors and shoot to GPU
    x = torch.from_numpy(np.asarray(x)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
    inst = torch.from_numpy(np.asarray(inst)).pin_memory().to(device, non_blocking=True)
    
    return x, inst

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

num_instruments = len(mlb.classes_)
model_args = dict(
    num_instruments=num_instruments,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
    in_chans=1,
    # depths=[3, 3, 9, 3], dims=[64, 128, 256, 512],
    depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
    # depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
    drop_path_rate=0.1,
    num_heads=8,
    transformer_layers=2,
    time_length=time_length,
    frequency_length=frequency_length
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
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters)):
            X, targets = get_batch(split)
            with ctx:
                loss = model(X, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def estimate_mAP(split):
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for _ in tqdm(range(eval_iters)):
        X, targets = get_batch(split)
        with ctx:
            all_preds.append(model(X))
            all_targets.append(targets)
    model.train()
        
    # Concatenate all batches -> (Total_Items, Num_Classes, Samples)
    all_preds = torch.cat(all_preds, dim=0).cpu().detach().numpy()
    all_targets = torch.cat(all_targets, dim=0).cpu().detach().numpy()
    
    num_classes = all_targets.shape[-1]
    metrics = {}
    
    for c in range(num_classes):
        c_name = mlb.classes_[c]
        
        # Flatten the entire validation set for this specific class
        preds_c = all_preds[:, :, c].flatten()
        targets_c = all_targets[:, :, c].flatten()
        
        # Find exactly where the labels are NOT -1
        valid_mask = targets_c > -0.5
        
        # Protection against a class missing entirely from the validation batch
        if valid_mask.sum() == 0:
            continue
            
        valid_preds = preds_c[valid_mask]
        valid_targets = targets_c[valid_mask]
        
        ap = average_precision_score(valid_targets, valid_preds)
        metrics[c_name] = ap
    
    metrics['mAP'] = np.mean(list(metrics.values()))
    
    return metrics

@torch.no_grad()
def save_samples(iter_num):
    batch_dir = os.path.join(out_dir, str(iter_num))
    os.makedirs(batch_dir, exist_ok=True)
    
    X, targets = get_batch('val')
    X = X[:10]
    targets = targets[:10]
    B, _, total_samples = X.shape
    
    model.eval()
    with ctx:
        mels = model.to_mel(X)
        preds = model(X)
    model.train()
    
    X = X.cpu().numpy()
    mels = mels.cpu().numpy()
    targets = targets.cpu().numpy()
    preds = preds.cpu().numpy()
    print(X.shape, mels.shape, targets.shape, preds.shape)
    
    # Custom colormap for Ground Truth so -1 (Ignore) is visually distinct (Gray)
    # 0 = Blue (Negative), 0.5 = Gray (Ignore), 1 = Red (Positive)
    gt_cmap = LinearSegmentedColormap.from_list("gt_cmap", ["blue", "gray", "red"])
    
    # Loop through the requested number of samples in the batch
    for i in range(B):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # --- 1. Mel Spectrogram ---
        mel = mels[i, 0] # Drop the channel dimension
        im0 = axes[0].imshow(mel, aspect='auto', origin='lower', cmap='magma')
        axes[0].set_title(f"Sample {i}: Input Mel Spectrogram")
        axes[0].set_ylabel("Frequency (Mels)")
        
        # --- 2. Ground Truth ---
        # Remap targets so imshow handles the -1, 0, 1 scale perfectly
        gt = targets[i] # Shape: (20, Samples)
        im1 = axes[1].imshow(gt, aspect='auto', origin='lower', cmap=gt_cmap, vmin=-1, vmax=1)
        axes[1].set_title("Ground Truth Masks (-1=Gray, 0=Blue, 1=Red)")
        axes[1].set_ylabel("Instruments")
        
        # --- 3. Predictions ---
        pred = preds[i] # Shape: (20, Samples)
        im2 = axes[2].imshow(pred, aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=1)
        axes[2].set_title("Predicted Probabilities (0.0 to 1.0)")
        axes[2].set_ylabel("Instruments")
        axes[2].set_xlabel("Time (Raw Audio Samples)")
        
        # Note: We do not share the X-axis because axes[0] is in Mel Frames, 
        # while axes[1] and axes[2] are in raw audio samples.
        
        plt.tight_layout()
        plt.savefig(os.path.join(batch_dir, f"segmentation_eval_{i}.png"), dpi=150)
        plt.close()

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
X, targets = get_batch('train') # fetch the very first batch
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
        # losses = estimate_loss()
        # print(f"iter {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}")
        train_metrics = estimate_mAP('train')
        for k, v in train_metrics.items():
            print(f'train {k} = {v:.4f}')
        val_metrics = estimate_mAP('val')
        for k, v in val_metrics.items():
            print(f'val {k} = {v:.4f}')
        save_samples(iter_num)
        
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
                "tokens": tokens_trained,
                "train/loss": losses['train'],
                "val/loss": losses['val']
            }
            wandb.log(log_dict)
            
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
            loss = model(X, targets)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, targets = get_batch('train')
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
