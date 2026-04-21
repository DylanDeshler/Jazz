import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

import os
import math
import time
import numpy as np
import soundfile as sf
from torchinfo import summary
from mil import MIL as net

import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.signal

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
out_dir = 'MIL'
eval_interval = 2500
sample_interval = 2500
log_interval = 100
save_interval = 2500
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
batch_size = 8
# model
n_samples = 16383 * 30
sample_rate = 16000
n_fft = 1024
hop_length = 512
n_mels = 192
time_length=32*6
frequency_length=64
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
import librosa
paths = glob.glob('/home/ubuntu/Data/wavs/*.wav')
wavs = []

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
# train_idx, test_idx = next(kf.split(np.arange(len(paths))[:, np.newaxis], instrument_labels, artists))
train_idx, test_idx = next(kf.split(np.arange(len(paths))[:, np.newaxis], [inst for i, inst in enumerate(instrument_labels) if i in valid_idxs]))

import concurrent.futures
from multiprocessing import cpu_count
# wavs = [None] * len(paths)
durations = []

with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count() // 2) as executor:
    future_to_index = {
        executor.submit(lambda x: librosa.load(x, sr=sample_rate)[0], path): i 
        for i, path in enumerate(paths)
    }
    
    for future in tqdm(concurrent.futures.as_completed(future_to_index), desc='Loading wav files', total=len(paths)):
        original_index = future_to_index[future]
        wav = future.result()
        # wavs[original_index] = wav
        durations.append(len(wav))

durations = np.asarray(durations)
# durations = np.asarray([len(wav) for wav in wavs])
train_durations = durations[train_idx] / np.sum(durations[train_idx])
test_durations = durations[test_idx] / np.sum(durations[test_idx])

def get_batch(split='train'):
    if split == 'train':
        idxs = np.random.choice(train_idx, batch_size, p=train_durations).tolist()
    else:
        idxs = np.random.choice(test_idx, batch_size, p=test_durations).tolist()
    
    x = []
    inst = []
    for idx in idxs:
        # wav = wavs[idx]
        # wav = librosa.load(paths[idx], sr=None)[0]
        wav = torchaudio.load(paths[idx])[0]
        url = paths[idx].split('/')[-1].split('.')[0]
        
        start = np.random.randint(len(wav) - n_samples)
        inst.append(instrument_labels[url_map[url]])
        x.append(wav[start:start+n_samples])
        
    x = torch.from_numpy(np.asarray(x).astype(np.float32)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
    inst = torch.from_numpy(np.asarray(inst).astype(np.float32)).pin_memory().to(device, non_blocking=True)
    
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
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.1,
    num_heads=12,
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
def save_samples(iter_num, activation_threshold=0.5, min_length_sec=1.0, apply_post_processing=True, median_filter_size_frames=11):
    """
    Takes raw waveforms, generates Mels internally, runs the model, extracts dynamic 
    audio segments, and generates a two-tier temporally aligned plot.
    
    Args:
        model: Trained MIL model.
        preprocessor: The AudioPreprocessor module to format Mels for the model.
        raw_waveforms: (B, 1, Total_Samples) tensor.
        labels: (B, Num_Classes) binary ground truth tensor.
    """
    model.eval()
    batch_dir = os.path.join(out_dir, str(iter_num))
    os.makedirs(batch_dir, exist_ok=True)
    
    X, targets = get_batch('val')
    B, _, total_samples = X.shape
    total_duration_sec = total_samples / sample_rate
    min_length_samples = int(min_length_sec * sample_rate)
    
    # 1. Generate Mel Spectrograms
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=40.0,
        f_max=sample_rate // 2,
        power=2.0,
        normalized=True,      # Normalizes the STFT to be magnitude invariant
        center=True,          # Padding to keep time/length consistent
        pad_mode='reflect'    # Better for audio boundary artifacts
    ).to(device)
    amplitude_to_db = T.AmplitudeToDB(top_db=80.0).to(device)
    
    with ctx:
        raw_mels = amplitude_to_db(mel_transform(X))
        frame_probs = model(X)['frame_probs']
        
    frame_probs_np = frame_probs.cpu().numpy()
    
    # 4. Post-Processing
    if apply_post_processing:
        frame_probs_np = scipy.signal.medfilt(
            frame_probs_np, 
            kernel_size=(1, median_filter_size_frames, 1)
        )
        
    probs_tensor = torch.from_numpy(frame_probs_np).transpose(1, 2)
    
    # Interpolate to sample level
    sample_level_probs = F.interpolate(
        probs_tensor, size=total_samples, mode='linear', align_corners=False
    )
    
    for b in range(B):
        present_classes = torch.where(targets[b] == 1.0)[0].cpu().tolist()
        
        for c in present_classes:
            class_name = mlb.classes_[c]
            prob_curve = sample_level_probs[b, c, :].numpy()
            
            # --- Dynamic Extraction Logic ---
            binary_mask = (prob_curve >= activation_threshold).astype(int)
            labeled_mask, num_features = scipy.ndimage.label(binary_mask)
            
            best_start, best_end, highest_mean_prob = 0, 0, -1.0
            
            if num_features > 0:
                for feature_idx in range(1, num_features + 1):
                    block_indices = np.where(labeled_mask == feature_idx)[0]
                    start_idx, end_idx = block_indices[0], block_indices[-1]
                    
                    if (end_idx - start_idx) >= min_length_samples:
                        mean_prob = prob_curve[start_idx:end_idx].mean()
                        if mean_prob > highest_mean_prob:
                            highest_mean_prob = mean_prob
                            best_start, best_end = start_idx, end_idx
                            
            # Fallback if no block meets minimum length
            if highest_mean_prob == -1.0:
                peak_idx = np.argmax(prob_curve)
                half_window = min_length_samples // 2
                best_start = max(0, peak_idx - half_window)
                best_end = min(total_samples, peak_idx + half_window)
                if best_end - best_start < min_length_samples:
                    if best_start == 0: best_end = min(total_samples, min_length_samples)
                    else: best_start = max(0, total_samples - min_length_samples)
            
            # --- Save Audio Snippet ---
            audio_snippet = X[b, :, best_start:best_end]
            wav_path = os.path.join(batch_dir, f"{class_name}_{b}.wav")
            torchaudio.save(wav_path, audio_snippet.cpu(), sample_rate)
            
            # --- Two-Tier Aligned Plotting ---
            best_start_sec = best_start / sample_rate
            best_end_sec = best_end / sample_rate
            time_axis = np.linspace(0, total_duration_sec, total_samples)
            mel_to_plot = raw_mels[b, 0].cpu().numpy()
            
            # Create subplots sharing the X-axis (Time)
            fig, (ax_prob, ax_mel) = plt.subplots(
                2, 1, 
                figsize=(12, 6), 
                sharex=True, 
                gridspec_kw={'height_ratios': [1, 2.5]}
            )
            
            # Top Plot: Probability Curve
            ax_prob.plot(time_axis, prob_curve, color='#00aaff', linewidth=2)
            ax_prob.axhline(activation_threshold, color='red', linestyle='--', alpha=0.5, label='Threshold')
            ax_prob.axvspan(best_start_sec, best_end_sec, color='lime', alpha=0.2, label='Extracted Region')
            
            ax_prob.set_title(f"Detection Timeline: {class_name}", fontsize=12, fontweight='bold')
            ax_prob.set_ylabel("Probability", fontsize=10)
            ax_prob.set_ylim(0, 1.05)
            ax_prob.legend(loc="upper right", fontsize=9)
            ax_prob.grid(True, alpha=0.3)
            
            # Bottom Plot: Mel Spectrogram
            # Use 'extent' to map the 2D array exactly to physical seconds
            ax_mel.imshow(
                mel_to_plot, 
                aspect='auto', 
                origin='lower', 
                cmap='magma', 
                extent=[0, total_duration_sec, 0, 128]
            )
            ax_mel.axvspan(best_start_sec, best_end_sec, color='lime', alpha=0.2)
            
            ax_mel.set_ylabel("Mel Bins", fontsize=10)
            ax_mel.set_xlabel("Time (seconds)", fontsize=10)
            
            # Clean up layout to remove space between plots
            plt.subplots_adjust(hspace=0.05)
            
            plot_path = os.path.join(batch_dir, f"{class_name}_{b}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    model.train()

@torch.no_grad
def estimate_mAP(split):
    """
    Evaluates the MIL model on a validation set and calculates
    BCE Loss, Average Precision (AP) for each class, and the overall mAP.
    
    Args:
        model: The trained MusicTaggingMIL model.
        val_loader: DataLoader for your validation set.
        class_names: List of strings containing instrument names.
        device: 'cuda' or 'cpu'.
        
    Returns:
        mAP: The Mean Average Precision across all valid classes (float).
        class_metrics: Dictionary containing 'loss' and 'ap' for each class.
    """
    model.eval()
    
    all_clip_logits = []
    all_labels = []
    
    for _ in tqdm(range(eval_iters)):
        X, targets = get_batch(split)
        with ctx:
            clip_logits = model(X, targets)['clip_logits']
            all_clip_logits.append(clip_logits.cpu().detach())
            all_labels.append(targets.cpu().detach())
    model.train()
    
    # Concatenate all batches into single tensors
    all_clip_logits = torch.cat(all_clip_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 1. Calculate Per-Class BCE Loss
    raw_losses = F.binary_cross_entropy_with_logits(all_clip_logits, all_labels, reduction='none')
    per_class_loss = raw_losses.mean(dim=0).numpy()
    
    # 2. Calculate Per-Class Average Precision (AP)
    all_clip_probs = torch.sigmoid(all_clip_logits).numpy()
    all_labels_np = all_labels.numpy()
    
    per_class_ap = np.zeros(num_instruments)
    for c in range(num_instruments):
        # Edge case guard: AP is undefined if a class never appears in the validation set
        if np.sum(all_labels_np[:, c]) == 0:
            per_class_ap[c] = np.nan 
        else:
            per_class_ap[c] = average_precision_score(
                y_true=all_labels_np[:, c], 
                y_score=all_clip_probs[:, c],
                average='macro'
            )
            
    # 3. Calculate Mean Average Precision (mAP)
    valid_aps = per_class_ap[~np.isnan(per_class_ap)]
    map_score = np.mean(valid_aps) if len(valid_aps) > 0 else 0.0
    
    
    metrics = {}
    metrics['mAP'] = map_score
    
    for c in range(num_instruments):
        c_name = mlb.classes_[c]
        c_loss = per_class_loss[c]
        c_ap = per_class_ap[c]
        
        metrics[c_name] = {'loss': c_loss, 'ap': c_ap}
    
    return metrics

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters)):
            X, targets = get_batch(split)
            with ctx:
                loss = model(X, targets)['loss']
            losses[k] = loss.item()
        out[split] = losses.mean()
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
        losses = estimate_loss()
        print(f"iter {iter_num}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}")
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
            log_dict |= {f'train/{k}': v for k, v in train_metrics.items()}
            log_dict |= {f'val/{k}': v for k, v in val_metrics.items()}
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
                'mAP': val_metrics['mAP']
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
                'mAP': val_metrics['mAP']
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
            loss = model(X, targets)['loss']['total']
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
