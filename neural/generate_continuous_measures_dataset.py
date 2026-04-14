import os
import time
import math
import json
import pickle
from contextlib import nullcontext
from tqdm import tqdm
from torchinfo import summary

import numpy as np
import torch

from dito import DiToV5 as Tokenizer
from adapter import InvertibleAdapter

import soundfile as sf
import librosa
import glob

device = torch.device('cuda')

batch_size = 128
rate = 16000
n_samples = 24576

out_prefix = 'low_large_24576_subset_adapter'

ckpt_path = os.path.join('tokenizer_low_large_24576_subset', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
tokenizer_args = checkpoint['model_args']
vae_embed_dim = tokenizer_args['dimension']
model = Tokenizer(**tokenizer_args).to(device)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model = torch.compile(model)
encoder_ratios = math.prod(model.encoder.ratios)

ckpt_path = os.path.join('tokenizer_adapter_low_large_24576_subset', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
adapter_args = checkpoint['model_args']
adapter = InvertibleAdapter(**adapter_args).to(device)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
adapter.load_state_dict(state_dict)
adapter.eval()
adapter = torch.compile(adapter)
max_seq_len = adapter.max_seq_len

paths = glob.glob('/home/ubuntu/Data/measures/*')
with open('/home/ubuntu/Data/valid_files_by_bpm.json', 'r') as f:
    beat_paths = json.load(f)
paths = [os.path.join('/home/ubuntu/Data/wavs', os.path.basename(path)) for path in paths if os.path.basename(path) in beat_paths]
print(len(paths))

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
                            
                            # found an issue where 4/4 is frequently being annotated as 8/4 this fixes it and safe because were only annotating 4/4 songs
                            if bn > 0:
                                bn = ((bn - 1) % 4) + 1
                        except ValueError:
                            pass
                    
                    beat_data.append({'time': ts, 'beat': bn})
                except ValueError:
                    continue
    
    return beat_data

TARGET_SIG = 4

if True:

    test = False

    write_idx = 0
    write_paths = []
    total_write_batches = 48

    all_codes = []
    all_bpms = []
    with torch.no_grad():
        for idx, wav in enumerate(tqdm(wavs)):
            beat_path = os.path.join('/home/ubuntu/Data/beats', os.path.basename(paths[idx]))
            beat_data = parse_beat_file(beat_path)
            downbeat_indices = [i for i, b in enumerate(beat_data) if b['beat'] == 1]
            
            x, bpms = [], []
            for i in range(len(downbeat_indices) - 1):    
                start_idx = downbeat_indices[i]
                end_idx = downbeat_indices[i+1]
                
                t_start = beat_data[start_idx]['time']
                t_end = beat_data[end_idx]['time']
                
                frame_start = int(t_start * rate)
                frame_end = int(t_end * rate)
                
                if frame_end > len(wav):
                    break
                
                duration_sec = (frame_end - frame_start) / rate
                instant_bpm = (TARGET_SIG / duration_sec) * 60
                
                x.append(wav[frame_start:frame_end])
                bpms.append(instant_bpm)
            
            lengths = [len(raw) for raw in x]
            max_len_trunc = min(max(lengths), encoder_ratios * (max_seq_len - 1))
            max_len_trunc = encoder_ratios * math.ceil(max_len_trunc / encoder_ratios)

            indices = torch.arange(max_len_trunc // encoder_ratios).unsqueeze(0)
            lengths = torch.from_numpy(np.asarray(lengths)).unsqueeze(1)
            lengths = (lengths + encoder_ratios - 1) // encoder_ratios
            latent_mask = (indices < lengths).to(device)
            
            batch = torch.from_numpy(np.stack([np.pad(raw[:max_len_trunc], (0, max_len_trunc - len(raw[:max_len_trunc]))) for raw in x], axis=0).astype(np.float32)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
            
            for i in range(len(batch) // batch_size + 1):
                _, codes = model.encode(batch[i*batch_size:(i+1)*batch_size])
                codes, _ = adapter.encode(codes, mask=latent_mask[i*batch_size:(i+1)*batch_size])
                this_codes = codes.permute(0, 2, 1).cpu().detach().numpy()

                all_codes.append(this_codes)
            all_bpms.append(np.asarray(bpms))
            
            if (idx + 1) % (len(paths) // total_write_batches) == 0:
                print(f'Writing batch {write_idx}...')
                all_codes = np.concatenate(all_codes, axis=0)
                print(all_codes.shape)
                filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_{str(write_idx).zfill(2)}.bin')
                dtype = np.float32
                arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_codes.shape)
                arr[:] = all_codes
                arr.flush()
                
                all_bpms = np.concatenate(all_bpms, axis=0)
                print(all_bpms.shape)
                filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_bpm_{str(write_idx).zfill(2)}.bin')
                dtype = np.float32
                arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_bpms.shape)
                arr[:] = all_bpms
                arr.flush()

                write_idx += 1
                write_paths.append((filename, len(all_codes)))
                all_codes = []
                all_bpms = []
    
    # write the remaining batch
    print(f'Writing batch {write_idx}...')
    all_codes = np.concatenate(all_codes, axis=0)
    print(all_codes.shape)
    filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_{str(write_idx).zfill(2)}.bin')
    dtype = np.float32
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_codes.shape)
    arr[:] = all_codes
    arr.flush()
    
    all_bpms = np.concatenate(all_bpms, axis=0)
    print(all_bpms.shape)
    filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_bpm_{str(write_idx).zfill(2)}.bin')
    dtype = np.float32
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_bpms.shape)
    arr[:] = all_bpms
    arr.flush()

    write_idx += 1
    write_paths.append((filename, len(all_codes)))
    all_codes = []

## get token write paths
dtype = np.float32
write_paths = []
paths = [f'{out_prefix}_{str(i).zfill(2)}.bin' for i in range(total_write_batches + 1)]
for path in paths:
    data = np.memmap(path, dtype=np.float32, mode='r')
    data = data.reshape((-1, n_samples // encoder_ratios, vae_embed_dim))
    write_paths.append((path, data.shape))

# write tokens to train.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_train.bin')
train_length = np.sum([shape[0] for path, shape[0] in write_paths[:-2]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(train_length, n_samples // encoder_ratios, vae_embed_dim))
print(arr.shape)

for path, shape in write_paths[:-2]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)

    arr[cur_idx:cur_idx+shape[0]] = data
    arr.flush()

    cur_idx += shape[0]

# write tokens to val.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_val.bin')
val_length = np.sum([shape[0] for path, shape in write_paths[-2:]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(val_length, n_samples // encoder_ratios, vae_embed_dim))
print(arr.shape)

for path, shape in write_paths[-2:]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)

    arr[cur_idx:cur_idx+shape[0]] = data
    arr.flush()

    cur_idx += shape[0]

## get BPM write paths
dtype = np.float32
write_paths = []
paths = [f'{out_prefix}_bpm_{str(i).zfill(2)}.bin' for i in range(total_write_batches + 1)]
for path in paths:
    data = np.memmap(path, dtype=np.float32, mode='r')
    write_paths.append((path, data.shape))

# write BPM to train.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_bpm_train.bin')
train_length = np.sum([shape[0] for path, shape[0] in write_paths[:-2]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(train_length,))
print(arr.shape)

for path, shape in write_paths[:-2]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)

    arr[cur_idx:cur_idx+shape[0]] = data
    arr.flush()

    cur_idx += shape[0]

# write BPM to val.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_bpm_val.bin')
val_length = np.sum([shape[0] for path, shape in write_paths[-2:]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(val_length,))
print(arr.shape)

for path, shape in write_paths[-2:]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)

    arr[cur_idx:cur_idx+shape[0]] = data
    arr.flush()

    cur_idx += shape[0]