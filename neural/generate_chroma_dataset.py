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
hop_length = 1024 # average over time so large hop is fine
TARGET_SIG = 4
total_write_batches = 48

out_prefix = 'low_large_24576_subset_chroma_rms'

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

if True:

    test = False

    write_idx = 0
    write_paths = []

    all_chromas = []
    all_rms = []
    with torch.no_grad():
        for idx, wav in enumerate(tqdm(wavs)):
            beat_path = os.path.join('/home/ubuntu/Data/beats', os.path.basename(paths[idx]))
            beat_data = parse_beat_file(beat_path)
            downbeat_indices = [i for i, b in enumerate(beat_data) if b['beat'] == 1]
            
            chromas, rms = [], []
            wav_chroma = librosa.feature.chroma_cqt(y=wav, sr=rate, hop_length=hop_length)
            wav_rms = librosa.feature.rms(y=wav, hop_length=hop_length)[0]
            for i in range(len(downbeat_indices) - 1):    
                start_idx = downbeat_indices[i]
                end_idx = downbeat_indices[i+1]
                
                t_start = beat_data[start_idx]['time']
                t_end = beat_data[end_idx]['time']
                
                frame_start = librosa.time_to_frames(t_start, sr=rate, hop_length=hop_length)
                frame_end = librosa.time_to_frames(t_end, sr=rate, hop_length=hop_length)
                
                if frame_end > len(wav):
                    break
                    
                frame_end = min(frame_end, wav_chroma.shape[1])
                
                measure_chroma = wav_chroma[:, frame_start:frame_end]
                measure_rms = wav_rms[frame_start:frame_end]
                
                if measure_chroma.shape[1] > 0:
                    chromas.append(np.mean(measure_chroma, axis=1))
                    rms.append(np.mean(measure_rms))
            
            all_chromas.append(np.asarray(chromas))
            all_rms.append(np.asarray(rms))
            
            if (idx + 1) % (len(paths) // total_write_batches) == 0:
                print(f'Writing batch {write_idx}...')
                all_chromas = np.concatenate(all_chromas, axis=0)
                all_rms = np.concatenate(all_rms, axis=0)
                print(all_chromas.shape, all_rms.shape)
                all_data = np.stack([all_chromas, all_rms], axis=1)
                filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_{str(write_idx).zfill(2)}.bin')
                dtype = np.float32
                arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_data.shape)
                arr[:] = all_data
                arr.flush()
                

                write_idx += 1
                write_paths.append((filename, len(all_data)))
                all_chromas = []
                all_rms = []
    
    # write the remaining batch
    print(f'Writing batch {write_idx}...')
    all_chromas = np.concatenate(all_chromas, axis=0)
    all_rms = np.concatenate(all_rms, axis=0)
    print(all_chromas.shape, all_rms.shape)
    all_data = np.stack([all_chromas, all_rms], axis=1)
    filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_{str(write_idx).zfill(2)}.bin')
    dtype = np.float32
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_data.shape)
    arr[:] = all_data
    arr.flush()

    write_idx += 1
    write_paths.append((filename, len(all_data)))
    all_chromas = []
    all_rms = []

## get token write paths
dtype = np.float32
write_paths = []
paths = [f'{out_prefix}_{str(i).zfill(2)}.bin' for i in range(total_write_batches + 1)]
for path in paths:
    data = np.memmap(path, dtype=np.float32, mode='r')
    data = data.reshape(-1, 2)
    write_paths.append((path, data.shape))

# write to train.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_train.bin')
train_length = np.sum([shape[0] for path, shape in write_paths[:-2]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(train_length, 2))
print(arr.shape)

for path, shape in write_paths[:-2]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)

    arr[cur_idx:cur_idx+shape[0]] = data
    arr.flush()

    cur_idx += shape[0]

# write to val.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_val.bin')
val_length = np.sum([shape[0] for path, shape in write_paths[-2:]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(val_length, 2))
print(arr.shape)

for path, shape in write_paths[-2:]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)

    arr[cur_idx:cur_idx+shape[0]] = data
    arr.flush()

    cur_idx += shape[0]