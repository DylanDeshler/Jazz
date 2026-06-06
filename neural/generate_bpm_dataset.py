import os
import time
import math
import json
import glob
import numpy as np
from tqdm import tqdm

rate = 16000
TARGET_SIG = 4
total_write_batches = 48
out_prefix = 'low_large_24576_subset_adapter_longtrain_v2_64'

paths = glob.glob('/data/wavs/*')
with open('/data/valid_files_by_bpm.json', 'r') as f:
    beat_paths = json.load(f)
paths = [os.path.join('/data/wavs', os.path.basename(path)) for path in paths if os.path.basename(path) in beat_paths]
print(f"Total paths to process: {len(paths)}")

def parse_beat_file(beat_path):
    """
    Parses the beat_this output file.
    Expected format per line: <timestamp> <beat_number>
    """
    beat_data = []
    if not os.path.exists(beat_path):
        return beat_data
        
    with open(beat_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    ts = float(parts[0])
                    bn = 0 
                    if len(parts) >= 2:
                        try:
                            bn = int(float(parts[1]))
                            if bn > 0:
                                bn = ((bn - 1) % 4) + 1
                        except ValueError:
                            pass
                    
                    beat_data.append({'time': ts, 'beat': bn})
                except ValueError:
                    continue
    return beat_data

if True:
    write_idx = 0
    write_paths = []
    all_bpms = []

    for idx, path in enumerate(tqdm(paths)):
        beat_path = os.path.join('/data/beats', os.path.basename(paths[idx]))
        beat_data = parse_beat_file(beat_path)
        downbeat_indices = [i for i, b in enumerate(beat_data) if b['beat'] == 1]
        
        bpms = []
        for i in range(len(downbeat_indices) - 1):    
            start_idx = downbeat_indices[i]
            end_idx = downbeat_indices[i+1]
            
            t_start = beat_data[start_idx]['time']
            t_end = beat_data[end_idx]['time']
            
            duration_sec = t_end - t_start
            if duration_sec <= 0:
                continue
                
            instant_bpm = (TARGET_SIG / duration_sec) * 60
            bpms.append(instant_bpm)
        
        if bpms:
            all_bpms.append(np.asarray(bpms, dtype=np.float32))
        
        # Periodic shard writing
        if (idx + 1) % max(1, (len(paths) // total_write_batches)) == 0 and all_bpms:
            print(f'\nWriting batch {write_idx}...')
            all_bpms = np.concatenate(all_bpms, axis=0)
            print(f"Shape: {all_bpms.shape}")
            
            filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_bpm_{str(write_idx).zfill(2)}.bin')
            arr = np.memmap(filename, dtype=np.float32, mode='w+', shape=all_bpms.shape)
            arr[:] = all_bpms
            arr.flush()

            write_idx += 1
            all_bpms = []
    
    # Write the remaining shards
    if all_bpms:
        print(f'\nWriting final batch {write_idx}...')
        all_bpms = np.concatenate(all_bpms, axis=0)
        print(f"Shape: {all_bpms.shape}")
        
        filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_bpm_{str(write_idx).zfill(2)}.bin')
        arr = np.memmap(filename, dtype=np.float32, mode='w+', shape=all_bpms.shape)
        arr[:] = all_bpms
        arr.flush()
        write_idx += 1

## get BPM write paths and map shapes
dtype = np.float32
bpm_write_paths = []
paths_to_check = [f'{out_prefix}_bpm_{str(i).zfill(2)}.bin' for i in range(write_idx)]
for path in paths_to_check:
    if os.path.exists(path):
        data = np.memmap(path, dtype=dtype, mode='r')
        bpm_write_paths.append((path, data.shape))

# Compile BPM into train.bin
cur_idx = 0
filename_train = os.path.join(os.path.dirname(__file__), f'{out_prefix}_bpm_train.bin')
train_length = np.sum([shape[0] for path, shape in bpm_write_paths[:-2]])
arr_train = np.memmap(filename_train, dtype=dtype, mode='w+', shape=(train_length,))
print(f"\nFinal Train BPM Shape: {arr_train.shape}")

for path, shape in bpm_write_paths[:-2]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)
    arr_train[cur_idx:cur_idx+shape[0]] = data
    arr_train.flush()
    cur_idx += shape[0]

# Compile BPM into val.bin
cur_idx = 0
filename_val = os.path.join(os.path.dirname(__file__), f'{out_prefix}_bpm_val.bin')
val_length = np.sum([shape[0] for path, shape in bpm_write_paths[-2:]])
arr_val = np.memmap(filename_val, dtype=dtype, mode='w+', shape=(val_length,))
print(f"Final Val BPM Shape: {arr_val.shape}")

for path, shape in bpm_write_paths[-2:]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)
    arr_val[cur_idx:cur_idx+shape[0]] = data
    arr_val.flush()
    cur_idx += shape[0]