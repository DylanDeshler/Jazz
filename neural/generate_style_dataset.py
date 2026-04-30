import os
import math
import json
import numpy as np
import torch
import soundfile as sf
import librosa
import glob
from tqdm import tqdm
import concurrent.futures
from multiprocessing import cpu_count
from contrast import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
batch_size = 128
rate = 16000
total_write_batches = 48
n_samples = 16383 * 2

out_prefix = 'contrast_learntmep_instance_2s'
style_embed_dim = 768

ckpt_path = os.path.join('contrast_learntmep_instance_2s', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
tokenizer_args = checkpoint['model_args']
model = Transformer(**tokenizer_args).to(device)
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

# Gather paths using the same filtering logic as the original script
paths = glob.glob('/home/ubuntu/Data/measures/*')
with open('/home/ubuntu/Data/valid_files_by_bpm.json', 'r') as f:
    beat_paths = json.load(f)
paths = [os.path.join('/home/ubuntu/Data/wavs', os.path.basename(path)) for path in paths if os.path.basename(path) in beat_paths]
print(f"Total valid files: {len(paths)}")

wavs = [None] * len(paths)

# Load wavs concurrently
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
    beat_data = []
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

write_idx = 0
write_paths = []
all_styles = []

with torch.no_grad():
    for idx, wav in enumerate(tqdm(wavs, desc="Extracting Styles")):    
        beat_path = os.path.join('/home/ubuntu/Data/beats', os.path.basename(paths[idx]))
        beat_data = parse_beat_file(beat_path)
        downbeat_indices = [i for i, b in enumerate(beat_data) if b['beat'] == 1]
        
        start_stops = [(i*n_samples, (i+1)*n_samples) for i in range(math.ceil(len(wav) / n_samples))]
        starts, stops = zip(*start_stops)
        
        x = [np.pad(wav[i*n_samples:(i+1)*n_samples], (0, n_samples - len(wav[i*n_samples:(i+1)*n_samples]))) for i in range(math.ceil(len(wav) / n_samples))]
        x = torch.from_numpy(np.asarray(x).astype(np.float32)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
        
        styles = []
        for i in range(math.ceil(len(x) / batch_size)):
            style = model(x[i*batch_size:(i+1)*batch_size], features_only=True).cpu().numpy()
            styles.append(style)
        styles = np.stack(styles, axis=0)
        
        assert len(start_stops) == len(styles), f'{len(start_stops)} != {styles.shape}'
        
        for i in range(len(downbeat_indices) - 1):    
            start_idx = downbeat_indices[i]
            end_idx = downbeat_indices[i+1]
            
            t_start = beat_data[start_idx]['time']
            t_end = beat_data[end_idx]['time']
            
            frame_start = int(t_start * rate)
            frame_end = int(t_end * rate)
            
            if frame_end > len(wav):
                break
            
            weights = {}
            for j, (start, stop) in enumerate(start_stops):
                overlap_start = max(start, frame_start)
                overlap_end = min(stop, frame_end)
                
                if overlap_start < overlap_end:
                    weights[j] = overlap_end - overlap_start
            
            if not weights:
                continue
            
            style = np.zeros_like(styles[0])
            for k, v in weights.items():
                style += styles[k] * v / sum(list(weights.values()))
            
            all_styles.append(style)

        if (idx + 1) % (len(paths) // total_write_batches) == 0:
            print(f'Writing batch {write_idx}...')
            
            all_styles_arr = np.concatenate(all_styles, axis=0)
            print(all_styles_arr.shape)
            filename_styles = os.path.join(os.path.dirname(__file__), f'{out_prefix}_style_{str(write_idx).zfill(2)}.bin')
            arr_styles = np.memmap(filename_styles, dtype=np.float32, mode='w+', shape=all_styles_arr.shape)
            arr_styles[:] = all_styles_arr
            arr_styles.flush()

            write_paths.append((filename_styles, len(all_styles_arr)))
            write_idx += 1
            all_styles = []

# Write the remaining trailing batch
if len(all_styles) > 0:
    print(f'Writing batch {write_idx}...')
    all_styles_arr = np.concatenate(all_styles, axis=0)
    filename_styles = os.path.join(os.path.dirname(__file__), f'{out_prefix}_style_{str(write_idx).zfill(2)}.bin')
    arr_styles = np.memmap(filename_styles, dtype=np.float32, mode='w+', shape=all_styles_arr.shape)
    arr_styles[:] = all_styles_arr
    arr_styles.flush()

    write_paths.append((filename_styles, len(all_styles_arr)))
    write_idx += 1
    all_styles = []

# ==============================================================================
# COMPILE FINAL TRAIN/VAL BIN FILES
# ==============================================================================
print("Compiling final train and val datasets...")
dtype = np.float32

# Verify paths and shapes
compiled_write_paths = []
paths_to_check = [f'{out_prefix}_style_{str(i).zfill(2)}.bin' for i in range(total_write_batches + 1)]
for path in paths_to_check:
    if not os.path.exists(path): continue
    data = np.memmap(path, dtype=dtype, mode='r')
    # Reshape to 2D array: (length, style_embed_dim)
    data = data.reshape((-1, style_embed_dim))
    compiled_write_paths.append((path, data.shape))

train_length = np.sum([shape[0] for path, shape in compiled_write_paths[:-2]])
val_length = np.sum([shape[0] for path, shape in compiled_write_paths[-2:]])

# Write Train
arr_style_train = np.memmap(f'{out_prefix}_style_train.bin', dtype=dtype, mode='w+', shape=(train_length, style_embed_dim))
cur_train = 0
for path, shape in compiled_write_paths[:-2]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)
    arr_style_train[cur_train:cur_train+shape[0]] = data
    cur_train += shape[0]
arr_style_train.flush()

# Write Val
arr_style_val = np.memmap(f'{out_prefix}_style_val.bin', dtype=dtype, mode='w+', shape=(val_length, style_embed_dim))
cur_val = 0
for path, shape in compiled_write_paths[-2:]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)
    arr_style_val[cur_val:cur_val+shape[0]] = data
    cur_val += shape[0]
arr_style_val.flush()

print(f"Train array shape: {arr_style_train.shape}")
print(f"Val array shape: {arr_style_val.shape}")
print("Style dataset generation complete!")