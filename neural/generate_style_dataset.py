import os
import math
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch

from contrast import Transformer as net

device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto 
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

batch_size = 1024

ckpt_path = os.path.join('contrast_learntmep_instance', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']

hidden_size = model_args['hidden_size']
sample_rate = model_args['sample_rate']
n_seconds = 10
n_samples = 16383
time_length = 32
frequency_length = 64

model_args['time_length'] = time_length
model_args['frequency_length'] = frequency_length

model = net(**model_args).to(device)
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

def extract_centered_style_windows(audio, sr=16000, window_sec=10, hop_sec=1):
    """
    Extracts centered 10s windows with a 1s hop length from a 1D audio array.
    Automatically pads with zeros to maintain center focus for edge frames.
    
    Args:
        audio (np.ndarray): 1D array of audio samples.
        sr (int): Sample rate (default 16000).
        window_sec (int): Length of the extracted window in seconds.
        hop_sec (int): Hop length in seconds (aligns with latent frames).
        
    Returns:
        np.ndarray: 2D array of shape (num_frames, window_samples)
    """
    # 1. Convert seconds to samples
    window_samples = window_sec * sr        # e.g., 160,000
    hop_samples = hop_sec * sr              # e.g., 16,000
    
    # 2. Determine the total number of 1Hz latent frames we need to match
    num_frames = int(np.ceil(len(audio) / hop_samples))
    
    # 3. Calculate exact padding needed for centering
    # To center a 10s window over a 1s chunk, we need 4.5s on the left.
    pad_left = (window_samples - hop_samples) // 2
    
    # Calculate the total length the array *must* be to extract all frames safely
    required_length = (num_frames - 1) * hop_samples + window_samples
    
    # Right pad is whatever is leftover to reach the required length
    pad_right = max(0, required_length - (len(audio) + pad_left))
    
    # 4. Pad the audio array with 0s (silence)
    padded_audio = np.pad(audio, (pad_left, pad_right), mode='constant', constant_values=0.0)
    
    # 5. Extract the windows using insanely fast NumPy stride tricks
    # This creates a sliding view of 160,000 samples, moving 1 sample at a time
    view = sliding_window_view(padded_audio, window_shape=window_samples)
    
    # Slice the view to jump by our hop_length (16,000 samples)
    windows = view[::hop_samples]
    
    # .copy() forces NumPy to allocate contiguous memory. 
    # Highly recommended before feeding this into PyTorch or a contrastive model!
    return np.ascontiguousarray(windows)

file_offsets = np.memmap('/home/dylan.d/research/music/Jazz/file_offsets.bin', dtype=np.int64, mode='r', shape=(32939, 4))
n_files = len(file_offsets)

data = np.memmap("/home/dylan.d/research/music/Jazz/wavs_16khz.bin", dtype=np.float32, mode='r')

N = 0
for i in tqdm(range(n_files)):
    start = file_offsets[i, 0]
    length = file_offsets[i, 1]
    
    batch = extract_centered_style_windows(data[start:start+length].copy(), sr=n_samples)
    N += len(batch)

print(f'Counted {N} segments')
arr = np.memmap(f'/home/dylan.d/research/music/Jazz/style.bin', dtype=np.float16, mode='w+', shape=(N, hidden_size))

cur_i = 0
with torch.no_grad():
    for i in tqdm(range(n_files)):
        start = file_offsets[i, 0]
        length = file_offsets[i, 1]
        
        batch = extract_centered_style_windows(data[start:start+length].copy(), sr=n_samples)
        batch = torch.from_numpy(batch).pin_memory().to(device, non_blocking=True)
        
        with ctx:
            out = model(batch)['features']
        
        arr[cur_i:cur_i + len(batch)] = out.float().cpu().detach().numpy().astype(np.float16)
        cur_i += len(batch)

arr.flush()
