import os
import glob
import librosa
import soundfile as sf
from tqdm import tqdm

import concurrent.futures
import pyrubberband as pyrb
from multiprocessing import cpu_count

import torch
import numpy as np
from contextlib import nullcontext
from dito import DiToV4 as Tokenizer
from fad import MultiTaskFAD as FAD
from contrast import Transformer as Contrast

torch.manual_seed(0)
np.randon.seed(0)

rate = 16000
n_samples = 24576
TARGET_SIG = 4
TARGET_BPM = 60 * TARGET_SIG / (n_samples / rate)

device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def load_model(ckpt_path, ModelType):
    checkpoint = torch.load(ckpt_path, map_location=device)
    tokenizer_args = checkpoint['model_args']

    model = ModelType(**tokenizer_args).to(device)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model = torch.compile(model)
    return model

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

base1 = load_model(os.path.join('tokenizer_low_large_24576', 'ckpt.pt'), Tokenizer)
base2 = load_model(os.path.join('tokenizer_low_large_24576_2std_subset', 'ckpt.pt'), Tokenizer)
measure1 = load_model(os.path.join('tokenizer_low_measures_2std_subset', 'ckpt.pt'), Tokenizer)
fad = load_model(os.path.join('FAD', 'ckpt.pt'), FAD)
contrast = load_model(os.path.join('contrast_learntmep_instance', 'ckpt.pt'), Contrast)

paths = glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean/*.wav')
paths = glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_measures/*.wav')
paths = [path.replace('jazz_data_16000_full_clean_measures', 'jazz_data_16000_full_clean') for path in paths]

paths = np.random.choice(paths, 32)

# wavs = [None] * len(paths)

# with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count() // 2) as executor:
#     future_to_index = {
#         executor.submit(lambda x: librosa.load(x, sr=rate)[0], path): i 
#         for i, path in enumerate(paths)
#     }
    
#     for future in tqdm(concurrent.futures.as_completed(future_to_index), desc='Loading wav files', total=len(paths)):
#         original_index = future_to_index[future]
#         wav = future.result()
#         wavs[original_index] = wav

for path in tqdm(paths):
    wav, sr = librosa.load(path)
    x = [wav[chunk * n_samples:(chunk+1) * n_samples] for chunk in range(len(wav) // n_samples)]
    x = torch.from_numpy(np.asarray(x).astype(np.float32)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
    beat_path = path.replace('jazz_data_16000_full_clean_measures', 'jazz_data_16000_full_clean_beats').replace('.wav', '.beats')
        
    start = np.random.randint((len(wav) // n_samples) - 1)
    ratios = [TARGET_BPM / calculate_bpm(beat_path, start) for start in range(len(wav) // n_samples)]
    
    with ctx:
        y1 = base1.reconstruct(x, n_steps=100)
        y2 = base2.reconstruct(x, n_steps=100)
        y3 = measure1.reconstruct(x, n_steps=100)
    
    print(y1.shape, y2.shape, y3.shape)
    
    y1 = y1.cpu().detach().float().numpy()
    y2 = y2.cpu().detach().float().numpy()
    y3 = y3.cpu().detach().float().numpy()
    ratios = ratios.cpu().detach().numpy()

    for y, ratio in zip(y3, ratios):
        sf.write(
            file=os.path.join(out_dir, ''), 
            data=restore_measure(y.squeeze(), ratio.item()), 
            samplerate=rate,
            subtype='PCM_16'
        )

