import os
import math
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch

import essentia.standard as es
import librosa
from scipy.signal import medfilt

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
window_samples = 16383
time_length = 32
frequency_length = 64
n_fft = 1024
hop_length = 512

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

def extract_centered_style_windows(audio, hop_samples=16000, window_samples=163830):
    """
    Extracts perfectly centered style windows using exact sample counts 
    to align STFT bins with the 1 Hz (16,000 sample) latent frames.
    
    Args:
        audio (np.ndarray): 1D array of raw audio samples.
        hop_samples (int): Hop length in samples (16000 for 1Hz at 16kHz).
        window_samples (int): Exact window length in samples (163830 for STFT alignment).
        
    Returns:
        np.ndarray: 2D array of shape (num_frames, window_samples)
    """
    # 1. Determine the total number of 1Hz latent frames we need to match
    # np.ceil ensures we get a frame even for the final fractional second of audio
    num_frames = int(np.ceil(len(audio) / hop_samples))
    
    # 2. Calculate exact padding needed for centering
    # Pad = (Style_Window - Latent_Window) / 2
    # For 163830 and 16000, this naturally evaluates to exactly 73915
    pad_left = (window_samples - hop_samples) // 2
    
    # Calculate the absolute total length the array *must* be to extract all frames
    required_length = (num_frames - 1) * hop_samples + window_samples
    
    # Right pad is whatever is leftover to reach the required length safely
    pad_right = max(0, required_length - (len(audio) + pad_left))
    
    # 3. Pad the audio array with 0s (silence)
    padded_audio = np.pad(audio, (pad_left, pad_right), mode='constant', constant_values=0.0)
    
    # 4. Extract the windows using insanely fast NumPy stride tricks
    view = sliding_window_view(padded_audio, window_shape=window_samples)
    
    # Slice the view to jump strictly by our latent hop_length
    windows = view[::hop_samples]
    
    # Force memory contiguity before sending to your Contrastive Style Model
    return np.ascontiguousarray(windows)

def get_key_essentia(audio_path):
    # 1. Load audio natively (resamples to 44.1kHz automatically)
    audio = es.MonoLoader(filename=audio_path, sampleRate=44100)()
    
    # 2. Initialize the Key Extractor
    # 'temperley' is generally the most robust profile for Jazz/Acoustic
    key_extractor = es.KeyExtractor(profileType='temperley')
    
    # 3. Extract!
    key, scale, strength = key_extractor(audio)
    
    # Returns e.g., "C", "minor", 0.85
    return f"{key} {scale}", strength

def extract_local_keys(audio, window_sec=15.0, hop_sec=1.0, sr=44100):
    """
    Slides a 15-second window across the audio to track key modulations.
    """
    
    # Calculate samples
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    total_samples = len(audio)
    
    # Initialize the Essentia extractor once for speed
    key_extractor = es.KeyExtractor(profileType='temperley')
    
    local_keys = []
    timestamps = []
    
    # 2. Slide the window across the audio
    for start_sample in range(0, total_samples - window_samples + 1, hop_samples):
        # Extract the chunk
        chunk = audio[start_sample : start_sample + window_samples]
        
        # 3. Predict the "Global" key of this specific 15s chunk
        key, scale, strength = key_extractor(chunk)
        
        # Calculate the center time of this window
        center_time_sec = (start_sample + (window_samples / 2)) / sr
        
        local_keys.append(f"{key} {scale}")
        timestamps.append(center_time_sec)
        
    return timestamps, local_keys

def smooth_key_timeline(local_keys, smoothing_window=15):
    """
    Applies a median filter to remove rapid, erroneous key fluctuations.
    smoothing_window must be an odd integer (e.g., 15 seconds).
    """
    # 1. Map string labels to integer IDs
    unique_keys = list(dict.fromkeys(local_keys))
    key_to_id = {k: i for i, k in enumerate(unique_keys)}
    id_to_key = {i: k for k, i in key_to_id.items()}
    
    y_vals = np.array([key_to_id[k] for k in local_keys])
    
    # 2. Apply the Median Filter (forces the algorithm to "commit" to a key)
    smoothed_y = medfilt(y_vals, kernel_size=smoothing_window).astype(int)
    
    # 3. Convert back to string labels
    smoothed_keys = [id_to_key[y] for y in smoothed_y]
    
    return smoothed_keys

def create_conditioned_chromagram(smoothed_keys, harmonic_rms, sr=16000, hop_length=512):
    """
    Upsamples a 1Hz key timeline to match the STFT frame rate and 
    synthesizes a dynamic 12D chromagram modulated by RMS.
    
    Args:
        smoothed_keys: List of string keys (e.g., ['C major', 'C major', ...]) 1 per sec.
        harmonic_rms: 1D torch Tensor of shape (num_frames,) containing HPSS Harmonic RMS.
        sr: Sample rate used for your STFT.
        hop_length: STFT hop length (e.g., 512).
        
    Returns:
        dynamic_chroma: Tensor of shape (num_frames, 12) ready for the DiT.
    """
    num_frames = harmonic_rms.shape[0]
    
    # 1. Krumhansl-Schmuckler Profiles
    maj_profile = torch.tensor([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    min_profile = torch.tensor([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    maj_profile = (maj_profile - maj_profile.min()) / (maj_profile.max() - maj_profile.min())
    min_profile = (min_profile - min_profile.min()) / (min_profile.max() - min_profile.min())
    
    maj_profile = maj_profile ** 2
    min_profile = min_profile ** 2

    pitch_map = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 
                 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 
                 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    
    # 2. Pre-compute the 12D vector for every second in our smoothed timeline
    key_vectors = []
    for key_str in smoothed_keys:
        parts = key_str.strip().split()
        root = parts[0]
        mode = parts[1].lower()
        
        base_profile = maj_profile if mode == 'major' else min_profile
        shift = pitch_map[root]
        
        # Roll the 12D vector so the root note aligns correctly
        profile = torch.roll(base_profile, shifts=shift)
        key_vectors.append(profile)
        
    # Shape: (Num_Seconds, 12)
    key_vectors = torch.stack(key_vectors).to(harmonic_rms.device)
    
    # 3. Upsample to Frame Rate (Nearest Neighbor Interpolation)
    # Calculate the exact time in seconds for every single STFT frame
    frame_times_sec = torch.arange(num_frames, device=harmonic_rms.device) * (hop_length / sr)
    
    # Map the frame time to the correct index in our 1Hz keys list
    # e.g., frame at 4.7s -> index 4
    key_indices = torch.floor(frame_times_sec).long() 
    
    # Clamp to max index to safely handle the exact tail end of the audio file
    key_indices = torch.clamp(key_indices, max=len(smoothed_keys) - 1)
    
    # Gather the 12D vectors for every frame. Shape becomes (num_frames, 12)
    upsampled_chroma = key_vectors[key_indices]
    
    # 4. Modulate with the Harmonic RMS Envelope
    # This turns the static blocks into a breathing, rhythmic conditioning signal
    harmonic_rms = harmonic_rms.view(-1, 1) # Ensure shape is (num_frames, 1) for broadcasting
    dynamic_chroma = upsampled_chroma * harmonic_rms
    
    return dynamic_chroma

file_offsets = np.memmap('/home/dylan.d/research/music/Jazz/file_offsets.bin', dtype=np.int64, mode='r', shape=(32939, 4))
n_files = len(file_offsets)

data = np.memmap("/home/dylan.d/research/music/Jazz/wavs_16khz.bin", dtype=np.float32, mode='r')

# N = 0
# for i in tqdm(range(n_files)):
#     start = file_offsets[i, 0]
#     length = file_offsets[i, 1]
    
#     batch = extract_centered_style_windows(data[start:start+length].copy(), sr=n_samples)
#     N += len(batch)

N = 6381425
print(f'Counted {N} segments')
# style = np.memmap(f'/home/dylan.d/research/music/Jazz/style.bin', dtype=np.float16, mode='w+', shape=(N, hidden_size))
# key_chroma = np.memmap(f'/home/dylan.d/research/music/Jazz/meta.bin', dtype=np.float16, mode='w+', shape=(N, 12))
# true_chroma = np.memmap(f'/home/dylan.d/research/music/Jazz/meta.bin', dtype=np.float16, mode='w+', shape=(N, 12))
# rms = np.memmap(f'/home/dylan.d/research/music/Jazz/meta.bin', dtype=np.float16, mode='w+', shape=(N, 12))
# key_chroma = np.memmap(f'/home/dylan.d/research/music/Jazz/meta.bin', dtype=np.float16, mode='w+', shape=(N, 12))

cur_i = 0
with torch.no_grad():
    for i in tqdm(range(n_files)):
        start = file_offsets[i, 0]
        length = file_offsets[i, 1]
        
        batch = extract_centered_style_windows(data[start:start+length].copy(), hop_samples=sample_rate, window_samples=window_samples)
        
        # Compute features
        rms = []
        spectral_centroid = []
        onset_strength = []
        zcr = []
        for y in batch:
            rms.append(librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0])
            
            spectral_centroid.append(librosa.feature.spectral_centroid(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0])
            onset_strength.append(librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length))
            zcr.append(librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0])
        rms = np.stack(rms)
        spectral_centroid = np.stack(spectral_centroid)
        onset_strength = np.stack(onset_strength)
        zcr = np.stack(zcr)
        
        timestamps, keys = extract_local_keys(data[start:start+length].copy(), window_sec=15.0, hop_sec=1.0, sr=sample_rate)
        keys = smooth_key_timeline(keys, smoothing_window=15)
        key_chromagram = create_conditioned_chromagram(keys, torch.from_numpy(rms.flatten()), sr=sample_rate).view(len(rms), -1, 12)
        chroma = librosa.feature.chroma_cqt(y=data[start:start+length].copy(), sr=sample_rate, hop_length=hop_length).T#[:, None, :].reshape(len(rms), -1, 12)
        
        print(y.shape, rms.shape, key_chromagram.shape, chroma.shape, spectral_centroid.shape, onset_strength.shape, zcr.shape)
        
        # Compute latents
        batch = torch.from_numpy(batch).unsqueeze(1).pin_memory().to(device, non_blocking=True)
        with ctx:
            out = model(batch, features_only=True)
        print(batch.shape, out.shape)
        continue
        
        # key_chroma[cur_i:cur_i + len(key_chromagram)] = key_chromagram
        # true_chroma[cur_i:cur_i + len(true_chromagram)] = true_chromagram
        # rms[cur_i:cur_i + len(rms)] = rms.astype(np.float16)
        # spectral_centroid[cur_i:cur_i + len(rms)] = 
        # onset_strength[cur_i:cur_i + len()]
        
        
        
        style[cur_i:cur_i + len(batch)] = out.float().cpu().detach().numpy().astype(np.float16)
        cur_i += len(batch)

style.flush()
