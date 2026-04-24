import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os
import json
from contextlib import nullcontext
from tqdm import tqdm

import torch
import torch.nn.functional as F
from mil import MIL
import scipy.signal

import math
import soundfile as sf
import librosa
import glob

def calculate_gmm_thresholds(
    probabilities: np.ndarray, 
    class_name: str = "Instrument", 
    n_components: int = 3, 
    std_multiplier: float = 2.0,
    plot: bool = True
):
    """
    Fits a GMM to continuous probability predictions to automatically 
    find the optimal Positive and Negative boundaries for Tri-State labeling.
    
    Args:
        probabilities: Flat 1D numpy array of frame probabilities (0.0 to 1.0).
        class_name: Name of the instrument (for plotting).
        n_components: 3 is ideal (Silence, Uncertainty, Confident).
        std_multiplier: How many standard deviations from the mean to set the threshold.
        plot: Whether to display the histogram and threshold lines.
    """
    # 1. Reshape data for sklearn (requires 2D array)
    probs_reshaped = probabilities.reshape(-1, 1)
    
    # 2. Fit the Gaussian Mixture Model
    print(f"Fitting GMM for {class_name}...")
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(probs_reshaped)
    
    # 3. Extract Means and Standard Deviations
    # gmm.means_ is shape (n_components, 1)
    # gmm.covariances_ is shape (n_components, 1, 1)
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    
    # 4. Sort components by mean to identify Silence (lowest) and Confident (highest)
    sorted_indices = np.argsort(means)
    means_sorted = means[sorted_indices]
    stds_sorted = stds[sorted_indices]
    
    # The lowest mean is our "Silence" cluster
    mean_silence = means_sorted[0]
    std_silence = stds_sorted[0]
    
    # The highest mean is our "Confident Positive" cluster
    mean_confident = means_sorted[-1]
    std_confident = stds_sorted[-1]
    
    # 5. Calculate the thresholds
    # Negative threshold: Upper tail of the Silence cluster
    neg_threshold = mean_silence + (std_multiplier * std_silence)
    
    # Positive threshold: Lower tail of the Confident cluster
    pos_threshold = mean_confident - (std_multiplier * std_confident)
    
    # --- Failsafes & Bounds Checking ---
    # Clamp to valid probability ranges
    neg_threshold = np.clip(neg_threshold, 0.01, 0.40) 
    pos_threshold = np.clip(pos_threshold, 0.60, 0.99)
    
    # Ensure the Ignore Zone actually exists
    if neg_threshold >= pos_threshold:
        print(f"Warning: Overlapping thresholds for {class_name}. Falling back to percentiles.")
        neg_threshold = np.percentile(probabilities, 60)
        pos_threshold = np.percentile(probabilities, 95)

    print(f"[{class_name}] Negative Threshold (<): {neg_threshold:.3f}")
    print(f"[{class_name}] Positive Threshold (>): {pos_threshold:.3f}")
    print(f"[{class_name}] Ignore Zone: {neg_threshold:.3f} to {pos_threshold:.3f}")

    # 6. Optional Visualization
    if plot:
        plt.figure(figsize=(10, 5))
        
        # Plot the raw histogram
        plt.hist(probabilities, bins=100, density=True, alpha=0.5, color='gray', label='Raw Probabilities')
        
        # Plot the individual Gaussian bells
        x_axis = np.linspace(0, 1, 1000).reshape(-1, 1)
        logprob = gmm.score_samples(x_axis)
        pdf = np.exp(logprob)
        plt.plot(x_axis, pdf, '-k', label='Combined GMM Fit')
        
        # Add threshold lines
        plt.axvline(neg_threshold, color='red', linestyle='--', linewidth=2, label=f'Neg Threshold ({neg_threshold:.2f})')
        plt.axvline(pos_threshold, color='green', linestyle='--', linewidth=2, label=f'Pos Threshold ({pos_threshold:.2f})')
        
        # Highlight the zones
        plt.axvspan(0, neg_threshold, alpha=0.1, color='red')
        plt.axvspan(pos_threshold, 1, alpha=0.1, color='green')
        
        plt.title(f"GMM Tri-State Thresholds: {class_name}")
        plt.xlabel("Phase 1 Probability")
        plt.ylabel("Density")
        plt.legend()
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{class_name}_probs.png'))

    return neg_threshold, pos_threshold

device = torch.device('cuda')

batch_size = 16
rate = 16000
n_samples = 16383 * 30

out_prefix = 'MIL_labels'
ckpt_path = os.path.join('MIL', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model_args = checkpoint['model_args']

model = MIL(**model_args).to(device)
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
num_classes = model.queries.shape[0]

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

if True:
    write_idx = 0
    write_paths = []
    total_write_batches = 100

    all_codes = []
    all_instruments = []
    with torch.no_grad():
        for idx, x in enumerate(tqdm(wavs)):
            this_codes = []
            
            n_cuts = len(x) // n_samples
            
            batch = torch.from_numpy(np.stack([x[i * n_samples: (i + 1) * n_samples] for i in range(n_cuts)], axis=0)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
            
            for i in range(math.ceil(len(batch) / batch_size)):
                probs = model(batch[i*batch_size:(i+1)*batch_size])['frame_probs'].cpu().detach().float().numpy()
                probs = scipy.signal.medfilt(
                    probs, 
                    kernel_size=(1, 11, 1)
                )
                probs = torch.from_numpy(probs).transpose(1, 2)
                probs = F.interpolate(
                    probs, size=batch.shape[-1], mode='linear', align_corners=False
                ).transpose(1, 2).cpu().detach().numpy()
                this_codes.append(probs)
            
            # pad with 0s for last section
            if len(x) - n_cuts * n_samples > 0:
                batch = np.zeros((1, n_samples), dtype=np.float32)
                batch[0, :len(x) - n_cuts * n_samples] = x[n_cuts * n_samples:]
                batch = torch.from_numpy(batch).unsqueeze(1).pin_memory().to(device, non_blocking=True)
                probs = model(batch)['frame_probs'].cpu().detach().float().numpy()
                probs = scipy.signal.medfilt(
                    probs, 
                    kernel_size=(1, 11, 1)
                )
                probs = torch.from_numpy(probs).transpose(1, 2)
                probs = F.interpolate(
                    probs, size=batch.shape[-1], mode='linear', align_corners=False
                ).transpose(1, 2).cpu().detach().numpy()
                this_codes.append(probs)

            this_codes = np.concatenate(this_codes, axis=0)
            this_codes = this_codes.reshape(-1, num_classes)[:len(x)]
            this_codes = np.concatenate([x[:, np.newaxis], this_codes], axis=1)
            all_codes.append(this_codes)
            
            if (idx + 1) % (len(paths) // total_write_batches) == 0:
                print(f'Writing batch {write_idx}...')
                all_codes = np.concatenate(all_codes, axis=0)
                print(all_codes.shape)

                all_codes = all_codes.reshape(all_codes.shape[0] * all_codes.shape[1], all_codes.shape[2])
                print(all_codes.shape)

                filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_{str(write_idx).zfill(2)}.bin')
                dtype = np.float32
                arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_codes.shape)
                arr[:] = all_codes
                arr.flush()

                write_idx += 1
                write_paths.append((filename, len(all_codes)))
                all_codes = []

    # write the remaining batch
    print(f'Writing batch {write_idx}...')
    all_codes = np.concatenate(all_codes, axis=0)

    filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_{str(write_idx).zfill(2)}.bin')
    dtype = np.float32
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_codes.shape)
    arr[:] = all_codes
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
    write_paths.append((path, len(data)))

# write tokens to train.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_train.bin')
train_length = np.sum([length for path, length in write_paths[:-2]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(train_length, num_classes + 1))
print(arr.shape)

for path, length in write_paths[:-2]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=(length, num_classes + 1))

    arr[cur_idx:cur_idx+length] = data
    arr.flush()

    cur_idx += length

# write tokens to val.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_val.bin')
val_length = np.sum([length for path, length in write_paths[-2:]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(val_length, num_classes + 1))
print(arr.shape)

for path, length in write_paths[-2:]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=(length, num_classes + 1))

    arr[cur_idx:cur_idx+length] = data
    arr.flush()

    cur_idx += length

# neg_t, pos_t = calculate_gmm_thresholds(
#     probabilities=simulated_probs, 
#     class_name="Saxophone", 
#     n_components=3, 
#     std_multiplier=2.0
# )
    