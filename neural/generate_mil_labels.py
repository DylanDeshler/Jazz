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
import h5py

def calculate_gmm_thresholds(
    probabilities: np.ndarray, 
    class_name: str = "Instrument", 
    n_components: int = 3, 
    std_multiplier: float = 2.0,
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
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
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
    # neg_threshold = np.clip(neg_threshold, 0.01, 0.40) 
    # pos_threshold = np.clip(pos_threshold, 0.60, 0.99)
    
    # Ensure the Ignore Zone actually exists
    # if neg_threshold >= pos_threshold:
    #     print(f"Warning: Overlapping thresholds for {class_name}. Falling back to percentiles.")
    #     neg_threshold = np.percentile(probabilities, 60)
    #     pos_threshold = np.percentile(probabilities, 95)

    print(f"[{class_name}] Negative Threshold (<): {neg_threshold:.3f}")
    print(f"[{class_name}] Positive Threshold (>): {pos_threshold:.3f}")
    print(f"[{class_name}] Ignore Zone: {neg_threshold:.3f} to {pos_threshold:.3f}")

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
    plt.savefig(os.path.join(out_prefix, f'{class_name}_probs.png'))

    return neg_threshold, pos_threshold

device = 'cuda:0'
batch_size = 4
rate = 16000
n_samples = 16383 * 30

out_prefix = '/home/ubuntu/Data/MIL_labels'
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

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
torch.cuda.set_device(device)

paths = glob.glob('/home/ubuntu/Data/measures/*')
# with open('/home/ubuntu/Data/valid_files_by_bpm.json', 'r') as f:
#     beat_paths = json.load(f)
# paths = [os.path.join('/home/ubuntu/Data/wavs', os.path.basename(path)) for path in paths if os.path.basename(path) in beat_paths]
# print(len(paths))

# import concurrent.futures
# from multiprocessing import cpu_count
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

# with h5py.File(out_prefix + '.h5', 'w') as h5_file, torch.no_grad():
#     for idx, x in enumerate(tqdm(wavs)):
#         this_codes = []
#         n_cuts = len(x) // n_samples
            
#         batch = torch.from_numpy(np.stack([x[i * n_samples: (i + 1) * n_samples] for i in range(n_cuts)], axis=0)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
        
#         for i in range(math.ceil(len(batch) / batch_size)):
#             with ctx:
#                 probs = model(batch[i*batch_size:(i+1)*batch_size])['frame_probs']
    
#             probs = scipy.signal.medfilt(
#                 probs.cpu().detach().float().numpy(),
#                 kernel_size=(1, 11, 1)
#             )
#             probs = torch.from_numpy(probs).transpose(1, 2)
#             probs = F.interpolate(
#                 probs, size=batch.shape[-1], mode='linear', align_corners=False
#             ).transpose(1, 2).cpu().detach().numpy()
#             this_codes.append(probs)
        
#         # pad with 0s for last section
#         if len(x) - n_cuts * n_samples > 0:
#             batch = np.zeros((1, n_samples), dtype=np.float32)
#             batch[0, :len(x) - n_cuts * n_samples] = x[n_cuts * n_samples:]
#             batch = torch.from_numpy(batch).unsqueeze(1).pin_memory().to(device, non_blocking=True)
#             probs = model(batch)['frame_probs'].cpu().detach().float().numpy()
#             probs = scipy.signal.medfilt(
#                 probs, 
#                 kernel_size=(1, 11, 1)
#             )
#             probs = torch.from_numpy(probs).transpose(1, 2)
#             probs = F.interpolate(
#                 probs, size=batch.shape[-1], mode='linear', align_corners=False
#             ).transpose(1, 2).cpu().detach().numpy()
#             this_codes.append(probs)

#         this_codes = np.concatenate(this_codes, axis=0)
#         this_codes = this_codes.reshape(-1, num_classes)[:len(x)]
#         this_codes = np.concatenate([x[:, np.newaxis], this_codes], axis=1)
        
#         h5_file.create_dataset(
#             name=str(idx),
#             data=this_codes.astype(np.float16), 
#             compression='lzf',
#             dtype=np.float16
#         )

np.random.seed(0)
to_samples = np.random.choice(np.arange(len(paths)), 128, replace=False)
for i in range(1, num_classes + 1):
    file = h5py.File(out_prefix + '.h5', 'r')
    probs = [file[str(key)][:, i].astype(np.float32) for key in to_samples]
    probs = np.concatenate(probs, axis=0)
    print(probs.shape)
    neg_t, pos_t = calculate_gmm_thresholds(
        probabilities=probs, 
        class_name=str(i), 
        n_components=3, 
        std_multiplier=2.0
    )
    