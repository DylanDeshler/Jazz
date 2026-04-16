import os
import glob
import math
import json
import librosa
import soundfile as sf
from tqdm import tqdm
from collections import defaultdict

import torch
import numpy as np
from scipy import linalg
from scipy.signal import medfilt
from contextlib import nullcontext
import torch.nn.functional as F

from dito import DiToV5 as Tokenizer
from fad import MultiTaskFAD as FAD, BPMProbe
from adapter import InvertibleAdapter as Adapter
from contrast import Transformer as Contrast
from diffusion_forcing import UnconditionalModernDiT_small as DiT

torch.manual_seed(0)
np.random.seed(0)

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
    print(f'Loading model {ckpt_path} ...')
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
    if 'cuda' in device:
        model = torch.compile(model)
    return model

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

def calculate_embd_statistics(embd_lst):
    if isinstance(embd_lst, list):
        embd_lst = np.array(embd_lst)
    
    mu = np.mean(embd_lst, axis=0)
    sigma = np.cov(embd_lst, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
            representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
            representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2).astype(complex), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset).astype(complex))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def drop_to_multiple(a, multiple):
    a = a.flatten()
    a = a[:(a.shape[0] // (multiple)) * multiple]
    a = a.view(-1, 1, multiple)
    return a

def smooth_bpm_predictions(bpm_tensor: torch.Tensor, method: str = 'median', window_size: int = 3) -> torch.Tensor:
    """
    Smooths the instantaneous BPM predictions across the chunk dimension.
    bpm_tensor: shape (Batch, Chunks)
    method: One of median, global, moving_average
    """
    if method == 'global':
        # Collapse the sequence to the mean tempo per batch item
        mean_bpm = bpm_tensor.mean(dim=1, keepdim=True)
        return mean_bpm.expand_as(bpm_tensor)
    
    bpm_np = bpm_tensor.cpu().detach().numpy()
    smoothed = np.zeros_like(bpm_np)
    
    for i in range(bpm_np.shape[0]):
        if method == 'median':
            # medfilt requires odd window sizes
            smoothed[i] = medfilt(bpm_np[i], kernel_size=window_size)
            
        elif method == 'moving_average':
            kernel = np.ones(window_size) / window_size
            # Pad edges so the sequence length stays 15
            padded = np.pad(bpm_np[i], (window_size//2, window_size//2), mode='edge')
            smoothed[i] = np.convolve(padded, kernel, mode='valid')
            
    return torch.from_numpy(smoothed).to(bpm_tensor.device)

def crossfade_chunks(chunks: list, overlap_samples: int) -> np.ndarray:
    """
    Takes a list of 1D numpy arrays (raw audio measures) and stitches them 
    together with a linear crossfade to eliminate phase clicks.
    """
    if not chunks:
        return np.array([])
        
    # Create linear fade envelopes
    fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)
    
    # Calculate the exact length of the final continuous array
    total_length = sum(len(c) for c in chunks) - overlap_samples * (len(chunks) - 1)
    out = np.zeros(total_length, dtype=np.float32)
    
    current_idx = 0
    for i, chunk in enumerate(chunks):
        chunk_len = len(chunk)
        end_idx = current_idx + chunk_len
        
        if i == 0:
            # First chunk writes directly
            out[current_idx:end_idx] = chunk
        else:
            # 1. Fade the overlapping region
            out[current_idx : current_idx + overlap_samples] *= fade_out
            out[current_idx : current_idx + overlap_samples] += chunk[:overlap_samples] * fade_in
            
            # 2. Write the rest of the new chunk
            out[current_idx + overlap_samples : end_idx] = chunk[overlap_samples:]
            
        # Advance the pointer, but step back by the overlap amount
        current_idx += chunk_len - overlap_samples
        
    return out

def predict_measures(gen_shape, n_steps, gen_noise, method='median', window_size=3, overlap_samples=0):
    with ctx:
        y2 = measure_dit.generate(gen_shape, n_steps=n_steps, noise=gen_noise)
    
    bpm = probe(y2)
    bpm = smooth_bpm_predictions(bpm, method=method, window_size=window_size)
    seconds_per_beat = 60.0 / bpm
    measure_duration_sec = seconds_per_beat * TARGET_SIG
    
    target_samples = (measure_duration_sec * rate).long()
    max_len = min(target_samples.max().item(), encoder_ratios * (max_seq_len - 1))
    max_len = encoder_ratios * math.ceil(max_len / encoder_ratios)
    max_latent_len = max_len // encoder_ratios
    
    indices = torch.arange(max_latent_len, device=device).view(1, 1, -1)
    lengths = ((target_samples + encoder_ratios - 1) // encoder_ratios).unsqueeze(-1)
    mask = indices < lengths
    mask = mask.view(batch_size * n_chunks, max_latent_len)
    shape = (batch_size * n_chunks, 1, max_latent_len)
        
    with ctx:
        y2 = y2.transpose(2, 3).view(batch_size * n_chunks, vae_embed_dim, spatial_window)
        y2 = adapter.decode(y2, shape, mask=mask)
        y2 = tokenizer.decode(y2, shape=(1, max_len), n_steps=n_steps, noise=None)
    
    target_samples = target_samples.flatten().cpu().detach().numpy()
    y2 = y2.squeeze().cpu().detach().numpy()
    
    if overlap_samples > 0:
        y2_cross = crossfade_chunks([y[:min(int(samples), max_len)] for y, samples in zip(y2, target_samples)], overlap_samples)
        y2_cross = torch.from_numpy(y2_cross.astype(np.float32)).to(device, non_blocking=True)
    
    y2 = np.concatenate([y[:min(int(samples), max_len)] for y, samples in zip(y2, target_samples)], axis=0)
    y2 = torch.from_numpy(y2.astype(np.float32)).to(device, non_blocking=True)
    
    if overlap_samples > 0:
        return y2, y2_cross
    return y2

# Tokenizers
tokenizer = load_model(os.path.join('tokenizer_low_large_24576_subset', 'ckpt.pt'), Tokenizer)
encoder_ratios = math.prod(tokenizer.encoder.ratios)
adapter = load_model(os.path.join('tokenizer_adapter_low_large_24576_subset', 'ckpt.pt'), Adapter)
max_seq_len = adapter.max_seq_len
probe = load_model(os.path.join('tokenizer_low_measures_fix_subset_BPMProbe', 'ckpt.pt'), BPMProbe)

# DiTs
base_dit = load_model(os.path.join('UnconditionalModernDiT_small_24576_subset', 'ckpt.pt'), DiT)
measure_dit = load_model(os.path.join('UnconditionalModernDiT_small_24576_subset_adapter', 'ckpt.pt'), DiT)
n_chunks = 15
spatial_window = 48
vae_embed_dim = 16

# Feature Extractors
fad = load_model(os.path.join('FAD', 'ckpt.pt'), FAD)

# 4/4 with BPM in reasonable range
measure_paths = glob.glob('/home/ubuntu/Data/measures/*')
with open('/home/ubuntu/Data/valid_files_by_bpm.json', 'r') as f:
    beat_paths = json.load(f)
measure_paths = [path for path in measure_paths if os.path.basename(path) in beat_paths]
audio_paths = [os.path.join('/home/ubuntu/Data/wavs', os.path.basename(path)) for path in measure_paths]
audio_paths = [path for path in audio_paths if os.path.basename(path) in beat_paths]
beat_paths = [os.path.join('/home/ubuntu/Data/beats', path) for path in beat_paths]

idxs = np.random.choice(np.arange(len(measure_paths)), size=32, replace=False)
n_steps = 32
batch_size = 16
EVAL_ITERATIVE = False
USE_CLAP = False

real_embs = []
y1_embs = []
y2_embs = []
y3_embs = []
y4_embs = []
y2_cross_embs = []
y3_cross_embs = []
y4_cross_embs = []
out_dir = '/home/ubuntu/Data/FAD_generative_samples'
os.makedirs(out_dir, exist_ok=True)
with torch.no_grad():
    for idx in tqdm(idxs):
        measure_path, audio_path, beat_path = measure_paths[idx], audio_paths[idx], beat_paths[idx]
        
        wav, _ = librosa.load(audio_path, sr=None)
        wav = wav[:batch_size * n_chunks * rate]
        
        beat_data = parse_beat_file(beat_path)
        downbeat_indices = [i for i, b in enumerate(beat_data) if b['beat'] == 1]
        
        m_raw = []
        first_frame, last_frame = None, None
        for i in range(len(downbeat_indices) - 1):
            start_idx = downbeat_indices[i]
            end_idx = downbeat_indices[i+1]
            
            t_start = beat_data[start_idx]['time']
            t_end = beat_data[end_idx]['time']
            
            frame_start = int(t_start * rate)
            frame_end = int(t_end * rate)
            
            if frame_end > len(wav):
                break
            
            if first_frame is None:
                first_frame = frame_start
            last_frame = frame_end
            
            m_raw.append(wav[frame_start:frame_end])
            
        if last_frame - first_frame < 16383 * 5:
            continue
        
        x_raw = wav[first_frame:last_frame]
        
        ## Standard approach
        if not EVAL_ITERATIVE:
            gen_shape = (batch_size, n_chunks, spatial_window, vae_embed_dim)
            gen_noise = torch.randn(gen_shape, device=device)
            
            with ctx:
                y1 = base_dit.generate(gen_shape, n_steps=n_steps, noise=gen_noise)
                y1 = y1.transpose(2, 3).view(batch_size * n_chunks, vae_embed_dim, spatial_window)
                y1 = tokenizer.decode(y1, shape=(1, 24576), n_steps=n_steps, noise=None)
            
            overlap_samples = int(0.02 * rate)
            y2, y2_cross = predict_measures(gen_shape, n_steps, gen_noise, method='global', window_size=3, overlap_samples=overlap_samples)
            y3, y3_cross = predict_measures(gen_shape, n_steps, gen_noise, method='moving_average', window_size=3, overlap_samples=overlap_samples)
            y4, y4_cross = predict_measures(gen_shape, n_steps, gen_noise, method='median', window_size=3, overlap_samples=overlap_samples)
            
            x = torch.from_numpy(x_raw.astype(np.float32)).to(device, non_blocking=True)
            
            # Custom embs
            if not USE_CLAP:
                x = drop_to_multiple(x, 16383 * 5)
                y1 = drop_to_multiple(y1, 16383 * 5)
                y2 = drop_to_multiple(y2, 16383 * 5)
                y3 = drop_to_multiple(y3, 16383 * 5)
                y4 = drop_to_multiple(y4, 16383 * 5)
                
                y2_cross = drop_to_multiple(y2_cross, 16383 * 5)
                y3_cross = drop_to_multiple(y3_cross, 16383 * 5)
                y4_cross = drop_to_multiple(y4_cross, 16383 * 5)
                with ctx:
                    try:
                        real_emb = fad.forward_features(x)
                        y1_emb = fad.forward_features(y1)
                        y2_emb = fad.forward_features(y2)
                        y3_emb = fad.forward_features(y3)
                        y4_emb = fad.forward_features(y4)
                        
                        y2_cross_emb = fad.forward_features(y2_cross)
                        y3_cross_emb = fad.forward_features(y3_cross)
                        y4_cross_emb = fad.forward_features(y4_cross)
                    except Exception as e:
                        print(e)
                        continue
            
            real_embs.append(real_emb.cpu().detach().numpy())
            y1_embs.append(y1_emb.cpu().detach().numpy())
            y2_embs.append(y2_emb.cpu().detach().numpy())
            y3_embs.append(y3_emb.cpu().detach().numpy())
            y4_embs.append(y4_emb.cpu().detach().numpy())
            
            y2_cross_embs.append(y2_cross_emb.cpu().detach().numpy())
            y3_cross_embs.append(y3_cross_emb.cpu().detach().numpy())
            y4_cross_embs.append(y4_cross_emb.cpu().detach().numpy())
        
        ## Iterative for measuring impact of n_steps
        else:
            noise = torch.randn((max(x.shape[0], m.shape[0]), 1, n_samples), device=device)
            with ctx:
                y1 = base1.iterative_reconstruct(x, n_steps=n_steps, noise=noise[:x.shape[0]])
                y2 = base2.iterative_reconstruct(x, n_steps=n_steps, noise=noise[:x.shape[0]])
                y3 = measure1.iterative_reconstruct(m, n_steps=n_steps, noise=noise[:m.shape[0]])
                
                real_emb = fad.forward_features(drop_to_multiple(x, 16383 * 5))
            
            y3 = [np.concatenate([restore_measure(y.squeeze(), ratio) for y, ratio in zip(y3_element.cpu().detach().numpy(), ratios)], axis=0) for y3_element in y3]
            y3 = [torch.from_numpy(y3_element.astype(np.float32)).unsqueeze(1).pin_memory().to(device, non_blocking=True) for y3_element in y3]
            
            base1_emb, base2_emb, measure1_emb = [], [], []
            for exponent in range(math.floor(math.log2(n_steps)) + 1):
                step = 2 ** exponent - 1
                if step >= len(y1):
                    break
                
                with ctx:
                    base1_emb.append(fad.forward_features(drop_to_multiple(y1[step], 16383 * 5)))
                    base2_emb.append(fad.forward_features(drop_to_multiple(y2[step], 16383 * 5)))
                    measure1_emb.append(fad.forward_features(drop_to_multiple(y3[step], 16383 * 5)))
            
            real_embs.append(real_emb.cpu().detach().numpy())
            base1_embs.append(torch.stack(base1_emb, dim=1).cpu().detach().numpy())
            base2_embs.append(torch.stack(base2_emb, dim=1).cpu().detach().numpy())
            measure1_embs.append(torch.stack(measure1_emb, dim=1).cpu().detach().numpy())
        
        if idx % 4 == 0:
            name = os.path.basename(measure_path)
            to_sample = np.random.randint(len(x))
            sf.write(
                file=os.path.join(out_dir, f'{idx}_real_{name}'), 
                data=x[to_sample].flatten().cpu().detach().numpy(), 
                samplerate=rate,
                subtype='PCM_16'
            )
            sf.write(
                file=os.path.join(out_dir, f'{idx}_base_{name}'), 
                data=y1[to_sample].flatten().cpu().detach().numpy(), 
                samplerate=rate,
                subtype='PCM_16'
            )
            sf.write(
                file=os.path.join(out_dir, f'{idx}_measure_global_{name}'), 
                data=y2[to_sample].flatten().cpu().detach().numpy(), 
                samplerate=rate,
                subtype='PCM_16'
            )
            sf.write(
                file=os.path.join(out_dir, f'{idx}_measure_movingaverage_{name}'), 
                data=y3[to_sample].flatten().cpu().detach().numpy(), 
                samplerate=rate,
                subtype='PCM_16'
            )
            sf.write(
                file=os.path.join(out_dir, f'{idx}_measure_median_{name}'), 
                data=y4[to_sample].flatten().cpu().detach().numpy(), 
                samplerate=rate,
                subtype='PCM_16'
            )
            sf.write(
                file=os.path.join(out_dir, f'{idx}_measure_cross_global_{name}'), 
                data=y2_cross[to_sample].flatten().cpu().detach().numpy(), 
                samplerate=rate,
                subtype='PCM_16'
            )
            sf.write(
                file=os.path.join(out_dir, f'{idx}_measure_cross_movingaverage_{name}'), 
                data=y3_cross[to_sample].flatten().cpu().detach().numpy(), 
                samplerate=rate,
                subtype='PCM_16'
            )
            sf.write(
                file=os.path.join(out_dir, f'{idx}_measure_cross_median_{name}'), 
                data=y4_cross[to_sample].flatten().cpu().detach().numpy(), 
                samplerate=rate,
                subtype='PCM_16'
            )

if not EVAL_ITERATIVE:
    real_mu, real_sigma = calculate_embd_statistics(np.concatenate(real_embs, axis=0))
    y1_mu, y1_sigma = calculate_embd_statistics(np.concatenate(y1_embs, axis=0))
    y2_mu, y2_sigma = calculate_embd_statistics(np.concatenate(y2_embs, axis=0))
    y3_mu, y3_sigma = calculate_embd_statistics(np.concatenate(y3_embs, axis=0))
    y4_mu, y4_sigma = calculate_embd_statistics(np.concatenate(y4_embs, axis=0))
    
    y2_cross_mu, y2_cross_sigma = calculate_embd_statistics(np.concatenate(y2_cross_embs, axis=0))
    y3_cross_mu, y3_cross_sigma = calculate_embd_statistics(np.concatenate(y3_cross_embs, axis=0))
    y4_cross_mu, y4_cross_sigma = calculate_embd_statistics(np.concatenate(y4_cross_embs, axis=0))

    y1_fad = calculate_frechet_distance(y1_mu, y1_sigma, real_mu, real_sigma)
    y2_fad = calculate_frechet_distance(y2_mu, y2_sigma, real_mu, real_sigma)
    y3_fad = calculate_frechet_distance(y3_mu, y3_sigma, real_mu, real_sigma)
    y4_fad = calculate_frechet_distance(y4_mu, y4_sigma, real_mu, real_sigma)
    
    y2_cross_fad = calculate_frechet_distance(y2_cross_mu, y2_cross_sigma, real_mu, real_sigma)
    y3_cross_fad = calculate_frechet_distance(y3_cross_mu, y3_cross_sigma, real_mu, real_sigma)
    y4_cross_fad = calculate_frechet_distance(y4_cross_mu, y4_cross_sigma, real_mu, real_sigma)

    print('Base -> Real Samples FAD: ', y1_fad)
    print('Measure (Global BPM) -> Real Samples FAD: ', y2_fad)
    print('Measure (Moving Average BPM) -> Real Samples FAD: ', y3_fad)
    print('Measure (Median BPM) -> Real Samples FAD: ', y4_fad)
    print('Measure (Crossfade Global BPM) -> Real Samples FAD: ', y2_cross_fad)
    print('Measure (Crossfade Moving Average BPM) -> Real Samples FAD: ', y3_cross_fad)
    print('Measure (Crossfade Median BPM) -> Real Samples FAD: ', y4_cross_fad)

else: 
    fads = defaultdict(list)
    for exponent in range(math.floor(math.log2(n_steps)) + 1):
        base1_mu, base1_sigma = calculate_embd_statistics(np.concatenate(base1_embs, axis=0)[:, exponent])
        base2_mu, base2_sigma = calculate_embd_statistics(np.concatenate(base2_embs, axis=0)[:, exponent])
        measure1_mu, measure1_sigma = calculate_embd_statistics(np.concatenate(measure1_embs, axis=0)[:, exponent])

        base1_fad = calculate_frechet_distance(base1_mu, base1_sigma, real_mu, real_sigma)
        base2_fad = calculate_frechet_distance(base2_mu, base2_sigma, real_mu, real_sigma)
        measure1_fad = calculate_frechet_distance(measure1_mu, measure1_sigma, real_mu, real_sigma)
        
        fads['base1'].append(base1_fad)
        fads['base2'].append(base2_fad)
        fads['measure1'].append(measure1_fad)

        print('='*60)
        print(f'Step {2 ** exponent}')
        print('='*60)
        print('Base 1 FAD: ', base1_fad)
        print('Base 2 FAD: ', base2_fad)
        print('Measure 1 FAD: ', measure1_fad)