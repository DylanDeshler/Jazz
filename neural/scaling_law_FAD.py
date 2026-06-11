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
from einops import rearrange

from dito import DiToV5 as Tokenizer
from fad import MultiTaskFAD as FAD, BPMProbe
from adapter import InvertibleAdapter as Adapter
from diffusion_forcing import UnconditionalModernDiT_smedium_L1, UnconditionalModernDiT_smedium_L2, UnconditionalModernDiT_smedium_L3, UnconditionalModernDiT_smedium_L4, UnconditionalModernDiT_smedium_L5
import argparse
parser = argparse.ArgumentParser(description="Process a specific level argument.")
valid_levels = [f"L{i}" for i in range(1, 3)]

parser.add_argument(
    '--batch_size', 
    type=int, 
    required=True, 
)
parser.add_argument(
    '--n_samples', 
    type=int, 
    required=True, 
)
parser.add_argument(
    '--n_steps', 
    type=int, 
    default=32,
)
parser.add_argument(
    '--device', 
    type=str, 
    required=True,
    help="Specify the device. Like cuda:1."
)
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

rate = 16000
n_samples = 24576
TARGET_SIG = 4
TARGET_BPM = 60 * TARGET_SIG / (n_samples / rate)

device = args.device
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def load_model(ckpt_path, ModelType):
    print(f'Loading model {ckpt_path} ...')
    checkpoint = torch.load(ckpt_path, map_location=device)
    tokenizer_args = checkpoint['model_args']

    model = ModelType(**tokenizer_args).to(device)
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    else:
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
    
    bpm_np = bpm_tensor.float().cpu().detach().numpy()
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

def crossfade_segments(segment_a, segment_b, sample_rate, crossfade_ms=15):
    """
    Crossfades two 1D numpy audio arrays to prevent boundary clicks.
    
    Args:
        segment_a: numpy array of the first measure.
        segment_b: numpy array of the second measure.
        sample_rate: The sample rate of the audio (e.g., 44100 or 24000).
        crossfade_ms: Duration of the crossfade in milliseconds.
    """
    # Convert milliseconds to exact sample count
    crossfade_samples = int(sample_rate * (crossfade_ms / 1000.0))
    
    # Safety check: if segments are too short, just concatenate
    if len(segment_a) < crossfade_samples or len(segment_b) < crossfade_samples:
        return np.concatenate((segment_a, segment_b))
        
    # Create linear fade curves (can also use np.cos for equal-power crossfades)
    fade_out = np.linspace(1.0, 0.0, crossfade_samples)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples)
    
    # Apply fades to the overlapping edges
    overlap_a = segment_a[-crossfade_samples:] * fade_out
    overlap_b = segment_b[:crossfade_samples] * fade_in
    
    # Sum the overlapped audio
    mixed_overlap = overlap_a + overlap_b
    
    # Stitch the untouched beginnings/ends with the mixed overlap
    stitched_audio = np.concatenate((
        segment_a[:-crossfade_samples],
        mixed_overlap,
        segment_b[crossfade_samples:]
    ))
    
    return stitched_audio

def predict_measures(model, gen_shape, net_kwargs, uncond_net_kwargs, n_steps, guidance=1, gen_noise=None, decoder_noise=None, method='median', window_size=3, memory_efficient=False, rescale_phi=0, cfg_mode="independent", t_dist="uniform"):
    with ctx:
        y = model.generate(gen_shape, net_kwargs=net_kwargs, uncond_net_kwargs=uncond_net_kwargs, n_steps=n_steps, guidance=guidance, noise=gen_noise, memory_efficient=memory_efficient, rescale_phi=rescale_phi, cfg_mode=cfg_mode, t_dist=t_dist)
    
        bpm = probe(y)
    
    bpm = smooth_bpm_predictions(bpm, method=method, window_size=window_size)
    seconds_per_beat = 60.0 / bpm
    measure_duration_sec = seconds_per_beat * TARGET_SIG
    
    target_samples = (measure_duration_sec * 16000).long()
    max_len = min(target_samples.max().item(), encoder_ratios * (max_seq_len - 1))
    max_len = encoder_ratios * math.ceil(max_len / encoder_ratios)
    max_latent_len = max_len // encoder_ratios
    
    indices = torch.arange(max_latent_len, device=device).view(1, 1, -1)
    lengths = ((target_samples + encoder_ratios - 1) // encoder_ratios).unsqueeze(-1)
    mask = indices < lengths
    mask = mask.view(gen_shape[0] * measure_chunks, max_latent_len)
    shape = (gen_shape[0] * measure_chunks, vae_embed_dim, max_latent_len)
        
    with ctx:
        y = rearrange(y, 'b t n c -> (b t) c n')
        y = adapter.decode(y, shape, mask=mask)
        y = tokenizer.decode(y, shape=(1, max_len), n_steps=n_steps, noise=decoder_noise[:, :, :max_len] if decoder_noise is not None else None)
    
    target_samples = target_samples.flatten().cpu().detach().numpy()
    y = y.squeeze().cpu().detach().numpy()
    
    target_samples = target_samples.reshape(gen_shape[0], measure_chunks)
    
    out = []
    for i in range(gen_shape[0]):
        temp = y[i*measure_chunks][:min(int(target_samples[i][0]), max_len)]
        for j in range(1, measure_chunks):
            temp = crossfade_segments(temp, y[i*measure_chunks+j][:min(int(target_samples[i][j]), max_len)], sample_rate=16000, crossfade_ms=20)
        out.append(temp.astype(np.float32))
        
    out = np.concatenate(out, axis=0)
    out = torch.from_numpy(out.astype(np.float32)).to(device, non_blocking=True)
    
    return out

# Tokenizers
tokenizer = load_model(os.path.join('tokenizer_low_large_24576_subset_longtrain', 'ckpt.pt'), Tokenizer)
encoder_ratios = math.prod(tokenizer.encoder.ratios)
adapter = load_model(os.path.join('tokenizer_adapter_low_large_24576_subset_longtrain_v2', 'ckpt.pt'), Adapter)
max_seq_len = adapter.max_seq_len
probe = load_model(os.path.join('tokenizer_low_measures_fix_subset_longtrain_v2_64_BPMProbe_tiny', 'ckpt.pt'), BPMProbe)

# DiTs
levels = [f"L{i}" for i in range(1, 3)]
dits = [UnconditionalModernDiT_smedium_L1, UnconditionalModernDiT_smedium_L2, UnconditionalModernDiT_smedium_L3, UnconditionalModernDiT_smedium_L4, UnconditionalModernDiT_smedium_L5]
base_dits = [load_model(f'UnconditionalModernDiT_smedium_{level}_24576_subset_longtrain_32chunks/ckpt.pt', DiT) for level, DiT in zip(levels, dits)]
measure_dits = [load_model(f'UnconditionalModernDiT_smedium_{level}_24576_subset_adapter_longtrain_24chunks/ckpt.pt', DiT) for level, DiT in zip(levels, dits)]
base_chunks = 32
base_window = 48
measure_chunks = 24
measure_window = 64
vae_embed_dim = 16
assert base_chunks * base_window * vae_embed_dim == measure_chunks * measure_window * vae_embed_dim

# Feature Extractors
fad = load_model(os.path.join('FAD', 'ckpt.pt'), FAD)

paths = glob.glob('/data/wavs/*')
paths = paths[-int(len(paths) * 2/48):] # test set

n_generations = args.n_samples
batch_size = args.batch_size
n_steps = args.n_steps
assert n_generations % batch_size == 0

idxs = np.random.choice(np.arange(len(paths)), size=n_generations, replace=False)

real_embs = []
y1_embs = defaultdict(list)
y2_embs = defaultdict(list)
with torch.no_grad():
    for idx in tqdm(idxs, desc='Embedding Real Samples'):
        path = paths[idx]
        wav, _ = librosa.load(path, sr=rate)
        wav = wav[:batch_size * base_chunks * rate]
        x_raw = wav
        
        x = torch.from_numpy(x_raw.astype(np.float32)).to(device, non_blocking=True)
        x = drop_to_multiple(x, 16383 * 5)
        
        with ctx:
            real_emb = fad.forward_features(x)
            print(real_emb.shape)
        
        real_embs.append(real_emb.cpu().detach().numpy())
        
    for _ in tqdm(range(n_generations // batch_size), desc='Embedding Generated Samples'):
        base_gen_shape = (batch_size, base_chunks, base_window, vae_embed_dim)
        base_gen_noise = torch.randn(base_gen_shape, device=device)
        base_decoder_noise = torch.randn((batch_size * base_chunks, 1, 24576), device=device)
        
        with ctx:
            for i, base_dit in enumerate(base_dits):
                y1 = base_dit.generate(base_gen_shape, n_steps=n_steps, noise=base_gen_noise)
                y1 = y1.transpose(2, 3).view(batch_size * base_chunks, vae_embed_dim, base_window)
                y1 = tokenizer.decode(y1, shape=(1, 24576), n_steps=n_steps, noise=base_decoder_noise)
                
                y1 = drop_to_multiple(y1, 16383 * 5)
                y1_emb = fad.forward_features(y1)
                print(y1_emb.shape)
        
                y1_embs[levels[i]].append(y1_emb.cpu().detach().numpy())
        
        measure_gen_shape = (batch_size, measure_chunks, measure_window, vae_embed_dim)
        measure_gen_noise = torch.randn(measure_gen_shape, device=device)
        measure_decoder_noise = torch.randn((batch_size * measure_chunks, 1, encoder_ratios * (max_seq_len - 1)), device=device)
        for i, measure_dit in enumerate(measure_dits):
            y2 = predict_measures(measure_dit, measure_gen_shape, None, None, n_steps, guidance=1.0, gen_noise=measure_gen_noise, decoder_noise=measure_decoder_noise, method='median', window_size=3)
            with ctx:
                y2 = drop_to_multiple(y2, 16383 * 5)
                y2_emb = fad.forward_features(y2)
        
            y2_embs[levels[i]].append(y2_emb.cpu().detach().numpy())
        

real_mu, real_sigma = calculate_embd_statistics(np.concatenate(real_embs, axis=0))
for k, v in y1_embs:
    y1_mu, y1_sigma = calculate_embd_statistics(np.concatenate(v, axis=0))
    y1_fad = calculate_frechet_distance(y1_mu, y1_sigma, real_mu, real_sigma)
    
    print(f'Base {k} -> Real Samples FAD: ', y1_fad)

for k, v in y2_embs:
    y2_mu, y2_sigma = calculate_embd_statistics(np.concatenate(v, axis=0))
    y2_fad = calculate_frechet_distance(y2_mu, y2_sigma, real_mu, real_sigma)

    print(f'Measure (Median BPM) {k} -> Real Samples FAD: ', y2_fad)
