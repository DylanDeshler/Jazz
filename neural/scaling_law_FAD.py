import os
import glob
import math
import json
import librosa
import soundfile as sf
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import time

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
from diffusion_forcing import UnconditionalModernDiT_smedium_W0, UnconditionalModernDiT_smedium_W1, UnconditionalModernDiT_smedium_W2, UnconditionalModernDiT_smedium_W3, UnconditionalModernDiT_smedium_W4, UnconditionalModernDiT_smedium_W5
from diffusion_forcing import UnconditionalModernDiT_smedium_D0, UnconditionalModernDiT_smedium_D1, UnconditionalModernDiT_smedium_D2, UnconditionalModernDiT_smedium_D3, UnconditionalModernDiT_smedium_D4, UnconditionalModernDiT_smedium_D5
import argparse

parser = argparse.ArgumentParser(description="Process a specific level argument.")

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
parser.add_argument(
    '--axis',
    type=str,
    required=True,
    choices=['width', 'depth']
)
parser.add_argument(
    '--recompute_only',
    action='store_true',
    help="If set, aggregates all saved embeddings across any batch_size/n_samples matching this axis and n_steps."
)

args = parser.parse_args()

valid_levels = [f"{args.axis[0].upper()}{i}" for i in range(0, 6)]

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
    # if 'cuda' in device:
    #     model = torch.compile(model, mode="reduce-overhead")
    return model

def calculate_embd_statistics(embd_lst):
    if isinstance(embd_lst, list):
        embd_lst = np.array(embd_lst)
    
    mu = np.mean(embd_lst, axis=0)
    sigma = np.cov(embd_lst, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2).astype(complex), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset).astype(complex))

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
    if method == 'global':
        mean_bpm = bpm_tensor.mean(dim=1, keepdim=True)
        return mean_bpm.expand_as(bpm_tensor)
    
    bpm_np = bpm_tensor.float().cpu().detach().numpy()
    smoothed = np.zeros_like(bpm_np)
    
    for i in range(bpm_np.shape[0]):
        if method == 'median':
            smoothed[i] = medfilt(bpm_np[i], kernel_size=window_size)
        elif method == 'moving_average':
            kernel = np.ones(window_size) / window_size
            padded = np.pad(bpm_np[i], (window_size//2, window_size//2), mode='edge')
            smoothed[i] = np.convolve(padded, kernel, mode='valid')
            
    return torch.from_numpy(smoothed).to(bpm_tensor.device)

def crossfade_segments(segment_a, segment_b, sample_rate, crossfade_ms=15):
    crossfade_samples = int(sample_rate * (crossfade_ms / 1000.0))
    if len(segment_a) < crossfade_samples or len(segment_b) < crossfade_samples:
        return np.concatenate((segment_a, segment_b))
        
    fade_out = np.linspace(1.0, 0.0, crossfade_samples)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples)
    
    overlap_a = segment_a[-crossfade_samples:] * fade_out
    overlap_b = segment_b[:crossfade_samples] * fade_in
    mixed_overlap = overlap_a + overlap_b
    
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

real_embs = []
y1_embs = defaultdict(list)
y2_embs = defaultdict(list)

if args.recompute_only:
    print(f">>> Mode: [Recompute Only] Aggregating all runs matching axis={args.axis} and n_steps={args.n_steps}")
    
    search_pattern = os.path.join("/data/binaries/FAD_embeddings", f"axis_{args.axis}_nsteps_{args.n_steps}_nsamples_*_*")
    matching_dirs = glob.glob(search_pattern)
    
    if not matching_dirs:
        raise FileNotFoundError(f"No output folders found matching axis={args.axis} and n_steps={args.n_steps}")
        
    print(f"Found {len(matching_dirs)} directories to aggregate.")
    
    for run_dir in matching_dirs:
        real_path = os.path.join(run_dir, "real_embs.npy")
        if os.path.exists(real_path):
            real_embs.append(np.load(real_path))
            
        for level in valid_levels:
            y1_path = os.path.join(run_dir, f"y1_embs_{level}.npy")
            y2_path = os.path.join(run_dir, f"y2_embs_{level}.npy")
            
            if os.path.exists(y1_path):
                y1_embs[level].append(np.load(y1_path))
            if os.path.exists(y2_path):
                y2_embs[level].append(np.load(y2_path))

    # Validate that we successfully grabbed data
    if not real_embs:
        raise FileNotFoundError("Could not find any 'real_embs.npy' files across the matched directories.")

else:
    # Mode: Standard Compute & Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("/data/binaries/FAD_embeddings", f"axis_{args.axis}_nsteps_{args.n_steps}_nsamples_{args.n_samples}_{timestamp}")
    os.makedirs(run_dir, exist_ok=False)
    print(f">>> Mode: [Compute Embeddings & FAD] Creating unique output folder: {run_dir}")

    real_embs_path = os.path.join(run_dir, "real_embs.npy")
    y1_embs_path = {level: os.path.join(run_dir, f"y1_embs_{level}.npy") for level in valid_levels}
    y2_embs_path = {level: os.path.join(run_dir, f"y2_embs_{level}.npy") for level in valid_levels}

    # Tokenizers
    tokenizer = load_model(os.path.join('tokenizer_low_large_24576_subset_longtrain', 'ckpt.pt'), Tokenizer)
    encoder_ratios = math.prod(tokenizer.encoder.ratios)
    adapter = load_model(os.path.join('tokenizer_adapter_low_large_24576_subset_longtrain_v2_48', 'ckpt.pt'), Adapter)
    max_seq_len = adapter.max_seq_len
    probe = load_model(os.path.join('tokenizer_low_measures_fix_subset_longtrain_v2_48_BPMProbe_tiny_lstm', 'ckpt.pt'), BPMProbe)

    # DiTs
    if args.axis == 'width':
        dits = [UnconditionalModernDiT_smedium_W0, UnconditionalModernDiT_smedium_W1, UnconditionalModernDiT_smedium_W2, UnconditionalModernDiT_smedium_W3, UnconditionalModernDiT_smedium_W4, UnconditionalModernDiT_smedium_W5]
    else:
        dits = [UnconditionalModernDiT_smedium_D0, UnconditionalModernDiT_smedium_D1, UnconditionalModernDiT_smedium_D2, UnconditionalModernDiT_smedium_D3, UnconditionalModernDiT_smedium_D4, UnconditionalModernDiT_smedium_D5]
    
    base_dits = [load_model(f'UnconditionalModernDiT_smedium_{level}_24576_subset_longtrain_32chunks/ckpt.pt', DiT) for level, DiT in zip(valid_levels, dits)]
    measure_dits = [load_model(f'UnconditionalModernDiT_smedium_{level}_24576_subset_adapter_longtrain_32chunks_v2/ckpt.pt', DiT) for level, DiT in zip(valid_levels, dits)]
    base_chunks, base_window, measure_chunks, measure_window, vae_embed_dim = 32, 48, 32, 48, 16
    assert base_chunks * base_window * vae_embed_dim == measure_chunks * measure_window * vae_embed_dim

    # Feature Extractors
    fad = load_model(os.path.join('FAD_v2_30drop', 'ckpt.pt'), FAD)

    paths = glob.glob('/data/wavs/*')
    paths = paths[-int(len(paths) * 2/48):] # test set

    n_generations = args.n_samples
    batch_size = args.batch_size
    n_steps = args.n_steps
    assert n_generations % batch_size == 0

    durations = []
    for path in tqdm(paths):
        durations.append(sf.info(path).duration)
    durations = np.asarray(durations)
    idxs = np.random.choice(np.arange(len(paths)), size=n_generations, p=durations / np.sum(durations))

    with torch.inference_mode():
        for idx in tqdm(idxs, desc='Embedding Real Samples'):
            path = paths[idx]
            wav, _ = librosa.load(path, sr=rate)
            wav = wav[:batch_size * base_chunks * n_samples]
            x_raw = wav
            
            x = torch.from_numpy(x_raw.astype(np.float32)).to(device, non_blocking=True)
            x = drop_to_multiple(x, 16383 * 5)
            
            with ctx:
                real_emb = fad.forward_features(x)
            
            real_embs.append(real_emb.cpu().detach().numpy())
        np.save(real_embs_path, np.concatenate(real_embs, axis=0))

        total_batches = n_generations // batch_size
        for b in tqdm(range(total_batches), desc='Embedding Generated Samples'):
            base_gen_shape = (batch_size, base_chunks, base_window, vae_embed_dim)
            base_gen_noise = torch.randn(base_gen_shape, device=device)
            base_decoder_noise = torch.randn((batch_size * base_chunks, 1, n_samples), device=device)
            
            for i, base_dit in enumerate(base_dits):
                with ctx:
                    y1 = base_dit.generate(base_gen_shape, n_steps=n_steps, noise=base_gen_noise, memory_efficient=False, cfg_mode='joint', t_dist='logit')
                    y1 = y1.transpose(2, 3).view(batch_size * base_chunks, vae_embed_dim, base_window)
                    y1 = tokenizer.decode(y1, shape=(1, n_samples), n_steps=n_steps, noise=base_decoder_noise)
                    
                    y1 = drop_to_multiple(y1, 16383 * 5)
                    y1_emb = fad.forward_features(y1)
            
                y1_embs[valid_levels[i]].append(y1_emb.cpu().detach().numpy())
                np.save(y1_embs_path[valid_levels[i]], np.concatenate(y1_embs[valid_levels[i]], axis=0))
            
            measure_gen_shape = (batch_size, measure_chunks, measure_window, vae_embed_dim)
            measure_gen_noise = torch.randn(measure_gen_shape, device=device)
            measure_decoder_noise = torch.randn((batch_size * measure_chunks, 1, encoder_ratios * (max_seq_len - 1)), device=device)
            
            for i, measure_dit in enumerate(measure_dits):
                y2 = predict_measures(measure_dit, measure_gen_shape, None, None, n_steps, guidance=1.0, gen_noise=measure_gen_noise, decoder_noise=measure_decoder_noise, method='median', window_size=3)
                with ctx:
                    y2 = drop_to_multiple(y2, 16383 * 5)
                    y2_emb = fad.forward_features(y2)
            
                y2_embs[valid_levels[i]].append(y2_emb.cpu().detach().numpy())
                np.save(y2_embs_path[valid_levels[i]], np.concatenate(y2_embs[valid_levels[i]], axis=0))

real_embs = np.concatenate(real_embs, axis=0)
N = len(real_embs)
for level in valid_levels:
    y1_embs[level] = np.concatenate(y1_embs[level], axis=0)
    y2_embs[level] = np.concatenate(y2_embs[level], axis=0)
        
    N = np.min([N, len(y1_embs[level]), len(y2_embs[level])]).item()
    print(level, f"Real matrix: {real_embs.shape} | Y1 matrix: {y1_embs[level].shape} | Y2 matrix: {y2_embs[level].shape}")

hours = N * 5 / 60 / 60
print(f'\nAggregated down to {N} common embeddings (approx {hours:.2f} hours of data)')

for level in valid_levels:
    level_real_embs = real_embs[np.random.choice(np.arange(len(real_embs)), size=N, replace=False)]
    real_mu, real_sigma = calculate_embd_statistics(level_real_embs)
    
    level_y1_embs = y1_embs[level][np.random.choice(np.arange(len(y1_embs[level])), size=N, replace=False)]
    y1_mu, y1_sigma = calculate_embd_statistics(level_y1_embs)
    y1_fad = calculate_frechet_distance(y1_mu, y1_sigma, real_mu, real_sigma)
    print(f'Base {level} -> Real Samples FAD: ', y1_fad)

    level_y2_embs = y2_embs[level][np.random.choice(np.arange(len(y2_embs[level])), size=N, replace=False)]
    y2_mu, y2_sigma = calculate_embd_statistics(level_y2_embs)
    y2_fad = calculate_frechet_distance(y2_mu, y2_sigma, real_mu, real_sigma)
    print(f'Measure (Median BPM) {level} -> Real Samples FAD: ', y2_fad)