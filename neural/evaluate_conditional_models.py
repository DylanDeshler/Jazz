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
import optuna

from dito import DiToV5 as Tokenizer
from fad import MultiTaskFAD as FAD, BPMProbe
from adapter import InvertibleAdapter as Adapter
from diffusion_forcing import MetaConditionalModernDiT_large as DiT

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

@torch.no_grad()
def predict_measures(gen_shape, net_kwargs, uncond_net_kwargs, n_steps, guidance=1, gen_noise=None, decoder_noise=None, method='median', window_size=3, memory_efficient=False):
    with ctx:
        y = model.generate(gen_shape, net_kwargs=net_kwargs, uncond_net_kwargs=uncond_net_kwargs, n_steps=n_steps, guidance=guidance, noise=gen_noise, memory_efficient=memory_efficient)
        
    if isinstance(net_kwargs, list):
        bpm = net_kwargs[0]['bpm']
    else:
        bpm = net_kwargs['bpm']
    
    seconds_per_beat = 60.0 / bpm
    measure_duration_sec = seconds_per_beat * TARGET_SIG
    
    target_samples = (measure_duration_sec * 16000).long()
    max_len = min(target_samples.max().item(), encoder_ratios * (max_seq_len - 1))
    max_len = encoder_ratios * math.ceil(max_len / encoder_ratios)
    max_latent_len = max_len // encoder_ratios
    
    indices = torch.arange(max_latent_len, device=device).view(1, 1, -1)
    lengths = ((target_samples + encoder_ratios - 1) // encoder_ratios).unsqueeze(-1)
    mask = indices < lengths
    mask = mask.view(gen_shape[0] * n_chunks, max_latent_len)
    shape = (gen_shape[0] * n_chunks, 1, max_latent_len)
        
    with ctx:
        y = y.transpose(2, 3).view(gen_shape[0] * n_chunks, vae_embed_dim, spatial_window)
        y = adapter.decode(y, shape, mask=mask)
        y = tokenizer.decode(y, shape=(1, max_len), n_steps=n_steps, noise=decoder_noise[:, :, :max_len] if decoder_noise is not None else None)
    
    target_samples = target_samples.flatten().cpu().detach().numpy()
    y = y.squeeze().cpu().detach().numpy()
    
    target_samples = target_samples.reshape(gen_shape[0], n_chunks)
    # y = [np.concatenate([y_[:min(int(samples), max_len)] for y_, samples in zip(y[i*n_chunks:(i+1)*n_chunks], target_samples[i])], axis=0).astype(np.float32) for i in range(gen_shape[0])]
    
    y = [[y_[:min(int(samples), max_len)].astype(np.float32) for y_, samples in zip(y[i*n_chunks:(i+1)*n_chunks], target_samples[i])] for i in range(gen_shape[0])]
    
    return y

# Tokenizers
tokenizer = load_model(os.path.join('tokenizer_low_large_24576_subset_longtrain', 'ckpt.pt'), Tokenizer)
adapter = load_model(os.path.join('tokenizer_adapter_low_large_24576_subset_longtrain', 'ckpt.pt'), Adapter)
encoder_ratios = math.prod(tokenizer.encoder.ratios)
max_seq_len = adapter.max_seq_len

# DiTs
model = load_model(os.path.join('MetaConditionalModernDiT_large_24576_subset_adapter_longtrain_32chunks', 'ckpt.pt'), DiT)
n_chunks = 32
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

measure_paths = glob.glob('/home/ubuntu/Data/wavs/*')
measure_paths = measure_paths[-int(len(measure_paths) * 2/48):] # test set

hop_length = 1024 # average over time so large hop is fine
n_fft = 2048

data = np.memmap('/home/ubuntu/Data/low_large_24576_subset_adapter_longtrain_val.bin', dtype=np.float32, mode='r', shape=(88303, spatial_window, vae_embed_dim))
styles = np.memmap('/home/ubuntu/Data/contrast_learntmep_instance_10s_style_val.bin', dtype=np.float32, mode='r', shape=(88303, 128))
meta = np.memmap('/home/ubuntu/Data/low_large_24576_subset_meta_val.bin', dtype=np.float32, mode='r', shape=(88303, 29))
bpms = np.memmap('/home/ubuntu/Data/low_large_24576_subset_adapter_longtrain_bpm_val.bin', dtype=np.float32, mode='r')

@torch.no_grad()
def run_optuna_experiments(batch_size, micro_batch_size, n_steps):
    assert batch_size % micro_batch_size == 0
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    def mse(x, y):
        return np.mean((x - y) ** 2)
    
    x = []
    path_idxs = np.random.choice(np.arange(len(measure_paths)), size=batch_size, replace=False)
    for i in path_idxs:
        wav, _ = librosa.load(measure_paths[i], sr=None)
        start = np.random.randint(len(wav) - n_chunks * rate)
        wav = wav[start:start + n_chunks * rate]
        x.append(wav)
    
    x = torch.from_numpy(np.stack(x).astype(np.float32)).to(device, non_blocking=True)
    x = drop_to_multiple(x, 16383 * 5)
    
    with ctx:
        emb = fad.forward_features(x)

    real_mu, real_sigma = calculate_embd_statistics(emb.cpu().detach().numpy())
    
    idxs = torch.randint(len(data) - n_chunks, (batch_size,))
    
    style = torch.from_numpy(np.stack([styles[idx:idx+n_chunks] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    chroma = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, :12] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    rms_low = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 12] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    rms_mid = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 13] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    rms_high = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 14] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    density = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 15] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    zcr = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 16] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    mfcc = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 17:] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    bpm = torch.from_numpy(np.stack([bpms[idx:idx+n_chunks] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    
    gen_shape = (micro_batch_size, n_chunks, spatial_window, vae_embed_dim)
    gen_noise = torch.randn(gen_shape).to(device)
    decoder_noise = torch.randn(micro_batch_size * n_chunks, 1, encoder_ratios * (max_seq_len - 1)).to(device)
    
    @torch.no_grad()  
    def objective(trial, batch_size, micro_batch_size, n_steps):
        scales = {
            'bpm': trial.suggest_float('w_bpm', 0, 5),
            'rms_low': trial.suggest_float('w_rms_low', 0, 5),
            'rms_mid': trial.suggest_float('w_rms_mid', 0, 5),
            'rms_high': trial.suggest_float('w_rms_high', 0, 5),
            'density': trial.suggest_float('w_density', 0, 5),
            'zcr': trial.suggest_float('w_zcr', 0, 5),
            'mfcc': trial.suggest_float('w_mfcc', 0, 5),
            'chroma': trial.suggest_float('w_chroma', 0, 5),
            'style': trial.suggest_float('w_style', 0, 5)
        }
        
        cfg_guidances = list(scales.values())
        errors, embs = [], []
        for micro_batch in range(batch_size // micro_batch_size):
            start_idx = micro_batch * micro_batch_size
            end_idx = start_idx + micro_batch_size
            
            mb_bpm = bpm[start_idx:end_idx]
            mb_rms_low = rms_low[start_idx:end_idx]
            mb_rms_mid = rms_mid[start_idx:end_idx]
            mb_rms_high = rms_high[start_idx:end_idx]
            mb_density = density[start_idx:end_idx]
            mb_zcr = zcr[start_idx:end_idx]
            mb_mfcc = mfcc[start_idx:end_idx]
            mb_chroma = chroma[start_idx:end_idx]
            mb_style = style[start_idx:end_idx]

            net_kwargs = {
                'bpm': mb_bpm,
                'rms_low': mb_rms_low,
                'rms_mid': mb_rms_mid,
                'rms_high': mb_rms_high,
                'density': mb_density,
                'zcr': mb_zcr,
                'mfcc': mb_mfcc,
                'chroma': mb_chroma,
                'style': mb_style,
            }
            
            unconditional_mask = {
                'bpm': torch.ones(*mb_bpm.shape, 1, device=device, dtype=torch.bool),
                'rms_low': torch.ones(*mb_rms_low.shape, 1, device=device, dtype=torch.bool),
                'rms_mid': torch.ones(*mb_rms_mid.shape, 1, device=device, dtype=torch.bool),
                'rms_high': torch.ones(*mb_rms_high.shape, 1, device=device, dtype=torch.bool),
                'density': torch.ones(*mb_density.shape, 1, device=device, dtype=torch.bool),
                'zcr': torch.ones(*mb_zcr.shape, 1, device=device, dtype=torch.bool),
                'mfcc': torch.ones(*mb_mfcc.shape[:-1], 1, device=device, dtype=torch.bool),
                'chroma': torch.ones(*mb_chroma.shape[:-1], 1, device=device, dtype=torch.bool),
                'style': torch.ones(*mb_style.shape[:-1], 1, device=device, dtype=torch.bool),
            }
            
            cfg_net_kwargs = []
            for k, v in unconditional_mask.items():
                temp_mask = unconditional_mask.copy()
                temp_mask[k] = ~v
                cfg_net_kwargs.append(net_kwargs | {'unconditional_mask': temp_mask})
                
                uncond_net_kwargs = net_kwargs | {'unconditional_mask': unconditional_mask}
                
            y = predict_measures(
                gen_shape, 
                cfg_net_kwargs, 
                uncond_net_kwargs, 
                n_steps, 
                guidance=cfg_guidances, 
                gen_noise=gen_noise, 
                decoder_noise=decoder_noise, 
                method='median', 
                window_size=3,
                memory_efficient=False
            )
            
            error = 0
            for batch, measure_list in enumerate(y):
                wav = np.concatenate(measure_list, axis=0)
                wav_chroma = librosa.feature.chroma_cqt(y=wav, sr=rate, hop_length=hop_length)
                stft_mag = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
                freqs = librosa.fft_frequencies(sr=rate, n_fft=n_fft)
                
                low_mask = (freqs < 250)
                mid_mask = (freqs >= 250) & (freqs < 4000)
                high_mask = (freqs >= 4000)
                
                wav_rms_low = np.sqrt(np.mean(stft_mag[low_mask, :]**2, axis=0))
                wav_rms_mid = np.sqrt(np.mean(stft_mag[mid_mask, :]**2, axis=0))
                wav_rms_high = np.sqrt(np.mean(stft_mag[high_mask, :]**2, axis=0))
                
                wav_onset_env = librosa.onset.onset_strength(y=wav, sr=rate, hop_length=hop_length)
                wav_onset_frames = librosa.onset.onset_detect(onset_envelope=wav_onset_env, sr=rate, hop_length=hop_length)
                wav_zcr = librosa.feature.zero_crossing_rate(wav, hop_length=hop_length)[0]
                
                # Extract 13 MFCCs, but strictly slice [1:13] to discard the 0th energy coefficient
                wav_mfccs = librosa.feature.mfcc(y=wav, sr=rate, hop_length=hop_length, n_mfcc=13)[1:13, :]
                
                current_sample = 0
                for i in range(len(measure_list) - 1):
                    t_start = current_sample / rate
                    t_end = (current_sample + len(measure_list[i])) / rate
                    current_sample += len(measure_list[i])
                    
                    frame_start = librosa.time_to_frames(t_start, sr=rate, hop_length=hop_length)
                    frame_end = librosa.time_to_frames(t_end, sr=rate, hop_length=hop_length)
                    
                    if frame_end > wav_chroma.shape[1]:
                        break
                        
                    frame_end = min(frame_end, wav_chroma.shape[1])
                    
                    measure_chroma = wav_chroma[:, frame_start:frame_end]
                    
                    measure_rms_low = wav_rms_low[frame_start:frame_end]
                    measure_rms_mid = wav_rms_mid[frame_start:frame_end]
                    measure_rms_high = wav_rms_high[frame_start:frame_end]
                    
                    measure_zcr = wav_zcr[frame_start:frame_end]
                    
                    measure_mfcc = wav_mfccs[:, frame_start:frame_end]
                    
                    if measure_chroma.shape[1] > 0:
                        chroma_error = mse(np.mean(measure_chroma, axis=1), mb_chroma[batch, i].cpu().numpy())
                        rms_low_error = mse(np.mean(measure_rms_low), mb_rms_low[batch, i].cpu().numpy())
                        rms_mid_error = mse(np.mean(measure_rms_mid), mb_rms_mid[batch, i].cpu().numpy())
                        rms_high_error = mse(np.mean(measure_rms_high), mb_rms_high[batch, i].cpu().numpy())
                        
                        onsets_in_measure = np.sum((wav_onset_frames >= frame_start) & (wav_onset_frames < frame_end))
                        measure_duration_sec = (frame_end - frame_start) / (rate / hop_length)
                        density_error = mse(onsets_in_measure / measure_duration_sec if measure_duration_sec > 0 else 0.0, density[batch, i].cpu().numpy())
                        
                        zcr_error = mse(np.mean(measure_zcr), mb_zcr[batch, i].cpu().numpy())
                        mfcc_error = mse(np.mean(measure_mfcc, axis=1), mb_mfcc[batch, i].cpu().numpy())
                        
                        error += chroma_error + rms_low_error + rms_mid_error + rms_high_error + density_error + zcr_error + mfcc_error
            
            error /= len(y)
            errors.append(error)
            
            y = torch.from_numpy(np.concatenate([np.concatenate(y_, axis=0) for y_ in y], axis=0).astype(np.float32)).to(device, non_blocking=True)
            y = drop_to_multiple(y, 16383 * 5)
                
            with ctx:
                emb = fad.forward_features(y)
            embs.append(emb.cpu().detach().numpy())
            
            del y, emb, temp_mask, measure_list
            del cfg_net_kwargs, uncond_net_kwargs, net_kwargs, unconditional_mask
            del mb_bpm, mb_rms_low, mb_rms_mid, mb_rms_high, mb_density, mb_zcr, mb_mfcc, mb_chroma, mb_style
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        y_mu, y_sigma = calculate_embd_statistics(np.concatenate(embs, axis=0))
        fad_score = calculate_frechet_distance(y_mu, y_sigma, real_mu, real_sigma)

        return np.mean(errors).item(), fad_score
    
    study = optuna.create_study(
        study_name='cfg',
        storage='sqlite:///cfg_optimization.db',
        directions=['minimize', 'minimize'],
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, batch_size, micro_batch_size, n_steps), n_trials=100)

    best_trials = study.best_trials
    for t in best_trials:
        print(f"DSP Error: {t.values[0]:.2f}, FAD: {t.values[1]:.2f} | Scales: {t.params}")

def run_eval(batch_size, n_steps):
    torch.manual_seed(0)
    np.random.seed(0)
    
    idxs = np.random.choice(np.arange(len(measure_paths)), size=batch_size, replace=False)
    
    real_embs = []
    y6_embs = []
    with torch.no_grad():
        for idx in tqdm(idxs):
            measure_path = measure_paths[idx]
            wav, _ = librosa.load(measure_path, sr=None)
            wav = wav[:batch_size * n_chunks * rate]
            x_raw = wav
            
            gen_shape = (batch_size, n_chunks, spatial_window, vae_embed_dim)
            gen_noise = torch.randn(gen_shape).to(device)
            decoder_noise = torch.randn(batch_size * n_chunks, 1, encoder_ratios * (max_seq_len - 1)).to(device)
            
            idxs = np.random.randint(len(styles) - n_chunks, size=(batch_size,))
            style = torch.from_numpy(np.stack([styles[idx:idx+n_chunks] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
            chroma = torch.from_numpy(np.stack([chroma_rms[idx:idx+n_chunks, :12] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
            rms = torch.from_numpy(np.stack([chroma_rms[idx:idx+n_chunks, -1] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
            bpm = torch.from_numpy(np.stack([meta[idx:idx+n_chunks] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
            
            unconditional_mask = {
                'bpm': torch.ones(*bpm.shape, 1).to(device).bool(),
                'rms': torch.ones(*rms.shape, 1).to(device).bool(),
                'chroma': torch.ones(*chroma.shape[:-1], 1).to(device).bool(),
                'style': torch.ones(*style.shape[:-1], 1).to(device).bool(),
            }
            net_kwargs = {
                'bpm': bpm,
                'rms': rms,
                'chroma': chroma,
                'style': style,
            }
            
            cfg_net_kwargs = []
            for k, v in unconditional_mask.items():
                temp_mask = unconditional_mask.copy()
                temp_mask[k] = ~v
                cfg_net_kwargs.append(net_kwargs | {'unconditional_mask': temp_mask})
            
            uncond_net_kwargs = net_kwargs | {'unconditional_mask': unconditional_mask}
            
            cfg_guidances = [2, 2, 3, 7]
            y_cfg = predict_measures(
                gen_shape, 
                cfg_net_kwargs, 
                uncond_net_kwargs, 
                n_steps, 
                guidance=cfg_guidances, 
                gen_noise=gen_noise, 
                decoder_noise=decoder_noise, 
                method='median', 
                window_size=3
            )
            
            x = torch.from_numpy(x_raw.astype(np.float32)).to(device, non_blocking=True)
            
            x = drop_to_multiple(x, 16383 * 5)
            y6 = drop_to_multiple(y6, 16383 * 5)
            
            with ctx:
                real_emb = fad.forward_features(x)
                y6_emb = fad.forward_features(y6)
            
            real_embs.append(real_emb.cpu().detach().numpy())
            y6_embs.append(y6_emb.cpu().detach().numpy())

    real_mu, real_sigma = calculate_embd_statistics(np.concatenate(real_embs, axis=0))
    y6_mu, y6_sigma = calculate_embd_statistics(np.concatenate(y6_embs, axis=0))
    y6_fad = calculate_frechet_distance(y6_mu, y6_sigma, real_mu, real_sigma)

    print('Style Measure (Median BPM) -> Real Samples FAD: ', y6_fad)

if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="CFG Tuning and Evaluation")
    
    parser.add_argument(
        "mode", 
        choices=["run_optuna_experiments", "run_eval"], 
        help="The primary function to execute."
    )
    
    parser.add_argument(
        "--n_steps", 
        type=int, 
        default=20, 
        help="Number of reverse diffusion steps (default: 20)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for generating audio/FAD calculation (default: 32)"
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=None,
        help="Batch size used for generation (defaults to batch_size)"
    )
    
    args = parser.parse_args()
    if args.micro_batch_size is None:
        args.micro_batch_size = args.batch_size
    
    # Route to the appropriate function
    if args.mode == "run_optuna_experiments":
        run_optuna_experiments(args.batch_size, args.micro_batch_size, args.n_steps)
    elif args.mode == "run_eval":
        run_eval(args.batch_size, args.micro_batch_size, args.n_steps)
        