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
from einops import rearrange

from contrast import Transformer as Contrast
from dito import DiToV5 as Tokenizer
from fad import MultiTaskFAD as FAD, BPMProbe
from adapter import InvertibleAdapter as Adapter
from diffusion_forcing import MetaConditionalModernDiTV2_smedium as DiT

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

@torch.no_grad()
def predict_measures(gen_shape, net_kwargs, uncond_net_kwargs, n_steps, guidance=1, gen_noise=None, decoder_noise=None, method='median', window_size=3, memory_efficient=False, rescale_phi=0, cfg_mode="independent", t_dist="uniform"):
    with ctx:
        y = model.generate(gen_shape, net_kwargs=net_kwargs, uncond_net_kwargs=uncond_net_kwargs, n_steps=n_steps, guidance=guidance, noise=gen_noise, memory_efficient=memory_efficient, rescale_phi=rescale_phi, cfg_mode=cfg_mode, t_dist=t_dist)
    
    if isinstance(net_kwargs, list):
        bpm = net_kwargs[0]['bpm']
    else:
        bpm = net_kwargs['bpm']
    
    seconds_per_beat = 60.0 / bpm
    measure_duration_sec = seconds_per_beat * TARGET_SIG
    
    target_samples = (measure_duration_sec * 16000).long()
    max_len = min(target_samples.max().item(), encoder_ratios * (max_adapter_len - 1))
    max_len = encoder_ratios * math.ceil(max_len / encoder_ratios)
    max_latent_len = max_len // encoder_ratios
    
    indices = torch.arange(max_latent_len, device=device).view(1, 1, -1)
    lengths = ((target_samples + encoder_ratios - 1) // encoder_ratios).unsqueeze(-1)
    mask = indices < lengths
    mask = mask.view(gen_shape[0] * n_chunks, max_latent_len)
    shape = (gen_shape[0] * n_chunks, vae_embed_dim, max_latent_len)
        
    with ctx:
        y = rearrange(y, 'b t n c -> (b t) c n')
        y = adapter.decode(y, shape, mask=mask)
        y = tokenizer.decode(y, shape=(1, max_len), n_steps=n_steps, noise=decoder_noise[:, :, :max_len] if decoder_noise is not None else None)
    
    target_samples = target_samples.flatten().cpu().detach().numpy()
    y = y.squeeze().cpu().detach().numpy().astype(np.float32)
    
    target_samples = target_samples.reshape(gen_shape[0], n_chunks)
    
    out = []
    for i in range(gen_shape[0]):
        temp = [y[i*n_chunks][:min(int(target_samples[i][0]), max_len)]]
        for j in range(1, n_chunks):
            temp.append(crossfade_segments(temp[-1], y[i*n_chunks+j][:min(int(target_samples[i][j]), max_len)], sample_rate=16000, crossfade_ms=20))
        out.append(temp)
    
    return out

# Tokenizers
tokenizer = load_model(os.path.join('tokenizer_low_large_24576_subset_longtrain', 'ckpt.pt'), Tokenizer)
adapter = load_model(os.path.join('tokenizer_adapter_low_large_24576_subset_longtrain_v2', 'ckpt.pt'), Adapter)
encoder_ratios = math.prod(tokenizer.encoder.ratios)
max_adapter_len = adapter.max_seq_len

# DiTs
ckpt_path = os.path.join('Stage2_MetaConditionalModernDiTV2_smedium_24576_subset_adapter_longtrain_24chunks', 'ckpt.pt')
print(f'Loading model {ckpt_path} ...')
checkpoint = torch.load(ckpt_path, map_location=device)
tokenizer_args = checkpoint['model_args']

model = DiT(**tokenizer_args).to(device)
state_dict = checkpoint['ema']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
if 'cuda' in device:
    model = torch.compile(model)

n_chunks = 24
spatial_window = 64
vae_embed_dim = 16

# Feature Extractors
fad = load_model(os.path.join('FAD', 'ckpt.pt'), FAD)
contrast = load_model(os.path.join('contrast_learntmep_instance_10s', 'ckpt.pt'), Contrast)

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

data = np.memmap('/home/ubuntu/Data/low_large_24576_subset_adapter_longtrain_v2_64_val.bin', dtype=np.float32, mode='r', shape=(88303, spatial_window, vae_embed_dim))
styles = np.memmap('/home/ubuntu/Data/contrast_learntmep_instance_10s_style_val.bin', dtype=np.float32, mode='r', shape=(88303, 128))
meta = np.memmap('/home/ubuntu/Data/low_large_24576_subset_chroma_rms_density_zcr_flatness_val.bin', dtype=np.float32, mode='r', shape=(88303, 16))
bpms = np.memmap('/home/ubuntu/Data/low_large_24576_subset_adapter_longtrain_v2_64_bpm_val.bin', dtype=np.float32, mode='r')

# DSP Error: 10354.72, FAD: 46.39 | Scales: {'w_bpm': 3.9373842460434645, 'w_rms_low': 3.4393841207062614, 'w_rms_mid': 0.7827845665866684, 'w_rms_high': 1.0842041672977343, 'w_density': 3.618813544402218, 'w_zcr': 1.0507734375953, 'w_mfcc': 4.09447388787254, 'w_chroma': 3.4437712554227327, 'w_style': 1.839268747645621}
# DSP Error: 9911.62, FAD: 47.15 | Scales: {'w_bpm': 2.9757548862211, 'w_rms_low': 1.6357210298917813, 'w_rms_mid': 1.1282261871766086, 'w_rms_high': 0.03202175277198516, 'w_density': 4.344732034109373, 'w_zcr': 0.25650236008780636, 'w_mfcc': 3.5936278524604393, 'w_chroma': 4.189428251806468, 'w_style': 0.022038777442222046}
# DSP Error: 15889.04, FAD: 43.82 | Scales: {'w_bpm': 4.4264080009987, 'w_rms_low': 3.283656337130307, 'w_rms_mid': 3.707876645874035, 'w_rms_high': 2.4815314659351877, 'w_density': 1.1081407656234061, 'w_zcr': 1.0670885384460864, 'w_mfcc': 0.28568961916008273, 'w_chroma': 4.94783978080251, 'w_style': 2.754625245386409}
# DSP Error: 10836.34, FAD: 46.19 | Scales: {'w_bpm': 3.710699914324043, 'w_rms_low': 3.2953789551401425, 'w_rms_mid': 1.9033044141510613, 'w_rms_high': 0.09824734544354286, 'w_density': 0.3520998684197113, 'w_zcr': 3.7219045152299524, 'w_mfcc': 4.4502983526041895, 'w_chroma': 4.81149841700109, 'w_style': 3.9142999037339976}
# DSP Error: 11161.89, FAD: 44.40 | Scales: {'w_bpm': 3.710699914324043, 'w_rms_low': 4.028431549774051, 'w_rms_mid': 4.42472970788414, 'w_rms_high': 0.09824734544354286, 'w_density': 3.9796203268408687, 'w_zcr': 2.074415557615488, 'w_mfcc': 4.404573167810174, 'w_chroma': 4.81149841700109, 'w_style': 3.9142999037339976}

@torch.no_grad()
def run_optuna_experiments(batch_size, micro_batch_size, n_steps, n_trials):
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
        fad_emb = fad.forward_features(x)

    real_mu, real_sigma = calculate_embd_statistics(fad_emb.cpu().detach().numpy())
    
    idxs = torch.randint(len(data) - n_chunks, (batch_size,))
    
    style = torch.from_numpy(np.stack([styles[idx:idx+n_chunks] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    chroma = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, :12] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    rms = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 12] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    density = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 13] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    zcr = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 14] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    flatness = torch.from_numpy(np.stack([meta[idx:idx+n_chunks, 15] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    bpm = torch.from_numpy(np.stack([bpms[idx:idx+n_chunks] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
    
    gen_shape = (micro_batch_size, n_chunks, spatial_window, vae_embed_dim)
    gen_noise = torch.randn(gen_shape).to(device)
    decoder_noise = torch.randn(micro_batch_size * n_chunks, 1, encoder_ratios * (max_adapter_len - 1)).to(device)
    
    @torch.no_grad()  
    def objective(trial, batch_size, micro_batch_size, n_steps):
        scales = {
            'guidance': trial.suggest_float('guidance', 0, 5),
        }
        
        cfg_guidances = list(scales.values())
        errors, embs, style_errors = [], [], []
        for micro_batch in range(batch_size // micro_batch_size):
            start_idx = micro_batch * micro_batch_size
            end_idx = start_idx + micro_batch_size
            
            mb_bpm = bpm[start_idx:end_idx]
            mb_rms = rms[start_idx:end_idx]
            mb_density = density[start_idx:end_idx]
            mb_zcr = zcr[start_idx:end_idx]
            mb_flatness = flatness[start_idx:end_idx]
            mb_chroma = chroma[start_idx:end_idx]
            mb_style = style[start_idx:end_idx]
            
            t_dist = 'logit'
            cfg_mode = 'joint'
            
            unconditional_mask = {
                'bpm': torch.ones(*mb_bpm.shape, 1).to(device).bool(),
                'rms': torch.ones(*mb_rms.shape, 1).to(device).bool(),
                'density': torch.ones(*mb_density.shape, 1).to(device).bool(),
                'zcr': torch.ones(*mb_zcr.shape, 1).to(device).bool(),
                'flatness': torch.ones(*mb_flatness.shape, 1).to(device).bool(),
                'chroma': torch.ones(*mb_chroma.shape[:-1], 1).to(device).bool(),
                'style': torch.ones(*mb_style.shape[:-1], 1).to(device).bool(),
            }
            net_kwargs = {
                'bpm': bpm,
                'rms': rms,
                'density': density,
                'zcr': zcr,
                'flatness': flatness,
                'chroma': chroma,
                'style': style,
            }
            uncond_net_kwargs = net_kwargs | {'unconditional_mask': unconditional_mask}
            
            if cfg_mode == 'joint':
                joint_conditional_mask = {k: ~v for k, v in unconditional_mask.items()}
                cfg_net_kwargs = [net_kwargs | {'unconditional_mask': joint_conditional_mask}]
            elif cfg_mode == 'independent':
                cfg_net_kwargs = []
                for k, v in unconditional_mask.items():
                    temp_mask = unconditional_mask.copy()
                    temp_mask[k] = ~v
                    cfg_net_kwargs.append(net_kwargs | {'unconditional_mask': temp_mask})
            
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
                memory_efficient=False, 
                rescale_phi=0, 
                cfg_mode=cfg_mode, 
                t_dist=t_dist
            )
            
            error = 0
            for batch, measure_list in enumerate(y):
                wav = np.concatenate(measure_list, axis=0)
                wav_chroma = librosa.feature.chroma_cqt(y=wav, sr=rate, hop_length=hop_length)
                
                wav_rms = librosa.feature.rms(y=wav, hop_length=hop_length)[0]
                
                wav_onset_env = librosa.onset.onset_strength(y=wav, sr=rate, hop_length=hop_length)
                wav_onset_frames = librosa.onset.onset_detect(onset_envelope=wav_onset_env, sr=rate, hop_length=hop_length)
                wav_zcr = librosa.feature.zero_crossing_rate(wav, hop_length=hop_length)[0]
                wav_flatness = librosa.feature.spectral_flatness(y=wav, n_fft=n_fft, hop_length=hop_length)[0]
                
                # Extract 13 MFCCs, but strictly slice [1:13] to discard the 0th energy coefficient
                # wav_mfccs = librosa.feature.mfcc(y=wav, sr=rate, hop_length=hop_length, n_mfcc=13)[1:13, :]
                
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
                    
                    measure_rms = wav_rms[frame_start:frame_end]
                    measure_zcr = wav_zcr[frame_start:frame_end]
                    measure_flatness = wav_flatness[frame_start:frame_end]
                    
                    # measure_mfcc = wav_mfccs[:, frame_start:frame_end]
                    
                    if measure_chroma.shape[1] > 0:
                        chroma_error = mse(np.mean(measure_chroma, axis=1), mb_chroma[batch, i].cpu().numpy())
                        rms_error = mse(np.mean(measure_rms), mb_rms[batch, i].cpu().numpy())
                        flatness_error = mse(np.mean(measure_flatness), mb_flatness[batch, i].cpu().numpy())
                        
                        onsets_in_measure = np.sum((wav_onset_frames >= frame_start) & (wav_onset_frames < frame_end))
                        measure_duration_sec = (frame_end - frame_start) / (rate / hop_length)
                        density_error = mse(onsets_in_measure / measure_duration_sec if measure_duration_sec > 0 else 0.0, density[batch, i].cpu().numpy())
                        
                        zcr_error = mse(np.mean(measure_zcr), mb_zcr[batch, i].cpu().numpy())
                        # mfcc_error = mse(np.mean(measure_mfcc, axis=1), mb_mfcc[batch, i].cpu().numpy())
                        
                        error += chroma_error + rms_error + density_error + zcr_error + flatness_error
            
            error /= len(y)
            errors.append(error)
            
            y = torch.from_numpy(np.concatenate([np.concatenate(y_, axis=0) for y_ in y], axis=0).astype(np.float32)).to(device, non_blocking=True)
            y = drop_to_multiple(y, 16383 * 5)
                
            with ctx:
                emb = fad.forward_features(y)
                style_emb = contrast(y, features_only=True)
            embs.append(emb.cpu().detach().numpy())
            
            sim = np.matmul(mb_style.cpu().detach().numpy(), style_emb.cpu().detach().numpy().T)
            style_errors.append(1 - (np.trace(sim) / mb_style.shape[0]))
            
            del y, emb, measure_list, style_emb
            del cfg_net_kwargs, uncond_net_kwargs, net_kwargs, unconditional_mask
            del mb_bpm, mb_rms, mb_density, mb_zcr, mb_chroma, mb_style
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        y_mu, y_sigma = calculate_embd_statistics(np.concatenate(embs, axis=0))
        fad_score = calculate_frechet_distance(y_mu, y_sigma, real_mu, real_sigma)

        return np.mean(errors).item(), fad_score.item(), np.mean(style_errors).item()
    
    study = optuna.create_study(
        study_name=f'cfg_{batch_size}bs_{micro_batch_size}mbs_{n_steps}steps_{n_trials}trials',
        storage=f'sqlite:///cfg_{batch_size}bs_{micro_batch_size}mbs_{n_steps}steps_{n_trials}trials_optimization.db',
        directions=['minimize', 'minimize', 'minimize'],
        load_if_exists=False
    )
    study.optimize(lambda trial: objective(trial, batch_size, micro_batch_size, n_steps), n_trials=n_trials)

    best_trials = study.best_trials
    for t in best_trials:
        print(f"DSP Error: {t.values[0]:.2f}, FAD: {t.values[1]:.2f} | Scales: {t.params}")

@torch.no_grad()
def run_eval(batch_size, micro_batch_size, n_steps):
    torch.manual_seed(0)
    np.random.seed(0)
    
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
    
    scales = {'w_bpm': 1.0851264588839231, 'w_rms_low': 0.659259594828685, 'w_rms_mid': 3.448296776845229, 'w_rms_high': 0.4130761136569244, 'w_density': 4.437366136337984, 'w_zcr': 4.235012657009647, 'w_mfcc': 2.8912681805245666, 'w_chroma': 4.489133559225099, 'w_style': 1.3087819574454755}
    cfg_guidances = list(scales.values())
    # cfg_guidances = [2, 0.5, 1, 0.5, 2, 0.5, 2, 3, 5]
    # cfg_guidances = [0, 0, 0, 0, 0, 0, 0, 0, 3]
    cfg_guidances = [3]
    embs = []
    for micro_batch in tqdm(range(batch_size // micro_batch_size)):
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
            memory_efficient=False,
            rescale_phi=0.7,
            cfg_mode="joint"
        )
        
        sf.write(
            file=f'{micro_batch}.wav', 
            data=np.concatenate(y[0], axis=0).flatten(), 
            samplerate=rate,
            subtype='PCM_16'
        )
        
        y = torch.from_numpy(np.concatenate([np.concatenate(y_, axis=0) for y_ in y], axis=0).astype(np.float32)).to(device, non_blocking=True)
        y = drop_to_multiple(y, 16383 * 5)
            
        with ctx:
            emb = fad.forward_features(y)
        embs.append(emb.cpu().detach().numpy())

    y_mu, y_sigma = calculate_embd_statistics(np.concatenate(embs, axis=0))
    fad_score = calculate_frechet_distance(y_mu, y_sigma, real_mu, real_sigma)

    print('FAD: ', fad_score)

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
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of optuna experiments to run (default: 100)"
    )
    
    args = parser.parse_args()
    if args.micro_batch_size is None:
        args.micro_batch_size = args.batch_size
    
    if args.mode == "run_optuna_experiments":
        run_optuna_experiments(args.batch_size, args.micro_batch_size, args.n_steps, args.n_trials)
    elif args.mode == "run_eval":
        run_eval(args.batch_size, args.micro_batch_size, args.n_steps)
        