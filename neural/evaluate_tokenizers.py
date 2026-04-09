import os
import glob
import math
import json
import librosa
import soundfile as sf
from tqdm import tqdm
from collections import defaultdict

import concurrent.futures
import pyrubberband as pyrb
from multiprocessing import cpu_count

import torch
import numpy as np
from scipy import linalg
from contextlib import nullcontext

from dito import DiToV5 as Tokenizer
from fad import MultiTaskFAD as FAD
from contrast import Transformer as Contrast

# from transformers import ClapModel, ClapProcessor

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
                            
                            # found an issue where 4/4 is frequently being annotated as 8/4 this fixes it and safe because were only annotating 4/4 songs
                            if bn > 0:
                                bn = ((bn - 1) % 4) + 1
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

def process_measure(y):
    current_samples = len(y)
    stretch_factor = current_samples / n_samples
    duration_sec = current_samples / rate
    instant_bpm = (TARGET_SIG / duration_sec) * 60
    
    y_warped = pyrb.time_stretch(y, rate, stretch_factor)

    if len(y_warped) > n_samples:
        y_warped = y_warped[:n_samples]
    elif len(y_warped) < n_samples:
        y_warped = np.pad(y_warped, (0, n_samples - len(y_warped)))
        
    return y_warped, stretch_factor, instant_bpm

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

base1 = load_model(os.path.join('tokenizer_low_large_24576', 'ckpt.pt'), Tokenizer)
# base2 = load_model(os.path.join('tokenizer_low_large_24576_2std_subset', 'ckpt.pt'), Tokenizer)
measure1 = load_model(os.path.join('tokenizer_low_measures_fix_subset', 'ckpt.pt'), Tokenizer)
fad = load_model(os.path.join('FAD', 'ckpt.pt'), FAD)
# contrast = load_model(os.path.join('contrast_learntmep_instance', 'ckpt.pt'), Contrast)
# clap_model = ClapModel.from_pretrained("laion/larger_clap_music").to(device)
# clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music")

## Only use 4/4 songs
# measure_paths = glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_measures/*.wav')
# audio_paths = [path.replace('jazz_data_16000_full_clean_measures', 'jazz_data_16000_full_clean') for path in measure_paths]
# beat_paths = [path.replace('jazz_data_16000_full_clean_measures', 'jazz_data_16000_full_clean_beats').replace('.wav', '.beats') for path in measure_paths]
# measure_paths = glob.glob('/home/ubuntu/Data/measures/*')
# audio_paths = [os.path.join('/home/ubuntu/Data/wavs', os.path.basename(path)) for path in measure_paths]
# beat_paths = [os.path.join('/home/ubuntu/Data/beats', os.path.basename(path)) for path in measure_paths]

# With BPM within reasonable range
measure_paths = glob.glob('/home/ubuntu/Data/measures/*')
with open('/home/ubuntu/Data/valid_files_by_bpm.json', 'r') as f:
    beat_paths = json.load(f)
measure_paths = [path for path in measure_paths if os.path.basename(path) in beat_paths]
audio_paths = [os.path.join('/home/ubuntu/Data/wavs', os.path.basename(path)) for path in measure_paths]
audio_paths = [path for path in audio_paths if os.path.basename(path) in beat_paths]
beat_paths = [os.path.join('/home/ubuntu/Data/beats', path) for path in beat_paths]

idxs = np.random.choice(np.arange(len(measure_paths)), size=128, replace=False)
n_steps = 32
batch_size = 64
EVAL_ITERATIVE = False
USE_CLAP = False

# FAD requires 16383 * 5 samples and contrast requires 16383 * 10 samples
# fad.forward_features()
# contrast(features_only=True)

real_embs = []
base1_embs = []
base2_embs = []
measure1_embs = []
out_dir = '/home/ubuntu/Data/FAD'
os.makedirs(out_dir, exist_ok=True)
with torch.no_grad():
    for idx in tqdm(idxs):
        measure_path, audio_path, beat_path = measure_paths[idx], audio_paths[idx], beat_paths[idx]
        
        wav, _ = librosa.load(audio_path, sr=None)
        wav = wav[:batch_size * rate]
        
        beat_data = parse_beat_file(beat_path)
        downbeat_indices = [i for i, b in enumerate(beat_data) if b['beat'] == 1]
        
        m, ratios = [], []
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
            
            measure, stretch_ratio, instant_bpm = process_measure(wav[frame_start:frame_end])
            m.append(measure)
            ratios.append(stretch_ratio)
        
        if last_frame - first_frame < 16383 * 5:
            continue
        
        m = torch.from_numpy(np.asarray(m).astype(np.float32)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
        x_raw = wav[first_frame:last_frame]
        
        remainder = len(x_raw) % n_samples
        pad_length = (n_samples - remainder) if remainder != 0 else 0
        
        x_padded = x_raw
        if pad_length > 0:
            x_padded = np.pad(x_raw, (0, pad_length))
        
        x = [x_padded[chunk * n_samples:(chunk+1) * n_samples] for chunk in range(len(x_padded) // n_samples)]
        x = torch.from_numpy(np.asarray(x).astype(np.float32)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
        
        ## Standard approach
        if not EVAL_ITERATIVE:
            noise = torch.randn((1, 1, n_samples), device=device).repeat(max(x.shape[0], m.shape[0]), 1, 1)
            with ctx:
                y1 = base1.reconstruct(x, n_steps=n_steps, noise=noise[:x.shape[0]])
                # y2 = base2.reconstruct(x, n_steps=n_steps, noise=noise[:x.shape[0]])
                y3 = measure1.reconstruct(m, n_steps=n_steps, noise=noise[:m.shape[0]])
            
            x = torch.from_numpy(x_raw.astype(np.float32)).to(device, non_blocking=True)
            
            y1 = y1.cpu().detach().numpy().flatten()
            if pad_length > 0:
                y1 = y1[:-pad_length]
            y1 = torch.from_numpy(y1.astype(np.float32)).to(device, non_blocking=True)
            
            y3 = np.concatenate([restore_measure(y.squeeze(), ratio) for y, ratio in zip(y3.cpu().detach().numpy(), ratios)], axis=0)
            y3 = torch.from_numpy(y3.astype(np.float32)).to(device, non_blocking=True)
            
            # Custom embs
            if not USE_CLAP:
                x = drop_to_multiple(x, 16383 * 5)
                y1 = drop_to_multiple(y1, 16383 * 5)
                y3 = drop_to_multiple(y3, 16383 * 5)
                with ctx:
                    try:
                        real_emb = fad.forward_features(x)
                        base1_emb = fad.forward_features(y1)
                        # base2_emb = fad.forward_features(drop_to_multiple(y2, 16383 * 5))
                        measure1_emb = fad.forward_features(y3)
                    except Exception as e:
                        print(e)
                        continue
            
            # Laion-CLAP embs
            else:
                real_inputs = librosa.resample(drop_to_multiple(x, rate * 10).squeeze().cpu().detach().numpy(), orig_sr=rate, target_sr=48000)
                real_emb = [clap_model.get_audio_features(**clap_processor(audios=inputs, sampling_rate=48000, return_tensors="pt").to(device)) for inputs in real_inputs]
                real_emb = torch.cat(real_emb, dim=0)
                
                base1_inputs = librosa.resample(drop_to_multiple(y1, rate * 10).squeeze().cpu().detach().numpy(), orig_sr=rate, target_sr=48000)
                base1_emb = [clap_model.get_audio_features(**clap_processor(audios=inputs, sampling_rate=48000, return_tensors="pt").to(device)) for inputs in base1_inputs]
                base1_emb = torch.cat(base1_emb, dim=0)
                
                base2_inputs = librosa.resample(drop_to_multiple(y2, rate * 10).squeeze().cpu().detach().numpy(), orig_sr=rate, target_sr=48000)
                base2_emb = [clap_model.get_audio_features(**clap_processor(audios=inputs, sampling_rate=48000, return_tensors="pt").to(device)) for inputs in base2_inputs]
                base2_emb = torch.cat(base2_emb, dim=0)
                
                measure1_inputs = librosa.resample(drop_to_multiple(y3, rate * 10).squeeze().cpu().detach().numpy(), orig_sr=rate, target_sr=48000)
                measure1_emb = [clap_model.get_audio_features(**clap_processor(audios=inputs, sampling_rate=48000, return_tensors="pt").to(device)) for inputs in measure1_inputs]
                measure1_emb = torch.cat(measure1_emb, dim=0)
            
            real_embs.append(real_emb.cpu().detach().numpy())
            base1_embs.append(base1_emb.cpu().detach().numpy())
            # base2_embs.append(base2_emb.cpu().detach().numpy())
            measure1_embs.append(measure1_emb.cpu().detach().numpy())
        
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
            
            inputs = clap_processor(audios=y1, return_tensors="pt").to(device)
            audio_embed = clap_model.get_audio_features(**inputs)
            
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
        
        # np.save(os.path.join(out_dir, 'real.npy'), np.concatenate(real_embs, axis=0))
        # np.save(os.path.join(out_dir, 'base1.npy'), np.concatenate(base1_embs, axis=0))
        # np.save(os.path.join(out_dir, 'base2.npy'), np.concatenate(base2_embs, axis=0))
        # np.save(os.path.join(out_dir, 'measure1.npy'), np.concatenate(measure1_embs, axis=0))

        print(real_emb.shape, base1_emb.shape, measure1_emb.shape)
        real_emb = fad.forward_features(drop_to_multiple(x, 16383 * 5))
        base1_emb = fad.forward_features(drop_to_multiple(y1, 16383 * 5))
        # base2_emb = fad.forward_features(drop_to_multiple(y2, 16383 * 5))
        measure1_emb = fad.forward_features(drop_to_multiple(y3, 16383 * 5))
        name = os.path.basename(measure_path)
        sf.write(
            file=os.path.join(out_dir, f'{idx}_real_{name}'), 
            data=x[8].flatten().cpu().detach().numpy(), 
            samplerate=rate,
            subtype='PCM_16'
        )
        sf.write(
            file=os.path.join(out_dir, f'{idx}_base1_{name}'), 
            data=y1[8].flatten().cpu().detach().numpy(), 
            samplerate=rate,
            subtype='PCM_16'
        )
        # sf.write(
        #     file=os.path.join(out_dir, f'{idx}_base2_{name}'), 
        #     data=y2.flatten(), 
        #     samplerate=rate,
        #     subtype='PCM_16'
        # )
        sf.write(
            file=os.path.join(out_dir, f'{idx}_measures_{name}'), 
            data=y3[8].flatten().cpu().detach().numpy(), 
            samplerate=rate,
            subtype='PCM_16'
        )

if not EVAL_ITERATIVE:
    real_mu, real_sigma = calculate_embd_statistics(np.concatenate(real_embs, axis=0))
    base1_mu, base1_sigma = calculate_embd_statistics(np.concatenate(base1_embs, axis=0))
    # base2_mu, base2_sigma = calculate_embd_statistics(np.concatenate(base2_embs, axis=0))
    measure1_mu, measure1_sigma = calculate_embd_statistics(np.concatenate(measure1_embs, axis=0))

    base1_fad = calculate_frechet_distance(base1_mu, base1_sigma, real_mu, real_sigma)
    # base2_fad = calculate_frechet_distance(base2_mu, base2_sigma, real_mu, real_sigma)
    measure1_fad = calculate_frechet_distance(measure1_mu, measure1_sigma, real_mu, real_sigma)

    print('Base 1 FAD: ', base1_fad)
    # print('Base 2 FAD: ', base2_fad)
    print('Measure 1 FAD: ', measure1_fad)
    # Base 1 FAD:  46.068807655471866694
    # Base 2 FAD:  46.19946344047392614
    # Measure 1 FAD:  49.728033691220890164
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