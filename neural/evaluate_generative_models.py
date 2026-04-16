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

idxs = np.random.choice(np.arange(len(measure_paths)), size=64, replace=False)
n_steps = 32
batch_size = 32
EVAL_ITERATIVE = False
USE_CLAP = False

real_embs = []
base_embs = []
measure_embs = []
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
            with ctx:
                shape = (batch_size, n_chunks, spatial_window, vae_embed_dim)
                noise = torch.randn(shape, device=device)
                
                y1 = base_dit.generate(shape, n_steps=n_steps, noise=noise)
                print(y1.shape)
                y2 = measure_dit.generate(shape, n_steps=n_steps, noise=noise)
                print(y2.shape)
                
                bpm = probe(y2)
                print(bpm.shape, bpm.mean().item(), bpm.std().item())
                seconds_per_beat = 60.0 / bpm
                measure_duration_sec = seconds_per_beat * TARGET_SIG
                
                target_samples = (measure_duration_sec * rate).long()
                max_len = min(target_samples.max().item(), encoder_ratios * (max_seq_len - 1))
                max_len = encoder_ratios * math.ceil(max_len / encoder_ratios)
                max_latent_len = max_len // encoder_ratios
                
                indices = torch.arange(max_latent_len, device=device).view(1, 1, -1)
                lengths = ((target_samples + encoder_ratios - 1) // encoder_ratios).unsqueeze(-1)
                print(indices.shape, lengths.shape)
                mask = indices < lengths
                mask = mask.view(batch_size * n_chunks, max_latent_len)
                shape = (batch_size * n_chunks, 1, max_latent_len)
                
                y2 = y2.transpose(2, 3).view(batch_size * n_chunks, vae_embed_dim, spatial_window)
                print(y2.shape, shape, mask.shape)
                y2 = adapter.decode(y2, shape, mask=mask)
                y1 = y1.transpose(2, 3).view(batch_size * n_chunks, vae_embed_dim, spatial_window)
                print(y1.shape, y2.shape)
                # .view(B, T, 1, 24576 * cut_seconds)
                # print(target_samples.max().item(), max_len_trunc)
                y1 = tokenizer.decode(y1, shape=(1, 24576), n_steps=n_steps, noise=None)
                y2 = tokenizer.decode(y2, shape=(1, max_len), n_steps=n_steps, noise=None)
                print(y1.shape, y2.shape)
            
            x = torch.from_numpy(x_raw.astype(np.float32)).to(device, non_blocking=True)
            
            # y1 = y1.cpu().detach().numpy().flatten()
            # if pad_length > 0:
            #     y1 = y1[:-pad_length]
            # y1 = torch.from_numpy(y1.astype(np.float32)).to(device, non_blocking=True)
            
            print(target_samples.shape, y1.shape, y2.shape)
            target_samples = target_samples.flatten().cpu().detach().numpy()
            y2 = y2.squeeze().cpu().detach().numpy()
            # y2 = np.concatenate([y[:(max_len - min(samples, max_len))] for y, samples in zip(y2, target_samples)], axis=0)
            
            out = []
            for y, samples in zip(y2, target_samples):
                print(y.shape, samples)
                out.append(y[:(max_len - min(samples, max_len))])
                print(out[-1].shape)
            y2 = np.concatenate(out, axis=0)
            
            y2 = torch.from_numpy(y2.astype(np.float32)).to(device, non_blocking=True)
            
            # Custom embs
            if not USE_CLAP:
                x = drop_to_multiple(x, 16383 * 5)
                y1 = drop_to_multiple(y1, 16383 * 5)
                y2 = drop_to_multiple(y2, 16383 * 5)
                with ctx:
                    try:
                        print(x.shape, y1.shape, y2.shape)
                        real_emb = fad.forward_features(x)
                        base_emb = fad.forward_features(y1)
                        measure_emb = fad.forward_features(y2)
                    except Exception as e:
                        print(e)
                        continue
            
            real_embs.append(real_emb.cpu().detach().numpy())
            base_embs.append(base_emb.cpu().detach().numpy())
            measure_embs.append(measure_emb.cpu().detach().numpy())
        
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
                file=os.path.join(out_dir, f'{idx}_measure_{name}'), 
                data=y2[to_sample].flatten().cpu().detach().numpy(), 
                samplerate=rate,
                subtype='PCM_16'
            )

if not EVAL_ITERATIVE:
    real_mu, real_sigma = calculate_embd_statistics(np.concatenate(real_embs, axis=0))
    base_mu, base_sigma = calculate_embd_statistics(np.concatenate(base_embs, axis=0))
    measure_mu, measure_sigma = calculate_embd_statistics(np.concatenate(measure_embs, axis=0))

    base_fad = calculate_frechet_distance(base_mu, base_sigma, real_mu, real_sigma)
    measure_fad = calculate_frechet_distance(measure_mu, measure_sigma, real_mu, real_sigma)

    print('Base -> Real Samples FAD: ', base_fad)
    print('Measure -> Real Samples FAD: ', measure_fad)

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