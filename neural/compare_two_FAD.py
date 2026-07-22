import os
import re
import glob
import math
import librosa
import soundfile as sf
from tqdm import tqdm

import torch
import numpy as np
from scipy import linalg
from scipy.signal import medfilt
from contextlib import nullcontext
from einops import rearrange

from contrast import Transformer as Contrast
from dito import DiToV5 as Tokenizer
from fad import MultiTaskFAD as FAD, BPMProbe
from adapter import InvertibleAdapter as Adapter
import diffusion_forcing
import argparse

parser = argparse.ArgumentParser(description="Directly compare two models by FAD.")

parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--n_samples', type=int, required=True)
parser.add_argument('--n_steps', type=int, default=32)
parser.add_argument('--device', type=str, required=True, help="Specify the device. Like cuda:1.")
parser.add_argument('--contrast', action='store_true', help='To extract features from a contrastive model')

# Model A
parser.add_argument('--ckpt_a', type=str, required=True, help="Checkpoint path for model A.")
parser.add_argument('--class_a', type=str, required=True, help="DiT factory name in diffusion_forcing for model A.")
parser.add_argument('--type_a', type=str, required=True, choices=['base', 'measure'], help="Generation pipeline for model A.")
parser.add_argument('--label_a', type=str, default=None, help="Display label for model A.")

# Model B
parser.add_argument('--ckpt_b', type=str, required=True, help="Checkpoint path for model B.")
parser.add_argument('--class_b', type=str, required=True, help="DiT factory name in diffusion_forcing for model B.")
parser.add_argument('--type_b', type=str, required=True, choices=['base', 'measure'], help="Generation pipeline for model B.")
parser.add_argument('--label_b', type=str, default=None, help="Display label for model B.")

args = parser.parse_args()

label_a = args.label_a or args.class_a
label_b = args.label_b or args.class_b

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
        print('Loaded EMA')
    else:
        state_dict = checkpoint['model']
        print('Loaded raw model')
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
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

def predict_measures(model, gen_shape, net_kwargs, uncond_net_kwargs, n_steps, measure_chunks, guidance=1, gen_noise=None, decoder_noise=None, method='median', window_size=3, memory_efficient=False, rescale_phi=0, cfg_mode="independent", t_dist="uniform"):
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

# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------
if args.contrast:
    fad = load_model(os.path.join('contrast_learntmep_instance_10s', 'ckpt.pt'), Contrast)
    samples_mult = 10
else:
    fad = load_model(os.path.join('FAD_v2_30drop', 'ckpt.pt'), FAD)
    samples_mult = 5

# ---------------------------------------------------------------------------
# Shared generation infrastructure
# ---------------------------------------------------------------------------
base_window = measure_window = 48
vae_embed_dim = 16

def extract_chunks(ckpt_path):
    """Pull the chunk count from a checkpoint path, e.g. '..._32chunks/ckpt.pt' -> 32."""
    matches = re.findall(r'(\d+)chunks', ckpt_path)
    if not matches:
        raise ValueError(f"Could not find a '_<N>chunks' token in checkpoint path: {ckpt_path}")
    return int(matches[-1])

chunks_a = extract_chunks(args.ckpt_a)
chunks_b = extract_chunks(args.ckpt_b)
print(f"Model A chunks: {chunks_a} | Model B chunks: {chunks_b}")

need_measure = 'measure' in (args.type_a, args.type_b)

# Tokenizer is required by both pipelines' decode step.
tokenizer = load_model(os.path.join('tokenizer_low_large_24576_subset_longtrain', 'ckpt.pt'), Tokenizer)
encoder_ratios = math.prod(tokenizer.encoder.ratios)

# Adapter + BPM probe are only needed by the measure pipeline.
if need_measure:
    adapter = load_model(os.path.join('tokenizer_adapter_low_large_24576_subset_longtrain_v2_48', 'ckpt.pt'), Adapter)
    max_seq_len = adapter.max_seq_len
    probe = load_model(os.path.join('tokenizer_low_measures_fix_subset_longtrain_v2_48_BPMProbe_tiny_lstm', 'ckpt.pt'), BPMProbe)

paths = glob.glob('/data/wavs/*')
paths = paths[-int(len(paths) * 2/48):]  # test set

n_generations = args.n_samples
batch_size = args.batch_size
n_steps = args.n_steps
assert n_generations % batch_size == 0
total_batches = n_generations // batch_size

durations = []
for path in tqdm(paths, desc='Scanning durations'):
    durations.append(sf.info(path).duration)
durations = np.asarray(durations)
idxs = np.random.choice(np.arange(len(paths)), size=n_generations, p=durations / np.sum(durations))

max_chunks = max(chunks_a, chunks_b)

def compute_real_embeddings():
    embs = []
    for idx in tqdm(idxs, desc='Embedding Real Samples'):
        path = paths[idx]
        wav, _ = librosa.load(path, sr=rate)
        wav = wav[:batch_size * max_chunks * n_samples]
        x = torch.from_numpy(wav.astype(np.float32)).to(device, non_blocking=True)
        x = drop_to_multiple(x, 16383 * samples_mult)
        with ctx:
            real_emb = fad.forward_features(x)
        embs.append(real_emb.cpu().detach().numpy())
    return np.concatenate(embs, axis=0)

def generate_embeddings(dit, gen_type, chunks, label):
    embs = []
    for b in tqdm(range(total_batches), desc=f'Generating [{label}]'):
        if gen_type == 'base':
            gen_shape = (batch_size, chunks, base_window, vae_embed_dim)
            gen_noise = torch.randn(gen_shape, device=device)
            decoder_noise = torch.randn((batch_size * chunks, 1, n_samples), device=device)
            with ctx:
                y = dit.generate(gen_shape, n_steps=n_steps, noise=gen_noise, memory_efficient=False, cfg_mode='joint', t_dist='logit')
                y = y.transpose(2, 3).view(batch_size * chunks, vae_embed_dim, base_window)
                y = tokenizer.decode(y, shape=(1, n_samples), n_steps=n_steps, noise=decoder_noise)
                y = drop_to_multiple(y, 16383 * samples_mult)
                emb = fad.forward_features(y)
            embs.append(emb.cpu().detach().numpy())
        else:  # measure
            gen_shape = (batch_size, chunks, measure_window, vae_embed_dim)
            gen_noise = torch.randn(gen_shape, device=device)
            decoder_noise = torch.randn((batch_size * chunks, 1, encoder_ratios * (max_seq_len - 1)), device=device)
            y = predict_measures(dit, gen_shape, None, None, n_steps, chunks, guidance=1.0, gen_noise=gen_noise, decoder_noise=decoder_noise, method='median', window_size=3, memory_efficient=False, cfg_mode='joint', t_dist='logit')
            with ctx:
                y = drop_to_multiple(y, 16383 * samples_mult)
                emb = fad.forward_features(y)
            embs.append(emb.cpu().detach().numpy())
    return np.concatenate(embs, axis=0)

# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------
with torch.inference_mode():
    real_embs = compute_real_embeddings()
    real_mu, real_sigma = calculate_embd_statistics(real_embs)

    results = {}
    for tag, ckpt, class_name, gen_type, chunks, label in [
        ('A', args.ckpt_a, args.class_a, args.type_a, chunks_a, label_a),
        ('B', args.ckpt_b, args.class_b, args.type_b, chunks_b, label_b),
    ]:
        dit = load_model(ckpt, getattr(diffusion_forcing, class_name))
        embs = generate_embeddings(dit, gen_type, chunks, label)
        mu, sigma = calculate_embd_statistics(embs)
        fad_score = calculate_frechet_distance(mu, sigma, real_mu, real_sigma)
        results[tag] = {'label': label, 'gen_type': gen_type, 'chunks': chunks, 'fad': fad_score,
                        'n_real': len(real_embs), 'n_gen': len(embs)}
        del dit
        if 'cuda' in device:
            torch.cuda.empty_cache()

print("\n" + "=" * 68)
print(f"{'Model':<28} {'Type':<8} {'Chunks':>7} {'FAD':>12}")
print("-" * 68)
for tag in ('A', 'B'):
    r = results[tag]
    print(f"{r['label']:<28} {r['gen_type']:<8} {r['chunks']:>7} {r['fad']:>12.4f}")
print("-" * 68)
winner = min(results, key=lambda t: results[t]['fad'])
loser = 'B' if winner == 'A' else 'A'
delta = results[loser]['fad'] - results[winner]['fad']
print(f"Lower FAD (better): {results[winner]['label']}  "
      f"(by {delta:.4f})")
print("=" * 68)
