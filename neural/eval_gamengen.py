import torch
import numpy as np
from einops import rearrange

from gamengen import LAM_M as LAM
from dito import DiToV4 as Tokenizer

import os
import math
import json
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda:1')

batch_size = 128
eval_batch_size = 128
eval_iters = eval_batch_size // batch_size

temporal_window = 16
spatial_window = 32
decoder_window = 32
cut_seconds = 1
cut_len = decoder_window * cut_seconds
max_seq_len = temporal_window * cut_len
vae_embed_dim = 16

out_path = 'LAM_M_analysis_history.json'

## load tokenizer
ckpt_path = os.path.join('tokenizer_low_large', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
tokenizer_args = checkpoint['model_args']

tokenizer = Tokenizer(**tokenizer_args).to(device)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
tokenizer.load_state_dict(state_dict)
tokenizer.eval()
tokenizer = torch.compile(tokenizer)

## load generative model
ckpt_path = os.path.join('LAM_M', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']

model = LAM(**model_args).to(device)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()

@torch.no_grad()
def generate_lam_vs_random_actions(n_autoregressive_steps, n_diffusion_steps, guidance, alpha):
    n_autoregressive_steps = n_autoregressive_steps * decoder_window // spatial_window
    
    data = np.memmap('/home/dylan.d/research/music/Jazz/latents/low_large_val.bin', dtype=np.float32, mode='r', shape=(4446944, vae_embed_dim))
    idxs = torch.randint(len(data) - max_seq_len - n_autoregressive_steps, (batch_size,))
    x = torch.from_numpy(np.stack([np.stack([data[idx+i*spatial_window:idx+(i+1)*spatial_window] for i in range(temporal_window + n_autoregressive_steps)], axis=0) for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)

    B, T, N, D = x.shape

    alpha = torch.ones(B, device=x.device) * alpha
    recon, random_recon = model.lam_vs_random_actions(x[:, :-n_autoregressive_steps].clone(), alpha, n_autoregressive_steps=n_autoregressive_steps, n_diffusion_steps=n_diffusion_steps, guidance=guidance, force_drop_actions=force_drop_actions, force_drop_history=force_drop_history)
    
    if decoder_window > spatial_window:
        t2 = decoder_window // spatial_window
        t1 = (temporal_window // t2) + (n_autoregressive_steps // t2)
        x = rearrange(x, 'b (t1 t2) n c -> b t1 (t2 n) c', t1=t1, t2=t2)
        recon = rearrange(recon, 'b (t1 t2) n c -> b t1 (t2 n) c', t1=t1, t2=t2)
        random_recon = rearrange(random_recon, 'b (t1 t2) n c -> b t1 (t2 n) c', t1=t1, t2=t2)
        
        B, T, N, D = x.shape
    
    batches = []
    for cut in tqdm(range(T - n_autoregressive_steps, T), desc='Decoding'):
        batch = torch.cat([x[:, cut], recon[:, cut], random_recon[:, cut]], dim=0).permute(0, 2, 1)
        batches.append(tokenizer.decode(batch, shape=(1, 16384 * cut_seconds), n_steps=n_diffusion_steps))
    x, recon, random_recon = [res.cpu().detach().numpy().squeeze(1) for res in torch.cat(batches, dim=-1).split(B, dim=0)]

    recon_psnr = psnr(x[:, -n_autoregressive_steps * 16000:], recon[:, -n_autoregressive_steps * 16000:])
    random_psnr = psnr(x[:, -n_autoregressive_steps * 16000:], random_recon[:, -n_autoregressive_steps * 16000:])
    
    return recon_psnr - random_psnr

def psnr(y_true, y_pred, max_val=1.):
    mse = np.mean((y_true - y_pred) ** 2, axis=1)  # (B,)
    res = 10 * np.log10((max_val ** 2) / (mse + 1e-8))
    return res

guidances = [1, 2, 3, 5, 10]
alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
n_autoregressive_steps = 10
n_diffusion_steps = 50

force_drop_actions = True
force_drop_history = False

res = {}
for i, (guidance, alpha) in enumerate(itertools.product(guidances, alphas)):
    temp = np.zeros(eval_batch_size)
    for iter in range(eval_iters):
        delta_psnrs = generate_lam_vs_random_actions(n_autoregressive_steps, n_diffusion_steps, guidance, alpha)
        temp[iter*batch_size:(iter+1)*batch_size] = delta_psnrs
    res[f'{guidance},{alpha}'] = temp.tolist()
    
    print(f'[{i} / {len(guidances) * len(alphas)}] Guidance {guidance} Alpha {alpha}: Delta PSNR {np.mean(temp).item():.2f} +- {np.std(temp).item():.4f}')
    with open(out_path, 'w') as f:
        json.dump(res, f)