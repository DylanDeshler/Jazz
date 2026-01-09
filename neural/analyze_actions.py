import os
import math
from contextlib import nullcontext
from tqdm import tqdm

import pyrubberband as pyrb
import soundfile as sf
import numpy as np
import torch

from style import IDM_S as net
from dito import DiToV5 as Tokenizer

device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto 
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

batch_size = 2**10

checkpoint = torch.load(os.path.join('Style_1024_adaln_measures_bpm_S_nobias_poolfirst_norm_nohistory_1head', 'ckpt.pt'), map_location=device)
model_args = checkpoint['model_args']
vae_embed_dim = model_args['in_channels']
spatial_window = model_args['spatial_window']
n_encoder_chunks = model_args['n_encoder_chunks']
n_decoder_chunks = model_args['n_decoder_chunks']
n_chunks = n_encoder_chunks + n_decoder_chunks
n_style_embeddings = model_args['n_style_embeddings']

model = net(**model_args).to(device)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model = torch.compile(model)
hidden_size = 768

checkpoint = torch.load('tokenizer_low_measures_large/ckpt.pt', map_location=device)
tokenizer_args = checkpoint['model_args']

tokenizer = Tokenizer(**tokenizer_args).to(device)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
tokenizer.load_state_dict(state_dict)
tokenizer = torch.compile(tokenizer)
tokenizer.eval()
del checkpoint

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

n_samples = 16
n_bpms = 15
bpms = torch.linspace(100, 200, n_bpms)
x = torch.randn(n_samples * n_bpms, n_decoder_chunks, spatial_window, vae_embed_dim).to(device)

out_dir = '/home/ubuntu/Data/action_wavs'
os.makedirs(out_dir, exist_ok=True)

with torch.no_grad():
    with ctx:
        for action in tqdm(range(n_style_embeddings)):
            action_dir = os.path.join(out_dir, str(action))
            os.makedirs(action_dir, exist_ok=True)
            
            bpm = torch.repeat_interleave(bpms, n_samples).unsqueeze(-1).repeat(1, 2).to(device)
            weights = torch.nn.functional.one_hot(torch.ones(n_samples * n_bpms).long() * action, n_style_embeddings).float().to(device)
            
            out = model.generate_from_actions(x, bpm, weights)
            out = tokenizer.decode(out.view(n_samples * n_bpms * n_decoder_chunks, spatial_window, vae_embed_dim).permute(0, 2, 1), shape=(1, 24576 * 1), n_steps=50).view(n_samples * n_bpms, n_decoder_chunks, 1, 24576 * 1)
        
            out = out.cpu().detach().float().numpy().squeeze(-2)
            bpm = bpm.cpu().detach().numpy()
            ratio = (4 * 60 * 16000) / (24576 * bpm)
            
            for i in range(out.shape[0]):
                y, r = out[i], ratio[i]
                
                wav = np.concatenate([restore_measure(y[j], r[j].item()) for j in range(n_chunks)])
                
                sf.write(os.path.join(action_dir, f'action_{action}_bpm_{bpm[i]}_sample{i % n_samples}.wav'), wav, 16000)