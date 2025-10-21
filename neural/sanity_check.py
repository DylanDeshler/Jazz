import numpy as np
import torch
import os

from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt

from fm import FM
from dito import DiToV4 as Tokenizer

device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

ckpt_path = os.path.join('tokenizer_low', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
tokenizer_args = checkpoint['model_args']

tokenizer = Tokenizer(**tokenizer_args).to(device)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
tokenizer.load_state_dict(state_dict)
tokenizer.eval()

temporal_window = 16
spatial_window = 32
cut_seconds = 1
cut_len = spatial_window * cut_seconds
max_seq_len = temporal_window * cut_len
vae_embed_dim = 16
batch_size = 4

diffusion = FM(sigma_min=1e-9, timescale=1000.0)

def get_batch(split='train'):
    if split == 'train':
        data = np.memmap('/home/dylan.d/research/music/Jazz/latents/low_train.bin', dtype=np.float32, mode='r', shape=(204654816, vae_embed_dim))
        idxs = torch.randint(len(data) - max_seq_len, (batch_size,))
        x = torch.from_numpy(np.stack([np.stack([data[idx+i*spatial_window:idx+(i+1)*spatial_window] for i in range(temporal_window)], axis=0) for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
        return x
    
    else:
        data = np.memmap('/home/dylan.d/research/music/Jazz/latents/low_val.bin', dtype=np.float32, mode='r', shape=(4446944, vae_embed_dim))
        idxs = torch.randint(len(data) - max_seq_len, (batch_size,))
        x = torch.from_numpy(np.stack([np.stack([data[idx+i*spatial_window:idx+(i+1)*spatial_window] for i in range(temporal_window)], axis=0) for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
        return x

x = get_batch()
B, T, N, D = x.shape

batches = []
for cut in tqdm(range(T), desc='Decoding'):
    batches.append(tokenizer.decode(x[:, cut].permute(0, 2, 1), shape=(1, 16384 * cut_seconds)))
x = torch.cat(batches, dim=-1)
print(x.shape)

sf.write('sanity.wav', x[0].squeeze().cpu().detach().numpy(), 16000)
