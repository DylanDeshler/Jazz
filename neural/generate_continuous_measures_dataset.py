import os
import time
import math
import json
import pickle
from contextlib import nullcontext
from tqdm import tqdm
from torchinfo import summary

import numpy as np
import torch

from dito import DiToV5 as Tokenizer

device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto 
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

batch_size = 128
rate = 16000
n_samples = 24576

out_prefix = 'low_measures_large'
ckpt_path = os.path.join('tokenizer_low_measures_large', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
tokenizer_args = checkpoint['model_args']
vae_embed_dim = tokenizer_args['dimension']

model = Tokenizer(**tokenizer_args).to(device)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()

N = 3693787
data = np.memmap('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_measures_audio.npy', dtype=np.float16, mode='r', shape=(N, n_samples))
arr = np.memmap('/home/dylan.d/research/music/Jazz/latents/low_measures_large.bin', dtype=np.float16, mode='w+', shape=(N, 48, 16))

with torch.no_grad():
    for i in tqdm(range(N // batch_size)):
        batch = torch.from_numpy(data[i*batch_size:(i+1)*batch_size].copy()).view(batch_size, n_samples).unsqueeze(1).pin_memory().to(device, non_blocking=True)
        with ctx:
            _, codes = model.encode(batch)
        codes = codes.permute(0, 2, 1).cpu().detach().numpy()
        arr[i*batch_size:(i+1)*batch_size] = codes.astype(np.float16)

arr.flush()
