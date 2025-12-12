import os
import math
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
import torch

from lapa2 import LAM_B as net

device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto 
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

batch_size = 128

ckpt_path = os.path.join('LAPA_measures_bpm_B_FSQ_16_3', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
vae_embed_dim = model_args['in_channels']
levels = model_args['levels']

model = net(**model_args).to(device)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model = torch.compile(model)

N = 3693787

data = np.memmap('/home/dylan.d/research/music/Jazz/latents/low_measures_large.bin', dtype=np.float16, mode='r', shape=(N, 48, vae_embed_dim))
meta = np.memmap('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_measures_meta.npy', dtype=np.float32, mode='r', shape=(N, 2))
arr = np.memmap(f'/home/dylan.d/research/music/Jazz/latents/low_measures_large_actions_{vae_embed_dim}_{math.prod(levels)}.bin', dtype=np.int8, mode='w+', shape=(N, math.prod(levels)))

with torch.no_grad():
    for i in tqdm(range(N // batch_size)):
        batch = torch.from_numpy(data[i*batch_size:(i+1)*batch_size].copy()).unsqueeze(1).pin_memory().to(device, non_blocking=True)
        bpm = torch.from_numpy(meta[i*batch_size:(i+1)*batch_size, 1].copy()).pin_memory().to(device, non_blocking=True)
        
        print(batch.shape, bpm.shape)
        with ctx:
           _, actions = model.enocde_actions(batch, bpm)
        
        print(actions.shape)
        arr[i*batch_size:(i+1)*batch_size] = actions.cpu().detach().numpy().astype(np.float16)

arr.flush()
