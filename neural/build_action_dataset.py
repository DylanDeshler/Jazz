import os
import time
import math
import pickle
from contextlib import nullcontext
from tqdm import tqdm
from torchinfo import summary

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from einops import rearrange

from lapa import LAM_B as net
from dito import DiToV4 as Tokenizer

import matplotlib.pyplot as plt
import soundfile as sf

import torch
import torchaudio
from sklearn.metrics.pairwise import paired_cosine_distances

device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto
torch.manual_seed(1337 + 0)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype] 
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

model_dir = 'LAPA_B_FS_64_noagg'
checkpoint = torch.load(os.path.join(model_dir, 'ckpt.pt'), map_location=device)
model_args = checkpoint['model_args']
spatial_window = model_args['spatial_window']
temporal_window = model_args['temporal_window']
max_seq_len = spatial_window * temporal_window
batch_size = 256
vae_embed_dim = 16

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

with torch.no_grad():
    for batch in tqdm(range(204654816 // batch_size)):
        data = np.memmap('/home/dylan.d/research/music/Jazz/latents/low_large_train.bin', dtype=np.float32, mode='r', shape=(204654816, vae_embed_dim))
        idxs = torch.arange(batch * batch_size, (batch + 1) * batch_size)
        x = torch.from_numpy(np.stack([np.stack([data[idx+i*spatial_window:idx+(i+1)*spatial_window] for i in range(temporal_window)], axis=0) for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
        
        z, indices = model.enocde_actions(x)
        print(indices.shape)
