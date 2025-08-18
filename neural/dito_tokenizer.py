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

from dito import DiTo as Transformer

import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import glob

device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True

batch_size = 96# * 5 * 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
# model
rate = 16000
n_samples = rate

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

paths = glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean/*.wav')
print(len(paths))

ckpt_path = os.path.join('tokenizer9', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']

model = Transformer(**model_args).to(device)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
# were loading the model directly from the trainer class
unwanted_prefix = 'model.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

# compile the model
if compile and 'cuda' in device:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

model.eval()

def get_batch(split='train'):
    if split == 'train':
        idxs = torch.randint(int(len(paths) * 0.98), (batch_size,))
        samples = [paths[idx] for idx in idxs]
        batch = []
        for sample in samples:
            x, sr = librosa.load(sample, sr=None)
            assert sr == rate

            start = np.random.randint(len(x) - n_samples)
            batch.append(x[start:start + n_samples])
        batch = torch.from_numpy(np.stack(batch, axis=0)).unsqueeze(1).to(device)
        return batch
    
    else:
        idxs = torch.randint(int(len(paths) * 0.98), len(paths), (batch_size,))
        samples = [paths[idx] for idx in idxs]
        batch = []
        for sample in samples:
            x, sr = librosa.load(sample, sr=None)
            assert sr == rate

            start = np.random.randint(len(x) - n_samples)
            batch.append(x[start:start + n_samples])
        batch = torch.from_numpy(np.stack(batch, axis=0)).unsqueeze(1).to(device)
        return batch

latents = []
with torch.no_grad():
    for path in tqdm(paths):
        x, sr = librosa.load(path, sr=None)
        assert sr == rate

        samples = []
        n_cuts = len(x) // n_samples
        for i in range(n_cuts):
            temp = x[i * n_samples : (i + 1) * n_samples]
            if len(temp) < n_samples:
                break
            samples.append(temp)
        
        n_batches = len(samples) // batch_size
        for i in range(n_batches):
            batch = torch.from_numpy(np.stack(samples[i * batch_size : (i + 1) * batch_size], axis=0)).unsqueeze(1).to(device)
            with ctx:
                z = model.encode(batch)
            latents.append(z.cpu().detach().numpy())
        
        if len(samples) - n_batches * batch_size > 0:
            batch = torch.from_numpy(np.stack(samples[n_batches * batch_size :], axis=0)).unsqueeze(1).to(device)
            with ctx:
                z = model.encode(batch)
            latents.append(z.cpu().detach().numpy())
        
latents = np.concatenate(latents, axis=0)
print(latents.shape)

filename = os.path.join(os.path.dirname(__file__), f'jazz.bin')
dtype = np.float32
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=latents.shape)
arr[:] = latents
arr.flush()