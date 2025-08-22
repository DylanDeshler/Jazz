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
compile = False

batch_size = 512
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

out_dir = '/home/dylan.d/research/music/Jazz/latents'
os.makedirs(out_dir, exist_ok=True)


def _extract_state_dict(checkpoint):
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict

def average_checkpoints(paths):
    sd0 = _extract_state_dict(torch.load(paths[0], map_location='cpu'))
    dtypes0 = {k: v.dtype for k, v in sd0.items()}
    acc = {}

    # Float tensors: accumulate in fp32; non-floats: copy from first
    for k, v in sd0.items():
        acc[k] = v.float().clone() if torch.is_floating_point(v) else v.clone()

    # Add remaining checkpoints
    for p in paths[1:]:
        sdi = _extract_state_dict(torch.load(p, map_location='cpu'))
        for k, v in sdi.items():
            if torch.is_floating_point(v):
                acc[k].add_(v.float())
            # non-floats ignored (kept from first)

    n = float(len(paths))
    # Finalize: average floats and cast back to original dtype
    for k in acc:
        if torch.is_floating_point(acc[k]):
            acc[k].div_(n)
            acc[k] = acc[k].to(dtypes0[k])
    return acc

ckpt_path = os.path.join('tokenizer10', 'ckpt.pt')
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
model.load_state_dict(state_dict)

# compile the model
if compile and 'cuda' in device:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

model.eval()

save_idx = 0
latents = []
batch_paths = []
batch_shapes = []
with torch.no_grad():
    for path_idx, path in enumerate(tqdm(paths)):
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

        if (path_idx + 1) % 5_000 == 0:
            latents = np.concatenate(latents, axis=0).swapaxes(1, 2)
            B, T, C = latents.shape
            latents = latents.reshape((B * T, C))

            print('Writing latents with shape: ', latents.shape)

            filename = os.path.join(out_dir, f'batch_{path_idx}.bin')
            arr = np.memmap(filename, dtype=np.float32, mode='w+', shape=latents.shape)
            arr[:] = latents
            arr.flush()

            latents = []
            batch_paths.append(filename)
            batch_shapes.append(latents.shape)

if len(latents) > 0:
    latents = np.concatenate(latents, axis=0).swapaxes(1, 2)
    B, T, C = latents.shape
    latents = latents.reshape((B * T, C))

    print('Writing latents with shape: ', latents.shape)

    filename = os.path.join(out_dir, f'batch_{path_idx + 1}.bin')
    arr = np.memmap(filename, dtype=np.float32, mode='w+', shape=latents.shape)
    arr[:] = latents
    arr.flush()

    batch_paths.append(filename)
    batch_shapes.append(latents.shape)

n_tokens = sum([shape[0] for shape in batch_shapes])
train_tokens = int(len(n_tokens) * 0.98)
val_tokens = n_tokens - train_tokens

train_start = 0
val_start = 0
train = np.memmap(os.path.join(out_dir, 'train.bin'), dtype=np.float32, mode='w+', shape=(train_tokens, C))
val = np.memmap(os.path.join(out_dir, 'val.bin'), dtype=np.float32, mode='w+', shape=(val_tokens, C))
for path, shape in zip(batch_paths, batch_shapes):
    arr = np.memmap(path, dtype=np.float32, mode='r', shape=shape)
    if train_start + len(arr) < train_tokens:
        train[train_start:train_start + len(arr)] = arr[:]
        train.flush()
        train_start += len(arr)
    elif val_start == 0:
        train_delta = train_tokens - train_start
        train[train_start:train_start + train_delta] = arr[:train_delta]
        train.flush()
        train_start += train_delta

        val_delta = len(arr) - train_delta
        val[:val_delta] = arr[train_delta:train_delta+val_delta]
        val.flush()
        val_start += val_delta
    else:
        val[val_start:val_start + len(arr)] = arr[:]
        val.flush()
        val_start += len(arr)

# latents = np.concatenate(latents, axis=0).swapaxes(1, 2)
# B, T, C = latents.shape
# latents = latents.reshape((B * T, C))

# train = latents[:int(len(latents) * 0.98)]
# print('Writing train with shape: ', train.shape)

# filename = os.path.join(out_dir, 'train.bin')
# arr = np.memmap(filename, dtype=np.float32, mode='w+', shape=train.shape)

# n_batches = 50
# n_samples = len(train) // n_batches
# for i in range(n_batches):
#     arr[i * n_samples : (i + 1) * n_samples] = train[i * n_samples : (i + 1) * n_samples]
#     arr.flush()

# arr[n_batches * n_samples :] = train[n_batches * n_samples :]
# arr.flush()

# val = latents[int(len(latents) * 0.98):]
# print('Writing val with shape: ', val.shape)

# filename = os.path.join(out_dir, 'val.bin')
# arr = np.memmap(filename, dtype=np.float32, mode='w+', shape=val.shape)
# arr[:] = val
# arr.flush()