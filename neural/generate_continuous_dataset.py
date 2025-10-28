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

from dito import DiToV4 as Tokenizer

import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import glob

device = torch.device('cuda')

batch_size = 64
rate = 16000
n_samples = rate

out_prefix = 'low_large'
ckpt_path = os.path.join('tokenizer_low_large', 'ckpt.pt')
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

paths = glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean/*.wav')
print(len(paths))

with open('/home/dylan.d/research/music/Jazz/song_instruments.json', 'r') as f:
    song_instruments = json.load(f)

if True:

    test = True

    write_idx = 0
    write_paths = []
    total_write_batches = 48

    all_codes = []
    all_instruments = []
    with torch.no_grad():
        for idx, path in enumerate(tqdm(paths)):
            instruments = song_instruments[path.split('/')[-1]]
            this_codes = []
            if test:
                this_samples = []
            x, sr = librosa.load(path, sr=None)

            n_cuts = len(x) // n_samples
            for i in range(n_cuts // batch_size):
                batch = torch.from_numpy(np.stack([x[(i*batch_size+j)*n_samples:(i*batch_size+j+1)*n_samples] for j in range(batch_size)], axis=0)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
                _, codes = model.encode(batch)
                codes = codes.permute(0, 2, 1).cpu().detach().numpy()
                if test:
                    # will introduce padding but thats fine for encode only
                    x_, z = model.encode(batch)
                    y = model.decode(z, x_.shape)
                    this_samples.append(y.cpu().detach().numpy())
                this_codes.append(codes)
            
            # remainder = len(x) - n_cuts * n_samples
            i = n_cuts // batch_size
            remainder = n_cuts - i * batch_size
            # print(i, remainder, n_cuts, i * batch_size)
            if remainder > 0:
                batch = torch.from_numpy(np.stack([x[(i*batch_size+j)*n_samples:(i*batch_size+j+1)*n_samples] for j in range(remainder)], axis=0)).unsqueeze(1).pin_memory().to(device, non_blocking=True)
                _, codes = model.encode(batch)
                codes = codes.permute(0, 2, 1).cpu().detach().numpy()
                if test:
                    x_, z = model.encode(batch)
                    y = model.decode(z, x_.shape)
                    this_samples.append(y.cpu().detach().numpy())
                this_codes.append(codes)

            # pad with 0s for last section
            if len(x) - (i * batch_size + remainder) * n_samples > 0:
                # print(len(x), ' - ', (i * batch_size + remainder) * n_samples, ' == ', len(x) - (i * batch_size + remainder) * n_samples)
                batch = np.zeros((1, n_samples), dtype=np.float32)
                batch[0, :len(x) - (i * batch_size + remainder) * n_samples] = x[(i * batch_size + remainder) * n_samples:]
                batch = torch.from_numpy(batch).unsqueeze(1).pin_memory().to(device, non_blocking=True)
                _, codes = model.encode(batch)
                codes = codes.permute(0, 2, 1).cpu().detach().numpy()
                if test:
                    x_, z = model.encode(batch)
                    y = model.decode(z, x_.shape)
                    this_samples.append(y.cpu().detach().numpy())
                this_codes.append(codes)

            this_codes = np.concatenate(this_codes, axis=0)
            binary_instruments = np.zeros(26, dtype=np.uint8)
            binary_instruments[instruments] = 1
            binary_instruments = np.tile(binary_instruments, (this_codes.shape[0], this_codes.shape[1], 1))

            all_codes.append(this_codes)
            all_instruments.append(binary_instruments)

            # for testing
            if test:
                this_samples = np.concatenate(this_samples, axis=0).reshape(-1)
                print(this_codes.shape, x.shape, this_samples.shape)

                sf.write('real.wav', x, rate)
                sf.write('recon.wav', this_samples, rate)
                
                import sys
                sys.exit()
            
            if (idx + 1) % (len(paths) // total_write_batches) == 0:
                print(f'Writing batch {write_idx}...')
                all_codes = np.concatenate(all_codes, axis=0)
                print(all_codes.shape)

                all_codes = all_codes.reshape(all_codes.shape[0] * all_codes.shape[1], all_codes.shape[2])
                print(all_codes.shape)

                filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_{str(write_idx).zfill(2)}.bin')
                dtype = np.float32
                arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_codes.shape)
                arr[:] = all_codes
                arr.flush()

                all_instruments = np.concatenate(all_instruments, axis=0)
                print(all_instruments.shape)
                all_instruments = all_instruments.reshape(all_instruments.shape[0] * all_instruments.shape[1], all_instruments.shape[2])
                print(all_instruments.shape)

                filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_instruments_{str(write_idx).zfill(2)}.bin')
                dtype = np.uint8
                arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_instruments.shape)
                arr[:] = all_instruments
                arr.flush()

                write_idx += 1
                write_paths.append((filename, len(all_codes)))
                all_codes = []
                all_instruments = []

    # write the remaining batch
    print(f'Writing batch {write_idx}...')
    all_codes = np.concatenate(all_codes, axis=0)
    all_instruments = np.concatenate(all_instruments, axis=0)
    print(all_codes.shape, all_instruments.shape)

    all_codes = all_codes.reshape(all_codes.shape[0] * all_codes.shape[1], all_codes.shape[2])
    all_instruments = all_instruments.reshape(all_instruments.shape[0] * all_instruments.shape[1], all_instruments.shape[2])
    print(all_codes.shape, all_instruments.shape)

    filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_{str(write_idx).zfill(2)}.bin')
    dtype = np.float32
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_codes.shape)
    arr[:] = all_codes
    arr.flush()

    filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_instruments_{str(write_idx).zfill(2)}.bin')
    dtype = np.uint8
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_instruments.shape)
    arr[:] = all_instruments
    arr.flush()

    write_idx += 1
    write_paths.append((filename, len(all_codes)))
    all_codes = []
    all_instruments = []

## get token write paths
dtype = np.float32
write_paths = []
paths = [f'{out_prefix}_{str(i).zfill(2)}.bin' for i in range(total_write_batches + 1)]
for path in paths:
    data = np.memmap(path, dtype=np.float32, mode='r')
    data = data.reshape((-1, vae_embed_dim))
    write_paths.append((path, len(data)))

# write tokens to train.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_train.bin')
train_length = np.sum([length for path, length in write_paths[:-2]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(train_length, vae_embed_dim))
print(arr.shape)

for path, length in write_paths[:-2]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=(length, vae_embed_dim))

    arr[cur_idx:cur_idx+length] = data
    arr.flush()

    cur_idx += length

# write tokens to val.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_val.bin')
val_length = np.sum([length for path, length in write_paths[-2:]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(val_length, vae_embed_dim))
print(arr.shape)

for path, length in write_paths[-2:]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=(length, vae_embed_dim))

    arr[cur_idx:cur_idx+length] = data
    arr.flush()

    cur_idx += length

## get instrument write paths
dtype = np.uint8
write_paths = []
paths = [f'{out_prefix}_instruments_{str(i).zfill(2)}.bin' for i in range(total_write_batches + 1)]
for path in paths:
    data = np.memmap(path, dtype=dtype, mode='r')
    data = data.reshape((-1, 26))
    write_paths.append((path, len(data)))

# write instruments to instruments_train.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_instruments_train.bin')
train_length = np.sum([length for path, length in write_paths[:-2]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(train_length, 26))
print(arr.shape)

for path, length in write_paths[:-2]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=(length, 26))

    arr[cur_idx:cur_idx+length] = data
    arr.flush()

    cur_idx += length

# write instruments to instruments_val.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_instruments_val.bin')
val_length = np.sum([length for path, length in write_paths[-2:]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(val_length, 26))
print(arr.shape)

for path, length in write_paths[-2:]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=(length, 26))

    arr[cur_idx:cur_idx+length] = data
    arr.flush()

    cur_idx += length