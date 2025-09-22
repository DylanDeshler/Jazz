import os
import time
import math
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

batch_size = 128
rate = 16000
n_samples = rate * 4

ckpt_path = os.path.join('tokenizer_high8_long', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
tokenizer_args = checkpoint['model_args']

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

test = False

all_codes = []
with torch.no_grad():
    for path in tqdm(paths):
        this_codes = []
        if test:
            this_samples = []
        x, sr = librosa.load(path, sr=None)

        n_cuts = len(x) // n_samples
        for i in range(n_cuts // batch_size):
            batch = torch.from_numpy(np.stack([x[(i*batch_size+j)*n_samples:(i*batch_size+j+1)*n_samples] for j in range(batch_size)], axis=0)).to(device)
            print(batch.shape)
            _, codes = model.encode(batch)
            codes = codes.cpu().detach().numpy()
            if test:
                this_samples.append(model.decode(model.encode(batch)[1]).cpu().detach().numpy())
            this_codes.append(codes)
        
        # remainder = len(x) - n_cuts * n_samples
        i = n_cuts // batch_size
        remainder = n_cuts - i * batch_size
        # print(i, remainder, n_cuts, i * batch_size)
        if remainder > 0:
            batch = torch.from_numpy(np.stack([x[(i*batch_size+j)*n_samples:(i*batch_size+j+1)*n_samples] for j in range(remainder)], axis=0)).to(device)
            print(batch.shape)
            _, codes = model.encode(batch)
            codes = codes.cpu().detach().numpy()
            if test:
                this_samples.append(model.decode(model.encode(batch)[1]).cpu().detach().numpy())
            this_codes.append(codes)

        # pad with 0s for last section
        if len(x) - (i * batch_size + remainder) * n_samples > 0:
            # print(len(x), ' - ', (i * batch_size + remainder) * n_samples, ' == ', len(x) - (i * batch_size + remainder) * n_samples)
            batch = np.zeros((1, n_samples), dtype=np.float32)
            batch[0, :len(x) - (i * batch_size + remainder) * n_samples] = x[(i * batch_size + remainder) * n_samples:]
            batch = torch.from_numpy(batch).to(device)
            _, codes = model.encode(batch)
            codes = codes.cpu().detach().numpy()
            if test:
                this_samples.append(model.decode(model.encode(batch)[1]).cpu().detach().numpy())
            this_codes.append(codes)

        this_codes = np.concatenate(this_codes, axis=0)
        # print(this_codes.shape, len(x) / n_samples * 160, len(this_codes) - len(x) / n_samples * 160)
        this_scales = torch.cat([torch.stack([t1, t2], dim=1) for t1, t2 in this_scales], dim=0)

        all_codes.append(this_codes)

        # for testing
        if test:
            this_samples = np.concatenate(this_samples, axis=0).reshape(-1)
            print(this_codes.shape, x.shape, this_samples.shape)

            sf.write('real.wav', x, rate)
            sf.write('recon.wav', this_samples, rate)
        


# all_codes = np.concatenate(all_codes, axis=0)
# print(all_codes.shape)
# # all_codes = all_codes.reshape(all_codes.shape[0] * all_codes.shape[1], all_codes.shape[2]);print(all_codes.shape)
# filename = os.path.join(os.path.dirname(__file__), f'jazz.bin')
# dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
# arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_codes.shape)
# arr[:] = all_codes
# arr.flush()

# (64696714, 8)