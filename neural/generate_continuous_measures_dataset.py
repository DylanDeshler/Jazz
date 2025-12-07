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
arr = np.memmap('/home/dylan.d/research/music/Jazz/latents/low_measures_large.bin', dtype=np.float16, mode='w+', shape=(N, 48))

for i in range(N // batch_size):
    batch = torch.from_numpy(data[i*batch_size:(i+1)*batch_size]).view(batch_size, n_samples).unsqueeze(1).pin_memory().to(device, non_blocking=True)
    _, codes = model.encode(batch)
    codes = codes.permute(0, 2, 1).cpu().detach().numpy()
    arr[i*batch_size:(i+1)*batch_size] = codes.astype(np.float16)

arr.flush()
