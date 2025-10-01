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

from llama_model import ModelArgs, VQTransformer4 as Transformer

import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import glob

device = torch.device('mps')

batch_size = 2 * 12# * 5
block_size = 256
# model
rate = 16000
n_samples = rate * 1
n_fft = 512
hop_length = int(rate * 10 / 1000)
win_length = int(rate * 25 / 1000)
patch_height = 32 * 2
patch_width = 10 * 1
n_layer = 6
n_head = 6
n_embd = 384
multiple_of = 128
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

checkpoint = torch.load('/Users/dylan.d/Documents/research/music/tokenizer7/ckpt.pt')
model_args = ModelArgs(**checkpoint['model_args'])
model = Transformer(model_args).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

paths = glob.glob('/Users/dylan.d/Documents/research/music/jazz_data_16000_full_clean/*.wav')

CODEBOOK_DEPTH = 12

test = False

all_codes, all_scales = [], []
with torch.no_grad():
    for path in tqdm(paths):
        this_codes, this_scales = [], []
        if test:
            this_samples = []
        x, sr = librosa.load(path, sr=None)

        n_cuts = len(x) // n_samples
        for i in range(n_cuts // batch_size):
            batch = torch.from_numpy(np.stack([x[(i*batch_size+j)*n_samples:(i*batch_size+j+1)*n_samples] for j in range(batch_size)], axis=0)).to(device)
            codes, scales = model.encode(batch)
            codes = codes.cpu().detach().numpy()
            if test:
                this_samples.append(model.decode(*model.encode(batch)).cpu().detach().numpy())
            this_codes.append(codes)
            this_scales.append(scales)
        
        # remainder = len(x) - n_cuts * n_samples
        i = n_cuts // batch_size
        remainder = n_cuts - i * batch_size
        # print(i, remainder, n_cuts, i * batch_size)
        if remainder > 0:
            batch = torch.from_numpy(np.stack([x[(i*batch_size+j)*n_samples:(i*batch_size+j+1)*n_samples] for j in range(remainder)], axis=0)).to(device)
            codes, scales = model.encode(batch)
            codes = codes.cpu().detach().numpy()
            if test:
                this_samples.append(model.decode(*model.encode(batch)).cpu().detach().numpy())
            this_codes.append(codes)
            this_scales.append(scales)

        # pad with 0s for last section
        if len(x) - (i * batch_size + remainder) * n_samples > 0:
            # print(len(x), ' - ', (i * batch_size + remainder) * n_samples, ' == ', len(x) - (i * batch_size + remainder) * n_samples)
            batch = np.zeros((1, n_samples), dtype=np.float32)
            batch[0, :len(x) - (i * batch_size + remainder) * n_samples] = x[(i * batch_size + remainder) * n_samples:]
            batch = torch.from_numpy(batch).to(device)
            codes, scales = model.encode(batch)
            codes = codes.cpu().detach().numpy()
            if test:
                this_samples.append(model.decode(*model.encode(batch)).cpu().detach().numpy())
            this_codes.append(codes)
            this_scales.append(scales)

        # TODO: append EOS token

        this_codes = np.concatenate(this_codes, axis=0)
        # print(this_codes.shape, len(x) / n_samples * 160, len(this_codes) - len(x) / n_samples * 160)
        this_scales = torch.cat([torch.stack([t1, t2], dim=1) for t1, t2 in this_scales], dim=0)

        all_codes.append(this_codes)
        all_scales.append(this_scales)

        print(torch.cat(all_scales, dim=0)[:, 0].mean().item(), torch.cat(all_scales, dim=0)[:, 1].mean().item(), torch.cat(all_scales, dim=0)[:, 1].std().item(), torch.cat(all_scales, dim=0)[:, 1].std().item())
        
        # for testing
        if test:
            this_samples = [librosa.griffinlim(this_sample, n_iter=10, hop_length=hop_length, win_length=n_fft, n_fft=n_fft, init=None) for this_sample in this_samples]
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