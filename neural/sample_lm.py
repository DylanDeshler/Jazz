
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

device = torch.device('mps')

from llama_model import ModelArgs, VQTransformer4 as Transformer
checkpoint = torch.load('/Users/dylan.d/Documents/research/music/tokenizer7/ckpt.pt', map_location='cpu')
tokenizer_args = ModelArgs(**checkpoint['model_args'])
tokenizer = Transformer(tokenizer_args).to(device)
tokenizer.load_state_dict(checkpoint['model'])
tokenizer.eval()

rate = tokenizer_args.rate
n_fft = tokenizer_args.n_fft
win_length = tokenizer_args.win_length
hop_length = tokenizer_args.hop_length

from rq_llama_model import RQTransformer as Transformer, ModelArgs

import matplotlib.pyplot as plt
import soundfile as sf
import librosa

checkpoint = torch.load('/Users/dylan.d/Documents/research/music/rqtransformer/ckpt.pt', map_location='cpu')
args = ModelArgs(**checkpoint['model_args'])
model = Transformer(args).to(device)
unwanted_prefix = '_orig_mod.'
state_dict = checkpoint['model']
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()

out_dir = '/Users/dylan.d/Documents/research/music/rqtransformer'

batch_size = 2
block_size = args.max_seq_len

def get_batch(split='train'):
    data = np.memmap('/Users/dylan.d/Documents/research/music/Jazz/jazz.bin', mode='r', dtype=np.uint16, shape=(6534430, 40, 12))
    if split == 'train':
        ix = torch.randint(int(len(data) * 0.98) - block_size // 40 - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(np.concatenate(data[i:i+block_size // 40, :, :args.depth_seq_len], axis=0).astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(np.concatenate(data[i+1:i+1+block_size // 40, :, :args.depth_seq_len], axis=0).astype(np.int64)) for i in ix]).to(device)

        return x, y
    
    else:
        ix = torch.randint(int(len(data) * 0.98) - block_size // 40 - 1, len(data) - block_size // 40 - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(np.concatenate(data[i:i+block_size // 40, :, :args.depth_seq_len], axis=0).astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(np.concatenate(data[i+1:i+1+block_size // 40, :, :args.depth_seq_len], axis=0).astype(np.int64)) for i in ix]).to(device)

        return x, y

# -14.075304985046387 2.681837558746338 1.5652638673782349 1.5652638673782349
SCALES = -14.07530, 2.68184

@torch.no_grad()
def show_depth():
    model.eval()

    X, _ = get_batch('val')
    X = X[:, :40]

    scales = [torch.full((X.shape[0], 1, 1), fill_value=SCALES[0], device=device), torch.full((X.shape[0], 1, 1), fill_value=SCALES[1], device=device)]

    # Magnitude
    plt.figure(figsize=(24, 12))

    depths = [4, 6, 8, 12]
    for j, i in enumerate(depths):
        Y = tokenizer.decode(X[:, :, :i], scales)[0].cpu().detach().numpy()

        plt.subplot(1, len(depths), j + 1)
        mag, phase = np.abs(Y), np.angle(Y)
        img1 = librosa.display.specshow(librosa.amplitude_to_db(mag, ref=np.max),
                                sr=rate, hop_length=hop_length, y_axis='linear', x_axis='time',
                                cmap='magma')

    plt.savefig(os.path.join(out_dir, f'compare_depth.png'))
    plt.close('all')

@torch.no_grad()
def save_samples():
    model.eval()

    X, Y = get_batch('val')
    X = X[:2, :40*2]
    X = X.flatten(1)
    
    # with ctx:
    samples = model.generate(X, max_seq_len=160, filter_thres = 1, temperature = 0.5)
    B, L, D = samples.shape
    samples = samples.view(B * L // 40, 40, D)
    scales = [torch.full((samples.shape[0], 1, 1), fill_value=SCALES[0], device=device), torch.full((samples.shape[0], 1, 1), fill_value=SCALES[1], device=device)]
    samples = tokenizer.decode(samples, scales)
    samples = samples.view(B, -1, samples.shape[-2], samples.shape[-1])
    
    for i in range(min(4, X.shape[0])):
        # sample = librosa.griffinlim(tokenizer.decode(model.generate(X, max_seq_len=160).cpu().detach()).cpu().detach().numpy(), n_iter=1000, hop_length=hop_length, win_length=win_length, n_fft=n_fft, init=None)
        sample = librosa.griffinlim(samples[i].cpu().detach().numpy(), n_iter=1000, hop_length=hop_length, win_length=win_length, n_fft=n_fft)#, init=None)
        sample = sample.flatten()

        # save .wavs
        sf.write(os.path.join(out_dir, f'{i}.wav'), sample, rate)

        x = librosa.stft(sample, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        # Magnitude
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        mag, phase = np.abs(x), np.angle(x)
        img1 = librosa.display.specshow(librosa.amplitude_to_db(mag, ref=np.max),
                                sr=rate, hop_length=hop_length, y_axis='linear', x_axis='time',
                                cmap='magma')
        plt.title('Log-magnitude Spectrogram')
        plt.colorbar(img1, format="%+2.0f dB")

        freqs = librosa.fft_frequencies(sr=rate, n_fft=n_fft)
        times = librosa.times_like(x)

        # Phase
        plt.subplot(3, 1, 2)
        mag, phase = np.abs(x), np.angle(x)
        unwrapped_phase = np.unwrap(phase, axis=1)
        unwrapped_phase = np.diff(unwrapped_phase, prepend=0) * np.clip(np.abs(mag) / np.max(mag), 0.1, 1)
        librosa.display.specshow(unwrapped_phase, sr=rate, hop_length=hop_length,
                                x_axis='time', y_axis='linear', cmap='twilight_shifted')
        plt.title('Weighted Unwrapped Phase Spectrogram')
        plt.colorbar(label='Phase (radians)')

        phase_exp = 2*np.pi*np.multiply.outer(freqs,times)

        # Rainbowgram
        plt.subplot(3, 1, 3)
        mag, phase = np.abs(x), np.angle(x)
        img = librosa.display.specshow(np.diff(np.unwrap(np.angle(phase)-phase_exp, axis=1), axis=1, prepend=0),
                                cmap='hsv',
                                alpha=librosa.amplitude_to_db(mag, ref=np.max)/80 + 1,
                                y_axis='linear',
                                x_axis='time')
        # plt.facecolor('#000')
        cbar = plt.colorbar(img, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.ax.set(yticklabels=['-π', '-π/2', "0", 'π/2', 'π'])
        plt.title('Rainbowgram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        plt.savefig(os.path.join(out_dir, f'{i}.png'))
        plt.close('all')

    model.train()

if __name__ == '__main__':
    save_samples()
    # show_depth()