
import os
import time
import math
import pickle
from contextlib import nullcontext
from tqdm import tqdm
from torchinfo import summary

import numpy as np
import torch

from dit import LAM_B_2 as dit
from dito import DiTo as Tokenizer

import matplotlib.pyplot as plt
import soundfile as sf

out_dir = 'lam_dit'
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

ckpt_path = os.path.join('tokenizer10', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
tokenizer_args = checkpoint['model_args']

tokenizer = Tokenizer(**tokenizer_args).to(device)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
tokenizer.load_state_dict(state_dict)
tokenizer.eval()

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']

model = dit(**model_args)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
iter_num = checkpoint['iter_num']
tokens_trained = checkpoint['tokens']
best_val_loss = checkpoint['best_val_loss']
model.to(device)
model.eval()

data = np.memmap('/home/dylan.d/research/music/Jazz/latents/val.bin', dtype=np.float32, mode='r', shape=(6501543, 128))

with torch.no_grad():
    n_cuts = 200
    for cut in range(n_cuts):
        x = data[cut * 150: (cut + 1) * 150]
        global_indices, local_indices = model.encode_action_indices(x)

        print(global_indices.shape, local_indices.shape)