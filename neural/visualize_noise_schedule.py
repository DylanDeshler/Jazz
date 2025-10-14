import numpy as np
import torch
import os

from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt

from fm import FM
from dito import DiToV4 as Tokenizer

device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

ckpt_path = os.path.join('tokenizer_low', 'ckpt.pt')
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

batch_size = 4
vae_embed_dim = 16
tokens_per_second = 32
seconds_per_tokenizer_window = 1
tokens_per_tokenizer_window = tokens_per_second * seconds_per_tokenizer_window
max_seq_len = 8 * tokens_per_second

diffusion = FM(sigma_min=1e-5, timescale=1000.0)

batch_dir = 'noise_schedule'
os.makedirs(batch_dir, exist_ok=True)

def get_batch(split='train'):
    if split == 'train':
        data = np.memmap('/home/dylan.d/research/music/Jazz/latents/low_train.bin', dtype=np.float32, mode='r', shape=(204654816, vae_embed_dim))
        idxs = torch.randint(len(data) - max_seq_len, (batch_size,))
        x = torch.from_numpy(np.stack([data[idx:idx+max_seq_len] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
        return x
    
    else:
        data = np.memmap('/home/dylan.d/research/music/Jazz/latents/low_val.bin', dtype=np.float32, mode='r', shape=(4446944, vae_embed_dim))
        idxs = torch.randint(len(data) - max_seq_len, (batch_size,))
        x = torch.from_numpy(np.stack([data[idx:idx+max_seq_len] for idx in idxs], axis=0)).pin_memory().to(device, non_blocking=True)
        return x

x = get_batch()
B, L, D = x.shape

out = []
steps = 10
ts = torch.linspace(0, 1, steps=steps)
for t in ts:
    x_t, noise = diffusion.add_noise(x, t.unsqueeze(0).expand(B, -1))
    out.append(x_t)
x = torch.cat(x_t, dim=0)
print(x.shape)

n_cuts = L // tokens_per_tokenizer_window
batches = []
for cut in tqdm(range(n_cuts), desc='Decoding'):
    batch = x[:, cut * tokens_per_tokenizer_window: (cut + 1) * tokens_per_tokenizer_window].permute(0, 2, 1)
    batches.append(tokenizer.decode(batch, shape=(1, 16384 * seconds_per_tokenizer_window)))
x = torch.cat(batches, dim=-1).chunk(steps, dim=0)

for j, chunk in enumerate(x):
    chunk = chunk.cpu().detach().numpy()
    for i in range(len(chunk)):
        sf.write(os.path.join(batch_dir, f't={ts[j].item():.3f}_sample_{i}.wav'), chunk[i].squeeze(), 16000)