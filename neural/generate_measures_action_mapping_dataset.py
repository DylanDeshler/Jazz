import os
import math
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
import torch

from mapping import DiffusionMLP_B as net

device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto 
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

batch_size = 2**10

ckpt_path = os.path.join('style_diffusion_mapping_256top5_64_redo', 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']

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
model = torch.compile(model)

N = 4403211
data = np.memmap(f'/home/ubuntu/Data/low_measures_large_actions_256.bin', dtype=np.float16, mode='r', shape=(N, 768))
arr = np.memmap(f'/home/ubuntu/Data/low_measures_large_actions_256top5_64_redo.bin', dtype=np.float16, mode='w+', shape=(N, 768))

with torch.no_grad():
    for i in tqdm(range(N // batch_size)):
        batch = torch.from_numpy(np.stack([data[j:j+1] for j in range(i*batch_size, (i+1)*batch_size)], axis=0)).pin_memory().to(device, non_blocking=True)
        
        with ctx:
            actions = model.sample(batch, batch).squeeze(1)

        arr[i*batch_size:(i+1)*batch_size] = actions.float().cpu().detach().numpy().astype(np.float16)

arr.flush()
