import os
import math
from collections import defaultdict
from contextlib import nullcontext
from itertools import combinations
from tqdm import tqdm
import argparse
import json

from scipy.sparse import csr_matrix
import numpy as np
import torch

from style import IDM_S as net

def generate_action_weights():
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto 
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    batch_size = 64#2**10

    ckpt_path = os.path.join('Style_256_adaln_1measures_bpm_S_nobias_poolfirst_norm_nohistory_1head_top5', 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model_args = checkpoint['model_args']
    vae_embed_dim = model_args['in_channels']
    spatial_window = model_args['spatial_window']
    n_encoder_chunks = model_args['n_encoder_chunks']
    n_decoder_chunks = model_args['n_decoder_chunks']
    n_chunks = n_encoder_chunks + n_decoder_chunks
    n_style_embeddings = model_args['n_style_embeddings']

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
    hidden_size = 768

    N = 4403211
    data = np.memmap('/home/ubuntu/Data/low_measures_large.bin', dtype=np.float16, mode='r', shape=(N, 48, vae_embed_dim))
    meta = np.memmap('/home/ubuntu/Data/measures_meta.bin', dtype=np.float32, mode='r', shape=(N, 2))
    arr = np.memmap(f'/home/ubuntu/Data/low_measures_large_actions_{n_style_embeddings}_stats.bin', dtype=np.float16, mode='w+', shape=(N, n_style_embeddings))

    with torch.no_grad():
        for i in tqdm(range(N // batch_size)):
            batch = torch.from_numpy(np.stack([data[j:j+n_decoder_chunks] for j in range(i*batch_size, (i+1)*batch_size)], axis=0)).pin_memory().to(device, non_blocking=True)
            bpm = torch.from_numpy(np.stack([meta[j:j+n_decoder_chunks, 1] for j in range(i*batch_size, (i+1)*batch_size)], axis=0)).pin_memory().to(device, non_blocking=True)
            
            with ctx:
                actions, weights = model.encode_actions(batch, bpm, force_manual=True, force_transfer=False, return_weights=True)
            
            arr[i*batch_size:(i+1)*batch_size] = weights.squeeze().float().cpu().detach().numpy().astype(np.float16)

    arr.flush()

def analyze():
    ckpt_path = os.path.join('Style_256_adaln_1measures_bpm_S_nobias_poolfirst_norm_nohistory_1head_top5', 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model_args = checkpoint['model_args']
    n_style_embeddings = model_args['n_style_embeddings']
    
    N = 4403211
    meta = np.memmap('/home/ubuntu/Data/measures_meta.bin', dtype=np.float32, mode='r', shape=(N, 2))
    arr = np.memmap(f'/home/ubuntu/Data/low_measures_large_actions_{n_style_embeddings}_stats.bin', dtype=np.float16, mode='r', shape=(N, n_style_embeddings))
    
    stats = {
        'mean': np.mean(arr, axis=0).tolist(),
        'median': np.median(arr, axis=0).tolist(),
        'max': np.max(arr, axis=0).tolist(),
        'min': np.min(arr, axis=0).tolist(),
        'std': np.std(arr, axis=0).tolist(),
    }
    
    joint_probs = defaultdict(float)
    for row in tqdm(arr):
        indices = np.nonzero(row)[0]
        values = row[indices]
        
        n_items = len(indices)
        if n_items < 2:
            continue
        
        max_k = min(n_items, 5) # top-k
        
        for k in range(2, max_k + 1):
            for combo in combinations(zip(indices, values), k):
                class_ids = tuple(sorted(c[0] for c in combo))
                
                weight_product = 1.0
                for _, val in combo:
                    weight_product *= val
                
                joint_probs[f'{class_ids}'] += weight_product * math.factorial(k)

    results = {k: v / N for k, v in joint_probs.items()}
    stats['probs'] = results

    print(f"Total Combinations Found: {len(results)}")
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)

    print("\n--- Top Co-occurrences ---")
    for classes, prob in sorted_results[:5]:
        print(f"Classes {classes}: P = {prob:.5f}")
    
    for i in range(n_style_embeddings):
        idxs = arr[:, i].nonzero()[0]
        
        try:
            stats[f'action {i}'] = {
                'mean': np.mean(arr[idxs, i]).item(),
                'median': np.median(arr[idxs, i]).item(),
                'max': np.max(arr[idxs, i]).item(),
                'min': np.min(arr[idxs, i]).item(),
                'std': np.std(arr[idxs, i]).item(),
            }
            
            stats[f'action {i} bpm'] = {
                'mean': np.mean(meta[idxs, 1]).item(),
                'median': np.median(meta[idxs, 1]).item(),
                'max': np.max(meta[idxs, 1]).item(),
                'min': np.min(meta[idxs, 1]).item(),
                'std': np.std(meta[idxs, 1]).item(),
                'quantile': np.quantile(meta[idxs, 1], (0.125, 0.25, 0.75, 87.5)).tolist()
            }
        except:
            stats[f'action {i}'] = {
                'mean': 0,
                'median': 0,
                'max': 0,
                'min': 0,
                'std': 0,
            }
            
            stats[f'action {i} bpm'] = {
                'mean': 0,
                'median': 0,
                'max': 0,
                'min': 0,
                'std': 0,
                'quantile': [0, 0, 0, 0]
            }
    
    with open('/home/ubuntu/Data/stats.json', 'w') as f:
        json.dump(stats, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Action Statistics")
    
    parser.add_argument("--generate", action='store_true', default=False, help="True to generate action weights")
    parser.add_argument("--analyze", action='store_true', default=False, help="True to analyze dataset action weights and bpms")
    
    args = parser.parse_args()
    
    if args.generate:
        generate_action_weights()
    if args.analyze:
        analyze()