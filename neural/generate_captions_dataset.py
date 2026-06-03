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

from transformers import T5Tokenizer, T5EncoderModel

device = torch.device('cuda')

batch_size = 32
total_write_batches = 16
max_tokens = 256

out_prefix = 'caption_embeddings'

with open('/home/dylandeshler/Jazz/preprocess/final_llm_captions.jsonl', 'r', encoding='utf-8') as f:
    captions = [json.loads(line) for line in f]
print(len(captions))

print("Loading T5-large model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5EncoderModel.from_pretrained("t5-large")
model.to(device)
model.eval()

embed_dim = model.config.d_model

shape = (num_samples, len(caption_types), max_tokens, embed_dim)
# embedding_memmap = np.memmap(output_memmap_path, dtype=np.float32, mode='w+', shape=shape)

if True:

    write_idx = 0
    write_paths = []

    all_codes = []
    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(captions)):
            path = data_dict['file_path']
            short = data_dict['llm_output']['short_caption']
            medium = data_dict['llm_output']['medium_caption']
            long = data_dict['llm_output']['long_caption']
            
            texts = [short, medium, long]
            
            inputs = tokenizer(
                flat_texts,
                padding="max_length",
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt"
            ).to(device)
            
            # Forward pass through the text encoder
            outputs = model(**inputs)
            # Extracted last hidden state shape: [Batch * 3, max_tokens, 1024]
            embeddings = outputs.last_hidden_state.cpu().numpy()
            print(embeddings.shape)
            import sys
            sys.exit(1)
            
            # Reshape back to unflattened batch space: [Batch, 3, max_tokens, 1024]
            reshaped_embeddings = embeddings.reshape(-1, len(caption_types), max_tokens, embed_dim)
            
            # Write directly to the memory-mapped file space
            end_idx = start_idx + len(batch_list)
            embedding_memmap[start_idx:end_idx] = reshaped_embeddings
                
                this_codes = codes.permute(0, 2, 1).cpu().detach().numpy() # (B, n_queries, vae_embed_dim)

                all_codes.append(this_codes)
            
            if (idx + 1) % (len(paths) // total_write_batches) == 0:
                print(f'Writing batch {write_idx}...')
                all_codes = np.concatenate(all_codes, axis=0)
                print(all_codes.shape)
                filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_{str(write_idx).zfill(2)}.bin')
                dtype = np.float32
                arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_codes.shape)
                arr[:] = all_codes
                arr.flush()
                
                all_bpms = np.concatenate(all_bpms, axis=0)
                print(all_bpms.shape)
                filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_bpm_{str(write_idx).zfill(2)}.bin')
                dtype = np.float32
                arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_bpms.shape)
                arr[:] = all_bpms
                arr.flush()

                write_idx += 1
                write_paths.append((filename, len(all_codes)))
                all_codes = []
                all_bpms = []
    
    # write the remaining batch
    print(f'Writing batch {write_idx}...')
    all_codes = np.concatenate(all_codes, axis=0)
    print(all_codes.shape)
    filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_{str(write_idx).zfill(2)}.bin')
    dtype = np.float32
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_codes.shape)
    arr[:] = all_codes
    arr.flush()
    
    all_bpms = np.concatenate(all_bpms, axis=0)
    print(all_bpms.shape)
    filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_bpm_{str(write_idx).zfill(2)}.bin')
    dtype = np.float32
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=all_bpms.shape)
    arr[:] = all_bpms
    arr.flush()

    write_idx += 1
    write_paths.append((filename, len(all_codes)))
    all_codes = []

## get token write paths
dtype = np.float32
write_paths = []
paths = [f'{out_prefix}_{str(i).zfill(2)}.bin' for i in range(total_write_batches + 1)]
for path in paths:
    data = np.memmap(path, dtype=np.float32, mode='r')
    data = data.reshape((-1, n_queries, vae_embed_dim))
    write_paths.append((path, data.shape))

# write tokens to train.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_train.bin')
train_length = np.sum([shape[0] for path, shape in write_paths[:-2]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(train_length, n_queries, vae_embed_dim))
print(arr.shape)

for path, shape in write_paths[:-2]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)

    arr[cur_idx:cur_idx+shape[0]] = data
    arr.flush()

    cur_idx += shape[0]

# write tokens to val.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_val.bin')
val_length = np.sum([shape[0] for path, shape in write_paths[-2:]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(val_length, n_queries, vae_embed_dim))
print(arr.shape)

for path, shape in write_paths[-2:]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)

    arr[cur_idx:cur_idx+shape[0]] = data
    arr.flush()

    cur_idx += shape[0]

## get BPM write paths
dtype = np.float32
write_paths = []
paths = [f'{out_prefix}_bpm_{str(i).zfill(2)}.bin' for i in range(total_write_batches + 1)]
for path in paths:
    data = np.memmap(path, dtype=np.float32, mode='r')
    write_paths.append((path, data.shape))

# write BPM to train.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_bpm_train.bin')
train_length = np.sum([shape[0] for path, shape in write_paths[:-2]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(train_length,))
print(arr.shape)

for path, shape in write_paths[:-2]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)

    arr[cur_idx:cur_idx+shape[0]] = data
    arr.flush()

    cur_idx += shape[0]

# write BPM to val.bin
cur_idx = 0
filename = os.path.join(os.path.dirname(__file__), f'{out_prefix}_bpm_val.bin')
val_length = np.sum([shape[0] for path, shape in write_paths[-2:]])
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(val_length,))
print(arr.shape)

for path, shape in write_paths[-2:]:
    data = np.memmap(path, dtype=dtype, mode='r', shape=shape)

    arr[cur_idx:cur_idx+shape[0]] = data
    arr.flush()

    cur_idx += shape[0]