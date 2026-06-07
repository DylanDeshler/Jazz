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

device = torch.device('cuda:0')

NUM_VARS = 5 + 1
NUM_TIERS = 3
batch_size = 32
total_write_batches = 16
max_tokens = 256

out_prefix = 'caption_embeddings_expanded'

with open('/home/dylandeshler/Jazz/preprocess/final_llm_captions_expanded.jsonl', 'r', encoding='utf-8') as f:
    captions = [json.loads(line) for line in f]
print(f'Found {len(captions)} captions')

print("Loading T5-large model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5EncoderModel.from_pretrained("t5-large")
model.to(device)
model.eval()

embedding_memmap = np.memmap(
    os.path.join(os.path.dirname(__file__), f'{out_prefix}.bin'), 
    dtype=np.float32, 
    mode='w+', 
    shape=(len(captions) + 1, NUM_TIERS, NUM_VARS, max_tokens, model.config.d_model)
)

# from collections import defaultdict
# lengths = defaultdict(list)

# for data_dict in tqdm(captions, desc='Calculating Token Lengths'):
#     data = data_dict.get('llm_output', [])
#     if isinstance(data, list):
#         print('skipping bad data')
#         continue
    
#     short = data.get('short_caption', '')
#     medium = data.get('medium_caption', '')
#     long = data.get('long_caption', '')
    
#     inputs = tokenizer(short, return_tensors="pt")
#     if inputs['input_ids'].shape[-1] > 1:
#         lengths['short'].append(inputs['input_ids'].shape[-1])
    
#     inputs = tokenizer(medium, return_tensors="pt")
#     if inputs['input_ids'].shape[-1] > 1:
#         lengths['medium'].append(inputs['input_ids'].shape[-1])
    
#     inputs = tokenizer(long, return_tensors="pt")
#     if inputs['input_ids'].shape[-1] > 1:
#         lengths['long'].append(inputs['input_ids'].shape[-1])

# for k, v in lengths.items():
#     print(f'[{k} token stats] min: {np.min(v)} mean: {np.mean(v)} std: {np.std(v)} max: {np.max(v)}')

with torch.no_grad():
    for idx, data_dict in enumerate(tqdm(captions)):
        path = data_dict.get('file_path', '')
        data = data_dict.get('llm_output', {})
        
        # Safety catch for bad data formats
        if isinstance(data, list) or not data:
            data = {}
        
        # Safely get the lists, defaulting to a list of empty strings if missing
        short_list = data.get('short_caption', [''] * NUM_VARS)
        medium_list = data.get('medium_caption', [''] * NUM_VARS)
        long_list = data.get('long_caption', [''] * NUM_VARS)
        
        # Safety padding: just in case a list is shorter than NUM_VARS
        short_list = (short_list + [''] * NUM_VARS)[:NUM_VARS]
        medium_list = (medium_list + [''] * NUM_VARS)[:NUM_VARS]
        long_list = (long_list + [''] * NUM_VARS)[:NUM_VARS]
        
        # Flatten into a single list of 18 strings
        texts = short_list + medium_list + long_list
        
        inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt"
        ).to(device)
        
        # Forward pass on all 18 strings at once
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state.cpu().numpy()
        
        # NEW: Reshape the (18, max_tokens, dim) array back into (3, 6, max_tokens, dim)
        reshaped_states = hidden_states.reshape(NUM_TIERS, NUM_VARS, max_tokens, model.config.d_model)
        embedding_memmap[idx] = reshaped_states

    # Process the null embedding
    inputs = tokenizer(
        [''],
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt"
    ).to(device)

    outputs = model(**inputs)
    null_state = outputs.last_hidden_state.cpu().numpy() # Shape: (1, max_tokens, dim)
    
    # NEW: Broadcast the single null embedding to fill the entire multi-dimensional block
    null_broadcasted = np.broadcast_to(
        null_state, 
        (NUM_TIERS, NUM_VARS, max_tokens, model.config.d_model)
    )
    embedding_memmap[-1] = null_broadcasted

embedding_memmap.flush()