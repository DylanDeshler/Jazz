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

device = torch.device('cuda:3')

batch_size = 32
total_write_batches = 16
max_tokens = 256

out_prefix = 'caption_embeddings'

with open('/home/dylandeshler/Jazz/preprocess/final_llm_captions.jsonl', 'r', encoding='utf-8') as f:
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
    shape=(len(captions) + 1, 3, max_tokens, model.config.d_model)
)

from collections import defaultdict
lengths = defaultdict(list)

for data_dict in tqdm(captions, desc='Calculating Token Lengths'):
    data = data_dict.get('llm_output', [])
    if isinstance(data, list):
        print('skipping bad data')
        continue
    
    short = data.get('short_caption', '')
    medium = data.get('medium_caption', '')
    long = data.get('long_caption', '')
    
    inputs = tokenizer(short, return_tensors="pt")
    if inputs['input_ids'].shape[-1] > 1:
        lengths['short'].append(inputs['input_ids'].shape[-1])
    
    inputs = tokenizer(medium, return_tensors="pt")
    if inputs['input_ids'].shape[-1] > 1:
        lengths['medium'].append(inputs['input_ids'].shape[-1])
    
    inputs = tokenizer(long, return_tensors="pt")
    if inputs['input_ids'].shape[-1] > 1:
        lengths['long'].append(inputs['input_ids'].shape[-1])

for k, v in lengths.items():
    print(f'[{k} token stats] min: {np.min(v)} mean: {np.mean(v)} std: {np.std(v)} max: {np.max(v)}')

with torch.no_grad():
    for idx, data_dict in enumerate(tqdm(captions)):
        path = data_dict['file_path']
        data = data_dict.get('llm_output', [])
        if isinstance(data, list):
            data = {'short_caption': '', 'medium_caption': '', 'long_caption': ''}
        
        short = data.get('short_caption', '')
        medium = data.get('medium_caption', '')
        long = data.get('long_caption', '')
        
        texts = [short, medium, long]
        
        inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt"
        ).to(device)
        
        outputs = model(**inputs)
        embedding_memmap[idx] = outputs.last_hidden_state.cpu().numpy()

    inputs = tokenizer(
        [''],
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt"
    ).to(device)

    outputs = model(**inputs)
    embedding_memmap[-1] = outputs.last_hidden_state.cpu().numpy()

embedding_memmap.flush()