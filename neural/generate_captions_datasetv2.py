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

NUM_VARS = 5 + 1        # 6
NUM_TIERS = 3           # 3
batch_size = 32
total_write_batches = 16
max_tokens = 256

out_prefix = 'caption_embeddings_shuffled'

with open('/home/dylandeshler/Jazz/preprocess/final_llm_captions_expanded.jsonl', 'r', encoding='utf-8') as f:
    captions = [json.loads(line) for line in f]
num_captions = len(captions)
print(f'Found {num_captions} captions')

print("Loading T5-large model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5EncoderModel.from_pretrained("t5-large")
model.to(device)
model.eval()

# -------------------------------------------------------------------------
# CALCULATE NEW SHAPE & GENERATE SHUFFLED INDEX MAPS (Captions Only)
# -------------------------------------------------------------------------
# Exclude the null embedding from the main flattened count
total_matrices = num_captions * NUM_TIERS * NUM_VARS  # N * 3 * 6
orig_sub_shape = (num_captions, NUM_TIERS, NUM_VARS)

print("Generating deterministic shuffle indices...")
# Lock the seed so you can always mathematically reproduce this mapping sequence
rng = np.random.default_rng(seed=42)
shuffled_indices = rng.permutation(total_matrices)

print("Generating inverse lookup tables for indexing tracking...")
inverse_mapping = np.argsort(shuffled_indices)

# Save the index arrays to disk as lightweight lookups for later tracking
mapping_meta_path = os.path.join(os.path.dirname(__file__), f'{out_prefix}_index_map.npz')
np.savez_compressed(
    mapping_meta_path, 
    shuffled_indices=shuffled_indices, 
    inverse_mapping=inverse_mapping,
    orig_sub_shape=orig_sub_shape
)
print(f"Saved indexing tracking metadata to {mapping_meta_path}")

# Initialize the flat memmap directly on disk (Exactly 40138 * 3 * 6 rows)
embedding_memmap = np.memmap(
    os.path.join(os.path.dirname(__file__), f'{out_prefix}.bin'), 
    dtype=np.float32, 
    mode='w+', 
    shape=(total_matrices, max_tokens, model.config.d_model)
)

# -------------------------------------------------------------------------
# GENERATION LOOP WITH DIRECT SHUFFLED WRITING
# -------------------------------------------------------------------------
with torch.no_grad():
    for idx, data_dict in enumerate(tqdm(captions, desc="Processing Embeddings")):
        path = data_dict.get('file_path', '')
        data = data_dict.get('llm_output', {})
        
        if isinstance(data, list) or not data:
            data = {}
        
        short_list = data.get('short_caption', [''] * NUM_VARS)
        medium_list = data.get('medium_caption', [''] * NUM_VARS)
        long_list = data.get('long_caption', [''] * NUM_VARS)
        
        short_list = (short_list + [''] * NUM_VARS)[:NUM_VARS]
        medium_list = (medium_list + [''] * NUM_VARS)[:NUM_VARS]
        long_list = (long_list + [''] * NUM_VARS)[:NUM_VARS]
        
        texts = short_list + medium_list + long_list
        
        inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt"
        ).to(device)
        
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state.cpu().numpy()  # (18, 256, 1024)
        
        # Scatter the 18 generated matrices into their randomly assigned positions
        for tier_j in range(NUM_TIERS):
            for var_k in range(NUM_VARS):
                # Calculate what the row index would have been in the old layout
                orig_flat_idx = np.ravel_multi_index((idx, tier_j, var_k), orig_sub_shape)
                
                # Look up its new randomized row index destination
                new_dest_row = inverse_mapping[orig_flat_idx]
                
                # Fetch matrix index from current batch (0 to 17)
                batch_matrix_idx = (tier_j * NUM_VARS) + var_k
                
                # Direct assignment right to its randomized home on disk
                embedding_memmap[new_dest_row] = hidden_states[batch_matrix_idx]

    # -------------------------------------------------------------------------
    # GENERATE AND SAVE THE NULL EMBEDDING SEPARATELY
    # -------------------------------------------------------------------------
    print("\nGenerating separate null embedding...")
    inputs = tokenizer(
        [''],
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt"
    ).to(device)

    outputs = model(**inputs)
    # Shape: (256, 1024)
    null_state = outputs.last_hidden_state.cpu().numpy()[0] 
    
    null_embed_path = os.path.join(os.path.dirname(__file__), 'null_embedding.npy')
    np.save(null_embed_path, null_state)
    print(f"Saved standalone null embedding to {null_embed_path}")

embedding_memmap.flush()
print("Finished generation! Binary file on disk matches your exact reshaped size and is fully randomized.")