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
import zarr

from transformers import T5Tokenizer, T5EncoderModel

device = torch.device('cuda:2')

NUM_VARS = 5 + 1
NUM_TIERS = 3
total_write_batches = 16
max_tokens = 256

model_id = "google/t5-v1_1-xxl" #t5-large
print(f"Loading {model_id} model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5EncoderModel.from_pretrained(model_id)
model.to(device)
model.eval()
print(f'Hidden dim {model.config.d_model}')

# Configuration prefixes matching your audio script
text_prefix = f"caption_embeddings_expanded_shuffled_{model_id.split('-')[-1]}"

dir_path = '/data/binaries'

# -------------------------------------------------------------------------
# 1. LOAD AUDIO METADATA SPLIT DICTIONARIES
# -------------------------------------------------------------------------
try:
    with open('/data/binaries/low_large_24576_subset_chroma_rms_density_zcr_flatness_train_map.json', 'r') as f:
        audio_train_map = json.load(f)
    with open('/data/binaries/low_large_24576_subset_chroma_rms_density_zcr_flatness_val_map.json', 'r') as f:
        audio_val_map = json.load(f)
except FileNotFoundError as e:
    raise FileNotFoundError("Please run the audio metadata script first to generate the train/val json maps!") from e

# Load all captions
with open('/home/dylandeshler/Jazz/preprocess/final_llm_captions_expanded.jsonl', 'r', encoding='utf-8') as f:
    all_captions = [json.loads(line) for line in f]
print(f'Found {len(all_captions)} raw captions')

# 2. GROUP CAPTIONS BY TRAIN/VAL CODES VIA DIRECT PATH MATCHING
train_captions = []
val_captions = []

for cap in all_captions:
    path = cap.get('file_path', '')
    if path in audio_train_map:
        train_captions.append((cap, path))
    elif path in audio_val_map:
        val_captions.append((cap, path))

print(f"Matched Captions Split -> Train: {len(train_captions)} songs, Val: {len(val_captions)} songs")

# -------------------------------------------------------------------------
# 3. RESOURCE ALLOCATION FUNCTION (ZARR V3 COMPATIBLE)
# -------------------------------------------------------------------------
def setup_split_resources(split_name, split_list):
    num_songs = len(split_list)
    total_matrices = num_songs * NUM_TIERS * NUM_VARS # Total rows for this split binary
    orig_sub_shape = (num_songs, NUM_TIERS, NUM_VARS)
    
    # Independent deterministic seeds per split
    rng = np.random.default_rng(seed=42 if split_name == 'train' else 24)
    shuffled_indices = rng.permutation(total_matrices)
    inverse_mapping = np.argsort(shuffled_indices)
    
    # Zarr v3 directory path
    zarr_path = os.path.join(dir_path, f'{text_prefix}_{split_name}.zarr')
    
    # Zarr v3 requires explicit dimensions and standard dtypes instead of object arrays.
    # Chunking by 1 row ensures fast retrieval and random shuffling during training batches.
    zarr_arr = zarr.open(
        store=zarr_path, 
        mode='w', 
        shape=(total_matrices, max_tokens, model.config.d_model), 
        chunks=(1, max_tokens, model.config.d_model),
        dtype='float16'
    )
    
    return zarr_arr, shuffled_indices, inverse_mapping, orig_sub_shape

print("Initializing dynamic text binaries...")
train_zarr, train_shuffled, train_inverse, train_shape = setup_split_resources('train', train_captions)
val_zarr, val_shuffled, val_inverse, val_shape = setup_split_resources('val', val_captions)

# Track mappings linking song paths directly to text binary coordinates
text_train_file_map = {}
text_val_file_map = {}

# -------------------------------------------------------------------------
# 4. CORE PROCESSING AND SHUFFLED MATRIX TRANSFORMATION
# -------------------------------------------------------------------------
def process_split_embeddings(split_list, zarr_arr, inverse_mapping, orig_sub_shape, file_tracking_map):
    with torch.no_grad():
        for song_idx, (data_dict, path) in enumerate(tqdm(split_list)):
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
            
            # Using max_length padding to map onto a clean Zarr array.
            # Trailing zeros compress down to essentially nothing on disk under LZ4.
            inputs = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state.cpu().numpy().astype(np.float16)
            
            # Scatter each matrix row individually into its global shuffled home on disk
            assigned_rows = []
            for tier_j in range(NUM_TIERS):
                for var_k in range(NUM_VARS):
                    # Compute what the index sequence would be in un-shuffled space
                    orig_flat_idx = np.ravel_multi_index((song_idx, tier_j, var_k), orig_sub_shape)
                    new_dest_row = inverse_mapping[orig_flat_idx]
                    
                    batch_matrix_idx = (tier_j * NUM_VARS) + var_k
                    
                    zarr_arr[new_dest_row] = hidden_states[batch_matrix_idx]
                    assigned_rows.append(int(new_dest_row))
            
            # Keep note of where this song's 18 rows ended up inside the text binary file
            file_tracking_map[path] = {
                "original_song_index": song_idx,
                "shuffled_flat_rows": sorted(assigned_rows)
            }

print("\n--> Generating and Shuffling TRAINING Text Embeddings...")
process_split_embeddings(train_captions, train_zarr, train_inverse, train_shape, text_train_file_map)

print("\n--> Generating and Shuffling VALIDATION Text Embeddings...")
process_split_embeddings(val_captions, val_zarr, val_inverse, val_shape, text_val_file_map)

# -------------------------------------------------------------------------
# 5. GENERATE STANDALONE NULL EMBEDDING 
# -------------------------------------------------------------------------
print("\nGenerating separate null embedding...")
with torch.no_grad():
    inputs = tokenizer([''], padding="max_length", truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
    outputs = model(**inputs)
    null_state = outputs.last_hidden_state.cpu().numpy()[0].astype(np.float16)
np.save(os.path.join(dir_path, 'null_embedding.npy'), null_state)

# -------------------------------------------------------------------------
# 6. SAVE COMPREHENSIVE INDEX TRACKING DICTIONARIES
# -------------------------------------------------------------------------
meta_save_path = os.path.join(dir_path, f'{text_prefix}_split_metadata.pkl')
metadata_payload = {
    "train": {
        "shuffled_indices": train_shuffled,
        "inverse_mapping": train_inverse,
        "orig_sub_shape": train_shape,
        "audio_file_to_text_rows": text_train_file_map
    },
    "val": {
        "shuffled_indices": val_shuffled,
        "inverse_mapping": val_inverse,
        "orig_sub_shape": val_shape,
        "audio_file_to_text_rows": text_val_file_map
    }
}

with open(meta_save_path, 'wb') as f:
    pickle.dump(metadata_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\n[SUCCESS] Split processing completed.")
print(f"Tracking coordinates saved to: {meta_save_path}")