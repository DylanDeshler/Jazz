import numpy as np

def restructure_binary():
    # 1. Map the original source array
    src_shape = (40138, 3, 6, 256, 1024)
    print("Mapping source binary...")
    src_data = np.memmap(
        '/data/binaries/caption_embeddings_expanded.bin', 
        dtype=np.float32, 
        mode='r', 
        shape=src_shape
    )
    
    # 2. Total unique combinations (722,484 items)
    total_combinations = src_shape[0] * src_shape[1] * src_shape[2]
    dst_shape = (total_combinations, 256, 1024)
    
    # 3. Initialize the target destination binary
    print(f"Creating restructured destination binary with shape {dst_shape}...")
    dst_data = np.memmap(
        '/data/binaries/caption_embeddings_expanded_shuffled.bin', 
        dtype=np.float32, 
        mode='w+', 
        shape=dst_shape
    )
    
    # 4. Create all valid coordinate pairs for (song, caption, variant)
    print("Generating index coordinates...")
    song_grid, cap_grid, var_grid = np.meshgrid(
        np.arange(src_shape[0]), 
        np.arange(src_shape[1]), 
        np.arange(src_shape[2]), 
        indexing='ij'
    )
    
    # Flatten coordinates to shape (722484,)
    song_coords = song_grid.ravel()
    cap_coords = cap_grid.ravel()
    var_coords = var_grid.ravel()
    
    # 5. Generate a random permutation to shuffle everything globally
    print("Generating shuffle permutation...")
    shuffle_indices = np.random.permutation(total_combinations)
    
    # Apply the shuffle to our coordinates
    shuffled_songs = song_coords[shuffle_indices]
    shuffled_caps = cap_coords[shuffle_indices]
    shuffled_vars = var_coords[shuffle_indices]
    
    # 6. Write to the new binary in optimized chunks to prevent RAM bloat
    print("Writing shuffled data to disk...")
    chunk_size = 1000  # Adjust based on your available CPU RAM
    for i in tqdm(range(0, total_combinations, chunk_size)):
        end_idx = min(i + chunk_size, total_combinations)
        
        # Grab coordinates for this chunk
        s_c = shuffled_songs[i:end_idx]
        c_c = shuffled_caps[i:end_idx]
        v_c = shuffled_vars[i:end_idx]
        
        # Read non-contiguously from source, write perfectly contiguously to destination
        dst_data[i:end_idx] = src_data[s_c, c_c, v_c]
        
        if i % 20000 == 0 or end_idx == total_combinations:
            print(f"Progress: {end_idx}/{total_combinations} combinations written.")
            dst_data.flush()  # Force write to disk memory

    print("Restructuring complete! Data is now physically contiguous on disk.")

if __name__ == "__main__":
    restructure_binary()