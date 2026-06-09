import numpy as np
from tqdm import tqdm
import os

def main():
    # -------------------------------------------------------------------------
    # 1. SETUP PATHS AND SHAPES
    # -------------------------------------------------------------------------
    src_bin_path = '/data/binaries/caption_embeddings_expanded.bin'
    dst_bin_path = '/data/binaries/caption_embeddings_shuffled.bin'
    
    orig_shape = (40138, 3, 6, 256, 1024)
    orig_sub_shape = orig_shape[:3] # (40138, 3, 6)
    
    total_matrices = int(np.prod(orig_sub_shape))  # 722,484
    matrix_shape = (256, 1024)
    
    # Final desired shape on disk
    shuffled_shape = (total_matrices, 256, 1024)
    
    if not os.path.exists(src_bin_path):
        raise FileNotFoundError(f"Could not find the source file at: {src_bin_path}")

    # -------------------------------------------------------------------------
    # 2. OPEN SOURCE AND DESTINATION MEMMAPS
    # -------------------------------------------------------------------------
    print("--> Mapping source 5D array (Read-Only)...")
    data_orig = np.memmap(src_bin_path, dtype=np.float32, mode='r', shape=orig_shape)
    
    # Flatten the first 3 dimensions conceptually without moving data yet
    data_orig_flat = data_orig.reshape(total_matrices, 256, 1024)

    print("--> Creating destination shuffled array on disk...")
    # 'w+' creates or overwrites the file, allocating the required size automatically
    data_shuffled = np.memmap(dst_bin_path, dtype=np.float32, mode='w+', shape=shuffled_shape)

    # -------------------------------------------------------------------------
    # 3. GENERATE DETERMINISTIC INDEX MAPPING
    # -------------------------------------------------------------------------
    print("--> Generating deterministic shuffle indices...")
    rng = np.random.default_rng(seed=42)
    
    # shuffled_indices[new_index] -> gives the original_flat_index
    shuffled_indices = rng.permutation(total_matrices)
    
    # Create the inverse map so you can instantly map a 5D coordinate to its new home
    print("--> Generating inverse index lookup table...")
    inverse_mapping = np.argsort(shuffled_indices)

    # -------------------------------------------------------------------------
    # 4. CHUNKED STREAMING (THE SHUFFLE PROCESS)
    # -------------------------------------------------------------------------
    print("\n--> Starting physical shuffling process...")
    # Adjust chunk_size depending on available RAM. 5,000 matrices ~ 5.2 GB of RAM during transit.
    chunk_size = 500 
    
    for start_idx in tqdm(range(0, total_matrices, chunk_size)):
        end_idx = min(start_idx + chunk_size, total_matrices)
        
        # 1. Grab the cluster of original flat indices destined for this chunk
        chunk_orig_indices = shuffled_indices[start_idx:end_idx]
        
        # 2. Pull randomly distributed matrices from source into RAM
        # (NumPy handles the advanced indexing cleanly here)
        buffer = data_orig_flat[chunk_orig_indices]
        
        # 3. Write them sequentially into the new file
        data_shuffled[start_idx:end_idx] = buffer
        
        # Print progress
        progress = (end_idx / total_matrices) * 100
        print(f"Progress: {progress:.2f}% | Processed {end_idx}/{total_matrices} matrices", end='\r')
        
    print("\n--> Shuffling complete! Flushing changes to disk...")
    data_shuffled.flush()

    # -------------------------------------------------------------------------
    # 5. BIDIRECTIONAL INDEX MAPPING FUNCTIONS
    # -------------------------------------------------------------------------
    def original_to_new_index(i, j, k, d4, d5):
        """ Maps an original 5D coordinate to the new shuffled 3D index """
        orig_flat_idx = np.ravel_multi_index((i, j, k), orig_sub_shape)
        new_flat_idx = inverse_mapping[orig_flat_idx]
        return (new_flat_idx, d4, d5)

    def new_to_original_index(new_flat_idx, d4, d5):
        """ Maps a shuffled 3D index back to its original 5D coordinate """
        orig_flat_idx = shuffled_indices[new_flat_idx]
        i, j, k = np.unravel_index(orig_flat_idx, orig_sub_shape)
        return (i, j, k, d4, d5)

    # -------------------------------------------------------------------------
    # 6. VERIFICATION TEST
    # -------------------------------------------------------------------------
    print("\n--- Running Verification Check ---")
    
    # Pick a random point in our new array
    test_new_idx = (500000, 12, 45) # (flat_index, d4, d5)
    
    # Find where it came from
    orig_coord = new_to_original_index(*test_new_idx)
    print(f"New Index {test_new_idx} points back to Original Coordinate: {orig_coord}")
    
    # Map back forward to make sure round-trip works
    round_trip = original_to_new_index(*orig_coord)
    
    # Read values from both files
    val_from_shuffled = data_shuffled[test_new_idx]
    val_from_original = data_orig[orig_coord]
    
    print(f"Value in Shuffled File: {val_from_shuffled}")
    print(f"Value in Original File: {val_from_original}")
    
    # Asserts
    assert round_trip == test_new_idx, "Index mapping round-trip mismatch!"
    assert val_from_shuffled == val_from_original, "Data integrity mismatch between files!"
    
    print("\n[SUCCESS] New binary file successfully created.")
    print("Contiguous slices are now inherently randomized, and index mapping works flawlessly.")

if __name__ == '__main__':
    main()