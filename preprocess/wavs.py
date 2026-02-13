import torch
import torchaudio
import soundfile as sf
import numpy as np
import os
from tqdm import tqdm  # For progress bar

def create_resampled_memmap(wav_files, output_filename, target_sr=16000, force_mono=True):
    """
    Resamples a list of wav files to a target sample rate and consolidates them 
    into a single continuous numpy memmap.
    
    Args:
        wav_files (list): List of file paths to .wav files.
        output_filename (str): Path to save the output .dat file.
        target_sr (int): Target sample rate (default 16000).
        force_mono (bool): If True, mixes stereo to mono.
    """
    if not wav_files:
        raise ValueError("The list of wav files is empty.")

    print(f"--- Pass 1: Calculating total size for {len(wav_files)} files ---")
    
    total_samples = 0
    file_offsets = [] # Store (filename, start_index, num_samples) for Pass 2

    # Pass 1: fast scan of metadata
    for f in tqdm(wav_files, desc="Scanning Metadata"):
        try:
            # torchaudio.info is much faster than loading the file
            info = sf.info(f)
            orig_sr = info.samplerate
            orig_frames = info.frames
            channels = info.channels

            # Calculate new duration
            if orig_sr != target_sr:
                # Calculate integer number of samples after resampling
                # floor(frames * target / source) is standard for resampling logic
                new_frames = int(orig_frames * target_sr / orig_sr)
            else:
                new_frames = orig_frames
            
            # Record where this file will go
            file_offsets.append((f, total_samples, new_frames, orig_sr, channels))
            total_samples += new_frames
            
        except Exception as e:
            print(f"Skipping corrupt file {f}: {e}")

    # Define shape
    # We default to float32 because resampling usually outputs float
    if force_mono:
        final_shape = (total_samples,)
    else:
        # Warning: If you don't force mono, this assumes ALL files have same channel count
        # taking the channel count of the first file as truth
        final_shape = (total_samples, file_offsets[0][4])

    print(f"\nTotal Target Samples: {total_samples}")
    print(f"Memmap Shape: {final_shape}")
    print(f"Memmap Size (GB): {total_samples * 4 / (1024**3):.2f} GB")

    # Create the file on disk
    fp = np.memmap(output_filename, dtype=np.float32, mode='w+', shape=final_shape)

    print(f"\n--- Pass 2: Resampling and Writing ---")
    
    # Pass 2: Processing
    # We define resamplers on the fly or cache them if specific SR pairs are common
    # but for simplicity/robustness we use the functional API here.
    
    for f_path, start_idx, num_samples, orig_sr, orig_channels in tqdm(file_offsets, desc="Processing"):
        
        # Load audio
        waveform, sr = sf.read(f_path)
        waveform = torch.from_numpy(waveform)
        
        # Mix to Mono if requested
        if force_mono and waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=False)
        
        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)

        # Ensure we strictly match the pre-calculated length
        # (Resampling can sometimes be off by 1 sample due to rounding)
        current_frames = waveform.shape[0]
        
        if current_frames != num_samples:
            print('AHHH ERROR')
            # Pad or Trim to match the space we reserved
            if current_frames > num_samples:
                waveform = waveform[:num_samples]
            else:
                padding = num_samples - current_frames
                waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Write to memmap
        # .squeeze() to drop the channel dim if it's 1 (mono) and our shape is 1D
        data_np = waveform.numpy().squeeze()
        
        # Assign to slice
        fp[start_idx : start_idx + num_samples] = data_np

    # Flush changes
    fp.flush()
    print(f"Done! Saved to {output_filename}")
    return fp

if __name__ == "__main__":
    import glob
    
    wavs = glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean/*.wav')
    print(f'Found {len(wavs)} wavs')
    output_path = "/home/dylan.d/research/music/Jazz/wavs_16khz.bin"
    mmap = create_resampled_memmap(wavs, output_path, target_sr=16000, force_mono=True)

