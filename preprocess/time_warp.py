import os
import glob
import librosa
import numpy as np
import pyrubberband as pyrb
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

TARGET_SR = 16000
TARGET_SAMPLES = 24576 
NUM_WORKERS = 32

def parse_beat_file(beat_path):
    """
    Parses the beat_this output file.
    Expected format per line: <timestamp> <beat_number>
    
    Returns a list of dictionaries: {'time': float, 'beat': int}
    """
    beat_data = []
    with open(beat_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    ts = float(parts[0])
                    # specific beat number (1, 2, 3, 4)
                    # Default to 0 if not present
                    bn = 0 
                    if len(parts) >= 2:
                        try:
                            bn = int(float(parts[1]))
                        except ValueError:
                            pass
                    
                    beat_data.append({'time': ts, 'beat': bn})
                except ValueError:
                    continue
    
    return beat_data

def get_time_signature(beat_data):
    """
    Estimates the numerator of the time signature (e.g., 4 for 4/4) 
    based on the most frequent measure length found in the file.
    """
    if not beat_data:
        return 0
        
    beat_nums = np.array([b['beat'] for b in beat_data])
    downbeat_indices = np.where(beat_nums == 1)[0]
    
    if len(downbeat_indices) < 2:
        return 0
        
    beats_per_measure = []
    for i in range(len(downbeat_indices) - 1):
        start = downbeat_indices[i]
        end = downbeat_indices[i+1]
        count = end - start
        beats_per_measure.append(count)
        
    if not beats_per_measure:
        return 0
        
    vals, counts = np.unique(beats_per_measure, return_counts=True)
    mode_time_sig = vals[np.argmax(counts)]
    return mode_time_sig

def process_measure(y):
    current_samples = len(y)
    stretch_factor = current_samples / TARGET_SAMPLES
    
    y_warped = pyrb.time_stretch(y, TARGET_SR, stretch_factor)

    if len(y_warped) > TARGET_SAMPLES:
        y_warped = y_warped[:TARGET_SAMPLES]
    elif len(y_warped) < TARGET_SAMPLES:
        y_warped = np.pad(y_warped, (0, TARGET_SAMPLES - len(y_warped)))
        
    return y_warped

def generate_audio_measures(audio_path, beat_path):
    """
    Clips full measures (Downbeat '1' to the next Downbeat '1').
    """
    beat_data = parse_beat_file(beat_path)
    
    downbeat_indices = [i for i, b in enumerate(beat_data) if b['beat'] == 1]
    
    if len(downbeat_indices) < 10:
        return
    
    y, sr = librosa.load(audio_path, sr=None)
    assert sr == TARGET_SR
    
    if len(y) < 1000:
        return
    
    if np.max(np.abs(y)) < 0.001:
        return

    res = []
    for i in range(len(downbeat_indices) - 1):
        start_idx = downbeat_indices[i]
        end_idx = downbeat_indices[i+1]
        
        t_start = beat_data[start_idx]['time']
        t_end = beat_data[end_idx]['time']
        
        frame_start = int(t_start * sr)
        frame_end = int(t_end * sr)
        
        res.append(process_measure(y[frame_start:frame_end]))
    res = np.stack(res, axis=0)
    out_path = audio_path.replace('.wav', '.npy')
    
    print(res.shape, out_path)
    np.save(out_path, res.astype(np.float16))

def main():
    print("Gathering files...")
    audio_paths = sorted(glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean/*.wav'))
    beat_paths = sorted(glob.glob('/home/dylan.d/research/music/Jazz/jazz_data_16000_full_clean_beats/*.beats'))

    valid_audio, valid_beats = [], []
    print(f"Filtering for songs with Time Signature: 4/4 ...")
    for audio_p, beat_p in zip(audio_paths, beat_paths):
        beat_data = parse_beat_file(beat_p)
        detected_sig = get_time_signature(beat_data)
        
        if detected_sig == 4:
            valid_audio.append(audio_p)
            valid_beats.append(beat_p)

    print(f"Found {len(valid_beats)} matching songs (out of {len(audio_paths)} total).")
    audio_paths = valid_audio
    beat_paths = valid_beats
    del valid_audio
    del valid_beats

    assert len(audio_paths) == len(beat_paths)
    
    tasks = []
    for audio_path, beat_path in zip(audio_paths, beat_paths):
        tasks.append(audio_path, beat_path)
        
    print(f"Found {len(tasks)} files. Processing with {NUM_WORKERS} cores...")
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(generate_audio_measures, tasks), total=len(tasks)))
        
    print("Done!")

if __name__ == "__main__":
    main()